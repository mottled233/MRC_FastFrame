import paddle.fluid as fluid
import numpy as np

epsilon = 1e-9


class CapsLayer(object):
    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            if not self.with_routing:
                # input: [batch_size, 256, 33, 50]
                capsules = fluid.layers.conv2d(input=input, num_filters=self.num_outputs * self.vec_len
                                               , filter_size=self.kernel_size, stride=self.stride,
                                               padding="VALID", act="relu")
                capsules = fluid.layers.reshape(capsules, [0, -1, self.vec_len, 1], inplace=True)
                # return tensor with shape [batch_size, 8736, 8, 1]
                capsules = squash(capsules)
                return (capsules)
        if self.layer_type == 'FC':
            if self.with_routing:
                # the DigitCaps layer, a fully connected layer
                # Reshape the input into [batch_size, 8736, 1, 8, 1]
                self.input = fluid.layers.reshape(input, shape=(0, -1, 1, input.shape[-2], 1),
                                                  inplace=True)
                # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                b_IJ = fluid.layers.fill_constant(
                    shape=[get_shape(input)[0], get_shape(input)[1], self.num_outputs, 1, 1],
                    value=0, dtype="float32")
                capsules = self.routing(self.input, b_IJ, num_outputs=self.num_outputs, num_dims=self.vec_len)
                capsules = fluid.layers.squeeze(capsules, axes=[1])
            return (capsules)

    def routing(self, input, b_IJ, num_outputs=3, num_dims=16):
        ''' The routing algorithm.
        Args:
            input: A Tensor with [batch_size, num_caps_l=8736, 1, length(u_i)=8, 1]
                   shape, num_caps_l meaning the number of capsule in the layer l.
            num_outputs: the number of output capsules.
            num_dims: the number of dimensions for output capsule.
        Returns:
            A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
            representing the vector output `v_j` in the layer l+1
        Notes:
            u_i represents the vector output of capsule i in the layer l, and
            v_j the vector output of capsule j in the layer l+1.
         '''

        # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
        input_shape = get_shape(input)
        # create_parameter不支持动态的shape，先设置成ones再将其设置成能梯度下降的变量
        W = fluid.layers.ones(shape=[1, input_shape[1], num_dims * num_outputs] + input_shape[-2:],
                              dtype="float32")
        W.stop_gradient = False
        biases = fluid.layers.create_parameter(name='bias', shape=[1, 1, num_outputs, num_dims, 1], dtype="float32")

        # Eq.2, calc u_hat
        # Since tf.matmul is a time-consuming op,
        # A better solution is using element-wise multiply, reduce_sum and reshape
        # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
        # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
        # reshape to [a, c]
        input = fluid.layers.expand(input, [1, 1, num_dims * num_outputs, 1, 1])
        # assert input.get_shape() == [cfg.batch_size, 8736, 48, 8, 1]

        u_hat = fluid.layers.reduce_sum(fluid.layers.elementwise_mul(input, W), dim=3, keep_dim=True)
        u_hat = fluid.layers.reshape(u_hat, shape=[-1, input_shape[1], num_outputs, num_dims, 1])
        # need to change this!
        u_hat.stop_gradient = True
        # assert u_hat.get_shape() == [batch_size, 8736, 3, 16, 1]

        # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
        # u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
        u_hat_stopped = fluid.layers.reduce_sum(fluid.layers.elementwise_mul(input, W), dim=3, keep_dim=True)
        u_hat_stopped = fluid.layers.reshape(u_hat_stopped, shape=[-1, input_shape[1], num_outputs, num_dims, 1])
        u_hat_stopped.stop_gradient = True

        # line 3,for r iterations do
        iter_routing = 3
        for r_iter in range(iter_routing):
            # line 4:
            # => [batch_size, 8736, 3, 1, 1]
            c_IJ = fluid.layers.softmax(b_IJ, axis=2)
            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 8736, 3, 16, 1]
                s_J = fluid.layers.elementwise_mul(u_hat, c_IJ)
                # then sum in the second dim, resulting in [batch_size, 1, 3, 16, 1]
                s_J = fluid.layers.reduce_sum(s_J, dim=1, keep_dim=True) + biases
                # assert s_J.get_shape() == [cfg.batch_size, 1, num_outputs, num_dims, 1]
                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                # assert v_J.get_shape() == [cfg.batch_size, 1, 3, 16, 1]
            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = fluid.layers.elementwise_mul(u_hat_stopped, c_IJ)
                s_J = fluid.layers.reduce_sum(s_J, dim=1, keep_dim=True) + biases
                v_J = squash(s_J)
                # line 7:
                # reshape & tile v_j from [batch_size ,1, 3, 16, 1] to [batch_size, 8736, 3, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 8736, 3, 1, 1]
                v_J_tiled = fluid.layers.expand(v_J, [1, input_shape[1], 1, 1, 1])
                u_produce_v = fluid.layers.reduce_sum(u_hat_stopped * v_J_tiled, dim=3, keep_dim=True)
                # assert u_produce_v.get_shape() == [cfg.batch_size, 8736, 3, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v
        return (v_J)


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = fluid.layers.reduce_sum(fluid.layers.square(vector), -2, keep_dim=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / fluid.layers.sqrt(vec_squared_norm + epsilon)
    vec_squashed = vector * scalar_factor  # element-wise
    return (vec_squashed)


def get_shape(inputs):
    static_shape = list(inputs.shape)
    dynamic_shape = fluid.layers.shape(inputs)
    shape = []
    for i, dim in enumerate(static_shape):
        dim = dim if (dim is not None and dim != -1) else dynamic_shape[i]
        shape.append(dim)
    return shape
