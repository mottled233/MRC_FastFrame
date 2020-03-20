from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid


def highway_layer(x, name, num_flatten_dims=1, bias_init_value=-1):
    trans_gate = fluid.layers.fc(
            input=x,
            size=x.shape[-1],
            act="sigmoid",
            num_flatten_dims=num_flatten_dims,
            param_attr=fluid.ParamAttr(
                name="{}_trans_gate_w".format(name),
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="{}_trans_gate_b".format(name), initializer=fluid.initializer.Constant(float(bias_init_value)))
            )

    trans = fluid.layers.fc(
            input=x,
            size=x.shape[-1],
            num_flatten_dims=num_flatten_dims,
            param_attr=fluid.ParamAttr(
                name="{}_trans_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="{}_trans_b".format(name), initializer=fluid.initializer.Constant(0.))
            )

    trans_out = fluid.layers.elementwise_mul(trans_gate, trans)
    x_out = fluid.layers.elementwise_mul(1-trans_gate, x)
    return x_out + trans_out
