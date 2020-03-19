from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid


def highway_layer(x, bias_init_value=-1):
    trans_gate = fluid.layers.fc(
            input=x,
            size=x.shape[1],
            act="sigmoid",
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(float(bias_init_value)))
            )

    trans = fluid.layers.fc(
            input=x,
            size=x.shape[1],
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(0.))
            )

    trans_out = fluid.layers.elementwise_mul(trans_gate, trans)
    x_out = fluid.layers.elementwise_mul(1-trans_gate, x)
    return x_out + trans_out
