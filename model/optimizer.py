import paddle
import paddle.fluid as fluid


def get_optimizer(learning_rate, args):
    """
    根据参数返回对应的优化器
    :param learning_rate: 学习率，数字或变量（vars）
    :param args: 参数集合
    :param regularization: 正则化设定
    :return: 优化器
    """
    OPTIMIZER = args["optimizer"]
    GRADIENT_CLIP = args['gradient_clip']
    CLIP_NORM = args['clip_norm']

    optimizers = {
        "sgd": get_sgd,
        "momentum": get_momentum,
        "adagrad": get_adagrad,
        "adam": get_adam,
    }
    optimizer = optimizers.get(OPTIMIZER, -1)

    #regularization = get_regularization(args)
    regularization = None

    # L2 Norm 裁剪梯度
    if GRADIENT_CLIP:
        fluid.clip.set_gradient_clip(
            clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=CLIP_NORM))

    if optimizer == -1:
        raise ValueError("Unknown strategies, expect following options:\n{}".format("\n\t".join(optimizers.keys())))
    return optimizer(learning_rate, regularization, args)


def get_sgd(learning_rate, regularization, args):
    return fluid.optimizer.SGD(learning_rate=learning_rate, regularization=regularization)


def get_momentum(learning_rate, regularization, args):
    MOMENTUM = args["momentum"]
    NESTEROV = args["use_nesterov"]
    return fluid.optimizer.Momentum(learning_rate=learning_rate, momentum=MOMENTUM, use_nesterov=NESTEROV, regularization=regularization)


def get_adagrad(learning_rate, regularization, args):
    EPSILON = args["adagrad_epsilon"]
    INIT_ACCUMULATOR_VAL = args["adagrad_accumulator_value"]
    return fluid.optimizer.Adagrad(learning_rate=learning_rate,
                                   epsilon=EPSILON,
                                   initial_accumulator_value=INIT_ACCUMULATOR_VAL,
                                   regularization=regularization)

def get_adam(learning_rate, regularization, args):
    BETA1 = args["adam_beta1"]
    BETA2 = args["adam_beta2"]
    EPSILON = args["adam_epsilon"]
    return fluid.optimizer.Adam(learning_rate=learning_rate,
                                   beta1=BETA1,
                                   beta2=BETA2,
                                   epsilon=EPSILON,
                                   regularization=regularization)


def get_regularization(args):
    REGULARIZATION = args['regularization']
    REGULARIZATION_COEFF = args['regularization_coeff']

    if REGULARIZATION == 'L1':
        regularization = fluid.regularizer.L1Decay(regularization_coeff=REGULARIZATION_COEFF)
    elif REGULARIZATION == 'L2':
        regularization = fluid.regularizer.L2Decay(regularization_coeff=REGULARIZATION_COEFF)
    else:
        regularization = None

    return regularization



