import paddle
import paddle.fluid as fluid


def get_strategy(args):
    """
    根据配置，返回相应的学习率变化策略
    :return: 学习率变量（Vars）
    """
    STRATEGY = args["learning_rate_strategy"]

    strategies = {
        "exponential_decay": get_exponential_decay_lr,
        "natural_exp_decay": get_natural_exp_decay_lr,
        "noam_worm_up_and_decay": get_noam_decay_lr,
        "piecewise_decay": get_piecewise_decay_lr,
        "linear_warm_up_and_decay": get_linear_warmup_decay,
        "fixed": get_fixed_learning_rate
    }
    strategy = strategies.get(STRATEGY, -1)
    if strategy == -1:
        raise ValueError("Unknown strategies, expect following options:\n{}".format("\n\t".join(strategies.keys())))
    return strategy(args)


def get_fixed_learning_rate(args):
    """
    直接返回默认的学习率
    """
    return args["base_learning_rate"]


def get_exponential_decay_lr(args):
    """
    在学习率上运用指数衰减.
    训练模型时，在训练过程中降低学习率。每 decay_steps 步骤中以 decay_rate 衰减学习率。
    if staircase == True:
        decayed_learning_rate = learning_rate * decay_rate ^ floor(global_step / decay_steps)
    else:
        decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    """
    LR = args["base_learning_rate"]
    DECAY_STEPS = args["decay_step"]
    DECAY_RATE = args["decay_rate"]
    STAIR_DECAY = args["stair_decay"]

    return fluid.layers.exponential_decay(learning_rate=LR, decay_steps=DECAY_STEPS,
                                                            decay_rate=DECAY_RATE,
                                                            staircase=STAIR_DECAY)


def get_natural_exp_decay_lr(args):
    """
    将自然指数衰减运用到初始学习率上。
    训练模型时，在训练过程中降低学习率。 自然指数衰减使用自然指数来计算衰减倍率，每 decay_steps 步衰减倍率的自然指数幂次项上增加 decay_rate 。
    if not staircase:
        decayed_learning_rate = learning_rate * exp(- decay_rate * (global_step / decay_steps))
    else:
        decayed_learning_rate = learning_rate * exp(- decay_rate * floor(global_step / decay_steps))
    """
    LR = args["base_learning_rate"]
    DECAY_STEPS = args["decay_step"]
    DECAY_RATE = args["decay_rate"]
    STAIR_DECAY = args["stair_decay"]

    return fluid.layers.natural_exp_decay(learning_rate=LR, decay_steps=DECAY_STEPS,
                                                            decay_rate=DECAY_RATE,
                                                            staircase=STAIR_DECAY)


def get_noam_decay_lr(args):
    """
    使用Noam衰减策略
    d_model = 2
    current_steps = 20
    warmup_steps = 200
    # 计算
    lr_value = np.power(d_model, -0.5) * np.min([
                           np.power(current_steps, -0.5),
                           np.power(warmup_steps, -1.5) * current_steps])
    """
    LR = args["base_learning_rate"]
    WARM_UP_STEP = args["warm_up_step"]

    return fluid.layers.learning_rate_scheduler.noam_decay(1/(WARM_UP_STEP * (LR ** 2)), WARM_UP_STEP)


def get_piecewise_decay_lr(args):
    """
    使用这个策略时，必须指定一系列学习率和对应的步数，阶梯式下降
    boundaries = [10000, 20000]
    values = [1.0, 0.5, 0.1]
    if step < 10000:
        learning_rate = 1.0
    elif 10000 <= step < 20000:
        learning_rate = 0.5
    else:
        learning_rate = 0.1
    """
    BOUND = [int(i) for i in args["decay_bound"].split(",")]
    LR = [float(i) for i in args["base_learning_rate"].split(",")]

    return fluid.layers.piecewise_decay(boundaries=BOUND, values=LR)


def get_linear_warmup_decay(args):
    """
    线性地将linear做升降变化，作为客制化修改的模板
    """
    START_LR = args["start_learning_rate"]
    WARM_UP_STEP = args["warm_up_step"]
    LR = args["base_learning_rate"]
    DECAY_STEP = args["decay_step"]
    END_LR = args["end_learning_rate"]

    with fluid.default_main_program()._lr_schedule_guard():
        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="scheduled_learning_rate")
        end_lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=END_LR,
            dtype='float32',
            persistable=True,
            name="end_learning_rate")
        global_step = fluid.layers.learning_rate_scheduler._decay_step_counter()

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(global_step < WARM_UP_STEP):
                warmup_lr = LR + (LR - START_LR) * (global_step / WARM_UP_STEP)
                fluid.layers.tensor.assign(warmup_lr, lr)
            with switch.case(global_step < WARM_UP_STEP + DECAY_STEP):
                decayed_lr = fluid.layers.learning_rate_scheduler.polynomial_decay(
                    learning_rate=LR,
                    decay_steps=WARM_UP_STEP + DECAY_STEP,
                    end_learning_rate=END_LR,
                    power=1.0,
                    cycle=False)
                fluid.layers.tensor.assign(decayed_lr, lr)
            with switch.default():
                fluid.layers.tensor.assign(end_lr, lr)

        return lr



