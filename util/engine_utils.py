import paddle.fluid as fluid
import os


def get_data_run_places(args):
    """
    根据获取数据层（dataloader）的运行位置
    :return: 运行位置
    """
    USE_PARALLEL = args["use_parallel"]
    USE_GPU = args["use_gpu"]
    NUM_OF_DEVICE = args["num_of_device"]

    if USE_PARALLEL and NUM_OF_DEVICE > 1:
        if USE_GPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(NUM_OF_DEVICE)
            places = fluid.cuda_places()
        else:
            places = fluid.cpu_places(NUM_OF_DEVICE)
    else:
        if USE_GPU:
            places = fluid.cuda_places(0)
        else:
            places = fluid.cpu_places(1)
    return places


def get_executor_run_places(args):
    """
    根据获取执行引擎（Executor）的运行位置
    :return: 运行位置
    """
    USE_GPU = args["use_gpu"]

    if USE_GPU:
        places = fluid.CUDAPlace(0)
    else:
        places = fluid.CPUPlace()
    return places
