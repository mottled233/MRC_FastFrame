import paddle
import paddle.fluid as fluid
import paddle.fluid.io as io
import numpy as np
import util.util_filepath as file_utils
import time
import os


def save_train_snapshot(executor, program, file_path=""):
    if file_path == "":
        name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        file_path = file_utils.get_fullurl("model", name, "pickle")

    io.save_persistables(executor=executor, dirname=file_path, main_program=program)
    return file_path


def load_train_snapshot(executor, program, file_path):
    assert os.path.exists(file_path), "[%s] cann't be found." % file_path
    io.load_persistables(executor=executor, dirname=file_path, main_program=program)


def save_model_as_whole(program, file_path=""):
    if file_path == "":
        name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        file_path = file_utils.get_fullurl("model", name, "pickle")

    io.save(program, file_path)
    return file_path

def load_model_params(exe, params_path, program):
    """
    加载模型参数路径下的Parameter类型的参数
    """
    #判断参数目录是否存在
    assert os.path.exists(params_path), "[%s] can't be found." % params_path
    #过滤器，两层过滤，一看参数是不是Parameter类型， 二是只加载路径下已经有的参数到网络中
    def existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(os.path.join(params_path, var.name))
    io.load_vars(exe, params_path, main_program=program, predicate=existed_params)

def load_persistable_params(exe, params_path, program, use_fp16=False):
    """
    加载模型参数路径下的持久化参数
    """
    # 判断参数目录是否存在
    assert os.path.exists(params_path), "[%s] can't be found." % params_path

    # 过滤器，两层过滤，一看参数是不是持久化参数， 二是只加载路径下已经有的参数到网络中
    def existed_persistables(var):
        if not fluid.io.is_persistable(var):
            return False
        return os.path.exists(os.path.join(params_path, var.name))

    fluid.io.load_vars(exe, params_path, main_program=program, predicate=existed_persistables)

def load_model(model_path, exe):
    """
    加载预测模型，此函数仅用于预测模块
    :param model_path: 模型二进制文件的存放地址
    :param exe: 加载模型的 executor
    :return: program，feed_target_names，fetch_targets
    """
    assert os.path.exists(model_path), "[%s] can't be found." % model_path
    return io.load_inference_model(dirname=model_path, executor=exe)

def save_model(model_path, feeded_var_names, target_vars, exe):
    """
    保存预测模型，此函数仅用于预测模块
    :param model_path:
    :param feeded_var_names: 预测模型的输入变量名的列表
    :param target_vars: 预测模型的输出参数列表
    :param exe:
    :return:
    """
    assert os.path.exists(model_path), "[%s] can't be found." % model_path
    io.save_inference_model(model_path, feeded_var_names, target_vars, exe)


