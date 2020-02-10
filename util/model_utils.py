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
    io.load_persistables(executor=executor, dirname=file_path, main_program=program)


def save_model_as_whole(program, file_path=""):
    if file_path == "":
        name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        file_path = file_utils.get_fullurl("model", name, "pickle")

    io.save(program, file_path)
    return file_path


def load_model_params(exe, params_path, program):
    assert os.path.exists(params_path), "[%s] cann't be found." % params_path
    io.load_params(exe, params_path, main_program=program)

def load_model(model_path, exe):
    assert os.path.exists(model_path), "[%s] cann't be found." % model_path
    io.load_inference_model(dirname=model_path, executor=exe)
