import paddle
import paddle.fluid as fluid
import paddle.fluid.io as io
import numpy as np
import os


def load_model_params(exe, params_path, program):
    assert os.path.exists(params_path), "[%s] cann't be found." % params_path
    io.load_params(exe, params_path, main_program=program)

def load_model(model_path, exe):
    assert os.path.exists(model_path), "[%s] cann't be found." % model_path
    io.load_inference_model(dirname=model_path, executor=exe)
