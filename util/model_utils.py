import paddle.fluid as fluid
import paddle.fluid.io as io
import util.util_filepath as file_utils
import time
import os


def save_train_snapshot(executor, program, file_name="", train_info={}):
    name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    file_name = file_name + "_" + name
    file_path = file_utils.get_fullurl("model", file_name, "pickle")
    file_utils.save_file(content=train_info, file_type="model", file_name=file_name, file_format="json")
    io.save_persistables(executor=executor, dirname=file_path, main_program=program)
    return file_path


def load_train_snapshot(executor, program, file_path):
    assert os.path.exists(file_path), "[%s] cann't be found." % file_path
    io.load_persistables(executor=executor, dirname=file_path, main_program=program)
    if os.path.exists(file_path + ".json"):
        info = file_utils.read_file(file_name=file_path + ".json", file_format="json")
        return info
    return False


def save_model_as_whole(program, file_name="", file_path=""):
    if file_path == "":
        name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        file_path = file_utils.get_fullurl("model", file_name + name, "pickle")

    io.save(program, file_path)
    return file_path


def load_model_params(exe, params_path, program):
    """
    加载模型参数路径下的Parameter类型的参数，可以用于模型初始化
    """
    # 判断参数目录是否存在
    assert os.path.exists(params_path), "[%s] can't be found." % params_path

    # 过滤器，两层过滤，一看参数是不是Parameter类型， 二是只加载路径下已经有的参数到网络中
    def existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        if os.path.exists(os.path.join(params_path, var.name)):
            return True
        print("missing layer: {}".format(var.name))
        return False

    io.load_vars(exe, params_path, main_program=program, predicate=existed_params)


def save_model_params(exe, params_path, program=None):
    """
    保存Parameter类型的模型参数
    :param exe:
    :param params_path:
    :param program:
    :return:
    """
    assert os.path.exists(params_path), "[%s] can't be found." % params_path
    io.save_params(executor=exe, dirname=params_path, main_program=program)


def load_persistable_params(exe, params_path, program):
    """
    加载模型参数路径下的持久化参数，主要用于断点续读
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
    加载预测模型，此函数仅用于预测模块！
    :param model_path: 模型二进制文件的存放地址
    :param exe: 加载模型的 executor
    :return: program，feed_target_names，fetch_targets
    """
    assert os.path.exists(model_path), "[%s] can't be found." % model_path
    return io.load_inference_model(dirname=model_path, executor=exe)


def save_model(model_path, feeded_var_names, target_vars, exe):
    """
    保存预测模型，此函数仅用于预测模块！
    :param model_path:
    :param feeded_var_names: 预测模型的输入变量名的列表
    :param target_vars: 预测模型的输出参数列表
    :param exe:
    :return:
    """
    assert os.path.exists(model_path), "[%s] can't be found." % model_path
    io.save_inference_model(model_path, feeded_var_names, target_vars, exe)


def cast_fp32_to_fp16(exe, main_program):
    """
    将fp32的模型转换为fp16
    """
    print("Cast parameters to float16 data format.")
    for param in main_program.global_block().all_parameters():
        if not param.name.endswith(".master"):
            param_t = fluid.global_scope().find_var(param.name).get_tensor()
            data = np.array(param_t)
            if param.name.find("layer_norm") == -1:
                param_t.set(np.float16(data).view(np.uint16), exe.place)
            master_param_var = fluid.global_scope().find_var(param.name +
                                                             ".master")
            if master_param_var is not None:
                master_param_var.get_tensor().set(data, exe.place)
