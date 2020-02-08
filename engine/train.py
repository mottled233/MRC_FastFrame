import paddle
import paddle.fluid as fluid
import numpy as np
import os
from model.network_test import network as network


class TrainEngine(object):
    """
    训练引擎，支持单机多卡训练
    一个所需参数示例：
        args = {
        "max_epoch": 20,
        "early_stopping": False,
        "warm_up": False,
        "continue_train": False,
        "model_path": "",
        "use_parallel": True,
        "use_gpu": False,
        "num_of_device": 2
    }

    """
    def __init__(self, train_data_reader, valid_data_reader, args):
        self.args = args

        '''
        创建训练过程
        '''
        self.train_main_prog = fluid.Program()
        self.train_startup_prog = fluid.Program()
        with fluid.program_guard(self.train_main_prog, self.train_startup_prog):
            # 使用 fluid.unique_name.guard() 实现与test program的参数共享
            with fluid.unique_name.guard():
                train_data_loader, optimizer, train_loss = network(self.args, train=True)  # 一些网络定义

        train_data_loader.set_sample_list_generator(train_data_reader, places=self.get_data_run_places(self.args))

        self.train_data_loader = train_data_loader
        self.optimizer = optimizer
        self.train_loss = train_loss

        '''
        创建验证过程
        '''
        self.valid_main_prog = fluid.Program()
        self.valid_startup_prog = fluid.Program()
        with fluid.program_guard(self.valid_main_prog, self.valid_startup_prog):
            # 使用 fluid.unique_name.guard() 实现与train program的参数共享
            with fluid.unique_name.guard():
                valid_data_loader, valid_loss = network(self.args, train=False)  # 一些网络定义

        valid_data_loader.set_sample_list_generator(valid_data_reader, places=self.get_data_run_places(self.args))

        self.valid_data_loader = valid_data_loader
        self.valid_loss = valid_loss

        '''
        过程并行化
        '''
        USE_PARALLEL = args["use_parallel"]
        if USE_PARALLEL:
            self.train_main_prog = fluid.CompiledProgram(self.train_main_prog)\
                                    .with_data_parallel(loss_name=self.train_loss.name)
            self.valid_main_prog = fluid.CompiledProgram(self.valid_main_prog)\
                                    .with_data_parallel(share_vars_from=self.train_main_prog)

    def train(self):
        """
        用于训练流程，根据参数完成训练，并使用验证数据对模型效果进行验证
        :return: 无
        """
        MAX_EPOCH = self.args["max_epoch"]
        EARLY_STOPPING = self.args["early_stopping"]
        WARM_UP = self.args["warm_up"]
        CONTINUE = self.args["continue_train"]
        if CONTINUE:
            MODEL_PATH = self.args["model_path"]
            # TODO 读取模型现有的参数并为继续训练进行相应处理

        # 定义执行器
        executor = fluid.Executor(self.get_executor_run_places(self.args))
        # 执行初始化
        executor.run(self.train_startup_prog)
        # TODO 应该有是否读入已有模型

        for epoch_id in range(MAX_EPOCH):
            # TODO 训练策略等
            self.run_train_iterable(self.train_main_prog, executor, self.train_loss, self.train_data_loader)
            self.run_valid_iterable(self.valid_main_prog, executor, self.valid_loss, self.valid_data_loader)

    def run_train_iterable(self, program, exe, loss, data_loader):
        total_loss = 0
        total_data = 0
        for data in data_loader():
            loss_value = exe.run(program=program, feed=data, fetch_list=[loss])
            total_loss += loss_value[0]
            total_data += data[0]['x'][0].shape()[0]
        # print('train mean loss is {}'.format(total_loss / total_data))

    def run_valid_iterable(self, program, exe, loss, data_loader):
        total_loss = 0
        total_data = 0
        for data in data_loader():
            loss_value = exe.run(program=program, feed=data, fetch_list=[loss])
            total_loss += loss_value[0]
            total_data += data[0]['x'][0].shape()[0]
        # print('valid mean loss is {}'.format(total_loss / total_data))

    def get_data_run_places(self, args):
        """
        根据获取数据层（dataloader）的运行位置
        :return: 运行位置
        """
        USE_PARALLEL = args["use_parallel"]
        USE_GPU = args["use_gpu"]
        NUM_OF_DEVICE = args["num_of_device"]

        if USE_PARALLEL:
            if USE_GPU:
                if NUM_OF_DEVICE != -1:
                    places = fluid.cuda_places(NUM_OF_DEVICE)
                else:
                    places = fluid.cuda_places()
            else:
                if NUM_OF_DEVICE != -1:
                    places = fluid.cpu_places(NUM_OF_DEVICE)
                else:
                    places = fluid.cpu_places()
        else:
            if USE_GPU:
                places = fluid.cuda_places(0)
            else:
                places = fluid.cpu_places(1)
        return places

    def get_executor_run_places(self, args):
        """
        根据获取执行引擎（Executor）的运行位置
        ! 注意该函数会根据参数修改环境变量中CPU_NUM以控制使用的CPU数，
        这是由于paddle中除非指定了CPU_NUM否则默认使用所有CPU
        且由于使用线程分发同一batch数据（gpu是每个卡一个batch），
        在CPU核心数太多的情况下会报错。
        :return: 运行位置
        """
        USE_PARALLEL = args["use_parallel"]
        USE_GPU = args["use_gpu"]
        NUM_OF_DEVICE = args["num_of_device"]

        if USE_PARALLEL:
            if USE_GPU:
                if NUM_OF_DEVICE != -1:
                    places = fluid.CUDAPlace(NUM_OF_DEVICE)
                else:
                    places = fluid.CUDAPlace()
            else:
                if NUM_OF_DEVICE != -1:
                    os.environ['CPU_NUM'] = str(NUM_OF_DEVICE)
                else:
                    os.environ['CPU_NUM'] = ""
                places = fluid.CPUPlace()
        else:
            if USE_GPU:
                places = fluid.CUDAPlace(0)
            else:
                places = fluid.CPUPlace()
        return places
