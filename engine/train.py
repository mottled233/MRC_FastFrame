import paddle
import paddle.fluid as fluid
import numpy as np
import os

import model.optimizer
import model.lr_stategy as lr_strategy
from model.network_test import network as network



class TrainEngine(object):
    """
    训练引擎，支持单机多卡训练
    参数：
        train_data_reader： batch化的训练数据集生成器，batch大小必须大于设备数
        valid_data_reader： batch化的验证数据集生成器，batch大小必须大于设备数
        args:
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
                train_data_loader, train_loss = network(self.args, train=True)  # 一些网络定义
                # 获取训练策略
                learning_rate = lr_strategy.get_strategy(self.args)
                optimizer = model.optimizer.get_optimizer(learning_rate, self.args, regularization=None)
                optimizer.minimize(train_loss)

        # 为训练过程设置数据集
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
        # 对训练状态的记录
        self.pre_epoch_valid_loss = float("inf")
        self.standstill_count = 0

        '''
        过程并行化
        '''
        USE_PARALLEL = args["use_parallel"]
        if USE_PARALLEL:
            print("use_parallel")
            # 设置并行训练的策略
            # 这里可以用参数配置，不过要改的东西很多，所以先写死吧
            buildStrategy = fluid.BuildStrategy()
            buildStrategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
            # 构建并行过程
            self.train_main_prog = fluid.CompiledProgram(self.train_main_prog)\
                                    .with_data_parallel(loss_name=self.train_loss.name,
                                                        places=self.get_data_run_places(self.args),
                                                        build_strategy=buildStrategy)
            self.valid_main_prog = fluid.CompiledProgram(self.valid_main_prog)\
                                    .with_data_parallel(share_vars_from=self.train_main_prog,
                                                        places=self.get_data_run_places(self.args),
                                                        build_strategy=buildStrategy)

    def train(self):
        """
        用于训练流程，根据参数完成训练，并使用验证数据对模型效果进行验证
        :return: 无
        """
        MAX_EPOCH = self.args["max_epoch"]
        EARLY_STOPPING = self.args["early_stopping"]
        if EARLY_STOPPING:
            THRESHOLD = self.args["early_stopping_threshold"]
            STANDSTILL_STEP = self.args["early_stopping_times"]
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
        # 执行MAX_EPOCH次迭代
        for epoch_id in range(MAX_EPOCH):
            # 一个epoch的训练过程，一个迭代
            self.__run_train_iterable(executor)
            # 进行一次验证集上的验证
            valid_loss = self.valid(executor)
            # 应用早停策略
            if EARLY_STOPPING:
                need_stop = self.early_stopping_strategy(valid_loss, threshold=THRESHOLD, standstill_step=STANDSTILL_STEP)
                if need_stop:
                    break
        # TODO 保存现有模型

    def __run_train_iterable(self, executor):
        """
        对训练过程的一个epoch的执行
        """
        total_loss = 0
        total_data = 0
        for data in self.train_data_loader():
            # 为获取字段名，这里需要改
            batch_size = data[0]['x'].shape()[0]
            if batch_size < 2:
                print("abort batch")
                continue
            loss_value = executor.run(program=self.train_main_prog, feed=data, fetch_list=[self.train_loss])
            total_loss += loss_value[0]
            total_data += batch_size
        mean_loss = total_loss / total_data
        print('train mean loss is {}'.format(mean_loss))

    def valid(self, exe):
        """
            对验证过程的一个epoch的执行
        """

        total_loss = 0
        total_data = 0
        for data in self.valid_data_loader():
            batch_size = data[0]['x'].shape()[0]
            if batch_size < 2:
                print("abort batch")
                continue
            loss_value = exe.run(program=self.valid_main_prog, feed=data, fetch_list=[self.valid_loss])
            total_loss += loss_value[0]
            total_data += batch_size
        mean_loss = total_loss / total_data
        print('train mean loss is {}'.format(mean_loss))
        return mean_loss

    def get_data_run_places(self, args):
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


    def get_executor_run_places(self, args):
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

    def early_stopping_strategy(self, current_valid_loss, threshold, standstill_step):
        """
        应用早停策略，当判断为停滞时及时中断训练过程
        :param current_valid_loss: 本次验证的loss（或准确率，此时传入的准确率应该加负号）
        :param threshold: 判断为停滞的阈值，当性能增长（指loss下降）不超过这个比例时判断为停滞
        :param standstill_step: 每次停滞都会被计数，连续停滞的次数超过几次后终止训练
        :return: 是否应该终止训练过程
        """
        current_valid_loss = np.sum(current_valid_loss)
        promote = self.pre_epoch_valid_loss - current_valid_loss
        promote_rate = promote / np.abs(self.pre_epoch_valid_loss)
        print(promote_rate)
        if promote_rate < threshold:
            self.standstill_count += 1
            if self.standstill_count > standstill_step:
                return True
        else:
            self.standstill_count = 0
            self.pre_epoch_valid_loss = current_valid_loss
        return False




