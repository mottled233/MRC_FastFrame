import paddle.fluid as fluid
import numpy as np

import util.model_utils as model_utils
import model.optimizer
import model.lr_stategy as lr_strategy
import util.engine_utils as engine_utils
from util.util_logging import UtilLogging
from util.util_filepath import *


class TrainEngine(object):
    """
    训练引擎，支持单机多卡训练
    继承指南：
        通过继承该类，根据需求重写一些分离出来的私有方法来完成改造
        _init_train_model：载入训练模型
        _init_train_strategy：初始化优化策略
        _init_validate_model：载入验证模型
        _load_process: 如何读取模型参数
        _set_data_source：如何给data_loader设置数据来源
        _run_train_iterable：在epoch内的训练过程
        _valid：验证过程
        train：训练的整个过程
        early_stopping_strategy：早停相关策略

    """

    def __init__(self, args, network, train_data_generator, valid_data_generator=None):
        """
        对训练过程进行初始化
        :param args: 训练过程的参数
        :param train_data_generator: 训练过程的数据生成器
        :param valid_data_generator: 验证过程的数据生成器，若为None则不执行validate阶段操作
        """
        self.args = args
        self.app_name = self.args["app_name"]
        self.do_validate = self.args["do_validate"]
        if valid_data_generator is None:
            self.do_validate = False
        self.logger = UtilLogging(args, __name__)
        self.train_data_generator = train_data_generator
        self.valid_data_generator = valid_data_generator
        self.network = network
        self.standstill_count = 0
        self.pre_epoch_valid_loss = float("inf")

        '''
        创建训练过程
        '''
        self.logger.info("Initializing training process...")
        self.train_main_prog = fluid.Program()
        self.train_startup_prog = fluid.Program()

        with fluid.program_guard(self.train_main_prog, self.train_startup_prog):
            # 使用 fluid.unique_name.guard() 实现与test program的参数共享
            with fluid.unique_name.guard():
                # 初始化网络结构
                self.logger.info("Initializing training neural network...")
                train_data_loader, train_loss, train_fetch_data = self._init_train_model()
                self.logger.info("Training neural network initialized.")

                # 设置训练策略
                self.logger.info("Setting training strategy...")
                optimizer, lr = self._init_train_strategy(train_loss)
                self.logger.info("Training strategy has been set.")

        # 属性化
        self.train_data_loader = train_data_loader
        self.optimizer = optimizer
        self.lr = lr
        self.train_loss = train_loss
        self.train_fetch_data = train_fetch_data
        self.logger.info("Training process initialized.")

        '''
        创建验证过程
        '''
        if self.do_validate:
            self.logger.info("Initializing validation process...")
            self.valid_main_prog = fluid.Program()
            self.valid_startup_prog = fluid.Program()
            with fluid.program_guard(self.valid_main_prog, self.valid_startup_prog):
                # 使用 fluid.unique_name.guard() 实现与train program的参数共享
                with fluid.unique_name.guard():
                    # 初始化网络定义
                    self.logger.info("Initializing validation neural network...")
                    valid_data_loader, valid_fetch_data = self._init_validate_model()
                    self.logger.info("Validation neural network initialized.")

            # 属性化
            self.valid_data_loader = valid_data_loader
            self.valid_fetch_data = valid_fetch_data

        '''
        读取保存的模型
        '''
        # 定义执行器
        self.executor = fluid.Executor(engine_utils.get_executor_run_places(self.args))
        # 执行初始化
        self.executor.run(self.train_startup_prog)
        # 读取保存的模型
        self.train_status = self._load_process(self.executor, self.train_main_prog)

        # 对训练状态的记录
        self.pre_epoch_valid_loss = float("inf")
        self.standstill_count = 0
        self.logger.info("Validation process initialized.")

        '''
        过程并行化
        '''
        self.use_parallel = self.args["use_parallel"]

        # 备份原program，因为compiled_program没有保存
        self.origin_train_prog = self.train_main_prog
        if self.use_parallel:
            self.logger.info("Initialize parallel processes...")
            # 设置并行训练的策略
            # 这里可以用参数配置，不过要改的东西很多，所以先写死吧
            build_strategy = fluid.BuildStrategy()
            build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.AllReduce
            # 构建并行过程
            self.train_main_prog = fluid.CompiledProgram(self.train_main_prog).with_data_parallel(
                loss_name=self.train_loss.name,
                places=engine_utils.get_data_run_places(self.args),
                build_strategy=build_strategy)

            if self.do_validate:
                self.valid_main_prog = fluid.CompiledProgram(self.valid_main_prog).with_data_parallel(
                    share_vars_from=self.train_main_prog,
                    places=engine_utils.get_data_run_places(self.args),
                    build_strategy=build_strategy)
            self.logger.info("Parallel processes initialized.")

    def _init_train_model(self):
        """
        定义训练过程中如何初始化网络
        :return: 必须为 reader, loss, fetch_data，其中fetch_data是一个字典，可以存放一些附加的信息，之后会被保存在
                 self.train_fetch_data里。
        """
        # 一些网络定义
        self.network.train()
        reader, train_fetch_data = self.network.create_model()
        try:
            loss = train_fetch_data['loss']
        except KeyError:
            raise KeyError("The key 'loss' is not found." +
                           "Please check if the network returns loss during training")
        return reader, loss, train_fetch_data

    def _init_validate_model(self):
        """
        定义验证过程中如何初始化网络
        :return: 必须为 reader, fetch_data，其中fetch_data是一个字典，可以存放一些附加的信息，之后会被保存在
                 self.valid_fetch_data里。
        """
        # 一些网络定义
        self.network.validate()
        reader, valid_fetch_data = self.network.create_model()
        return reader, valid_fetch_data

    def _init_train_strategy(self, train_loss):
        """
        定义训练的优化策略
        :param train_loss: 模型中的loss
        :return:
        """
        learning_rate = lr_strategy.get_strategy(self.args)
        optimizer = model.optimizer.get_optimizer(learning_rate, self.args)

        if self.args['regularization'] == "L2":
            # L2正则
            param_list = dict()
            for param in self.train_main_prog.global_block().all_parameters():
                param_list[param.name] = param * 1.0
                param_list[param.name].stop_gradient = True

            _, param_grads = optimizer.minimize(train_loss)

            if self.args['regularization_coeff'] > 0:
                for param, grad in param_grads:
                    if self._exclude_from_weight_decay(param.name):
                        continue
                    with param.block.program._optimized_guard(
                            [param, grad]), fluid.framework.name_scope("weight_decay"):
                        updated_param = param - param_list[
                            param.name] * self.args['regularization_coeff'] * learning_rate
                        fluid.layers.assign(output=param, input=updated_param)
        else:
            optimizer.minimize(train_loss)
        return optimizer, learning_rate

    def _load_process(self, executor, main_prog):
        """
        读取模型的过程，
        如果想从零开始，请将load_model_path设为空字符串，且read_checkpoint，continue_train为false
        如果想从预训练模型（或某个基线）开始训练，请设置continue_train为False，且read_checkpoint为False
        如果想从某个继续训练，请设置请设置continue_train为True，且read_checkpoint为False，此时会读取模型状态，包括飞桨
        内部储存的global_step信息，但本框架内的epoch和step信息不会被读取
        如果想从断点训练，请设置continue_train为True，且read_checkpoint为True。此时会读取之前训练的断点，包括所有信息
        :param executor:
        :param main_prog:
        :return: 字典，保存当前训练状态, 将保存在self.train_status中
        """
        CONTINUE = self.args["continue_train"]
        MODEL_PATH = self.args["load_model_path"]
        CHECK_POINT = self.args["read_checkpoint"]

        total_step = 0
        step_in_epoch = 0
        total_epoch = 1
        # 读取模型现有的参数并为继续训练进行相应处理
        if CONTINUE and CHECK_POINT:
            info = model_utils.load_train_snapshot(executor, main_prog, MODEL_PATH)
            self.logger.info("Model file in {} has been loaded".format(MODEL_PATH))
            if info:
                total_step = info.get("total_step", 0)
                step_in_epoch = info.get("step_in_epoch", 0)
                total_epoch = info.get("epoch", 1)
                self.logger.info("Load train info: {}".format(info))
        elif MODEL_PATH != "":
            # 若是第一次训练且预训练模型参数不为空，则加载预训练模型参数
            model_utils.load_model_params(exe=executor, program=main_prog, params_path=MODEL_PATH)
            self.logger.info("Pre-trained model file in {} has been loaded".format(MODEL_PATH))

        return {'total_step': total_step, 'total_epoch': total_epoch, 'step_in_epoch': step_in_epoch}

    def _set_data_source(self, data_loader, data_source):
        data_loader.set_sample_list_generator(data_source, places=engine_utils.get_data_run_places(self.args))

    def _save_model(self, file_name, train_info=None):
        """
        储存模型
        :param file_name: 会自动将其放在FileDir/model中
        :param train_info: 训练信息，保存断点信息
        :return:
        """
        return model_utils.save_train_snapshot(self.executor, self.origin_train_prog,
                                               file_name=file_name, train_info=train_info)

    def train(self, **kwargs):
        """
        用于训练流程，根据参数完成训练，并使用验证数据对模型效果进行验证
        :return: 无
        """
        APP_NAME = self.args["app_name"]
        MAX_EPOCH = self.args["max_epoch"]

        VALIDATE = self.args["do_validate"]
        if VALIDATE:
            EARLY_STOPPING = self.args["early_stopping"]
            if EARLY_STOPPING:
                THRESHOLD = self.args["early_stopping_threshold"]
                STANDSTILL_STEP = self.args["early_stopping_stand_times"]

        # 设置数据集
        self._set_data_source(self.train_data_loader, self.train_data_generator)
        if VALIDATE:
            self._set_data_source(self.valid_data_loader, self.valid_data_generator)

        # 定义执行器
        executor = self.executor

        total_step = self.train_status['total_step']
        step_in_epoch = self.train_status['step_in_epoch']
        total_epoch = self.train_status['total_epoch']

        self.logger.info("Ready to train the model.Executing...")
        # 执行MAX_EPOCH次迭代save_train_snapshot
        for epoch_id in range(total_epoch, MAX_EPOCH + 1):
            # 一个epoch的训练过程，一个迭代
            kwargs['total_step'] = total_step
            kwargs['step_in_epoch'] = step_in_epoch
            kwargs['epoch_id'] = epoch_id

            total_step, loss = self._run_train_iterable(executor, **kwargs)
            # 为了在训练过程中终止训练进程。这里不能写成not total_step
            if total_step == False:
                return

            step_in_epoch = 0
            self.logger.info('Epoch {epoch} done, train mean loss is {loss}'.format(epoch=epoch_id, loss=loss))
            if VALIDATE:
                # 进行一次验证集上的验证
                result = self._valid(executor, **kwargs)
                self.logger.info(' Epoch {epoch} Validated'.format(epoch=epoch_id))
            # 进行保存
            info = {
                "total_step": total_step,
                "epoch": epoch_id
            }
            file_path = self._save_model(file_name="{}_epoch{}".format(APP_NAME, epoch_id),
                                         train_info=info)
            self.logger.info("Snapshot of training process has been saved as folder {}".format(file_path))
            # 应用早停策略
            if VALIDATE and EARLY_STOPPING:
                if self.early_stopping_strategy(result, threshold=THRESHOLD, standstill_step=STANDSTILL_STEP):
                    self.logger.info("Training process has been standstill over the threshold epochs.")
                    break
            if os.path.isfile("stop_signal"):
                return

        self.logger.info("Training process completed.")

    def _run_train_iterable(self, executor, **kwargs):
        """
        对训练过程的一个epoch的执行
        :param executor: 执行器
        :return: 执行完该epoch后的total_step，以及该epoch的平均loss（用于打印）
        """
        mean_loss = 0
        total_step = 0
        return total_step, mean_loss

    def _valid(self, exe, **kwargs):
        """
        对验证过程的一个epoch的执行
        :param exe: 执行器
        :param epoch_id: 第几个epoch
        :return: 用于判断是否应该早停的指标，该指标越小表示模型越好。
        """
        mean_loss = 0
        return mean_loss

    def early_stopping_strategy(self, current_valid_loss, threshold, standstill_step):
        """
        应用早停策略，当判断为停滞时及时中断训练过程
        :param current_valid_loss: 本次验证的loss,该指标越小模型效果越好（或准确率，此时传入的准确率应该加负号）
        :param threshold: 判断为停滞的阈值，当性能增长（指loss下降）不超过这个比例时判断为停滞
        :param standstill_step: 每次停滞都会被计数，连续停滞的次数超过几次后终止训练
        :return: 是否应该终止训练过程
        """
        current_valid_loss = np.sum(current_valid_loss)
        promote = self.pre_epoch_valid_loss - current_valid_loss
        promote_rate = promote / np.abs(self.pre_epoch_valid_loss)
        self.logger.info("This epoch promote performance by {}".format(promote_rate))
        if promote_rate < threshold:
            self.standstill_count += 1
            if self.standstill_count > standstill_step:
                return True
        else:
            self.standstill_count = 0
            self.pre_epoch_valid_loss = current_valid_loss
        return False

    @staticmethod
    def _exclude_from_weight_decay(name):
        """exclude_from_weight_decay"""
        if name.find("layer_norm") > -1:
            return True
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return True
        return False
