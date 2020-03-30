import paddle
import paddle.fluid as fluid
import numpy as np
import os
import util.model_utils as model_utils
import json
import model.optimizer
import model.lr_stategy as lr_strategy
import model.classifier as classifier
from util.model_utils import load_model_params
from engine.train import TrainEngine
from model.classifier import create_model
import util.util_filepath as file_utils


class TrainEngineForMT(object):
    """
    训练引擎，支持单机多卡训练
    参数：
        train_data_reader： batch化的训练数据集生成器，batch大小必须大于设备数
        valid_data_reader： batch化的验证数据集生成器，batch大小必须大于设备数
        args: UtilParameter实例，
    一个所需train部分的参数示例：
    args = {
        "max_epoch": 100,
        "snapshot_frequency": 10,
        "early_stopping": True,
        "warm_up": False,
        "continue_train": False,
        "load_model_path": "",
        "use_parallel": True,
        "use_gpu": False,
        "num_of_device": 2,
        "batch_size": 32,
        "base_learning_rate": 0.01,
        "learning_rate_strategy": "fixed",
        "start_learning_rate": 1e-04,
        "warm_up_step": 50,
        "end_learning_rate": 1e-04,
        "decay_step": 1000,
        "optimizer": "sgd"
    }

    """

    def __init__(self, train_data_reader, train_vocab_size, valid_data_reader, valid_vocab_size, args, logger,
                 create_model=None):
        """
        对训练过程进行初始化
        :param train_data_reader:
        :param valid_data_reader:
        :param args:
        :param logger:
        """
        self.args = args.get_config(args.TRAIN)
        self.logger = logger

        '''
        创建训练过程
        '''
        self.logger.info("Initializing training process...")
        self.train_main_prog = fluid.Program()
        self.train_startup_prog = fluid.Program()
        with fluid.program_guard(self.train_main_prog, self.train_startup_prog):
            # 使用 fluid.unique_name.guard() 实现与test program的参数共享
            with fluid.unique_name.guard():
                self.logger.info("Initializing training neural network...")
                # train_data_loader, train_loss = network(self.args, train=True)  # 一些网络定义
                output = classifier.create_model_for_multi_task(
                    args.get_config(args.MODEL_BUILD),
                    vocab_size=train_vocab_size,
                    is_prediction=False)
                train_data_loader = output[0]
                train_loss = output[1]
                self.logger.info("Training neural network initialized.")
                # 获取训练策略
                self.logger.info("Setting training strategy...")

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
                self.logger.info("Training strategy has been set.")

        # 为训练过程设置数据集
        train_data_loader.set_sample_list_generator(train_data_reader, places=self.get_data_run_places(self.args))
        self.train_data_loader = train_data_loader
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.logger.info("Training process initialized.")

        '''
        创建验证过程
        '''
        self.logger.info("Initializing validation process...")
        self.valid_main_prog = fluid.Program()
        self.valid_startup_prog = fluid.Program()
        with fluid.program_guard(self.valid_main_prog, self.valid_startup_prog):
            # 使用 fluid.unique_name.guard() 实现与train program的参数共享
            with fluid.unique_name.guard():
                self.logger.info("Initializing validation neural network...")
                # valid_data_loader, valid_loss = network(self.args, train=False)  # 一些网络定义
                valid_data_loader, valid_loss, _, accuracy, accuracy_for_reverse, _ = classifier.create_model_for_multi_task(
                    args.get_config(args.MODEL_BUILD),
                    vocab_size=valid_vocab_size,
                    is_prediction=False)
                self.logger.info("Validation neural network initialized.")

        valid_data_loader.set_sample_list_generator(valid_data_reader, places=self.get_data_run_places(self.args))

        self.valid_data_loader = valid_data_loader
        self.valid_loss = valid_loss
        self.valid_accuracy = accuracy
        self.valid_accuracy_for_reverse = accuracy_for_reverse
        # 对训练状态的记录
        self.pre_epoch_valid_loss = float("inf")
        self.standstill_count = 0
        self.logger.info("Validation process initialized.")
        '''
        过程并行化
        '''
        USE_PARALLEL = self.args["use_parallel"]

        # 备份原program，因为compiled_program没有保存
        self.origin_train_prog = self.train_main_prog
        if USE_PARALLEL:
            self.logger.info("Initialize parallel processes...")
            # 设置并行训练的策略
            # 这里可以用参数配置，不过要改的东西很多，所以先写死吧
            build_strategy = fluid.BuildStrategy()
            build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.AllReduce
            # 构建并行过程
            self.train_main_prog = fluid.CompiledProgram(self.train_main_prog).with_data_parallel(
                loss_name=self.train_loss.name,
                places=self.get_data_run_places(self.args),
                build_strategy=build_strategy)
            self.valid_main_prog = fluid.CompiledProgram(self.valid_main_prog).with_data_parallel(
                share_vars_from=self.train_main_prog,
                places=self.get_data_run_places(self.args),
                build_strategy=build_strategy)
            self.logger.info("Parallel processes initialized.")

    def train(self):
        """
        用于训练流程，根据参数完成训练，并使用验证数据对模型效果进行验证
        :return: 无
        """
        APP_NAME = self.args["app_name"]
        MAX_EPOCH = self.args["max_epoch"]
        SNAPSHOT_FREQUENCY = self.args["snapshot_frequency"]

        EARLY_STOPPING = self.args["early_stopping"]
        if EARLY_STOPPING:
            THRESHOLD = self.args["early_stopping_threshold"]
            STANDSTILL_STEP = self.args["early_stopping_stand_times"]

        CONTINUE = self.args["continue_train"]
        MODEL_PATH = self.args["load_model_path"]
        CHECK_POINT = self.args["read_checkpoint"]

        # 定义执行器
        executor = fluid.Executor(self.get_executor_run_places(self.args))
        # 执行初始化
        executor.run(self.train_startup_prog)

        total_step = 0
        step_in_epoch = 0
        total_epoch = 0
        # 读取模型现有的参数并为继续训练进行相应处理
        if CONTINUE and CHECK_POINT:
            info = model_utils.load_train_snapshot(executor, self.origin_train_prog, MODEL_PATH)
            self.logger.info("Model file in {} has been loaded".format(MODEL_PATH))
            if info:
                total_step = info.get("total_step", 0)
                step_in_epoch = info.get("step_in_epoch", 0)
                total_epoch = info.get("epoch", 0)
                self.logger.info("Load train info: {}".format(info))
        elif MODEL_PATH != "":
            # 若是第一次训练且预训练模型参数不为空，则加载预训练模型参数
            model_utils.load_model_params(exe=executor, program=self.origin_train_prog, params_path=MODEL_PATH)
            self.logger.info("Pre-trained model file in {} has been loaded".format(MODEL_PATH))

        self.logger.info("Ready to train the model.Executing...")

        # 执行MAX_EPOCH次迭代save_train_snapshot
        for epoch_id in range(total_epoch, MAX_EPOCH):
            # 一个epoch的训练过程，一个迭代
            total_step, loss = self.__run_train_iterable(executor, total_step, epoch_id, step_in_epoch)
            step_in_epoch = 0
            self.logger.info('Epoch {epoch} done, train mean loss is {loss}'.format(epoch=epoch_id, loss=loss))
            # 进行一次验证集上的验证
            valid_loss, valid_acc, _ = self.__valid(executor)
            self.logger.info(' Epoch {epoch} Validated'.format(epoch=epoch_id))
            # 进行保存
            info = {
                "total_step": total_step,
                "epoch": epoch_id
            }
            file_path = model_utils.save_train_snapshot(executor, self.origin_train_prog,
                                                        file_name="{}_epoch{}".format(APP_NAME, epoch_id),
                                                        train_info=info)
            self.logger.info("Snapshot of training process has been saved as folder {}".format(file_path))
            # 应用早停策略
            if EARLY_STOPPING:
                need_stop = self.early_stopping_strategy(-valid_acc, threshold=THRESHOLD,
                                                         standstill_step=STANDSTILL_STEP)
                if need_stop:
                    self.logger.info("Performance improvement stalled, ending the training process")
                    break
        # 保存现有模型
        file_path = model_utils.save_train_snapshot(executor, self.origin_train_prog, APP_NAME)
        self.logger.info("Training process completed. model saved in {}".format(file_path))

    def __run_train_iterable(self, executor, total_step=0, epoch=0, step_in_epoch=0):
        """
        对训练过程的一个epoch的执行
        """
        PRINT_PER_STEP = self.args["print_per_step"]
        USE_PARALLEL = self.args["use_parallel"]
        NUM_OF_DEVICE = self.args["num_of_device"]
        APP_NAME = self.args["app_name"]
        SNAPSHOT_FREQUENCY = self.args["snapshot_frequency"]
        VALID_FREQUENCY = self.args["validate_frequency_step"]
        WHETHER_VALID = self.args["validate_in_epoch"]
        total_loss = 0
        total_data = 0
        for step, data in enumerate(self.train_data_loader()):
            # 为获取字段名，这里需要改
            if step <= step_in_epoch:
                continue
            batch_size = data[0]['qas_ids'].shape()[0]
            if USE_PARALLEL and batch_size < NUM_OF_DEVICE:
                self.logger.warning("Batch size less than the number of devices. Batch aborted.")
                continue
            loss_value = executor.run(program=self.train_main_prog, feed=data, fetch_list=[self.train_loss])
            total_loss += loss_value[0]
            total_data += 1
            # 打印逻辑
            if PRINT_PER_STEP > 0:
                if step % PRINT_PER_STEP == 0:
                    self.logger.info("Step {step}: loss = {loss}".
                                     format(step=total_step + step, loss=loss_value))
            # 保存逻辑
            info = {
                "total_step": total_step,
                "step_in_epoch": step,
                "epoch": epoch
            }
            if step % SNAPSHOT_FREQUENCY == 0 and step != 0:
                file_path = model_utils.save_train_snapshot(executor, self.origin_train_prog,
                                                            "{}_step{}".format(APP_NAME, total_step + step), info)
                self.logger.info("Snapshot of training process has been saved as folder {}".format(file_path))

            if WHETHER_VALID and step % VALID_FREQUENCY == 0 and step != 0:
                self.__valid(executor)

        mean_loss = total_loss / total_data
        return total_step + step, mean_loss

    def __valid(self, exe):
        """
            对验证过程的一个epoch的执行
        """
        PRINT_PER_STEP = self.args["print_per_step"]
        USE_PARALLEL = self.args["use_parallel"]
        NUM_OF_DEVICE = self.args["num_of_device"]

        total_loss = 0
        total_accuracy = 0
        total_data = 0
        total_acc_for_reverse = 0
        for step, data in enumerate(self.valid_data_loader()):
            batch_size = data[0]['qas_ids'].shape()[0]
            if USE_PARALLEL and batch_size < NUM_OF_DEVICE:
                self.logger.warning("Batch size less than the number of devices. Batch aborted.")
                continue
            loss_value = exe.run(program=self.valid_main_prog, feed=data, fetch_list=[self.valid_loss,
                                                                                      self.valid_accuracy,
                                                                                      self.valid_accuracy_for_reverse])
            total_loss += loss_value[0]
            total_accuracy += loss_value[1]
            total_acc_for_reverse += loss_value[2]
            total_data += 1
            if PRINT_PER_STEP > 0:
                if step % PRINT_PER_STEP == 0:
                    self.logger.info(
                        "Valid batch {step} in epoch: loss = {loss}".format(step=step, loss=total_loss / total_data))
        mean_loss = total_loss / total_data
        mean_accuracy = total_accuracy / total_data
        mean_accuracy_for_reverse = total_acc_for_reverse / total_data
        self.logger.info('valid mean loss is {loss}, mean accuracy is {acc}, {acc_for_reverse}'.format(
            loss=mean_loss,
            acc=mean_accuracy,
            acc_for_reverse=mean_accuracy_for_reverse))
        return mean_loss, mean_accuracy, mean_accuracy_for_reverse

    @staticmethod
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

    @staticmethod
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
        self.logger.info("This epoch promote performance by {}".format(promote_rate))
        if promote_rate < threshold:
            self.standstill_count += 1
            if self.standstill_count > standstill_step:
                return True
        else:
            self.standstill_count = 0
            self.pre_epoch_valid_loss = current_valid_loss
        return False

    def _exclude_from_weight_decay(self, name):
        """exclude_from_weight_decay"""
        if name.find("layer_norm") > -1:
            return True
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return True
        return False


class TrainEngineForGetCLSOutput(object):
    """
    训练引擎，支持单机多卡训练
    参数：
        train_data_reader： batch化的训练数据集生成器，batch大小必须大于设备数
        valid_data_reader： batch化的验证数据集生成器，batch大小必须大于设备数
        args: UtilParameter实例，
    一个所需train部分的参数示例：
    args = {
        "max_epoch": 100,
        "snapshot_frequency": 10,
        "early_stopping": True,
        "warm_up": False,
        "continue_train": False,
        "load_model_path": "",
        "use_parallel": True,
        "use_gpu": False,
        "num_of_device": 2,
        "batch_size": 32,
        "base_learning_rate": 0.01,
        "learning_rate_strategy": "fixed",
        "start_learning_rate": 1e-04,
        "warm_up_step": 50,
        "end_learning_rate": 1e-04,
        "decay_step": 1000,
        "optimizer": "sgd"
    }

    """

    def __init__(self, train_data_reader, train_vocab_size, args, logger, cache_path, is_prediction=False):
        """
        对训练过程进行初始化
        :param train_data_reader:
        :param valid_data_reader:
        :param args:
        :param logger:
        """
        self.args = args.get_config(args.TRAIN)
        self.logger = logger
        self.is_prediction = is_prediction
        self.cache_path = cache_path
        '''
        创建训练过程
        '''
        self.logger.info("Initializing training process...")
        self.train_main_prog = fluid.Program()
        self.train_startup_prog = fluid.Program()
        with fluid.program_guard(self.train_main_prog, self.train_startup_prog):
            # 使用 fluid.unique_name.guard() 实现与test program的参数共享
            with fluid.unique_name.guard():
                self.logger.info("Initializing training neural network...")
                # train_data_loader, train_loss = network(self.args, train=True)  # 一些网络定义
                output = classifier.creat_model_for_cls_output(
                    args.get_config(args.MODEL_BUILD),
                    vocab_size=train_vocab_size,
                    is_prediction=self.is_prediction)
                if not is_prediction:
                    train_data_loader, cls_feats, qas_id, labels = output
                else:
                    train_data_loader, cls_feats, qas_id = output
                self.logger.info("Training neural network initialized.")
                # 获取训练策略
                self.logger.info("Setting training strategy...")

        train_data_loader.set_sample_list_generator(train_data_reader, places=self.get_data_run_places(self.args))
        self.train_data_loader = train_data_loader
        self.qas_id = qas_id
        self.cls_feats = cls_feats
        if not is_prediction:
            self.labels = labels

        # 对训练状态的记录
        self.pre_epoch_valid_loss = float("inf")
        self.standstill_count = 0
        self.logger.info("Validation process initialized.")
        '''
        过程并行化
        '''
        USE_PARALLEL = self.args["use_parallel"]

        # 备份原program，因为compiled_program没有保存
        self.origin_train_prog = self.train_main_prog
        if USE_PARALLEL:
            self.logger.info("Initialize parallel processes...")
            # 设置并行训练的策略
            # 这里可以用参数配置，不过要改的东西很多，所以先写死吧
            build_strategy = fluid.BuildStrategy()
            build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.AllReduce
            # 构建并行过程
            self.train_main_prog = fluid.CompiledProgram(self.train_main_prog).with_data_parallel(
                loss_name=self.train_loss.name,
                places=self.get_data_run_places(self.args),
                build_strategy=build_strategy)
            self.logger.info("Parallel processes initialized.")

    def train(self):
        """
        用于训练流程，根据参数完成训练，并使用验证数据对模型效果进行验证
        :return: 无
        """
        APP_NAME = self.args["app_name"]
        MAX_EPOCH = self.args["max_epoch"]
        SNAPSHOT_FREQUENCY = self.args["snapshot_frequency"]

        EARLY_STOPPING = self.args["early_stopping"]
        if EARLY_STOPPING:
            THRESHOLD = self.args["early_stopping_threshold"]
            STANDSTILL_STEP = self.args["early_stopping_stand_times"]

        CONTINUE = self.args["continue_train"]
        MODEL_PATH = self.args["load_model_path"]
        CHECK_POINT = self.args["read_checkpoint"]

        # 定义执行器
        executor = fluid.Executor(self.get_executor_run_places(self.args))
        # 执行初始化
        executor.run(self.train_startup_prog)

        total_step = 0
        step_in_epoch = 0
        total_epoch = 0
        # 读取模型现有的参数并为继续训练进行相应处理
        if MODEL_PATH != "":
            # 若是第一次训练且预训练模型参数不为空，则加载预训练模型参数
            model_utils.load_model_params(exe=executor, program=self.origin_train_prog, params_path=MODEL_PATH)
            self.logger.info("Pre-trained model file in {} has been loaded".format(MODEL_PATH))

        self.logger.info("Ready to train the model.Executing...")
        self.__run_train_iterable(executor, total_step, 0, 0)

    def __run_train_iterable(self, executor, total_step=0, epoch=0, step_in_epoch=0):
        """
        对训练过程的一个epoch的执行
        """
        qas_id_list = []
        labels_list = []
        cls_feats_list = []
        PRINT_PER_STEP = self.args["print_per_step"]
        USE_PARALLEL = self.args["use_parallel"]
        NUM_OF_DEVICE = self.args["num_of_device"]
        APP_NAME = self.args["app_name"]
        SNAPSHOT_FREQUENCY = self.args["snapshot_frequency"]
        VALID_FREQUENCY = self.args["validate_frequency_step"]
        WHETHER_VALID = self.args["validate_in_epoch"]
        total_loss = 0
        total_data = 0
        for step, data in enumerate(self.train_data_loader()):
            # 为获取字段名，这里需要改ß
            batch_size = data[0]['qas_ids'].shape()[0]
            if USE_PARALLEL and batch_size < NUM_OF_DEVICE:
                self.logger.warning("Batch size less than the number of devices. Batch aborted.")
                continue
            if not self.is_prediction:
                qas_id, cls_feats, labels = executor.run(program=self.train_main_prog, feed=data,
                                                         fetch_list=[self.qas_id, self.cls_feats, self.labels])
                qas_id_list.extend(qas_id)
                labels_list.extend(labels)
                cls_feats_list.extend(cls_feats)
            else:
                qas_id, cls_feats = executor.run(program=self.train_main_prog, feed=data,
                                                 fetch_list=[self.qas_id, self.cls_feats])
                qas_id_list.extend(qas_id)
                cls_feats_list.extend(cls_feats)
            total_data += 1

            # 打印逻辑
            if PRINT_PER_STEP > 0:
                if step % PRINT_PER_STEP == 0:
                    self.logger.info("Step {step}".
                                     format(step=total_step + step))

        if not self.is_prediction:
            cls_feats_list = [x.tolist() for x in cls_feats_list]
            labels_list = [int(x[0]) for x in labels_list]
            qas_id_list = [int(x[0]) for x in qas_id_list]
            results = []
            for i in range(len(cls_feats_list)):
                result = {'qas_id': qas_id_list[i], 'label': labels_list[i], 'cls_feats': cls_feats_list[i]}
                results.append(result)
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(results, f)
        else:
            cls_feats_list = [x.tolist() for x in cls_feats_list]
            qas_id_list = [int(x[0]) for x in qas_id_list]
            results = []
            for i in range(len(cls_feats_list)):
                result = {'qas_id': qas_id_list[i], 'cls_feats': cls_feats_list[i]}
                results.append(result)
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(results, f)

    @staticmethod
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

    @staticmethod
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
        self.logger.info("This epoch promote performance by {}".format(promote_rate))
        if promote_rate < threshold:
            self.standstill_count += 1
            if self.standstill_count > standstill_step:
                return True
        else:
            self.standstill_count = 0
            self.pre_epoch_valid_loss = current_valid_loss
        return False


class TrainEngineForMergeModel(object):
    """
    训练引擎，支持单机多卡训练
    参数：
        train_data_reader： batch化的训练数据集生成器，batch大小必须大于设备数
        valid_data_reader： batch化的验证数据集生成器，batch大小必须大于设备数
        args: UtilParameter实例，
    一个所需train部分的参数示例：
    args = {
        "max_epoch": 100,
        "snapshot_frequency": 10,
        "early_stopping": True,
        "warm_up": False,
        "continue_train": False,
        "load_model_path": "",
        "use_parallel": True,
        "use_gpu": False,
        "num_of_device": 2,
        "batch_size": 32,
        "base_learning_rate": 0.01,
        "learning_rate_strategy": "fixed",
        "start_learning_rate": 1e-04,
        "warm_up_step": 50,
        "end_learning_rate": 1e-04,
        "decay_step": 1000,
        "optimizer": "sgd"
    }

    """

    def __init__(self, train_data_reader, valid_data_reader, args, logger):
        """
        对训练过程进行初始化
        :param train_data_reader:
        :param valid_data_reader:
        :param args:
        :param logger:
        """
        self.args = args.get_config(args.TRAIN)
        self.logger = logger

        '''
        创建训练过程
        '''
        self.logger.info("Initializing training process...")
        self.train_main_prog = fluid.Program()
        self.train_startup_prog = fluid.Program()
        with fluid.program_guard(self.train_main_prog, self.train_startup_prog):
            # 使用 fluid.unique_name.guard() 实现与test program的参数共享
            with fluid.unique_name.guard():
                self.logger.info("Initializing training neural network...")
                # train_data_loader, train_loss = network(self.args, train=True)  # 一些网络定义
                output = classifier.create_model_for_cls_merge(
                    args.get_config(args.MODEL_BUILD),
                    is_prediction=False)
                train_data_loader = output[0]
                train_loss = output[1]
                self.logger.info("Training neural network initialized.")
                # 获取训练策略
                self.logger.info("Setting training strategy...")

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
                self.logger.info("Training strategy has been set.")

        # 为训练过程设置数据集
        train_data_loader.set_sample_list_generator(train_data_reader, places=self.get_data_run_places(self.args))
        self.train_data_loader = train_data_loader
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.logger.info("Training process initialized.")

        '''
        创建验证过程
        '''
        self.logger.info("Initializing validation process...")
        self.valid_main_prog = fluid.Program()
        self.valid_startup_prog = fluid.Program()
        with fluid.program_guard(self.valid_main_prog, self.valid_startup_prog):
            # 使用 fluid.unique_name.guard() 实现与train program的参数共享
            with fluid.unique_name.guard():
                self.logger.info("Initializing validation neural network...")
                # valid_data_loader, valid_loss = network(self.args, train=False)  # 一些网络定义
                valid_data_loader, valid_loss, _, accuracy, _ = classifier.create_model_for_cls_merge(
                    args.get_config(args.MODEL_BUILD),
                    is_prediction=False)
                self.logger.info("Validation neural network initialized.")

        valid_data_loader.set_sample_list_generator(valid_data_reader, places=self.get_data_run_places(self.args))

        self.valid_data_loader = valid_data_loader
        self.valid_loss = valid_loss
        self.valid_accuracy = accuracy
        # 对训练状态的记录
        self.pre_epoch_valid_loss = float("inf")
        self.standstill_count = 0
        self.logger.info("Validation process initialized.")
        '''
        过程并行化
        '''
        USE_PARALLEL = self.args["use_parallel"]

        # 备份原program，因为compiled_program没有保存
        self.origin_train_prog = self.train_main_prog
        if USE_PARALLEL:
            self.logger.info("Initialize parallel processes...")
            # 设置并行训练的策略
            # 这里可以用参数配置，不过要改的东西很多，所以先写死吧
            build_strategy = fluid.BuildStrategy()
            build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.AllReduce
            # 构建并行过程
            self.train_main_prog = fluid.CompiledProgram(self.train_main_prog).with_data_parallel(
                loss_name=self.train_loss.name,
                places=self.get_data_run_places(self.args),
                build_strategy=build_strategy)
            self.valid_main_prog = fluid.CompiledProgram(self.valid_main_prog).with_data_parallel(
                share_vars_from=self.train_main_prog,
                places=self.get_data_run_places(self.args),
                build_strategy=build_strategy)
            self.logger.info("Parallel processes initialized.")

    def train(self):
        """
        用于训练流程，根据参数完成训练，并使用验证数据对模型效果进行验证
        :return: 无
        """
        APP_NAME = self.args["app_name"]
        MAX_EPOCH = self.args["max_epoch"]
        SNAPSHOT_FREQUENCY = self.args["snapshot_frequency"]

        EARLY_STOPPING = self.args["early_stopping"]
        if EARLY_STOPPING:
            THRESHOLD = self.args["early_stopping_threshold"]
            STANDSTILL_STEP = self.args["early_stopping_stand_times"]

        CONTINUE = self.args["continue_train"]
        MODEL_PATH = self.args["load_model_path"]
        CHECK_POINT = self.args["read_checkpoint"]

        # 定义执行器
        executor = fluid.Executor(self.get_executor_run_places(self.args))
        # 执行初始化
        executor.run(self.train_startup_prog)

        total_step = 0
        step_in_epoch = 0
        total_epoch = 0
        # 读取模型现有的参数并为继续训练进行相应处理
        if CONTINUE and CHECK_POINT:
            info = model_utils.load_train_snapshot(executor, self.origin_train_prog, MODEL_PATH)
            self.logger.info("Model file in {} has been loaded".format(MODEL_PATH))
            if info:
                total_step = info.get("total_step", 0)
                step_in_epoch = info.get("step_in_epoch", 0)
                total_epoch = info.get("epoch", 0)
                self.logger.info("Load train info: {}".format(info))
        elif MODEL_PATH != "":
            # 若是第一次训练且预训练模型参数不为空，则加载预训练模型参数
            model_utils.load_model_params(exe=executor, program=self.origin_train_prog, params_path=MODEL_PATH)
            self.logger.info("Pre-trained model file in {} has been loaded".format(MODEL_PATH))

        self.logger.info("Ready to train the model.Executing...")

        # 执行MAX_EPOCH次迭代save_train_snapshot
        for epoch_id in range(total_epoch, MAX_EPOCH):
            # 一个epoch的训练过程，一个迭代
            total_step, loss = self.__run_train_iterable(executor, total_step, epoch_id, step_in_epoch)
            step_in_epoch = 0
            self.logger.info('Epoch {epoch} done, train mean loss is {loss}'.format(epoch=epoch_id, loss=loss))
            # 进行一次验证集上的验证
            valid_loss, valid_acc = self.__valid(executor)
            self.logger.info(' Epoch {epoch} Validated'.format(epoch=epoch_id))
            # 进行保存
            info = {
                "total_step": total_step,
                "epoch": epoch_id
            }
            file_path = model_utils.save_train_snapshot(executor, self.origin_train_prog,
                                                        file_name="{}_epoch{}".format(APP_NAME, epoch_id),
                                                        train_info=info)
            self.logger.info("Snapshot of training process has been saved as folder {}".format(file_path))
            # 应用早停策略
            if EARLY_STOPPING:
                need_stop = self.early_stopping_strategy(-valid_acc, threshold=THRESHOLD,
                                                         standstill_step=STANDSTILL_STEP)
                if need_stop:
                    self.logger.info("Performance improvement stalled, ending the training process")
                    break
        # 保存现有模型
        file_path = model_utils.save_train_snapshot(executor, self.origin_train_prog, APP_NAME)
        self.logger.info("Training process completed. model saved in {}".format(file_path))

    def __run_train_iterable(self, executor, total_step=0, epoch=0, step_in_epoch=0):
        """
        对训练过程的一个epoch的执行
        """
        PRINT_PER_STEP = self.args["print_per_step"]
        USE_PARALLEL = self.args["use_parallel"]
        NUM_OF_DEVICE = self.args["num_of_device"]
        APP_NAME = self.args["app_name"]
        SNAPSHOT_FREQUENCY = self.args["snapshot_frequency"]
        VALID_FREQUENCY = self.args["validate_frequency_step"]
        WHETHER_VALID = self.args["validate_in_epoch"]
        total_loss = 0
        total_data = 0
        for step, data in enumerate(self.train_data_loader()):
            # 为获取字段名，这里需要改
            if step <= step_in_epoch:
                continue
            batch_size = data[0]['qas_ids'].shape()[0]
            if USE_PARALLEL and batch_size < NUM_OF_DEVICE:
                self.logger.warning("Batch size less than the number of devices. Batch aborted.")
                continue
            loss_value = executor.run(program=self.train_main_prog, feed=data, fetch_list=[self.train_loss])
            total_loss += loss_value[0]
            total_data += 1
            # 打印逻辑
            if PRINT_PER_STEP > 0:
                if step % PRINT_PER_STEP == 0:
                    self.logger.info("Step {step}: loss = {loss}".
                                     format(step=total_step + step, loss=loss_value))
            # 保存逻辑
            info = {
                "total_step": total_step,
                "step_in_epoch": step,
                "epoch": epoch
            }
            if step % SNAPSHOT_FREQUENCY == 0 and step != 0:
                file_path = model_utils.save_train_snapshot(executor, self.origin_train_prog,
                                                            "{}_step{}".format(APP_NAME, total_step + step), info)
                self.logger.info("Snapshot of training process has been saved as folder {}".format(file_path))

            if WHETHER_VALID and step % VALID_FREQUENCY == 0 and step != 0:
                self.__valid(executor)

        mean_loss = total_loss / total_data
        return total_step + step, mean_loss

    def __valid(self, exe):
        """
            对验证过程的一个epoch的执行
        """
        PRINT_PER_STEP = self.args["print_per_step"]
        USE_PARALLEL = self.args["use_parallel"]
        NUM_OF_DEVICE = self.args["num_of_device"]

        total_loss = 0
        total_accuracy = 0
        total_data = 0
        for step, data in enumerate(self.valid_data_loader()):
            batch_size = data[0]['qas_ids'].shape()[0]
            if USE_PARALLEL and batch_size < NUM_OF_DEVICE:
                self.logger.warning("Batch size less than the number of devices. Batch aborted.")
                continue
            loss_value = exe.run(program=self.valid_main_prog, feed=data, fetch_list=[self.valid_loss,
                                                                                      self.valid_accuracy])
            total_loss += loss_value[0]
            total_accuracy += loss_value[1]
            total_data += 1
            if PRINT_PER_STEP > 0:
                if step % PRINT_PER_STEP == 0:
                    self.logger.info(
                        "Valid batch {step} in epoch: loss = {loss}".format(step=step, loss=total_loss / total_data))
        mean_loss = total_loss / total_data
        mean_accuracy = total_accuracy / total_data
        self.logger.info('valid mean loss is {loss}, mean accuracy is {acc}'.format(
            loss=mean_loss,
            acc=mean_accuracy))
        return mean_loss, mean_accuracy

    @staticmethod
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

    @staticmethod
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
        self.logger.info("This epoch promote performance by {}".format(promote_rate))
        if promote_rate < threshold:
            self.standstill_count += 1
            if self.standstill_count > standstill_step:
                return True
        else:
            self.standstill_count = 0
            self.pre_epoch_valid_loss = current_valid_loss
        return False

    def _exclude_from_weight_decay(self, name):
        """exclude_from_weight_decay"""
        if name.find("layer_norm") > -1:
            return True
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return True
        return False


class PredictEngineForMergeModel(object):

    def __init__(self, param, logger,
                 prob_postprocess=lambda x: x,
                 yesno_postprocess=lambda x: x):
        """
        预测引擎
        :param param: 参数实体
        :param logger: 日志
        :param prob_postprocess:
        """
        self.exe = None
        self.args_model_build = param.get_config(param.MODEL_BUILD)
        self.args = param.get_config(param.PREDICT)
        self.logger = logger
        self.predict_program = fluid.Program()
        self.predict_startup = fluid.Program()

        self.loader, self.probs, self.qas_id = self.init_model()

        self.probs_list = []
        self.qas_id_list = []
        self.yesno_list = []
        self.prob_postprocess_op = prob_postprocess
        self.yesno_postprocess_op = yesno_postprocess
        self.answer_list = ["Yes", "No", "Depends"]
        self.logger = logger

    def init_model(self):
        """
        根据模型参数路径读入模型来初始化，包括预测程序编译，模型参数赋值，并行策略
        :param vocab_size: 词典大小
        :return:
        """
        model_path = self.args["load_model_path"]
        self.logger.info("Initializing predict model...")
        self.exe = fluid.Executor(TrainEngine.get_executor_run_places(self.args))
        with fluid.program_guard(self.predict_program, self.predict_startup):
            # 根据gzl的模型来定义网络，输出占位符
            loader, probs, qas_id = classifier.create_model_for_cls_merge(args=self.args_model_build,
                                                                          is_prediction=True)
            self.logger.info("Prediction neural network created.")

        self.logger.info("Prediction neural network parameter initialized.")

        # start_up程序运行初始参数
        self.exe.run(self.predict_startup)

        # 加载模型参数到网络中
        load_model_params(self.exe, model_path, self.predict_program)

        # 若并行，用并行编译program
        if self.args["use_parallel"]:
            build_strategy = fluid.BuildStrategy()
            # 并行策略暂时写死
            build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
            self.predict_program = fluid.CompiledProgram(self.predict_program). \
                with_data_parallel(places=TrainEngine.get_data_run_places(self.args),
                                   build_strategy=build_strategy)

        self.logger.info("Finish initializing predict model!")
        return loader, probs, qas_id

    def predict(self, data_generator, data_examples=[]):
        """
        一个完整的默认配置的预测流程
        :param data_generator: 传入batch化的数据生成器
        :param data_examples: 原始数据
        """
        # 预测
        self._predict(data_generator)
        # 概率后处理
        # self.prob_postprocess(self.prob_postprocess_op, data_examples)
        # 概率转答案
        self.probs_to_yesno()
        # 答案后处理
        # self.yesno_postprocess(self.yesno_postprocess_op, data_examples)

    def _predict(self, data_generator):
        """
        预测功能，根据生成器预测pros，将pros填充到probs_list中
        :param data_generator: 传入batch化的数据生成器
        :return:
        """
        self.logger.info("Start to predict data from input generator...")
        self.probs_list = []
        # 加载数据
        self.loader.set_sample_list_generator(data_generator,
                                              places=TrainEngine.get_data_run_places(self.args))
        self.logger.info("Has get batched data input.")
        # 喂入模型
        for step, feed_data in enumerate(self.loader()):
            probs_batch, qas_id_batch = self.exe.run(self.predict_program,
                                                     feed=feed_data,
                                                     fetch_list=[self.probs, self.qas_id])
            self.probs_list.extend(probs_batch)
            self.qas_id_list.extend(qas_id_batch)
            if step % 10 == 0:
                self.logger.info("Step {} finshed".format(step))
        # 将numpy数据转化成普通的str和int方便打印到文件中
        self.probs_list = [x.tolist() for x in self.probs_list]
        self.qas_id_list = [int(x[0]) for x in self.qas_id_list]
        assert len(self.probs_list) == len(self.qas_id_list)
        self.logger.info("Length of result is {}".format(len(self.probs_list)))
        self.logger.info("Finish calculate the probs!")

    def predict_for_one_sample(self, example):
        # TODO 预测一个样例
        return

    def prob_postprocess(self, op, examples=[]):
        """
        传入一个函数对概率矩阵进行后处理操作，如适当增大Depends小类的概率等
        :param op:
        :return:
        """
        self.logger.info("Start postprocess probs...")
        self.probs_list = list(map(op, self.probs_list))
        self.logger.info("Finish postprocess probs")

    def yesno_postprocess(self, op, examples=[]):
        self.logger.info("Start postprocess answer...")
        self.answer_list = list(map(op, self.answer_list))
        self.logger.info("Finish postprocess answer")

    def probs_to_yesno(self):
        """
        将概率矩阵转化为是否类答案列表
        :return:
        """
        self.logger.info("Start transform probs to yesno answer...")

        answers = []
        for probs in self.probs_list:
            max_index = probs.index(max(probs))
            answers.append(self.answer_list[max_index])
        self.yesno_list = answers

        assert (len(self.yesno_list) == len(self.qas_id_list),
                "num of data from generator and num of data from test_examples should be same")
        self.logger.info("Finish predict the Yesno answer, num of predict examples is {}."
                         .format(len(self.yesno_list)))

    def write_to_json(self, headers=("id", "yesno_answer"), attach_data={}):
        """

        :param headers:  要写入的表头，必须在data_dict是定义好的
        :param attach_data: 附加的信息，也按表头：数据的形式。注意长度必须与预测结果列表一致
        :return: 保存文件路径
        """
        header_dict = {"id": self.qas_id_list,
                       "yesno_answer": self.yesno_list}
        attach_data_len = len(self.qas_id_list)
        for key in attach_data:
            assert (attach_data_len == len(attach_data[key]), "Length of attach data must be equal to result list.")
        data_dict = {}
        for header in headers:
            data_dict[header] = header_dict[header]

        data_dict.update(attach_data)

        predict_file = file_utils.get_default_filename(self.args)
        result_list = []
        result_len = len(self.qas_id_list)
        for i in range(result_len):
            record = {}
            for key in data_dict:
                record[key] = data_dict[key][i]
            result_list.append(record)

        return file_utils.save_file(result_list, file_name=predict_file, file_type="result", file_format="json")

    def write_full_info(self, attach_data={}):
        """
        将预测的结果以完整形式（包括概率）写入csv
        :return:
        """
        probs_list = np.array(self.probs_list)
        print(probs_list)
        attach = {}
        for i in range(probs_list.shape[1]):
            attach[self.answer_list[i]] = probs_list[:, i]

        attach.update(attach_data)
        return self.write_to_json(attach_data=attach)
