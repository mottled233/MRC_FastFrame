import paddle.fluid as fluid
import numpy as np
import util.model_utils as model_utils
from model.network.network import Network
import model.optimizer
import model.lr_stategy as lr_strategy
from util.util_filepath import *
from postprocess.postprocess import Postprocess
import collections


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

    def __init__(self, params, logger, train_data_generator, valid_data_generator=None, valid_data=None):
        """
        对训练过程进行初始化
        :param params:
        :param logger:
        """
        self.params = params
        self.args = params.get_config(params.TRAIN)
        self.app_name = self.args["app_name"]
        self.logger = logger
        self.max_seq_length = params.get_config(params.MODEL_BUILD)['max_seq_length']
        self.train_data_generator = train_data_generator
        self.valid_data_generator = valid_data_generator
        self.valid_data = valid_data
        output_dir = get_fullurl('result_valid', self.app_name, 'dir')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
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
                train_data_loader, train_loss, train_fetch_data, enc_output_list = self._init_train_model()
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
        self.enc_output_list = enc_output_list
        self.logger.info("Training process initialized.")

        '''
        创建验证过程
        '''
        VALIDATE = self.args["do_validate"]
        if VALIDATE:
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
        self.executor = fluid.Executor(self.get_executor_run_places(self.args))
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

            if VALIDATE:
                self.valid_main_prog = fluid.CompiledProgram(self.valid_main_prog).with_data_parallel(
                    share_vars_from=self.train_main_prog,
                    places=self.get_data_run_places(self.args),
                    build_strategy=build_strategy)
            self.logger.info("Parallel processes initialized.")

    def _init_train_model(self):
        """
        定义训练过程中如何初始化网络
        :param vocab_size: 词典大小，注意当参数设置词典大小时该项无效
        :return: 必须为 reader, loss, fetch_data，其中fetch_data是一个字典，可以存放一些附加的信息，之后会被保存在
                 self.train_fetch_data里。
        """
        # 一些网络定义
        reader, loss, _, _, _, enc_output_list = mrc.create_model(
            self.params.get_config(self.params.MODEL_BUILD),
            is_prediction=False,
            is_validate=False,
            is_soft_label=self.args['is_soft_label'],
            t=self.args['temperature']
        )
        return reader, loss, {}, enc_output_list

    def _init_validate_model(self):
        """
        定义验证过程中如何初始化网络
        :param vocab_size: 词典大小，注意当参数设置词典大小时该项无效
        :return: 必须为 reader, fetch_data，其中fetch_data是一个字典，可以存放一些附加的信息，之后会被保存在
                 self.valid_fetch_data里。
        """
        # 一些网络定义
        reader, loss, unique_ids, start_logits, end_logits, _ = mrc.create_model(
            self.params.get_config(self.params.MODEL_BUILD),
            is_prediction=False,
            is_validate=True
        )
        return reader, {'loss': loss, 'unique_ids': unique_ids, 'start_logits': start_logits, 'end_logits': end_logits}

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
        如果想从预训练模型（或某个基线）开始训练，请设置continue_train为False，
        如果想继续训练，请设置请设置continue_train为True，
        如果想从断点训练，请设置read_checkpoint为true。
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
        data_loader.set_sample_list_generator(data_source, places=self.get_data_run_places(self.args))

    def train(self):
        """
        用于训练流程，根据参数完成训练，并使用验证数据对模型效果进行验证
        :return: 无
        """
        APP_NAME = self.args["app_name"]
        MAX_EPOCH = self.args["max_epoch"]
        EARLY_STOPPING = self.args["early_stopping"]
        GRADUAL_UNFREEZE = self.args["gradual_unfreeze"]
        FREEZE = self.args["freeze"]

        if EARLY_STOPPING:
            THRESHOLD = self.args["early_stopping_threshold"]
            STANDSTILL_STEP = self.args["early_stopping_stand_times"]
        VALIDATE = self.args["do_validate"]

        if GRADUAL_UNFREEZE:
            NUM_LAYERS = self.args["num_hidden_layers"]
            for i in range(NUM_LAYERS):
                self.enc_output_list[i].stop_gradient = True
            self.logger.info('Use GRADUAL_UNFREEZE, before training, all encode layer freeze!')

        if FREEZE:
            NUM_LAYERS = self.args["num_hidden_layers"]
            UNFREEZE = self.args["unfreeze_layers"]
            self.enc_output_list[NUM_LAYERS - 1 - UNFREEZE].stop_gradient = True

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

            if GRADUAL_UNFREEZE:
                study_process = (epoch_id + 1 - total_epoch) / (MAX_EPOCH + 1 - total_epoch)
                num_unfreeze = int(NUM_LAYERS * study_process)
                for i in range(NUM_LAYERS, NUM_LAYERS - num_unfreeze, -1):
                    self.enc_output_list[i - 1].stop_gradient = False
                    self.logger.info(
                        'Epoch {epoch} start, {layer_index} encode layer unfreezed'.format(
                            epoch=epoch_id, layer_index=i))

            # 一个epoch的训练过程，一个迭代
            total_step, loss = self._run_train_iterable(executor, total_step, epoch_id, step_in_epoch)

            if total_step == False:
                return

            step_in_epoch = 0
            self.logger.info('Epoch {epoch} done, train mean loss is {loss}'.format(epoch=epoch_id, loss=loss))
            if VALIDATE:
                # 进行一次验证集上的验证
                result = self._valid(executor, (self.app_name + '/predictions_epoch_' + str(epoch_id)))
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

            if os.path.isfile("stop_signal"):
                return
        # 保存现有模型
        file_path = model_utils.save_train_snapshot(executor, self.origin_train_prog, APP_NAME)
        self.logger.info("Training process completed. model saved in {}".format(file_path))

    def _run_train_iterable(self, executor, total_step=0, epoch=0, step_in_epoch=0):
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
        valid_step_loss = 0
        batch_loss = 0
        for step, data in enumerate(self.train_data_loader()):
            # 为获取字段名，这里需要改
            if step_in_epoch != 0 and step <= step_in_epoch:
                continue
            data_key = list(data[0].keys())[0]
            batch_size = data[0][data_key].shape()[0]
            if USE_PARALLEL and batch_size < NUM_OF_DEVICE:
                self.logger.warning("Batch size less than the number of devices. Batch aborted.")
                continue
            fetch_value = executor.run(program=self.train_main_prog, feed=data, fetch_list=[self.train_loss, self.lr])
            total_loss += fetch_value[0]
            total_data += 1
            batch_loss += fetch_value[0]
            valid_step_loss += fetch_value[0]
            # 打印逻辑
            if PRINT_PER_STEP > 0:
                if step % PRINT_PER_STEP == 0:
                    self.logger.info("Step {step}: loss = {loss}, lr = {lr}".
                                     format(step=total_step + step, loss=batch_loss / PRINT_PER_STEP,
                                            lr=fetch_value[1]))
                    batch_loss = 0
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
                self._valid(executor, output_file_name=self.app_name + '/predictions_step_' + str(total_step + step))
                self.logger.info("from {}-{} step: mean loss = {}"
                                 .format(total_step + step - VALID_FREQUENCY,
                                         total_step + step, valid_step_loss / VALID_FREQUENCY))
                valid_step_loss = 0
                if os.path.isfile("stop_signal"):
                    return False, False

        mean_loss = total_loss / total_data
        return total_step + step, mean_loss

    def _valid(self, exe, output_file_name=''):
        """
            对验证过程的一个epoch的执行
        """
        RawResult = collections.namedtuple("RawResult",
                                           ["unique_id", "start_logits", "end_logits"])
        output_prediction_file = get_fullurl(file_type='result_valid', file_name=output_file_name, file_format='json')
        output_nbest_file = output_prediction_file.replace('predictions', 'nbest')
        PRINT_PER_STEP = self.args["print_per_step"]
        USE_PARALLEL = self.args["use_parallel"]
        NUM_OF_DEVICE = self.args["num_of_device"]

        total_loss = 0
        total_data = 0
        batch_loss = 0
        all_results = []
        for step, data in enumerate(self.valid_data_loader()):
            data_key = list(data[0].keys())[0]
            batch_size = data[0][data_key].shape()[0]
            if USE_PARALLEL and batch_size < NUM_OF_DEVICE:
                self.logger.warning("Batch size less than the number of devices. Batch aborted.")
                continue
            fetch_value = exe.run(program=self.valid_main_prog, feed=data,
                                  fetch_list=[self.valid_fetch_data['loss'],
                                              self.valid_fetch_data['unique_ids'],
                                              self.valid_fetch_data['start_logits'],
                                              self.valid_fetch_data['end_logits']])
            total_loss += fetch_value[0]
            batch_loss += fetch_value[0]
            unique_ids = [int(x[0]) for x in fetch_value[1]]
            start_logits = [x.tolist() for x in fetch_value[2]]
            end_logits = [x.tolist() for x in fetch_value[3]]
            for i in range(len(unique_ids)):
                all_results.append(RawResult(unique_id=unique_ids[i],
                                             start_logits=start_logits[i],
                                             end_logits=end_logits[i]))
            total_data += 1
            if PRINT_PER_STEP > 0:
                if step % PRINT_PER_STEP == 0:
                    self.logger.info("Valid batch {step} in epoch: loss = {loss}"
                                     .format(step=step, loss=batch_loss / PRINT_PER_STEP))
                    batch_loss = 0
        prediction_data = {'examples': self.valid_data[1], 'features': self.valid_data[2], 'results': all_results,
                           'original_file': self.valid_data[0], 'prediction_file': output_prediction_file}
        prediction_param = {'n_best_size': self.args['n_best_size'],
                            'max_answer_length': self.args['max_answer_length'],
                            'do_lower_case': True,
                            'output_prediction_file': output_prediction_file,
                            'output_nbest_file': output_nbest_file}

        write_prediction(write_prediction_data, write_prediction_param)
        eval_data = {'original_file': self.valid_data[0], 'prediction_file': output_prediction_file}
        results = do_eval(None, eval_data)
        # tmp_result = get_eval(self.valid_processor.raw_data_path, output_prediction_file)
        # best_f1 = float(tmp_result['F1'])
        # best_em = float(tmp_result['EM'])
        mean_loss = total_loss / total_data

        self.logger.info('valid mean loss is {loss}, f1={f1}, em={em}'.format(loss=mean_loss, f1=results['F1'],
                                                                              em=results['EM']))
        return mean_loss

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


class PredictEngine(object):
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

    def __init__(self, params, logger, predict_data_generator, predict_data):
        """
        对训练过程进行初始化
        :param params:
        :param logger:
        """
        self.params = params
        self.args = params.get_config(params.TRAIN)
        self.logger = logger
        self.max_seq_length = params.get_config(params.MODEL_BUILD)['max_seq_length']
        self.predict_data_generator = predict_data_generator
        self.predict_data = predict_data
        self.app_name = self.args["app_name"]
        output_dir = get_fullurl('result_predict', self.app_name, 'dir')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        '''
        创建训练过程
        '''
        self.logger.info("Initializing training process...")
        self.predict_main_prog = fluid.Program()
        self.predict_startup_prog = fluid.Program()

        with fluid.program_guard(self.predict_main_prog, self.predict_startup_prog):
            # 使用 fluid.unique_name.guard() 实现与test program的参数共享
            with fluid.unique_name.guard():
                # 初始化网络结构
                self.logger.info("Initializing predict neural network...")
                predict_data_loader, predict_fetch_data = self._init_predict_model()
                self.logger.info("Training neural network initialized.")

        # 属性化
        self.predict_data_loader = predict_data_loader
        self.predict_fetch_data = predict_fetch_data
        self.logger.info("Training process initialized.")

        '''
        读取保存的模型
        '''
        # 定义执行器
        self.executor = fluid.Executor(self.get_executor_run_places(self.args))
        # 执行初始化
        self.executor.run(self.predict_startup_prog)
        # 读取保存的模型
        self.predict_status = self._load_process(self.executor, self.predict_main_prog)

        # 对训练状态的记录
        self.pre_epoch_valid_loss = float("inf")
        self.standstill_count = 0
        self.logger.info("Validation process initialized.")

        '''
        过程并行化
        '''
        USE_PARALLEL = self.args["use_parallel"]

        # 备份原program，因为compiled_program没有保存
        self.origin_train_prog = self.predict_main_prog
        if USE_PARALLEL:
            self.logger.info("Initialize parallel processes...")
            # 设置并行训练的策略
            # 这里可以用参数配置，不过要改的东西很多，所以先写死吧
            build_strategy = fluid.BuildStrategy()
            build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.AllReduce
            # 构建并行过程
            self.predict_main_prog = fluid.CompiledProgram(self.predict_main_prog).with_data_parallel(
                places=self.get_data_run_places(self.args),
                build_strategy=build_strategy)
            self.logger.info("Parallel processes initialized.")

    def _init_predict_model(self):
        """
        定义验证过程中如何初始化网络
        :param vocab_size: 词典大小，注意当参数设置词典大小时该项无效
        :return: 必须为 reader, fetch_data，其中fetch_data是一个字典，可以存放一些附加的信息，之后会被保存在
                 self.valid_fetch_data里。
        """
        # 一些网络定义
        reader, unique_ids, start_logits, end_logits = mrc.create_model(
            self.params.get_config(self.params.MODEL_BUILD),
            is_prediction=True,
            is_validate=False,
            t=self.args['temperature']
        )
        return reader, {'unique_ids': unique_ids, 'start_logits': start_logits, 'end_logits': end_logits}

    def _load_process(self, executor, main_prog):
        """
        读取模型的过程
        :param executor:
        :param main_prog:
        :return: 字典，保存当前训练状态, 将保存在self.train_status中
        """
        MODEL_PATH = self.args["load_model_path"]

        model_utils.load_model_params(exe=executor, program=main_prog, params_path=MODEL_PATH)
        self.logger.info("Pre-trained model file in {} has been loaded".format(MODEL_PATH))

    def _set_data_source(self, data_loader, data_source):
        data_loader.set_sample_list_generator(data_source, places=self.get_data_run_places(self.args))

    def predict(self):
        """
        用于训练流程，根据参数完成训练，并使用验证数据对模型效果进行验证
        :return: 无
        """
        APP_NAME = self.args["app_name"]

        # 设置数据集
        self._set_data_source(self.predict_data_loader, self.predict_data_generator)
        # 定义执行器
        executor = self.executor
        self._predict(executor, self.app_name + '/predictions')

    def _predict(self, exe, output_file_name=''):
        """
            对验证过程的一个epoch的执行
        """
        RawResult = collections.namedtuple("RawResult",
                                           ["unique_id", "start_logits", "end_logits"])
        output_prediction_file = get_fullurl(file_type='result_predict', file_name=output_file_name, file_format='json')
        output_nbest_file = output_prediction_file.replace('predictions', 'nbest')
        output_full_info = output_prediction_file.replace('predictions', 'full_info')
        PRINT_PER_STEP = self.args["print_per_step"]
        USE_PARALLEL = self.args["use_parallel"]
        NUM_OF_DEVICE = self.args["num_of_device"]

        total_data = 0
        all_results = []
        for step, data in enumerate(self.predict_data_loader()):
            data_key = list(data[0].keys())[0]
            batch_size = data[0][data_key].shape()[0]
            if USE_PARALLEL and batch_size < NUM_OF_DEVICE:
                self.logger.warning("Batch size less than the number of devices. Batch aborted.")
                continue
            fetch_value = exe.run(program=self.predict_main_prog, feed=data,
                                  fetch_list=[self.predict_fetch_data['unique_ids'],
                                              self.predict_fetch_data['start_logits'],
                                              self.predict_fetch_data['end_logits']])
            unique_ids = [int(x[0]) for x in fetch_value[0]]
            start_logits = [x.tolist() for x in fetch_value[1]]
            end_logits = [x.tolist() for x in fetch_value[2]]
            for i in range(len(unique_ids)):
                all_results.append(RawResult(unique_id=unique_ids[i],
                                             start_logits=start_logits[i],
                                             end_logits=end_logits[i]))
            total_data += 1
            if PRINT_PER_STEP > 0:
                if step % PRINT_PER_STEP == 0:
                    self.logger.info("Predict batch {step} in epoch"
                                     .format(step=step))

                write_prediction(self.predict_data[1], self.predict_data[2], all_results,
                          n_best_size=self.args['n_best_size'],
                          max_answer_length=self.args['max_answer_length'],
                          do_lower_case=True, output_prediction_file=output_prediction_file,
                          output_nbest_file=output_nbest_file)
        self.write_predictions_with_example(output_prediction_file, output_full_info)

    def write_predictions_with_example(self, results_file, output_file):
        examples = self.predict_processor.examples
        results = read_file(file_name=results_file)
        infos = []
        for example in examples:
            predict_answer = results[example['qid']]
            info = {'qid': example['qid'],
                    'doc': ''.join(example['doc_tokens']),
                    'question': example['question'],
                    'answer': example['answer'],
                    'predict_answer': predict_answer}
            infos.append(info)
        with open(output_file, "w") as writer:
            writer.write(json.dumps(infos, indent=4, ensure_ascii=False) + "\n")

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


class TrainEngineForMRCWithTag(TrainEngine):
    def _init_train_model(self):
        """
        定义训练过程中如何初始化网络
        :param vocab_size: 词典大小，注意当参数设置词典大小时该项无效
        :return: 必须为 reader, loss, fetch_data，其中fetch_data是一个字典，可以存放一些附加的信息，之后会被保存在
                 self.train_fetch_data里。
        """
        # 一些网络定义
        reader, loss, _, _, _, enc_output_list = mrc.create_model_with_lexical_inf(
            self.params.get_config(self.params.MODEL_BUILD),
            is_prediction=False,
            is_validate=False,
            is_soft_label=self.args['is_soft_label'],
            t=self.args['temperature']
        )
        return reader, loss, {}, enc_output_list

    def _init_validate_model(self):
        """
        定义验证过程中如何初始化网络
        :param vocab_size: 词典大小，注意当参数设置词典大小时该项无效
        :return: 必须为 reader, fetch_data，其中fetch_data是一个字典，可以存放一些附加的信息，之后会被保存在
                 self.valid_fetch_data里。
        """
        # 一些网络定义
        reader, loss, unique_ids, start_logits, end_logits, _ = mrc.create_model_with_lexical_inf(
            self.params.get_config(self.params.MODEL_BUILD),
            is_prediction=False,
            is_validate=True
        )
        return reader, {'loss': loss, 'unique_ids': unique_ids, 'start_logits': start_logits, 'end_logits': end_logits}


class PredictEngineWithTag(PredictEngine):
    def _init_predict_model(self):
        """
        定义验证过程中如何初始化网络
        :param vocab_size: 词典大小，注意当参数设置词典大小时该项无效
        :return: 必须为 reader, fetch_data，其中fetch_data是一个字典，可以存放一些附加的信息，之后会被保存在
                 self.valid_fetch_data里。
        """
        # 一些网络定义
        reader, unique_ids, start_logits, end_logits = mrc.create_model_with_lexical_inf(
            self.params.get_config(self.params.MODEL_BUILD),
            is_prediction=True,
            is_validate=False,
            t=self.args['temperature']
        )
        return reader, {'unique_ids': unique_ids, 'start_logits': start_logits, 'end_logits': end_logits}
