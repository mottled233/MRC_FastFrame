import paddle.fluid as fluid
import collections

import util.model_utils as model_utils
import util.engine_utils as engine_utils
from util.util_logging import UtilLogging
from util.util_filepath import *


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

    def __init__(self, args, network):
        """
        对训练过程进行初始化
        :param args: 预测参数
        :param network: 网络构建对象
        """
        self.args = args
        self.logger = UtilLogging(args, __name__)
        self.app_name = self.args["app_name"]
        self.network = network

        output_dir = get_fullurl('result_predict', self.app_name, 'dir')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        '''
        创建预测过程
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
        self.executor = fluid.Executor(engine_utils.get_executor_run_places(self.args))
        # 执行初始化
        self.executor.run(self.predict_startup_prog)
        # 读取保存的模型
        self._load_process(self.executor, self.predict_main_prog)

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
                places=engine_utils.get_data_run_places(self.args),
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
        self.network.predict()
        reader, fetch_data = self.network.create_model()
        return reader, fetch_data

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
        MODEL_PATH = self.args["load_model_path"]

        model_utils.load_model_params(exe=executor, program=main_prog, params_path=MODEL_PATH)
        self.logger.info("Pre-trained model file in {} has been loaded".format(MODEL_PATH))
        return

    def _set_data_source(self, data_loader, data_source):
        data_loader.set_sample_list_generator(data_source, places=engine_utils.get_data_run_places(self.args))

    def predict(self, predict_data_generator, **kwargs):
        """
        用于训练流程，根据参数完成训练，并使用验证数据对模型效果进行验证
        :return: 无
        """
        # 设置数据集
        self._set_data_source(self.predict_data_loader, predict_data_generator)
        # 定义执行器
        executor = self.executor
        kwargs['output_file'] = self.app_name + '/predictions'
        self._predict(executor, **kwargs)

    def _predict(self, exe, **kwargs):
        """

        :param exe:
        :param kwargs:
        :return:
        """
        return {}
