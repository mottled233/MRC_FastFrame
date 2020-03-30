import paddle.fluid as fluid
import numpy as np

from util.model_utils import load_model_params
from engine.train import TrainEngine
from model.classifier import create_model
import util.util_filepath as file_utils


class PredictEngine(object):
    def __init__(self, param, logger, vocab_size,
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
        self.vocab_size = vocab_size

        self.loader, self.probs, self.qas_id = self.init_model(vocab_size)

        self.probs_list = []
        self.qas_id_list = []
        self.yesno_list = []
        self.prob_postprocess_op = prob_postprocess
        self.yesno_postprocess_op = yesno_postprocess
        self.answer_list = ["Yes", "No", "Depends"]
        self.logger = logger

    def init_model(self, vocab_size):
        """
        根据模型参数路径读入模型来初始化，包括预测程序编译，模型参数赋值，并行策略
        :param vocab_size: 词典大小
        :return:
        """
        model_path = self.args["load_model_path"]
        self.logger.info("Initializing predict model...")
        self.exe = fluid.Executor(TrainEngine.get_executor_run_places(self.args))
        with fluid.program_guard(self.predict_program, self.predict_startup):
            with fluid.unique_name.guard():
                # 根据gzl的模型来定义网络，输出占位符
                loader, probs, qas_id = create_model(args=self.args_model_build, vocab_size=vocab_size,
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
        self.qas_id_list = []
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

    def write_to_json(self, name="", headers=("id", "yesno_answer"), attach_data={}):
        """

        :param headers:  要写入的表头，必须在data_dict是定义好的
        :param attach_data: 附加的信息，也按表头：数据的形式。注意长度必须与预测结果列表一致
        :return: 保存文件路径
        """
        header_dict = {"id": self.qas_id_list,
                     "yesno_answer": self.yesno_list}
        attach_data_len = len(self.qas_id_list)
        for key in attach_data:
            assert(attach_data_len == len(attach_data[key]), "Length of attach data must be equal to result list.")
        data_dict = {}
        for header in headers:
            data_dict[header] = header_dict[header]

        data_dict.update(attach_data)
        if name == "":
            predict_file = file_utils.get_default_filename(self.args)
        else:
            predict_file = name
        result_list = []
        result_len = len(self.qas_id_list)
        for i in range(result_len):
            record = {}
            for key in data_dict:
                record[key] = data_dict[key][i]
            result_list.append(record)

        return file_utils.save_file(result_list, file_name=predict_file, file_type="result", file_format="json_dump")

    def write_full_info(self, name="", attach_data={}):
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
        return self.write_to_json(name=name, attach_data=attach)




