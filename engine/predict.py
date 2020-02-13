import paddle.fluid as fluid
from util.model_utils import load_model_params
from engine.train import TrainEngine
from model.classifier import create_model
import json as js


class PredictEngine(object):
    def __init__(self, args, logger):
        """
        定义参数
        :param args:
        """
        self.exe = None
        self.param = args
        self.args = args.get_config(args.PREDICT)
        self.predict_program = fluid.Program()
        self.predict_startup = fluid.Program()
        self.loader = None
        self.probs = None
        self.qas_id = ""
        self.probs_list = []
        self.qas_id_list = []
        self.yesno_list = []
        self.logger = logger

    def init_model(self, vocab_size):
        """
        根据模型参数路径读入模型来初始化，包括预测程序编译，模型参数赋值，并行策略
        :param model_path:
        :return:
        """
        model_path = self.args["model_path"]
        self.logger.info("Initializing predict model...")
        self.exe = fluid.Executor(TrainEngine.get_executor_run_places(self.args))
        with fluid.program_guard(self.predict_program, self.predict_startup):
            # 根据gzl的模型来定义网络，输出占位符
            self.loader, self.probs, self.qas_id = create_model(args=self.param.get_config(self.param.MODEL_BUILD),
                                                                vocab_size=vocab_size,
                                                                is_prediction=True)
            self.logger.info("Prediction neural network created.")
        # start_up程序运行初始参数
        self.exe.run(self.predict_startup)
        # 加载模型参数到网络中
        load_model_params(self.exe, model_path, self.predict_program)
        self.logger.info("Prediction neural network parameter initialized.")
        # 若并行，用并行编译program
        if self.args["use_parallel"]:
            buildStrategy = fluid.BuildStrategy()
            # 并行策略暂时写死
            buildStrategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
            self.predict_program = fluid.CompiledProgram(self.predict_program). \
                with_data_parallel(places=TrainEngine.get_data_run_places(self.args),
                                   build_strategy=buildStrategy)
        self.logger.info("Finish initializing predict model!")

    def predict(self, data_generator):
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
        for feed_data in self.loader():
            probs_batch, qas_id_batch = self.exe.run(self.predict_program,
                                                     feed=feed_data,
                                                     fetch_list=[self.probs, self.qas_id])
            self.probs_list.extend(probs_batch)
            self.qas_id_list.extend(qas_id_batch)
        # 将numpy数据转化成普通的str和int方便打印到文件中
        self.probs_list = [x.tolist() for x in self.probs_list]
        self.qas_id_list = [int(x[0]) for x in self.qas_id_list]
        assert len(self.probs_list) == len(self.qas_id_list)
        self.logger.info("Length of result is {}".format(len(self.probs_list)))
        self.logger.info("Finish calculate the probs!")

    def predict_for_one_sample(self, example):
        # TODO 预测一个样例
        return

    def generate_result(self):
        """
        后处理的主函数，外部直接调用即可, 最终结果是生成答案写到本地
        :return:
        """
        self.probs_to_yesno()
        self.logger.info("Finish predict the Yesno answer, num of predict examples is {}."
                         .format(len(self.yesno_list)))
        assert len(self.yesno_list) == len(self.qas_id_list), \
            "num of data from generator and num of data from test_examples should be same"
        self.write_to_json()
        self.logger.info("Finish write data to {}, finish predict process!"
                         .format(self.args["result_file_path"]))

    def probs_reprocess(self, op):
        """
        传入一个函数对概率矩阵进行后处理操作，如适当增大Depends小类的概率等
        :param op:
        :return:
        """
        self.probs_list = list(map(op, self.probs_list))

    def probs_to_yesno(self):
        """
        将概率矩阵转化为是否类答案列表
        :return:
        """
        answer_list = ["Yes", "No", "Depends"]
        answers = []
        for probs in self.probs_list:
            max_index = probs.index(max(probs))
            answers.append(answer_list[max_index])
        self.yesno_list = answers

    def yesno_reprocess(self, test_examples):
        # TODO 根据examples里面一些特定的文字信息按规则对答案进行后处理修正
        return

    def write_to_json(self):
        """
        将结果写入到本地result路径下
        :return:
        """
        predict_file = self.args["result_file_path"]
        with open(predict_file, "w", encoding='utf-8') as f:
            for (i, qas_id) in enumerate(self.qas_id_list):
                yesno_answer = self.yesno_list[i]
                js.dump({"id": qas_id, "yesno_answer": yesno_answer}, f)
                f.write('\n')
