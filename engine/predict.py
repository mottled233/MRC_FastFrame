import paddle.fluid as fluid
import os
import numpy as np
from util.model_utils import load_model
from engine.train import TrainEngine
import json as js

class PredictEngine(object):
    def __init__(self, args):
        '''
        初始化参数
        '''
        self.exe = None
        self.args = args
        self.predict_program = fluid.Program()
        self.predict_startup = fluid.Program()
        self.feed_target_names =[]
        self.fetch_targets = []
        self.probs_list = []
        self.yesno_list = []

    def init_model(self, model_path):
        self.exe = fluid.Executor(fluid.CPUPlace())
        [self.predict_program, self.feed_target_names, self.fetch_targets] = (
            load_model(model_path=model_path, exe=self.exe))

        if self.args["use_parallel"]:
            buildStrategy = fluid.BuildStrategy()
            buildStrategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
            self.predict_program = fluid.CompiledProgram(self.predict_program).\
                with_data_parallel(places=TrainEngine.get_data_run_places(self.args)
                                   , build_strategy=buildStrategy)

    def predict(self, data_generator):
        #对batch_size的处理还有一点问题
        for data in data_generator():
            for sample in data:
                feed_data = dict(zip(self.feed_target_names, [sample]))
                prob = self.exe.run(self.predict_program,
                                feed=feed_data,
                                fetch_list=self.fetch_targets)
                self.probs_list.append(prob)
        print(self.probs_list)

    #传入一个函数对概率矩阵进行后处理操作，如适当增大Depends小类的概率等
    def probs_reprocess(self, op):
        self.probs_list = list(map(op, self.probs_list))

    def probs_to_yesno(self):
        answer_list = ["Yes", "No", "Depends"]
        answers = []
        for probs in self.probs_list:
            assert type(probs) == list
            max_index = probs.index(max(probs))
            answers.append(answer_list[max_index])
        self.yesno_list = answers

    def yesno_reprocess(self, test_examples):
        # TODO 根据examples里面一些特定的文字信息按规则对答案进行后处理修正
        return

    def write_to_json(self, predict_file, test_examples):
        with open(predict_file, "w", encoding= 'utf-8') as f:
            for (i, example) in enumerate(test_examples):
                yesno_answer = self.yesno_list[i]
                id = example.qas_id
                js.dump({"id":id, "yesno_answer":yesno_answer}, f)
                f.write('\n')

    def get_executor_run_places(self, args):
        USE_GPU = args["use_gpu"]
        if USE_GPU:
            places = fluid.CUDAPlace(0)
        else:
            places = fluid.CPUPlace()
        return places


if __name__ == '__main__':
    args = {
        "use_parallel": False,
        "use_gpu": False,
        "num_of_device": 1,
        "batch_size": 32
    }
    predict_engine = PredictEngine(args)
    predict_engine.init_model(os.getcwd() + "/infer_model")
    def fake_sample_generator():
        for _ in range(50):
            sample_x = np.array(np.random.random((1, 28, 28)), dtype=np.float32)
            yield sample_x
    generator = fluid.io.batch(fake_sample_generator, batch_size=args["batch_size"])
    predict_engine.predict(generator)

