import paddle
import paddle.fluid as fluid
import os
import numpy as np
from util.model_utils import load_model_params,load_model
import json as js
#from model.classifier import create_model
import time

class PredictEngine(object):

    def __init__(self, test_data_reader, args):
        '''
        创建预测过程
        '''
        self.args = args
        self.predict_prog = fluid.Program()
        self.predict_startup = fluid.Program()
        with fluid.program_guard(self.predict_prog, self.predict_startup):
            with fluid.unique_name.guard():
                test_data_loader, probs = self.create_model(self.args, is_prediction=True)  # 替换成gzl模型
        test_data_loader.set_sample_list_generator(test_data_reader, places=self.get_data_run_places(self.args))
        self.test_data_loader = test_data_loader
        self.probs = probs
        self.results = []

    def create_model(self, args, is_prediction):
        x = fluid.data(name="x", dtype='float32', shape=[None,4])
        y = fluid.data(name="y", dtype='float32', shape=[None,1])
        feed_list = [x, y]
        loader = fluid.io.DataLoader.from_generator(feed_list=feed_list, capacity=1, iterable=True)
        y_predict = fluid.layers.fc(input=x, size=10)
        if is_prediction:
            return loader, y_predict

    def predict(self):
        # 定义执行器
        predict_exe = fluid.Executor(self.get_executor_run_places(self.args))
        # 执行初始化
        predict_exe.run(self.predict_startup)
        # 加载模型或加载模型参数
        load_model_params(predict_exe, os.getcwd() + '/my_paddle_model', self.predict_prog)
        #[inference_program, feed_target_names, fetch_targets] = \
        #    (load_model(os.getcwd() + "/my_paddle_model", predict_exe))
        results = []
        for data in self.test_data_loader():
            batch_size = data[0]['x'].shape()[0]
            if batch_size < 2:
                print("abort batch")
                continue
            result = predict_exe.run(feed=data, fetch_list=[self.probs.name])
            results.append(result)
            self.results = results

        print("-------------- prediction results --------------")
        for index, result in enumerate(results):
            print(str(index) + '\t{}'.format(result))

    def write_to_json(self, predict_file, test_examples):
        answers = []
        for (i, probs) in enumerate(self.results):
            assert type(probs) == list
            max_index = probs.index(max(probs))
            answer_list = ["Yes", "No", "Depends"]
            answer = answer_list[max_index]
            answers.append(answer)
        with open(predict_file, "w", encoding= 'utf-8') as f:
            for (i, example) in enumerate(test_examples):
                yesno_answer = answers[i]
                id = example.qas_id
                js.dump({"id":id, "yesno_answer":yesno_answer}, f)
                f.write('\n')



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



if __name__ == '__main__':
    args = {
        "max_epoch": 100,
        "early_stopping": True,
        "warm_up": False,
        "continue_train": False,
        "model_path": "",
        "use_parallel": False,
        "use_gpu": False,
        "num_of_device": 2,
        "batch_size": 32,
        "base_learning_rate": 0.001,
        "learning_rate_strategy": "linear_warm_up_and_decay",
        "start_learning_rate": 1e-04,
        "warm_up_step": 200,
        "end_learning_rate": 1e-04,
        "decay_step": 500,
        "optimizer": "adagrad",
        "adagrad_epsilon": 1e-06,
        "adagrad_accumulator_value": 0,
        "early_stopping_threshold": 0.03,
        "early_stopping_times": 5
    }

    ground_truth = np.random.random(size=(1, 4)).astype('int64')
    def fake_sample_generator():
        for _ in range(1000):
            sample_x = np.random.random(size=(4,)).astype('float32')
            sample_y = np.dot(ground_truth, sample_x)
            yield sample_x, sample_y
    reader = fluid.io.batch(fake_sample_generator, batch_size=args["batch_size"])
    predict_engine = PredictEngine(reader, args)
    t3 = time.time()
    predict_engine.predict()
    t4 = time.time()
