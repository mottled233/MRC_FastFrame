import paddle
import paddle.fluid as fluid
import numpy as np
import time
import sys
from engine.train import TrainEngine as TrainEngine
from engine.predict import PredictEngine as PredictEngine
from data.Dataset import Dataset
from preprocess.preprocess import PreProcess
from preprocess.preprocess_for_mt import ProcessorForMultiTask
from util.util_parameter import UtilParameter as UParam
from util.util_logging import UtilLogging as ULog
import util.util_tool as util_tool

if __name__ == "__main__":
    # 设置参数
    param = UParam()
    param.read_config_file("config_test")
    param.set_config(sys.argv[1:])
    args = param.get_config(param.GLOBAL)
    # 初始化日志
    logger = ULog(param)

    app_name = args["app_name"]

    # corpus_cleaner = Corpus_cleaner()
    # # corpus_cleaner.read_from_json("pretrain_corpus.json")
    # corpus_cleaner.read_from_src()
    # docs = corpus_cleaner.get_docs()
    # for i in range(10):
    #     print(docs[i])
    #     print("###########################################################")

    # 读取数据集
    datasets = Dataset(logger=logger, args=param.get_config(param.DATASET))
    # datasets.read_dataset(div_nums=[7, 2, 1])
    datasets.load_examples()
    trainset, validset, testset = datasets.get_split()  # 这三个函数要修改，split应该检查是否已分割
    # datasets.save_example()


    # 预测过程
    predict_preprocess = ProcessorForMultiTask(logger=logger, args=param.get_config(param.DATASET), examples=validset,
                                               feature_file_name='test_feature_for_multi_task0',
                                               task_id=0,
                                               is_prediction=True)
    predict_preprocess.convert_examples_to_features()
    predict_vocab_size = predict_preprocess.get_vocab_size()
    predict_batch_reader = predict_preprocess.batch_generator()

    predict_engine = PredictEngine(param=param, logger=logger, vocab_size=predict_vocab_size)
    predict_engine.predict(predict_batch_reader)
    example_info = util_tool.trans_exam_list_to_colum(validset)
    predict_engine.write_full_info(attach_data=example_info)
