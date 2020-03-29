import paddle
import paddle.fluid as fluid
import numpy as np
import time
import sys
from engine.train_for_multitask import TrainEngineForMT as TrainEngine
from engine.predict import PredictEngine as PredictEngine
from data.Dataset import Dataset
from preprocess.preprocess_for_mt import ProcessorForMultiTask as PreProcess

from util.util_parameter import UtilParameter as UParam
from util.util_logging import UtilLogging as ULog
import util.util_tool as util_tool


if __name__ == "__main__":
    # 设置参数
    param = UParam()
    param.read_config_file("config_roberta_large")
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

    # 训练数据预处理
    train_preprocess = PreProcess(logger=logger, args=param.get_config(param.DATASET), examples=trainset,
                                  feature_file_name='train_feature_for_multi_task')
    train_preprocess.convert_examples_to_features()
    train_vocab_size = train_preprocess.get_vocab_size()
    train_batch_reader = train_preprocess.batch_generator()
    # 验证数据预处理
    valid_preprocess = PreProcess(logger=logger, args=param.get_config(param.DATASET), examples=validset,
                                  feature_file_name='valid_feature_for_multi_task')
    valid_preprocess.convert_examples_to_features()
    valid_vocab_size = valid_preprocess.get_vocab_size()
    valid_batch_reader = valid_preprocess.batch_generator()

    # 训练过程
    train_engine = TrainEngine(train_batch_reader, train_vocab_size, valid_batch_reader, valid_vocab_size,
                               args=param, logger=logger)
    t1 = time.time()
    train_engine.train()
    t2 = time.time()
    print(t2-t1)

    # # 预测过程
    # predict_engine = PredictEngine(param=param, logger=logger, vocab_size=predict_vocab_size)
    # predict_engine.predict(predict_batch_reader)
    # # example_info = util_tool.trans_exam_list_to_colum(validset)
    # predict_engine.write_to_json()

