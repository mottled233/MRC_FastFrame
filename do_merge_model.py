import paddle
import paddle.fluid as fluid
import numpy as np
import time
import sys
from engine.train_for_multitask import TrainEngineForMergeModel as TrainEngine
from engine.predict import PredictEngine as PredictEngine
from data.Dataset import Dataset
from preprocess.preprocess_for_mt import ProcessorForMergeModel as PreProcess

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

    train_preprocess = PreProcess(logger=logger, args=param.get_config(param.DATASET),
                                  file_name_1='roberta_large_mt1_cls.json',
                                  file_name_2='roberta_large_mt2_cls.json')

    train_batch_reader = train_preprocess.batch_generator()

    valid_preprocess = PreProcess(logger=logger, args=param.get_config(param.DATASET),
                                  file_name_1='roberta_large_valid_mt1_cls.json',
                                  file_name_2='roberta_large_valid_mt2_cls.json')
    valid_batch_reader = valid_preprocess.batch_generator()
    # 训练过程
    train_engine = TrainEngine(train_batch_reader, valid_batch_reader,
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

