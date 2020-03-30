import paddle
import paddle.fluid as fluid
import numpy as np
import time
import sys
from engine.train_for_multitask import PredictEngineForMergeModel
from engine.predict import PredictEngine as PredictEngine
from data.Dataset import Dataset
from preprocess.preprocess_for_mt import ProcessorForMergeModel as PreProcess

from util.util_parameter import UtilParameter as UParam
from util.util_logging import UtilLogging as ULog
import util.util_tool as util_tool

if __name__ == "__main__":
    # 设置参数
    param = UParam()
    param.read_config_file("config_merge_test")
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

    test_preprocess = PreProcess(logger=logger, args=param.get_config(param.DATASET),
                                 file_name_1='roberta_large_test_mt1_cls.json',
                                 file_name_2='roberta_large_test_mt2_cls.json', is_prediction=True)

    test_batch_reader = test_preprocess.batch_generator()

    predict_engine = PredictEngineForMergeModel(param=param, logger=logger)
    predict_engine.predict(test_batch_reader)
    predict_engine.write_to_json()
