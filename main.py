import sys
from dataset.dataset_for_mrc_squad import DatasetForMrcSquad as Dataset
from preprocess.preprocess_for_mrc import PreprocessForMRCChinese as Preprocess
from model.network.mrc_net import MRCNet as Net
from engine.mrc_train_engine import MRCTrainEngine as TrainEngine
from engine.mrc_predict_engine import MRCPredictEngine as PredictEngine
from util.util_parameter import UtilParameter as UParam
from util.util_logging import UtilLogging as ULog


if __name__ == "__main__":
    # 设置参数
    param = UParam()
    param.read_config_file("config_ernie")
    param.set_config(sys.argv[1:])
    args = param.get_config(param.GLOBAL)
    # 初始化日志
    logger = ULog(args, params=param)
    app_name = args["app_name"]
    dataset_args = param.get_config(param.DATASET)

    # 训练数据预处理
    train_dataset = Dataset(dataset_args)
    train_dataset.read_from_srcfile(dataset_args['train_file_path'], cache=dataset_args['train_example_file_name'],
                                    is_training=True)
    train_preprocess = Preprocess(args=dataset_args, examples=train_dataset.get_examples(),
                                  cache=dataset_args['train_feature_file_name'])
    train_data_generator = train_preprocess.do_preprocess()

    # # 验证数据预处理
    valid_dataset = Dataset(dataset_args)
    valid_dataset.read_from_srcfile(dataset_args['dev_file_path'], cache=dataset_args['dev_example_file_name'],
                                    is_training=True)
    valid_preprocess = Preprocess(args=dataset_args, examples=valid_dataset.get_examples(),
                                  cache=dataset_args['dev_feature_file_name'])
    valid_data_generator = valid_preprocess.do_preprocess()
    valid_data = {'features': valid_preprocess.get_features(), 'examples': valid_dataset.get_examples(),
                    'dev_file_path': dataset_args['dev_file_path']}

    # 预测数据预处理
    predict_dataset = Dataset(dataset_args)
    predict_dataset.read_from_srcfile(dataset_args['test_file_path'], cache=dataset_args['test_example_file_name'],
                                      is_training=False)
    predict_preprocess = Preprocess(args=dataset_args, examples=predict_dataset.get_examples(), is_prediction=True,
                                    cache=dataset_args['test_feature_file_name'])
    predict_data_generator = predict_preprocess.do_preprocess()

    predict_data = {'features': predict_preprocess.get_features(), 'examples': predict_dataset.get_examples(),
                    'test_file_path': dataset_args['test_file_path']}

    # 初始化网络
    net_args = param.get_config(param.MODEL_BUILD)
    net = Net(args=net_args)

    # 训练过程
    train_args = param.get_config(param.TRAIN)
    train_engine = TrainEngine(args=train_args, network=net,
                               train_data_generator=train_data_generator,
                               valid_data_generator=valid_data_generator, valid_data=valid_data)
    train_engine.train()

    # 预测过程
    test_args = param.get_config(param.PREDICT)
    predict_engine = PredictEngine(test_args, net)
    predict_engine.predict(predict_data_generator, predict_data=predict_data)




