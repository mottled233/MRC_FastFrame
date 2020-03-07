# MRC_FastFrame

A framework for machine reading comprehension based on paddlepaddle.

## 更新时间

2020/2/14

完成各个模块最初版本，核心模型基于bert-飞桨，修复了一些运行过程中的bug，能完整运行整个流程。

## 运行环境

python版本：3.7

python库：

- paddlepaddle
- paddlehub
- numpy
- logging

## 数据集读取

数据集读取模块读取从全局参数中读取数据集文件路径，划分比例数组，将读入数据集中的数据作有效性检查后组装成Example的格式，根据划分比例将数据进行划分。提供调用，缓存，加载examples的接口供主程序调用。使用方法如下：

```python
# 初始数据集类

 dataset = Dataset(logger=logger, args=param.get_config(param.DATASET))

# 读取数据集并划分

 dataset.read_dataset(div_nums=[7, 2, 1])

# 保存缓存文件

 dataset.save_examples()

# 读取缓存文件

 dataset.load_examples()
  # 读取划分好的数据集
 trainset, validset, testset = datasets.get_split()
```


生成的example类的内容如下：

- qas_id：此example的id，类型为int。

- question：此example的问题，类型为string。

- answer：此example的答案，类型为string。

- yes_or_no：此example的label，类型为string，取值范围为('Yes', 'No', 'Depends')。

- docs: 此example的文档集合，类型为string组成的list。

- docs_selected：此example的被选中的文档的集合，类型为string组成的list。


## 数据预处理

数据预处理模块的主要功能是将输入的以list形式存储的example集，经过诸如去除停用字等前处理操作后，进行padding、加入[CLS]与[SEP]等bert输入要求的变换后，返回成batch的数据的生成器。使用方法如下：

```python
    # 训练数据预处理
    train_preprocess = PreProcess(logger=logger, args=param.get_config(param.DATASET), examples=trainset)
    train_preprocess.prepare_batch_data(cache_filename="train_features")
    train_vocab_size = train_preprocess.get_vocab_size()
    train_batch_reader = train_preprocess.batch_generator()
    # 验证数据预处理
    valid_preprocess = PreProcess(logger=logger, args=param.get_config(param.DATASET), examples=validset)
    valid_preprocess.prepare_batch_data(cache_filename="valid_features")
    valid_vocab_size = valid_preprocess.get_vocab_size()
    valid_batch_reader = valid_preprocess.batch_generator()
    # 预测数据预处理
    predict_preprocess = PreProcess(logger=logger, args=param.get_config(param.DATASET), examples=testset,
                                    for_prediction=True)
    predict_preprocess.prepare_batch_data(cache_filename="")
    predict_vocab_size = predict_preprocess.get_vocab_size()
    predict_batch_reader = predict_preprocess.batch_generator()
```

生成数据的格式

```text
src_id, pos_id, segment_id, input_mask
```

`input_mask` 为 FLOAT32 类型，其余字段为 INT64 类型。

## 训练引擎

训练模块的主要功能是接受主程序所提供的相关参数、数据生成器和构建好的模型，制定训练策略，完成训练，并使用验证数据对模型效果进行验证，最后将模型保存。使用方法如下：

预训练模型下载：https://bert-models.bj.bcebos.com/chinese_L-12_H-768_A-12.tar.gz
```python
# 创建训练引擎
train_engine = TrainEngine(train_batch_reader, train_vocab_size, valid_batch_reader, \ valid_vocab_size,args=param, logger=logger)
t1 = time.time()
# 训练过程
train_engine.train()
t2 = time.time()
print(t2-t1)
```

生成的模型参数文件。默认保存模型参数的路径是是File_Directory/models。

## 预测引擎

预测模块的主要功能是接受主程序所提供的相关参数、数据生成器和构建好的模型，载入外部存储的训练好的模型参数，完成预测，并将结果经过后处理后输出到外部文件中。使用方法如下：

```python
# 创建预测引擎
predict_engine = PredictEngine(args=param, logger=logger)
# 预测引擎加载网络模型
predict_engine.init_model(vocab_size=predict_vocab_size)
# 预测引擎执行预测
predict_engine.predict(predict_batch_reader)
# 预测引擎生成文字结果
predict_engine.generate_result()
```

预测引擎生成的结果存放路径由全局参数指定。默认是在File_Directory/result/result.json。下面是一个demo生成的结果：

```json
{"yesno_answer": "Yes", "id": "178"}
{"yesno_answer": "Depends", "id": "511"}
{"yesno_answer": "No", "id": "293"}
```

## 运行整个项目

运行main.py文件，可通过命令行参数修改默认配置，优先级是：命令行>配置文件>默认值

```bash
python main.py --key1=value1 --key2=value2
```

