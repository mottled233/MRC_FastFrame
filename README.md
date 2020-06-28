# MRC_FastFrame_Reconfiguration

A framework for machine reading comprehension based on paddlepaddle.



## 项目背景

2020/6/26

本项目是一套通用的自然语言任务处理流程，基于百度的深度学习框架paddlepaddle。项目采取模块化总线式设计，将自然语言处理任务分为数据集整理、数据预处理、模型构建、训练与预测五个步骤实现，每一个步骤都具有灵活的可替换性和扩展性。

目前，本项目加入了处理SQuAD类型的机器阅读理解任务的逻辑代码，机器阅读理解任务的每个模块均继承自抽象的自然语言处理任务模块。

同样，如果需加入其他的任务，只需要继承自然语言处理任务模块，再加上该任务特有的逻辑代码即可，整体流程具有灵活的可替换性和扩展性。

系统控制流程如下图所示：

![img](https://uploader.shimo.im/f/VOh3ld8qSH8piLXE.jpg!thumbnail)



## 运行环境

python版本：3.7

python库：

- paddlepaddle 1.8.1
- paddlehub 1.6.1
- numpy
- logging

系统环境：Linux环境和Windows环境均可



## 快速开始
### 安装环境

```bash
!pip install paddlepaddle==1.8.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
!pip install paddlehub==1.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

可通过paddlehub下载预训练模型参数，ernie，roberta等，移动到本地目录，去除hub的标志。

通过其他途径下载预训练模型再复制到本地也可以，保证参数load_model_path是指向正确的预训练模型即可。

```bash
!hub install ernie==1.2.0
!cp -r /home/aistudio/.paddlehub/modules/ernie/ $HOME
!rename 's/\@HUB_ernie-stable\@(.*)/$1/' ernie/assets/params/*

!hub install chinese-roberta-wwm-ext==1.0.0
!cp -r /home/aistudio/.paddlehub/modules/chinese_roberta_wwm_ext/ $HOME
!rename 's/\@HUB_chinese-roberta-wwm-ext\@(.*)/$1/' chinese_roberta_wwm_ext/assets/params/*

!hub install chinese-roberta-wwm-ext-large==1.0.0
!cp -r /home/aistudio/.paddlehub/modules/chinese_roberta_wwm_ext_large/ $HOME
!rename 's/\@HUB_chinese-roberta-wwm-ext-large\@(.*)/$1/' chinese_roberta_wwm_ext_large/assets/params/*
```

### 设置参数管理和日志管理

```python
# 初始化参数管理，参数文件在File_Directory/config下。
param = UParam()
param.read_config_file("config_ernie") # 以ernie为例
param.set_config(sys.argv[1:])
args = param.get_config(param.GLOBAL)
# 初始化日志管理，可在File_Directory/logging下查看系统的运行情况
logger = ULog(args, params=param)
app_name = args["app_name"]
dataset_args = param.get_config(param.DATASET)
```

### 数据集模块

```python
# 初始化数据集类,从源文件中读取数据,文件路径参数指定
train_dataset = Dataset(dataset_args)
train_dataset.read_from_srcfile(dataset_args['train_file_path'],is_training=True)
```

### 数据预处理模块

```python
# 初始化预处理类,预处理类中需要传入数据集类生成的examples列表
train_preprocess=Preprocess(args=dataset_args,examples=train_dataset.get_examples())
train_data_generator = train_preprocess.do_preprocess()
```

### 模型构建模块

```python
# 通过参数指定网络类型,构建网络对象
net_args = param.get_config(param.MODEL_BUILD)
net = Net(args=net_args)
```

### 训练引擎模块

```python
# 创建训练引擎进行模型训练,训练引擎需传入模型构建模块的网络对象和预处理模块的数据生成器
train_args = param.get_config(param.TRAIN)
train_engine = TrainEngine(args=train_args, network=net,
train_data_generator=train_data_generator,
valid_data_generator=valid_data_generator, valid_data=valid_data)
train_engine.train()
```

### 预测引擎模块

```python
# 预测引擎,读取训练好的模型,将测试集数据进行预测操作,并将结果写入到File_Directory/result目录下
test_args = param.get_config(param.PREDICT)
predict_engine = PredictEngine(test_args, net)
predict_engine.predict(predict_data_generator, predict_data=predict_data)
```

### 运行整个项目

直接运行main.py文件，可通过命令行参数修改默认配置，优先级是：命令行>配置文件>默认值

该节只介绍了各个模块的主要使用

```bash
python main.py --key1=value1 --key2=value2
```



## 各模块功能点

### 数据集模块:

- 基本的数据读取功能,转化为examples列表

- examples列表缓存功能

- 数据集划分功能, 具体见dataset下的Spliter类, 用于只有一份训练集的情况

### 预处理模块
- 将examples转化为features列表

- features列表缓存功能

### 模型构建模块
- 构建模型网络

- 支持多种预训练模型, 参数控制

- 支持多种下游网络模型, 参数控制

### 训练引擎
- 对网络参数的读取, 训练, 保存, 支持断点续训

- 支持多种训练策略,如并行, cpu或gpu, 学习率变化策略等等

- 训练的到指定步长进行验证

### 预测引擎
- 读取训练好的模型, 计算预测结果
- 预测结果写入外部文件, 封装了多种写入格式



## 性能表现

本项目在Squad类型的中文机器阅读理解上作了开发与测试，下给出这两份数据集的下载链接：

[CMRC2018](https://www.kesci.com/home/dataset/5e7b180798d4a8002d2d3af6/files)
[2020LIC_MRC(2020语言与智能技术竞赛：机器阅读理解任务)](https://aistudio.baidu.com/aistudio/competition/detail/28)

下载数据集后放在File_Directory/dataset下即可开始运行本项目。

<table>
   <tr>
      <td>模型名</td>
      <td colspan="2">CMRC2018</td>
      <td colspan="2">2020LIC-MRC</td>
   </tr>
   <tr>
      <td></td>
      <td>F1</td>
      <td>EM</td>
      <td>F1</td>
      <td>EM</td>
   </tr>
   <tr>
      <td>ERNIE</td>
      <td>82.4</td>
      <td>70.3</td>
      <td>63.6</td>
      <td>49</td>
   </tr>
   <tr>
      <td>BERT</td>
      <td>83.7</td>
      <td>72.3</td>
      <td>65.9</td>
      <td>53.8</td>
   </tr>
   <tr>
      <td>RoBERTa</td>
      <td>86.1</td>
      <td>74.2</td>
      <td>66.6</td>
      <td>55.2</td>
   </tr>
   <tr>
      <td></td>
   </tr>
</table>


## Lincense

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)