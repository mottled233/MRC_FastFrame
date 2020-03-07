"""
bert分类model实现，对接训练模块和预测模块
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid

from model.bert import BertModel
from model.capsLayer import CapsLayer


# 搭建分类模型
# 被训练模块和预测模块直接调用
# 返回相关的计算结果和对应的dataloader对象
def create_model(args,
                 vocab_size,
                 is_prediction=False):
    # 处理词典大小
    if args['vocab_size'] > 0:
        vocab_size = args['vocab_size']

    # 输入定义
    qas_ids = fluid.data(name='qas_ids', dtype='int64', shape=[-1, 1])
    src_ids = fluid.data(name='src_ids', dtype='int64', shape=[-1, args['max_seq_length'], 1])
    pos_ids = fluid.data(name='pos_ids', dtype='int64', shape=[-1, args['max_seq_length'], 1])
    sent_ids = fluid.data(name='sent_ids', dtype='int64', shape=[-1, args['max_seq_length'], 1])
    input_mask = fluid.data(name='input_mask', dtype='float32', shape=[-1, args['max_seq_length'], 1])
    labels = fluid.data(name='labels', dtype='int64', shape=[-1, 1])
    # 根据任务的不同调整所需的数据，预测任务相比训练任务缺少label这一项数据
    if is_prediction:
        feed_list = [qas_ids, src_ids, pos_ids, sent_ids, input_mask]
    else:
        feed_list = [qas_ids, src_ids, pos_ids, sent_ids, input_mask, labels]
    reader = fluid.io.DataLoader.from_generator(feed_list=feed_list, capacity=64, iterable=True)

    # 模型部分
    # 由bert后接一层全连接完成预测任务

    # bert部分
    config = args
    config['vocab_size'] = vocab_size
    bert = BertModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        config=config,
        use_fp16=False)

    mrc_layer = config['mrc_layer']

    logits = None
    if mrc_layer == "cls_fc":
        # 取[CLS]的输出经全连接进行预测
        cls_feats = bert.get_pooled_output()
        cls_feats = fluid.layers.dropout(
            x=cls_feats,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")
        logits = fluid.layers.fc(
            input=cls_feats,
            size=args['num_labels'],
            param_attr=fluid.ParamAttr(
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(0.)))
    elif mrc_layer == "capsNet":
        # 取完整的bert_output，输入胶囊网络
        bert_output = bert.get_sequence_output()
        param_attr = fluid.ParamAttr(name='conv2d.weight', initializer=fluid.
                                     initializer.Xavier(uniform=False),
                                     learning_rate=0.001)
        bert_output = fluid.layers.unsqueeze(input=bert_output, axes=[1])
        capsules = fluid.layers.conv2d(input=bert_output, num_filters=256, filter_size=32, stride=15,
                                       padding="VALID", act="relu", param_attr=param_attr)
        # (batch_size, 256, 33, 50)
        primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
        caps1 = primaryCaps(capsules, kernel_size=9, stride=2)
        # (batch_size, 8736, 8, 1)
        classifierCaps = CapsLayer(num_outputs=args['num_labels'], vec_len=16, with_routing=True, layer_type='FC')
        caps2 = classifierCaps(caps1)
        # (batch_size, 3, 16, 1)

        epsilon = 1e-9
        v_length = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(caps2),
                                                             -2, keep_dim=True) + epsilon)
        logits = fluid.layers.squeeze(v_length, axes=[2, 3])

    # 根据任务返回不同的结果
    # 预测任务仅返回dataloader和预测出的每个label对应的概率
    if is_prediction:
        probs = fluid.layers.softmax(logits)
        return reader, probs, qas_ids

    # 训练任务则计算loss
    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=labels, return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)

    if args['use_fp16'] and args.loss_scaling > 1.0:
        loss *= args.loss_scaling

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)

    # 返回dataloader，loss，预测结果，和准确度
    return reader, loss, probs, accuracy, qas_ids
