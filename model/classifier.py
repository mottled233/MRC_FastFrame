"""
bert分类model实现，对接训练模块和预测模块
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid

from model.bert import BertModel
from model.capsLayer import CapsLayer
from model.highway import highway_layer



def create_model(args,
                 vocab_size,
                 is_prediction=False, is_validate=False):
    """
    搭建分类模型
    被训练模块和预测模块直接调用
    返回相关的计算结果和对应的dataloader对象
    :param args: 参数
    :param vocab_size: 词典大小，用于构建词嵌入层。注意当参数设置词典大小时，该项无效
    :param is_prediction: 是否是预测模式，将禁用dropout等。
    :param is_validate: 是否是验证模式，除了禁用dropout，还将返回loss和acc，如果输入数据中没有对应项，则会报错。
    :return:
    """
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
    # engineer_ids = fluid.data(name='engineer_ids', dtype='int64', shape=[-1, args['max_seq_length']+1, 1])
    engineer_ids = fluid.data(name='engineer_ids', dtype='int64', shape=[-1, args['max_seq_length'], 1])
    # 根据任务的不同调整所需的数据，预测任务相比训练任务缺少label这一项数据
    if is_prediction:
        feed_list = [qas_ids, src_ids, pos_ids, sent_ids, input_mask]
    else:
        feed_list = [qas_ids, src_ids, pos_ids, sent_ids, input_mask, labels]
    if config['use_engineer']:
        feed_list.append(engineer_ids)
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
        use_fp16=False,
        is_prediction=(is_prediction or is_validate))

    mrc_layer = config['mrc_layer']
    freeze_pretrained_model = config['freeze_pretrained_model']

    cls_feats = bert.get_pooled_output()
    bert_encode = bert.get_sequence_output()

    if freeze_pretrained_model:
        cls_feats.stop_gradient = True
        bert_encode.stop_gradient = True

    if config['use_engineer']:
        # entity_sim = engineer_ids[:,-1,:]
        # entity_sim_code = fluid.layers.one_hot(input=entity_sim, depth=2, allow_out_of_range=False)
        # engineer_emb = fluid.layers.embedding(input=engineer_ids[:,:-1,:], size=[32, 8])
        engineer_emb = fluid.layers.embedding(input=engineer_ids, size=[32, 8])
        bert_encode = fluid.layers.concat(input=[bert_encode, engineer_emb], axis=-1)

    logits = None
    if mrc_layer == "cls_fc":
        # 取[CLS]的输出经全连接进行预测
        cls_feats = fluid.layers.dropout(
            x=cls_feats,
            dropout_prob=0.1,
            is_test=(is_prediction or is_validate),
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
        bert_output = bert_encode
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

    elif mrc_layer == "lstm":
        hidden_size = args['lstm_hidden_size']

        cell = fluid.layers.LSTMCell(hidden_size=hidden_size)
        cell_r = fluid.layers.LSTMCell(hidden_size=hidden_size)
        encoded = bert_encode[:, 1:, :]
        encoded = fluid.layers.dropout(
            x=encoded,
            is_test=(is_prediction or is_validate),
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")
        outputs = fluid.layers.rnn(cell, encoded)[0][:, -1, :]
        outputs_r = fluid.layers.rnn(cell_r, encoded, is_reverse=True)[0][:, -1, :]
        outputs = fluid.layers.concat(input=[outputs, outputs_r], axis=1)

        cls_feats = outputs
        cls_feats = fluid.layers.dropout(
            x=cls_feats,
            is_test=(is_prediction or is_validate),
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")
        # fc = fluid.layers.fc(input=cls_feats, size=hidden_size*2)
        # fc = fluid.layers.dropout(
        #     x=fc,
        #     dropout_prob=0.1,
        #     dropout_implementation="upscale_in_train")
        logits = fluid.layers.fc(
            input=cls_feats,
            size=args['num_labels'],
            param_attr=fluid.ParamAttr(
                name="lstm_fc_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="lstm_fc_b", initializer=fluid.initializer.Constant(0.)))

    elif mrc_layer == "highway_lstm":

        hidden_size = 128

        cell = fluid.layers.LSTMCell(hidden_size=hidden_size)
        cell_r = fluid.layers.LSTMCell(hidden_size=hidden_size)
        encoded = bert_encode[:, 1:, :]
        encoded = fluid.layers.dropout(
            x=encoded,
            is_test=(is_prediction or is_validate),
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")

        encoded = highway_layer(encoded, name="highway1", num_flatten_dims=2)
        encoded = fluid.layers.dropout(
            x=encoded,
            is_test=(is_prediction or is_validate),
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")

        outputs = fluid.layers.rnn(cell, encoded)[0][:, -1, :]
        outputs_r = fluid.layers.rnn(cell_r, encoded, is_reverse=True)[0][:, -1, :]
        outputs = fluid.layers.concat(input=[outputs, outputs_r], axis=1)

        cls_feats = outputs
        cls_feats = fluid.layers.dropout(
            x=cls_feats,
            is_test=(is_prediction or is_validate),
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")
        # fc = fluid.layers.fc(input=cls_feats, size=hidden_size*2)
        # fc = fluid.layers.dropout(
        #     x=fc,
        #     dropout_prob=0.1,
        #     dropout_implementation="upscale_in_train")
        logits = fluid.layers.fc(
            input=cls_feats,
            size=args['num_labels'],
            param_attr=fluid.ParamAttr(
                name="lstm_fc_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="lstm_fc_b", initializer=fluid.initializer.Constant(0.)))

    # 根据任务返回不同的结果
    # 预测任务仅返回dataloader和预测出的每个label对应的概率
    if is_prediction and not is_validate:
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



def create_model_for_pretrain(args, vocab_size):
    if args['vocab_size'] > 0:
        vocab_size = args['vocab_size']

    # 输入定义
    src_ids = fluid.data(name='src_ids', dtype='int64', shape=[-1, args['max_seq_length'], 1])
    pos_ids = fluid.data(name='pos_ids', dtype='int64', shape=[-1, args['max_seq_length'], 1])
    sent_ids = fluid.data(name='sent_ids', dtype='int64', shape=[-1, args['max_seq_length'], 1])
    input_mask = fluid.data(name='input_mask', dtype='float32', shape=[-1, args['max_seq_length'], 1])
    mask_labels = fluid.data(name='mask_labels', dtype='int64', shape=[-1, 1])
    mask_pos = fluid.data(name='mask_pos', dtype='int64', shape=[-1, 1])
    otheranswer_labels = fluid.data(name='otheranswer_labels', dtype='int64', shape=[-1, 1])
    # 根据任务的不同调整所需的数据，预测任务相比训练任务缺少label这一项数据
    feed_list = [src_ids, pos_ids, sent_ids, input_mask, mask_labels, mask_pos, otheranswer_labels]
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

    results = bert.get_pretraining_output(mask_labels, mask_pos, otheranswer_labels)
    qa_acc, mean_mask_lm_loss, loss = results
    return reader, qa_acc, mean_mask_lm_loss, loss

def create_model_for_multi_task(args,
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
    labels_for_reverse = fluid.data(name='labels_for_reverse', dtype='int64', shape=[-1, 1])
    # 根据任务的不同调整所需的数据，预测任务相比训练任务缺少label这一项数据
    if is_prediction:
        feed_list = [qas_ids, src_ids, pos_ids, sent_ids, input_mask]
    else:
        feed_list = [qas_ids, src_ids, pos_ids, sent_ids, input_mask, labels, labels_for_reverse]
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
        use_fp16=False,
        is_prediction=is_prediction)

    mrc_layer = config['mrc_layer']
    freeze_pretrained_model = config['freeze_pretrained_model']

    cls_feats, reverse_feats = bert.get_pooled_outputs()
    bert_encode = bert.get_sequence_output()
    if freeze_pretrained_model:
        cls_feats.stop_gradient = True
        bert_encode.stop_gradient = True

    logits = None
    if mrc_layer == "cls_fc":
        # 取[CLS]的输出经全连接进行预测
        cls_feats = fluid.layers.dropout(
            x=cls_feats,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train",
            is_test=is_prediction)
        logits = fluid.layers.fc(
            input=cls_feats,
            size=args['num_labels'],
            param_attr=fluid.ParamAttr(
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(0.)))
        logits_for_reverse = fluid.layers.fc(
            input=reverse_feats,
            size=2,
            param_attr=fluid.ParamAttr(
                name="reverse_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="reverse_out_b", initializer=fluid.initializer.Constant(0.)))
    elif mrc_layer == "capsNet":
        # 取完整的bert_output，输入胶囊网络
        bert_output = bert_encode
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

    elif mrc_layer == "lstm":
        hidden_size = 128

        cell = fluid.layers.LSTMCell(hidden_size=hidden_size)
        cell_r = fluid.layers.LSTMCell(hidden_size=hidden_size)
        encoded = bert_encode[:, 1:, :]
        encoded = fluid.layers.dropout(
            x=encoded,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")
        outputs = fluid.layers.rnn(cell, encoded)[0][:, -1, :]
        outputs_r = fluid.layers.rnn(cell_r, encoded, is_reverse=True)[0][:, -1, :]
        outputs = fluid.layers.concat(input=[outputs, outputs_r], axis=1)

        cls_feats = outputs
        cls_feats = fluid.layers.dropout(
            x=cls_feats,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")
        # fc = fluid.layers.fc(input=cls_feats, size=hidden_size*2)
        # fc = fluid.layers.dropout(
        #     x=fc,
        #     dropout_prob=0.1,
        #     dropout_implementation="upscale_in_train")
        logits = fluid.layers.fc(
            input=cls_feats,
            size=args['num_labels'],
            param_attr=fluid.ParamAttr(
                name="lstm_fc_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="lstm_fc_b", initializer=fluid.initializer.Constant(0.)))

    # 根据任务返回不同的结果
    # 预测任务仅返回dataloader和预测出的每个label对应的概率
    if is_prediction:
        probs = fluid.layers.softmax(logits)
        return reader, probs, qas_ids

    # 训练任务则计算loss
    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=labels, return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)

    ce_loss_for_reverse, probs_for_reverse = fluid.layers.softmax_with_cross_entropy(
        logits=logits_for_reverse, label=labels_for_reverse, return_softmax=True)
    loss_for_reverse = fluid.layers.mean(x=ce_loss_for_reverse)

    if args['use_fp16'] and args.loss_scaling > 1.0:
        loss *= args.loss_scaling

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)
    accuracy_for_reverse = fluid.layers.accuracy(input=probs_for_reverse, label=labels_for_reverse, total=num_seqs)

    # 返回dataloader，loss，预测结果，和准确度
    return reader, loss + loss_for_reverse, probs, accuracy, accuracy_for_reverse, qas_ids


def creat_model_for_cls_output(args,
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
    labels_for_reverse = fluid.data(name='labels_for_reverse', dtype='int64', shape=[-1, 1])
    # 根据任务的不同调整所需的数据，预测任务相比训练任务缺少label这一项数据
    if is_prediction:
        feed_list = [qas_ids, src_ids, pos_ids, sent_ids, input_mask]
    else:
        feed_list = [qas_ids, src_ids, pos_ids, sent_ids, input_mask, labels, labels_for_reverse]
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
        use_fp16=False,
        is_prediction=is_prediction)

    cls_feats = bert.get_pooled_output()
    if is_prediction:
        return reader, cls_feats, qas_ids
    return reader, cls_feats, qas_ids, labels


def create_model_for_cls_merge(args, is_prediction=False):
    qas_ids = fluid.data(name='qas_ids', dtype='int64', shape=[-1, 1])
    cls_feats = fluid.data(name='cls_feats', dtype='float32', shape=[-1, 2 * args['hidden_size'], 1])
    labels = fluid.data(name='labels', dtype='int64', shape=[-1, 1])
    if is_prediction:
        feed_list = [qas_ids, cls_feats]
    else:
        feed_list = [qas_ids, cls_feats, labels]
    reader = fluid.io.DataLoader.from_generator(feed_list=feed_list, capacity=64, iterable=True)
    cls_feats = fluid.layers.dropout(
        x=cls_feats,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train",
        is_test=is_prediction)
    cls_feats_pooled = fluid.layers.fc(
        input=cls_feats,
        size=args['hidden_size'],
        param_attr=fluid.ParamAttr(
            name="first_pool_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="first_pool_b", initializer=fluid.initializer.Constant(0.)))

    logits = fluid.layers.fc(
        input=cls_feats_pooled,
        size=args['num_labels'],
        param_attr=fluid.ParamAttr(
            name="cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_out_b", initializer=fluid.initializer.Constant(0.)))

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







