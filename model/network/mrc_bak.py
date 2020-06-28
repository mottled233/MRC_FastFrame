from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid

from model.layer.bert import BertModel
from model.layer.highway import highway_layer


def create_model(args,
                 is_prediction=False, is_validate=False, is_soft_label=False, t=1):
    # 处理词典大小
    vocab_size = args['vocab_size']
    T = fluid.layers.fill_constant(shape=[1], value=t, dtype='float32')
    T2 = fluid.layers.fill_constant(shape=[1], value=t * t, dtype='float32')
    # 输入定义
    unique_ids = fluid.data(name='unique_ids', dtype='int64', shape=[-1, 1])
    src_ids = fluid.data(name='src_ids', dtype='int64', shape=[-1, args['max_seq_length'], 1])
    pos_ids = fluid.data(name='pos_ids', dtype='int64', shape=[-1, args['max_seq_length'], 1])
    sent_ids = fluid.data(name='sent_ids', dtype='int64', shape=[-1, args['max_seq_length'], 1])
    input_mask = fluid.data(name='input_mask', dtype='float32', shape=[-1, args['max_seq_length'], 1])
    if is_soft_label:
        start_positions = fluid.data(name='start_positions', dtype='float32', shape=[-1, args['max_seq_length'], 1])
        end_positions = fluid.data(name='end_positions', dtype='float32', shape=[-1, args['max_seq_length'], 1])
    else:
        start_positions = fluid.data(name='start_positions', dtype='int64', shape=[-1, 1])
        end_positions = fluid.data(name='end_positions', dtype='int64', shape=[-1, 1])
    # 根据任务的不同调整所需的数据，预测任务相比训练任务缺少label这一项数据
    if is_prediction:
        feed_list = [unique_ids, src_ids, pos_ids, sent_ids, input_mask]
    else:
        feed_list = [unique_ids, src_ids, pos_ids, sent_ids, input_mask, start_positions, end_positions]
    reader = fluid.io.DataLoader.from_generator(feed_list=feed_list, capacity=64, iterable=True)

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

    enc_out = bert.get_sequence_output()
    enc_output_list = bert.get_enc_output_list()

    freeze_pretrained_model = config['freeze_pretrained_model']
    if freeze_pretrained_model:
        enc_out.stop_gradient = True

    if config['mrc_layer'] == "cls_fc":

        enc_out = fluid.layers.dropout(
            x=enc_out,
            is_test=(is_prediction or is_validate),
            dropout_prob=args['lstm_dropout_prob'],
            dropout_implementation="upscale_in_train")
        logits = fluid.layers.fc(
            input=enc_out,
            size=2,
            num_flatten_dims=2,
            param_attr=fluid.ParamAttr(
                name="cls_squad_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_squad_out_b", initializer=fluid.initializer.Constant(0.)))

    elif config['mrc_layer'] == 'highway_lstm':
        enc_out = fluid.layers.dropout(
            x=enc_out,
            is_test=(is_prediction or is_validate),
            dropout_prob=args['lstm_dropout_prob'],
            dropout_implementation="upscale_in_train")
        encoded = highway_layer(enc_out, name="highway1", num_flatten_dims=2)
        encoded = fluid.layers.dropout(
            x=encoded,
            is_test=(is_prediction or is_validate),
            dropout_prob=args['lstm_dropout_prob'],
            dropout_implementation="upscale_in_train")
        encoded = highway_layer(encoded, name="highway2", num_flatten_dims=2)
        encoded = fluid.layers.dropout(
            x=encoded,
            is_test=(is_prediction or is_validate),
            dropout_prob=args['lstm_dropout_prob'],
            dropout_implementation="upscale_in_train")
        hidden_size = args['lstm_hidden_size']
        cell = fluid.layers.LSTMCell(hidden_size=hidden_size)
        cell_r = fluid.layers.LSTMCell(hidden_size=hidden_size)
        outputs = fluid.layers.rnn(cell, encoded)[0]
        outputs_r = fluid.layers.rnn(cell_r, encoded, is_reverse=True)[0]
        outputs = fluid.layers.concat(input=[outputs, outputs_r], axis=-1)
        enc_out = fluid.layers.dropout(
            x=outputs,
            is_test=(is_prediction or is_validate),
            dropout_prob=args['lstm_dropout_prob'],
            dropout_implementation="upscale_in_train")
        logits = fluid.layers.fc(
            input=enc_out,
            size=2,
            num_flatten_dims=2,
            param_attr=fluid.ParamAttr(
                name="lstm_squad_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="lstm_squad_out_b", initializer=fluid.initializer.Constant(0.)))

    logits = fluid.layers.transpose(x=logits, perm=[2, 0, 1])
    start_logits, end_logits = fluid.layers.unstack(x=logits, axis=0)
    if is_prediction:
        if not t == 1:
            start_logits = fluid.layers.elementwise_div(start_logits, T)
            end_logits = fluid.layers.elementwise_div(end_logits, T)
        return reader, unique_ids, start_logits, end_logits

    def compute_loss(logits, positions):
        if not t == 1:
            logits = fluid.layers.elementwise_div(logits, T)
            loss = fluid.layers.softmax_with_cross_entropy(
                logits=logits, label=positions, soft_label=is_soft_label)
            loss = fluid.layers.mean(x=loss)
            loss = fluid.layers.elementwise_mul(loss, T2)
        else:
            loss = fluid.layers.softmax_with_cross_entropy(
                logits=logits, label=positions, soft_label=is_soft_label)
            loss = fluid.layers.mean(x=loss)
        return loss

    start_loss = compute_loss(start_logits, start_positions)
    end_loss = compute_loss(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2.0
    return reader, total_loss, unique_ids, start_logits, end_logits, enc_output_list

def create_model_with_lexical_inf(args, is_prediction=False, is_validate=False, is_soft_label=False, t=1):
    # 处理词典大小
    vocab_size = args['vocab_size']
    T = fluid.layers.fill_constant(shape=[1], value=t, dtype='float32')
    T2 = fluid.layers.fill_constant(shape=[1], value=t * t, dtype='float32')
    # 输入定义
    unique_ids = fluid.data(name='unique_ids', dtype='int64', shape=[-1, 1])
    src_ids = fluid.data(name='src_ids', dtype='int64', shape=[-1, args['max_seq_length'], 1])
    pos_ids = fluid.data(name='pos_ids', dtype='int64', shape=[-1, args['max_seq_length'], 1])
    tag_ids = fluid.data(name='pos_tags', dtype='int64', shape=[-1, args['max_seq_length'], 1])
    sent_ids = fluid.data(name='sent_ids', dtype='int64', shape=[-1, args['max_seq_length'], 1])
    input_mask = fluid.data(name='input_mask', dtype='float32', shape=[-1, args['max_seq_length'], 1])
    if is_soft_label:
        start_positions = fluid.data(name='start_positions', dtype='float32', shape=[-1, args['max_seq_length'], 1])
        end_positions = fluid.data(name='end_positions', dtype='float32', shape=[-1, args['max_seq_length'], 1])
    else:
        start_positions = fluid.data(name='start_positions', dtype='int64', shape=[-1, 1])
        end_positions = fluid.data(name='end_positions', dtype='int64', shape=[-1, 1])
    # 根据任务的不同调整所需的数据，预测任务相比训练任务缺少label这一项数据
    if is_prediction:
        feed_list = [unique_ids, src_ids, pos_ids, sent_ids, tag_ids, input_mask]
    else:
        feed_list = [unique_ids, src_ids, pos_ids, sent_ids, tag_ids, input_mask, start_positions, end_positions]
    reader = fluid.io.DataLoader.from_generator(feed_list=feed_list, capacity=64, iterable=True)

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

    enc_out = bert.get_sequence_output()
    enc_output_list = bert.get_enc_output_list()

    tag_embedding = fluid.layers.embedding(input=tag_ids, size=[args['tag_nums'], args['tag_embedding_size']])
    enc_out = fluid.layers.concat(input=[enc_out, tag_embedding], axis=-1)

    freeze_pretrained_model = config['freeze_pretrained_model']
    if freeze_pretrained_model:
        enc_out.stop_gradient = True

    if config['mrc_layer'] == "cls_fc":
        logits = fluid.layers.fc(
            input=enc_out,
            size=2,
            num_flatten_dims=2,
            param_attr=fluid.ParamAttr(
                name="cls_squad_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_squad_out_b", initializer=fluid.initializer.Constant(0.)))
    elif config['mrc_layer'] == 'highway_lstm':
        enc_out = fluid.layers.dropout(
            x=enc_out,
            is_test=(is_prediction or is_validate),
            dropout_prob=args['lstm_dropout_prob'],
            dropout_implementation="upscale_in_train")
        encoded = highway_layer(enc_out, name="highway1", num_flatten_dims=2)
        encoded = fluid.layers.dropout(
            x=encoded,
            is_test=(is_prediction or is_validate),
            dropout_prob=args['lstm_dropout_prob'],
            dropout_implementation="upscale_in_train")
        encoded = highway_layer(encoded, name="highway2", num_flatten_dims=2)
        encoded = fluid.layers.dropout(
            x=encoded,
            is_test=(is_prediction or is_validate),
            dropout_prob=args['lstm_dropout_prob'],
            dropout_implementation="upscale_in_train")
        hidden_size = args['lstm_hidden_size']
        cell = fluid.layers.LSTMCell(hidden_size=hidden_size)
        cell_r = fluid.layers.LSTMCell(hidden_size=hidden_size)
        outputs = fluid.layers.rnn(cell, encoded)[0]
        outputs_r = fluid.layers.rnn(cell_r, encoded, is_reverse=True)[0]
        outputs = fluid.layers.concat(input=[outputs, outputs_r], axis=-1)
        enc_out = fluid.layers.dropout(
            x=outputs,
            is_test=(is_prediction or is_validate),
            dropout_prob=args['lstm_dropout_prob'],
            dropout_implementation="upscale_in_train")
        logits = fluid.layers.fc(
            input=enc_out,
            size=2,
            num_flatten_dims=2,
            param_attr=fluid.ParamAttr(
                name="lstm_squad_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="lstm_squad_out_b", initializer=fluid.initializer.Constant(0.)))

    logits = fluid.layers.transpose(x=logits, perm=[2, 0, 1])
    start_logits, end_logits = fluid.layers.unstack(x=logits, axis=0)
    if is_prediction:
        if not t == 1:
            start_logits = fluid.layers.elementwise_div(start_logits, T)
            end_logits = fluid.layers.elementwise_div(end_logits, T)
        return reader, unique_ids, start_logits, end_logits

    def compute_loss(logits, positions):
        if not t == 1:
            logits = fluid.layers.elementwise_div(logits, T)
            loss = fluid.layers.softmax_with_cross_entropy(
                logits=logits, label=positions, soft_label=is_soft_label)
            loss = fluid.layers.mean(x=loss)
            loss = fluid.layers.elementwise_mul(loss, T2)
        else:
            loss = fluid.layers.softmax_with_cross_entropy(
                logits=logits, label=positions, soft_label=is_soft_label)
            loss = fluid.layers.mean(x=loss)
        return loss

    start_loss = compute_loss(start_logits, start_positions)
    end_loss = compute_loss(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2.0
    return reader, total_loss, unique_ids, start_logits, end_logits, enc_output_list

