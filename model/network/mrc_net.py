from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid

from model.network.network import Network
from model.layer.bert import BertModel
from model.layer.highway import highway_layer


class MRCNet(Network):
    """
    搭建分类模型
    被训练模块和预测模块直接调用
    返回相关的计算结果和对应的dataloader对象
    :param args: 参数
    """
    def __init__(self, args):
        super().__init__(args)

    def create_model(self):
        """
        创建模型
        若是预测模式，将禁用dropout等，若不是prediction模式，需要返回loss。
        若是验证模式，除了禁用dropout，还将返回loss和acc。
        :return:
        """
        is_predict = self.is_predict
        is_train = self.is_train
        args = self.args

        # 处理词典大小
        vocab_size = args['vocab_size']
        # 输入定义
        unique_ids = fluid.data(name='unique_ids', dtype='int64', shape=[-1, 1])
        src_ids = fluid.data(name='src_ids', dtype='int64', shape=[-1, args['max_seq_length'], 1])
        pos_ids = fluid.data(name='pos_ids', dtype='int64', shape=[-1, args['max_seq_length'], 1])
        sent_ids = fluid.data(name='sent_ids', dtype='int64', shape=[-1, args['max_seq_length'], 1])
        input_mask = fluid.data(name='input_mask', dtype='float32', shape=[-1, args['max_seq_length'], 1])
        start_positions = fluid.data(name='start_positions', dtype='int64', shape=[-1, 1])
        end_positions = fluid.data(name='end_positions', dtype='int64', shape=[-1, 1])
        # 根据任务的不同调整所需的数据，预测任务相比训练任务缺少label这一项数据
        if is_predict:
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
            is_prediction=not is_train)

        enc_out = bert.get_sequence_output()
        enc_output_list = bert.get_enc_output_list()

        freeze_pretrained_model = config['freeze_pretrained_model']
        if freeze_pretrained_model:
            enc_out.stop_gradient = True

        if config['mrc_layer'] == "cls_fc":

            enc_out = fluid.layers.dropout(
                x=enc_out,
                is_test=not is_train,
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
                is_test=not is_train,
                dropout_prob=args['lstm_dropout_prob'],
                dropout_implementation="upscale_in_train")
            encoded = highway_layer(enc_out, name="highway1", num_flatten_dims=2)
            encoded = fluid.layers.dropout(
                x=encoded,
                is_test=not is_train,
                dropout_prob=args['lstm_dropout_prob'],
                dropout_implementation="upscale_in_train")
            encoded = highway_layer(encoded, name="highway2", num_flatten_dims=2)
            encoded = fluid.layers.dropout(
                x=encoded,
                is_test=not is_train,
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
                is_test=not is_train,
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

        else:
            raise Exception("Downstream model not found. Check the network structure or config file.")

        logits = fluid.layers.transpose(x=logits, perm=[2, 0, 1])
        start_logits, end_logits = fluid.layers.unstack(x=logits, axis=0)

        if is_predict:
            fetch_val = {'unique_ids': unique_ids,
                         'start_logits': start_logits,
                         'end_logits': end_logits}

            return reader, fetch_val

        def compute_loss(logits, positions):
            loss = fluid.layers.softmax_with_cross_entropy(
                logits=logits, label=positions)
            loss = fluid.layers.mean(x=loss)
            return loss

        start_loss = compute_loss(start_logits, start_positions)
        end_loss = compute_loss(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2.0
        fetch_val = {'total_loss': total_loss,
                     'unique_ids': unique_ids,
                     'start_logits': start_logits,
                     'end_logits': end_logits,
                     'enc_output_list': enc_output_list}

        return reader, fetch_val
