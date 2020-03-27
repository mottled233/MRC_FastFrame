"""
bert的paddle实现
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import json
import paddle.fluid as fluid
from model.transformer import transformer_encoder as encoder
from model.transformer import pre_process_layer


# 记录Bert相关参数的Config类
class BertConfig(object):
    """ 根据config_path来读取网络的配置 """

    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing bert model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict[key]

    def print_config(self):
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


# Bert模型
# 以(src_ids, position_ids, sentence_ids, input_mask）作为输入
# 主要结构为embedding、transformer_encoder两层
# 有三种返回结果，返回整个seq每个位置的输出，只返回[CLS]对应的输出，和返回预训练过程所进行的mask_prediction
# 和next_sentence_prediction两个任务的loss。
class BertModel(object):

    # 根据传入的config设置网络参数
    def __init__(self,
                 src_ids,
                 position_ids,
                 sentence_ids,
                 input_mask,
                 config,
                 weight_sharing=True,
                 use_fp16=False,
                 is_prediction=False):

        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._max_position_seq_len = config['max_seq_length']
        self._sent_types = config['type_vocab_size']
        self._hidden_act = config['hidden_act']
        self._post_and_pre_process_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_dropout_prob']
        self._weight_sharing = weight_sharing

        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._sent_emb_name = "sent_embedding"
        self._dtype = "float16" if use_fp16 else "float32"

        # 使用truncated normal initializer进行参数的初始化, 且biases
        # 默认初始化为0
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'])

        self._build_model(src_ids, position_ids, sentence_ids, input_mask, is_test=is_prediction)

    # 进行模型的搭建工作
    def _build_model(self, src_ids, position_ids, sentence_ids, input_mask, is_test=False):
        # padding对应的词表中的id必须为0
        # 模型中的三种embedding
        emb_out = fluid.layers.embedding(
            input=src_ids,
            size=[self._voc_size, self._emb_size],
            dtype=self._dtype,
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name, initializer=self._param_initializer),
            is_sparse=False)
        position_emb_out = fluid.layers.embedding(
            input=position_ids,
            size=[self._max_position_seq_len, self._emb_size],
            dtype=self._dtype,
            param_attr=fluid.ParamAttr(
                name=self._pos_emb_name, initializer=self._param_initializer))

        sent_emb_out = fluid.layers.embedding(
            sentence_ids,
            size=[self._sent_types, self._emb_size],
            dtype=self._dtype,
            param_attr=fluid.ParamAttr(
                name=self._sent_emb_name, initializer=self._param_initializer))

        emb_out = emb_out + position_emb_out
        emb_out = emb_out + sent_emb_out

        # 接下来是transformer的encoder部分(随机的mask，构造任务，过encoder等)
        emb_out = pre_process_layer(
            emb_out, 'nd', self._post_and_pre_process_dropout, name='pre_encoder', is_test=is_test)

        if self._dtype == "float16":
            input_mask = fluid.layers.cast(x=input_mask, dtype=self._dtype)

        self_attn_mask = fluid.layers.matmul(
            x=input_mask, y=input_mask, transpose_y=True)
        self_attn_mask = fluid.layers.scale(
            x=self_attn_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = fluid.layers.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        self._enc_out = encoder(
            inputs=emb_out,
            attention_bias=n_head_self_attn_mask,
            num_attention_layers=self._n_layer,
            num_attention_heads=self._n_head,
            key_size=self._emb_size // self._n_head,
            value_size=self._emb_size // self._n_head,
            hidden_state_size=self._emb_size,
            inner_hidden_size=self._emb_size * 4,
            post_and_pre_process_dropout=self._post_and_pre_process_dropout,
            attention_dropout=self._attention_dropout,
            activate_dropout=0,
            hidden_act=self._hidden_act,
            preprocess_cmd="",
            postprocess_cmd="dan",
            param_initializer=self._param_initializer,
            name='encoder',
            is_test=is_test)

    # 返回序列每个位置的token对应的输出
    def get_sequence_output(self):
        return self._enc_out

    # 只返回[CLS]的输出
    def get_pooled_output(self):
        """Get the first feature of each sequence for classification"""

        next_sent_feat = fluid.layers.slice(
            input=self._enc_out, axes=[1], starts=[0], ends=[1])
        next_sent_feat = fluid.layers.fc(
            input=next_sent_feat,
            size=self._emb_size,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="pooled_fc.w_0", initializer=self._param_initializer),
            bias_attr="pooled_fc.b_0")
        return next_sent_feat

    # 返回预训练任务的loss
    def get_pretraining_output(self, mask_label, mask_pos, labels, is_test=False):
        """Get the loss & accuracy for pretraining"""

        mask_pos = fluid.layers.cast(x=mask_pos, dtype='int32')

        # 得到[CLS]对应的输出
        next_sent_feat = self.get_pooled_output()
        reshaped_emb_out = fluid.layers.reshape(
            x=self._enc_out, shape=[-1, self._emb_size])

        # 得到被mask位置的输出
        mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=mask_pos)

        # 全连接
        mask_trans_feat = fluid.layers.fc(
            input=mask_feat,
            size=self._emb_size,
            act=self._hidden_act,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_fc.w_0',
                initializer=self._param_initializer),
            bias_attr=fluid.ParamAttr(name='mask_lm_trans_fc.b_0'))

        # layer norm
        mask_trans_feat = pre_process_layer(
            mask_trans_feat, 'n', name='mask_lm_trans', is_test=is_test)

        mask_lm_out_bias_attr = fluid.ParamAttr(
            name="mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))

        if self._weight_sharing:
            # 计算mask部分的预测结果
            fc_out = fluid.layers.matmul(
                x=mask_trans_feat,
                y=fluid.default_main_program().global_block().var(
                    self._word_emb_name),
                transpose_y=True)
            fc_out += fluid.layers.create_parameter(
                shape=[self._voc_size],
                dtype=self._dtype,
                attr=mask_lm_out_bias_attr,
                is_bias=True)

        else:
            # 计算mask部分的预测结果
            fc_out = fluid.layers.fc(input=mask_trans_feat,
                                     size=self._voc_size,
                                     param_attr=fluid.ParamAttr(
                                         name="mask_lm_out_fc.w_0",
                                         initializer=self._param_initializer),
                                     bias_attr=mask_lm_out_bias_attr)

        # 计算mask预测任务的loss
        mask_lm_loss = fluid.layers.softmax_with_cross_entropy(
            logits=fc_out, label=mask_label)
        mean_mask_lm_loss = fluid.layers.mean(mask_lm_loss)

        # 计算next_sentence部分的结果
        next_sent_fc_out = fluid.layers.fc(
            input=next_sent_feat,
            size=2,
            param_attr=fluid.ParamAttr(
                name="next_sent_fc.w_0", initializer=self._param_initializer),
            bias_attr="next_sent_fc.b_0")

        # 计算next_sentence部分的loss
        next_sent_loss, next_sent_softmax = fluid.layers.softmax_with_cross_entropy(
            logits=next_sent_fc_out, label=labels, return_softmax=True)

        next_sent_acc = fluid.layers.accuracy(
            input=next_sent_softmax, label=labels)

        mean_next_sent_loss = fluid.layers.mean(next_sent_loss)

        # 总得loss为两部分loss的和
        loss = mean_next_sent_loss + mean_mask_lm_loss
        return next_sent_acc, mean_mask_lm_loss, loss
