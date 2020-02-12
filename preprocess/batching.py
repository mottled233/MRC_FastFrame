from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def mask(batch_tokens, max_len, total_token_num, vocab_size, special_char=None):
    """
    为tokens添加mask，返回运行结果、mask标签、与mask位置
    :param batch_tokens: 待添加mask的tokens
    :param max_len: 最大句子长度
    :param total_token_num: 单个token总数
    :param vocab_size: 字典大小
    :param special_char: bert中特殊字符
    :return:
        batch_tokens: 打完mask的结果
        mask_label: 被mask覆盖的token（包括覆盖为原词的），输出为tuple
        mask_pos：被mask覆盖的token的所在位置（按最长句子统计），输出为tuple
    """
    if special_char is None:
        special_char = {"CLS": 1, "SEP": 2, "MASK": 3}

    mask_label = []
    mask_pos = []
    prob_mask = np.random.rand(total_token_num)
    replace_ids = np.random.randint(1, high=vocab_size, size=total_token_num)  # 不能使用字典中的第0号元素[PAD]进行覆盖

    pre_sent_len = 0  # 当前句子长度
    prob_index = 0  # 之前句子长度之和
    for sent_index, sent in enumerate(batch_tokens):
        mask_flag = False
        prob_index += pre_sent_len
        pre_sent_len = len(sent)
        for token_index, token in enumerate(sent):
            prob = prob_mask[prob_index + token_index]

            if prob > 0.15:
                # 85%的token不覆盖，其他均为覆盖
                continue
            elif 0.03 < prob <= 0.15:
                # 15%*80%的token用mask覆盖
                if token not in special_char.values():
                    mask_label.append(sent[token_index])
                    sent[token_index] = special_char["MASK"]
                    mask_flag = True
                    mask_pos.append(sent_index * max_len + token_index)
            elif 0.015 < prob <= 0.03:
                # 15%*10%的token用其他token替换
                if token not in special_char.values():
                    mask_label.append(sent[token_index])
                    sent[token_index] = replace_ids[prob_index + token_index]
                    mask_flag = True
                    mask_pos.append(sent_index * max_len + token_index)
            else:
                # 15%*10%的token保持不变
                if token not in special_char.values():
                    mask_label.append(sent[token_index])
                    mask_pos.append(sent_index * max_len + token_index)

        # 确保一定存在被改变的token
        while not mask_flag:
            token_index = int(np.random.randint(1, high=len(sent) - 1, size=1))
            if sent[token_index] not in special_char.values():
                mask_label.append(sent[token_index])
                sent[token_index] = special_char["MASK"]
                mask_flag = True
                mask_pos.append(sent_index * max_len + token_index)

    mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])
    mask_pos = np.array(mask_pos).astype("int64").reshape([-1, 1])
    return batch_tokens, mask_label, mask_pos


def pad_batch_data(batch_tokens,
                   max_len,
                   pad_idx=0,
                   return_pos=False,
                   return_sent=False, sep_id=2,
                   return_input_mask=False,
                   # return_max_len=False,
                   # return_num_token=False
                   ):
    """
    将句子统一填充到最大句子长度，并生成相应的位置数据和输入覆盖
    :param batch_tokens:
    :param max_len:
    :param pad_idx: dict中包含的任何标记都可用于填充，因为填充的损失将被权重掩盖，并且对参数梯度没有影响
    :param return_pos: position_id
    :param return_sent: sentence_id
    :param sep_id:
    :param return_input_mask: input_mask
    :return:
    """

    return_list = []

    inst_data = np.array([
        list(inst) + list([pad_idx] * (max_len - len(inst))) for inst in batch_tokens
    ])
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    if return_pos:
        # 位置数据，用于表示token在其句子中的位置
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in batch_tokens
        ])
        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_sent:
        # 句子数据，用于表示是否属于回答部分
        sep = [[] for _ in batch_tokens]
        for i in range(len(batch_tokens)):
            for j in range(len(batch_tokens[i])):
                if batch_tokens[i][j] == sep_id:
                    sep[i].append(j)
            sep[i][1] += 1
        inst_sent = np.array([
            [0] * sep[i][0] + [1] * (sep[i][1] - sep[i][0]) + [pad_idx] * (max_len - sep[i][1])
            for i in range(len(batch_tokens))
        ])
        return_list += [inst_sent.astype("int64").reshape([-1, max_len, 1])]

    if return_input_mask:
        # 输入覆盖
        input_mask_data = np.array([[1] * len(inst) + [0] * (max_len - len(inst))
                                    for inst in batch_tokens])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)  # 去除最内层的不必要维度
        return_list += [input_mask_data.astype("float32")]

    '''
    if return_max_len:
        # 最大句子长度
        return_list += [max_len]

    if return_num_token:
        # 单个token总数
        num_token = 0
        for inst in batch_tokens:
            num_token += len(inst)
        return_list += [num_token]
    '''

    return return_list if len(return_list) > 1 else return_list[0]


def prepare_batch_data(insts,
                       max_len,
                       total_token_num,
                       voc_size=0,
                       pad_id=None,
                       cls_id=None,
                       sep_id=None,
                       mask_id=None,
                       # return_input_mask=True,
                       # return_max_len=True,
                       # return_num_token=False
                       ):
    """
    创建数据张量、位置张量、自注意力覆盖（shape: batch_size*max_len*max_len）
    :param insts:
    :param max_len:
    :param total_token_num:
    :param voc_size:
    :param pad_id:
    :param cls_id:
    :param sep_id:
    :param mask_id:
    :return:
    """

    batch_src_ids = [inst[0] for inst in insts]
    batch_sent_ids = [inst[1] for inst in insts]
    batch_pos_ids = [inst[2] for inst in insts]
    labels_list = []

    for i in range(3, len(insts[0]), 1):
        labels = [inst[i] for inst in insts]
        labels = np.array(labels).astype("int64").reshape([-1, 1])
        labels_list.append(labels)

    # 第一步：在不进行填充的情况下完成mask
    if mask_id >= 0:
        out, mask_label, mask_pos = mask(
            batch_src_ids,
            max_len,
            total_token_num,
            vocab_size=voc_size,
            special_char={"CLS": cls_id, "SEP": sep_id, "MASK": mask_id})
    else:
        out = batch_src_ids
        mask_label, mask_pos = [], []

    # 第二步：进行句子的填充
    src_id, self_input_mask = pad_batch_data(
        out, max_len, pad_idx=pad_id, return_input_mask=True)
    pos_id = pad_batch_data(
        batch_pos_ids,
        max_len,
        pad_idx=pad_id,
        return_pos=False,
        return_input_mask=False)
    sent_id = pad_batch_data(
        batch_sent_ids,
        max_len,
        pad_idx=pad_id,
        return_pos=False,
        return_input_mask=False)

    if mask_id >= 0:
        return_list = [
            src_id, pos_id, sent_id, self_input_mask, mask_label, mask_pos
        ] + labels_list
    else:
        return_list = [src_id, pos_id, sent_id, self_input_mask] + labels_list

    return return_list if len(return_list) > 1 else return_list[0]
