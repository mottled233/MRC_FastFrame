import paddle.fluid as fluid

from util.util_filepath import *
from util.util_logging import UtilLogging as ULog
from preprocess.tokenizer_CHN import ChnTokenizer as CToken
from preprocess.batching import *
from tqdm import tqdm
import random
import json
import os


class FeatureForMultiTask(object):

    def __init__(self, qas_id, src_id, pos_id, sent_id, input_mask, label, label_1):
        self.qas_id = qas_id
        self.src_id = src_id
        self.pos_id = pos_id
        self.sent_id = sent_id
        self.input_mask = input_mask
        self.label = label
        self.label_1 = label_1


class ProcessorForMultiTask(object):
    def __init__(self, args, logger, examples, feature_file_name, task_id=0, is_prediction=False):
        self.logger = logger
        self.args = args
        self.examples = examples
        self.task_id = task_id
        self.is_prediction = is_prediction
        self.feature_file_name = feature_file_name
        self.max_seq_length = self.args["max_seq_length"]
        # self.logger.info("Prepare to build tokenizer ……")
        self.tokenizer = CToken(
            self.args["vocab_name"], self.args["vocab_format"], self.args["vocab_type"], self.args["do_lowercase"]
        )
        self.logger.info("Successfully build tokenizer")
        # 使用指定的字典，构建tokenizer
        self.batch_size = args['batch_size']
        self.features = []

    def get_vocab_size(self):
        """
        获取使用的词表的大小
        """

        return len(self.tokenizer.vocab)

    def merge_sentences(self, sentence_1, sentence_2, special_char=None):
        vocab = self.tokenizer.vocab
        if special_char is None:
            special_char = {"CLS": vocab["[CLS]"], "SEP": vocab["[SEP]"],
                            "MASK": vocab["[MASK]"], "PAD": vocab["[PAD]"]}

        if len(sentence_1) + len(sentence_2) > self.max_seq_length - 3:
            sentence_1, sentence_2 = self._truncate_seq_pair(sentence_1, sentence_2)

        sent = [special_char["CLS"]] + sentence_1 + [special_char["SEP"]] + sentence_2 + [special_char["SEP"]]
        return sent

    def mask(self, batch_tokens, max_len, total_token_num, special_char=None):
        """
        进行mask覆盖，返回覆盖后的结果和覆盖信息
        """
        vocab = self.tokenizer.vocab
        if special_char is None:
            special_char = {"CLS": vocab["[CLS]"], "SEP": vocab["[SEP]"],
                            "MASK": vocab["[MASK]"], "PAD": vocab["[PAD]"]}

        vocab_size = len(self.tokenizer.vocab)
        return mask(batch_tokens, max_len, total_token_num, vocab_size, special_char)

    def convert_examples_to_features(self):
        if os.path.exists(get_fullurl(file_type='datap', file_name=self.feature_file_name, file_format='pickle')):
            self.logger.info('load features from file')
            features = read_file('datap', self.feature_file_name, 'pickle')
            self.features = features
            self.logger.info('{} features loaded'.format(len(self.features)))
            return

        features = []
        labels = []
        labels_for_reverse = []
        src_ids = []
        qas_ids = []
        label_map = {"Yes": 0, "No": 1, "Depends": 2}
        for example in self.examples:
            if self.is_prediction:
                labels.append(0)
            else:
                labels.append(label_map[example.yes_or_no])
            question_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(example.question))
            answer_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(example.answer))
            if self.task_id == 0:
                prob_reverse = np.random.rand()
                if prob_reverse > 0.5:
                    src_id = self.merge_sentences(question_id, answer_id)
                    labels_for_reverse.append(1)
                else:
                    src_id = self.merge_sentences(answer_id, question_id)
                    labels_for_reverse.append(0)
            if self.task_id == 1:
                src_id = self.merge_sentences(question_id, answer_id)
                labels_for_reverse.append(0)
            if self.task_id == 2:
                src_id = self.merge_sentences(answer_id, question_id)
                labels_for_reverse.append(1)
            src_ids.append(src_id)
            qas_ids.append(example.qas_id)
        src_ids, pos_ids, sent_ids, input_masks = self.pad_batch_data(
            src_ids, self.max_seq_length,
            return_pos=True, return_sent=True,
            return_input_mask=True,
            sep_id=self.tokenizer.vocab['[SEP]'])
        labels = np.array(labels).reshape([-1, 1])
        labels_for_reverse = np.array(labels_for_reverse).reshape([-1, 1])
        for i in range(len(src_ids)):
            features.append(FeatureForMultiTask(qas_ids[i],
                                                src_ids[i],
                                                pos_ids[i],
                                                sent_ids[i],
                                                input_masks[i],
                                                labels[i],
                                                labels_for_reverse[i]))

        save_file(features, 'datap', self.feature_file_name, 'pickle')
        self.features = features

    def pad_batch_data(self,
                       batch_tokens,
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
        """

        return pad_batch_data(batch_tokens, max_len, pad_idx, return_pos, return_sent, sep_id, return_input_mask)

    def sample_generator(self):
        self.logger.info("Preprocessing a new round of data of {}".format(len(self.features)))
        if self.args["shuffle"]:
            random.shuffle(self.features)
        for feature in self.features:
            if self.is_prediction:
                yield feature.qas_id, feature.src_id, feature.pos_id, \
                    feature.sent_id, feature.input_mask
            else:
                yield feature.qas_id, feature.src_id, feature.pos_id, \
                      feature.sent_id, feature.input_mask, feature.label, feature.label_1

    def batch_generator(self):
        reader = fluid.io.batch(self.sample_generator, batch_size=self.args["batch_size"])
        return reader

    def _truncate_seq_pair(self, tokens_a, tokens_b):
        """截短过长的问答对."""
        max_length = self.max_seq_length - 3

        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

        return tokens_a, tokens_b


class ProcessorForMergeModel(object):
    def __init__(self, args, logger, file_name_1, file_name_2, feature_file_name=None, task_id=0, is_prediction=False):
        self.logger = logger
        self.args = args
        self.features = []
        self.task_id = task_id
        self.is_prediction = is_prediction
        self.feature_file_name = feature_file_name
        self.max_seq_length = self.args["max_seq_length"]
        # self.logger.info("Prepare to build tokenizer ……")
        # 使用指定的字典，构建tokenizer
        self.batch_size = args['batch_size']
        self.logger.info('start to create feature')

        with open(file_name_1, 'r') as f1:
            with open(file_name_2, 'r') as f2:
                feats_1 = json.load(f1)
                feats_2 = json.load(f2)
                for i in range(len(feats_1)):
                    if not self.is_prediction:
                        feat = {'qas_id': np.array(feats_1[i]['qas_id']),
                                'label': np.array(feats_1[i]['label']),
                                'cls_feats': np.array(feats_1[i]['cls_feats'] + feats_2[i]['cls_feats'],
                                                      dtype=np.float32)}
                    else:
                        feat = {'qas_id': np.array(feats_1[i]['qas_id']),
                                'cls_feats': np.array(feats_1[i]['cls_feats'] + feats_2[i]['cls_feats'],
                                                      dtype=np.float32)}
                    self.features.append(feat)
        self.logger.info('total features:{}'.format(len(self.features)))

    def sample_generator(self):
        self.logger.info("Preprocessing a new round of data of {}".format(len(self.features)))
        if self.args["shuffle"]:
            random.shuffle(self.features)
        for feature in self.features:
            if self.is_prediction:
                yield feature['qas_id'], feature['cls_feats']
            else:
                yield feature['qas_id'], feature['cls_feats'], feature['label']

    def batch_generator(self):
        reader = fluid.io.batch(self.sample_generator, batch_size=self.args["batch_size"])
        return reader
