import paddle.fluid as fluid

from util.util_filepath import *
from util.util_logging import UtilLogging as ULog
from preprocess.tokenizer_CHN import ChnTokenizer as CToken
from preprocess.batching import *
from tqdm import tqdm
import random
from random import randrange
import os


class FeatureForPretraining(object):

    def __init__(self, src_id, pos_id, sent_id, input_mask, label, mask_pos, mask_label):
        self.src_id = src_id
        self.pos_id = pos_id
        self.sent_id = sent_id
        self.input_mask = input_mask
        self.label = label


class ProcessorForPretrainingQa():
    def __init__(self, args, logger, questions, answers):
        self.logger = logger
        self.args = args
        assert len(questions) == len(answers)
        self.questions = questions
        self.answers = answers
        self.max_seq_length = self.args["max_seq_length"]
        self.qa_pair = []
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

    def merge_qa(self, question, answer, special_char=None):
        vocab = self.tokenizer.vocab
        if special_char is None:
            special_char = {"CLS": vocab["[CLS]"], "SEP": vocab["[SEP]"],
                            "MASK": vocab["[MASK]"], "PAD": vocab["[PAD]"]}

        sent = [special_char["CLS"]] + question + [special_char["SEP"]] + answer + [special_char["SEP"]]
        if len(sent) > self.max_seq_length:
            sent = sent[:self.max_seq_length - 1] + [special_char["SEP"]]
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

    def split_qa_to_qair(self):
        num = 0
        length = len(self.questions)
        for i in range(length):
            question = self.questions[i]
            answer = self.answers[i]
            question = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(question))
            answer = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(answer))
            question, answer = self._truncate_seq_pair(question, answer)
            self.qa_pair.append([question, answer])
            num += 1
            if num % 3000 == 0:
                self.logger.info('{} pairs generated'.format(num))
        self.logger.info('total split qa_pair {}'.format(len(self.qa_pair)))

    def convert_docs_to_features(self):
        if os.path.exists(get_fullurl(file_type='datap', file_name='pretrain_corpus_feature', file_format='pickle')):
            self.logger.info('load features from file')
            features = read_file('datap', 'pretrain_corpus_feature', 'pickle')
            self.features = features
            self.logger.info('{} features loaded'.format(len(self.features)))
            return

        self.split_qa_to_qair()
        features = []
        num = 0
        for index, qa_pair in enumerate(self.qa_pair):
            question, answer = qa_pair
            random_index = randrange(0, len(self.qa_pair))
            if random_index == index:
                random_index = randrange(0, len(self.qa_pair))
            _, otheranswer = self.qa_pair[random_index]
            prob_otheranswer = np.random.rand()
            if prob_otheranswer < 0.5:
                otheranswer_label = 0
                qa_merged = self.merge_qa(question, answer)
            else:
                otheranswer_label = 1
                qa_merged = self.merge_qa(question, otheranswer)
            features.append([qa_merged, otheranswer_label])
            num += 1
            if num % 5000 == 0:
                self.logger.info('{}features created'.format(num))
        save_file(features, 'datap', 'pretrain_corpus_feature', 'pickle')
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

    def data_generator(self):
        src_ids = []
        otheranswer_labels = []
        total_token_num = 0
        index = 0
        for sentence, otheranswer_label in self.features:
            index += 1
            otheranswer_labels.append(otheranswer_label)
            src_ids.append(sentence)
            total_token_num += len(sentence)
            if len(src_ids) == self.batch_size or index == len(self.features):
                src_ids, mask_labels, mask_pos = self.mask(src_ids, self.max_seq_length, total_token_num)
                out = self.pad_batch_data(src_ids, self.max_seq_length,
                                          return_pos=True, return_sent=True, return_input_mask=True,
                                          sep_id=self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0])
                src_ids, pos_ids, sent_ids, input_masks = out[0], out[1], out[2], out[3]
                otheranswer_labels = np.array(otheranswer_labels).reshape([-1, 1])
                yield src_ids, pos_ids, sent_ids, input_masks, mask_labels, mask_pos, otheranswer_labels

                src_ids = []
                otheranswer_labels = []
                total_token_num = 0

    def _truncate_seq_pair(self, tokens_a, tokens_b):
        """
        截短过长的问答对
        """
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



