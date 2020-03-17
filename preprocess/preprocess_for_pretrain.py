import paddle.fluid as fluid

from util.util_filepath import *
from util.util_logging import UtilLogging as ULog
from preprocess.tokenizer_CHN import ChnTokenizer as CToken
from preprocess.batching import *
from tqdm import tqdm
import random
import os


class FeatureForPretraining(object):

    def __init__(self, src_id, pos_id, sent_id, input_mask, label, mask_pos, mask_label):
        self.src_id = src_id
        self.pos_id = pos_id
        self.sent_id = sent_id
        self.input_mask = input_mask
        self.label = label


class ProcessorForPretraining():
    def __init__(self, args, logger, docs, ):
        self.logger = logger
        self.args = args
        self.docs = docs
        self.max_seq_length = self.args["max_seq_length"]
        self.sentence_split = []
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

        sent = [special_char["CLS"]] + sentence_1 + [special_char["SEP"]] + sentence_2 + [special_char["SEP"]]
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

    def split_docs_to_sentence(self):
        num = 0
        for doc in self.docs:
            sentences = doc.split('。')
            sentence_tokenized = []
            for sentence in sentences:
                if len(sentence) == 0:
                    continue
                sentence = self.tokenizer.tokenize(sentence)
                sentence = self.tokenizer.convert_tokens_to_ids(sentence)
                if len(sentence) > (self.max_seq_length/2 - 3):
                    split_length = int(self.max_seq_length/2) - 3
                    sentence_split = [sentence[i:i + split_length] for i in range(0, len(sentence), split_length)]
                    sentence_tokenized += sentence_split
                else:
                    sentence_tokenized.append(sentence)

            if len(sentence_tokenized) == 1:
                continue
            for i in range(len(sentence_tokenized) - 1):
                sentence_1 = sentence_tokenized[i]
                sentence_2 = sentence_tokenized[i+1]
                self.sentence_split.append([sentence_1, sentence_2])
            num += 1
            if num%3000 == 0:
                self.logger.info('{}docs splited'.format(num))
        self.logger.info('total split sentence {}'.format(len(self.sentence_split)))

    def convert_docs_to_features(self):
        if os.path.exists(get_fullurl(file_type='datap', file_name='pretrain_corpus_feature', file_format='pickle')):
            self.logger.info('load features from file')
            features = read_file('datap', 'pretrain_corpus_feature', 'pickle')
            self.features = features
            self.logger.info('{} features loaded'.format(len(self.features)))
            return

        self.split_docs_to_sentence()
        features = []
        num = 0
        for sentence_pair in self.sentence_split:
            sentence_1, sentence_2 = sentence_pair
            prob_reverse = np.random.rand()
            if prob_reverse < 0.5:
                reverse_label = 0
                sent_merged = self.merge_sentences(sentence_1, sentence_2)
            else:
                reverse_label = 1
                sent_merged = self.merge_sentences(sentence_2, sentence_1)
            features.append([sent_merged, reverse_label])
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
        reverse_labels = []
        total_token_num = 0
        for sentence, reverse_label in self.features:
            reverse_labels.append(reverse_label)
            src_ids.append(sentence)
            total_token_num += len(sentence)
            if len(src_ids) == self.batch_size:
                src_ids, mask_labels, mask_pos = self.mask(src_ids, self.max_seq_length, total_token_num)
                out = self.pad_batch_data(src_ids, self.max_seq_length,
                                          return_pos=True, return_sent=True, return_input_mask=True,
                                          sep_id=self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0])
                src_ids, pos_ids, sent_ids, input_masks = out[0], out[1], out[2], out[3]
                reverse_labels = np.array(reverse_labels).reshape([-1, 1])
                yield src_ids, pos_ids, sent_ids, input_masks, mask_labels, mask_pos, reverse_labels

                src_ids = []
                reverse_labels = []
                total_token_num = 0



