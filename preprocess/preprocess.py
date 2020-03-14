import paddle.fluid as fluid

from util.util_filepath import *
from util.util_logging import UtilLogging as ULog
from preprocess.tokenizer_CHN import ChnTokenizer as CToken
from preprocess.batching import *
from preprocess.util_preprocess import *
import random


class Feature(object):

    def __init__(self, qas_id, src_id, pos_id, sent_id, input_mask, label):

        self.qas_id = qas_id
        self.src_id = src_id
        self.pos_id = pos_id
        self.sent_id = sent_id
        self.input_mask = input_mask
        self.label = label


class PreProcess:

    def __init__(self, logger, args, examples, for_prediction=False):

        self.logger = logger
        self.args = args
        self.for_prediction = for_prediction

        reverse_qa = False
        self.examples = examples
        if reverse_qa:
            self.reverse_qa()

        self.max_seq_length = self.args["max_seq_length"]

        # self.logger.info("Prepare to build tokenizer ……")
        self.c_token = CToken(
            self.args["vocab_name"], self.args["vocab_format"], self.args["vocab_type"], self.args["do_lowercase"]
        )
        self.logger.info("Successfully build tokenizer")
        # 使用指定的字典，构建tokenizer

        self.func(half_width)
        self.func(lower)
        if self.args["pretrained_model_type"] == "ernie":
            self.func(punctuation_replace_for_ernie)
            self.func(translate_for_ernie)
        self.func(split_unk, self.c_token.vocab)

        self.features = []

    def get_vocab_size(self):
        """
        获取使用的词表的大小
        """

        return len(self.c_token.vocab)

    def func(self, util_func, vocab=None):
        """
        对question和answer的文本信息进行整理
        """

        if vocab is None:
            for i in range(len(self.examples)):
                self.examples[i].question = util_func(self.examples[i].question)
                self.examples[i].answer = util_func(self.examples[i].answer)
        else:
            for i in range(len(self.examples)):
                self.examples[i].question = util_func(self.examples[i].question, vocab)
                self.examples[i].answer = util_func(self.examples[i].answer, vocab)

    def exams_tokenize(self, examples, token_id=2):
        """
        完成对list of Example中question和answer的tokenize，并返回结果列表
        :param examples: 需要处理的Example列表
        :param token_id: token_id=1时返回tokens，token_id=2时返回ids
        """

        ques_tokens = []
        ans_tokens = []
        ques_ids = []
        ans_ids = []
        for example in examples:
            q_tokens = self.c_token.tokenize(example.question)
            ques_tokens.append(q_tokens)
            a_tokens = self.c_token.tokenize(example.answer)
            ans_tokens.append(a_tokens)
            if token_id == 2:
                q_ids = self.c_token.convert_tokens_to_ids(q_tokens)
                ques_ids.append(q_ids)
                a_ids = self.c_token.convert_tokens_to_ids(a_tokens)
                ans_ids.append(a_ids)
        if token_id == 1:
            return ques_tokens, ans_tokens
        elif token_id == 2:
            return ques_ids, ans_ids

    def save_tokens(self, tokens, file_name, file_format="pickle", file_type="datap"):
        """
        将tokens储存在指定位置
        """

        save_file(tokens, file_type, file_name, file_format)

    def splice_ques_ans(self, ques_ids, ans_ids, special_char=None):
        """
        对问题张量和答案张量进行拼接，并返回句子最大长度与单个token总数的信息
        """

        vocab = self.c_token.vocab
        if special_char is None:
            special_char = {"CLS": vocab["[CLS]"], "SEP": vocab["[SEP]"],
                            "MASK": vocab["[MASK]"], "PAD": vocab["[PAD]"]}

        l1 = len(ques_ids)
        l2 = len(ans_ids)
        if l1 != l2:
            raise Exception("Different number of Questions and Answers")
            # 发现问题答案数量不匹配，返回错误信息
        batch_tokens = []
        max_len = 0
        total_token_num = 0
        for i in range(l1):
            if len(ques_ids[i]) + len(ans_ids[i]) > self.max_seq_length - 3:
                ques_ids[i], ans_ids[i] = self._truncate_seq_pair(ques_ids[i], ans_ids[i])

            sent = [special_char["CLS"]] + ques_ids[i] + [special_char["SEP"]] + ans_ids[i] + [special_char["SEP"]]

            batch_tokens.append(sent)
            max_len = max(max_len, len(sent))
            total_token_num += len(sent)

        return batch_tokens, max_len, total_token_num

    def mask(self, batch_tokens, max_len, total_token_num, special_char=None):
        """
        进行mask覆盖，返回覆盖后的结果和覆盖信息
        """

        vocab = self.c_token.vocab
        if special_char is None:
            special_char = {"CLS": vocab["[CLS]"], "SEP": vocab["[SEP]"],
                            "MASK": vocab["[MASK]"], "PAD": vocab["[PAD]"]}

        vocab_size = len(self.c_token.vocab)
        return mask(batch_tokens, max_len, total_token_num, vocab_size, special_char)

    def pad_batch_data(self,
                       batch_tokens,
                       max_len,
                       pad_idx=0,
                       return_pos=False,
                       return_sent=False, sep_id=None,
                       return_input_mask=False,
                       # return_max_len=False,
                       # return_num_token=False
                       ):
        """
        将句子统一填充到最大句子长度，并生成相应的位置数据和输入覆盖
        """

        if sep_id is None:
            sep_id = self.c_token.vocab["[SEP]"]
        return pad_batch_data(batch_tokens, max_len, pad_idx, return_pos, return_sent, sep_id, return_input_mask)

    '''
    def prepare_batch_data(self,
                           insts,
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
        """

        return prepare_batch_data(insts, max_len, total_token_num, voc_size, pad_id, cls_id, sep_id, mask_id)
    '''

    def get_tokens(self, file_name, file_format="pickle", file_type="datap"):
        """
        获取tokenize结果并将之储存进指定文件，若文件已存在则直接读取
        file_name=""表示不进行文件缓存
        """

        if file_name != "" and os.path.exists(get_fullurl(file_type, file_name, file_format)):
            self.logger.info("Get tokens from file")
            self.logger.info("File location: " + get_fullurl(file_type, file_name, file_format))
            batch_tokens = read_file(file_type, file_name, file_format)
            total_token_num = 0
            for sent in batch_tokens:
                total_token_num += len(sent)

        else:
            self.logger.info("Start caching output of tokenizing")
            ques_ids, ans_ids = self.exams_tokenize(self.examples)
            self.logger.info("  - Complete tokenizing")
            batch_tokens, _, total_token_num = self.splice_ques_ans(ques_ids, ans_ids)
            self.logger.info("  - Complete splicing question and answer")
            if file_name != "":
                self.save_tokens(batch_tokens, file_name)
                self.logger.info("  - Complete cache of tokenize results")
                self.logger.info("    File location: " + "dataset_processed/" + file_name)
            self.logger.info("Finish caching")

        return batch_tokens, total_token_num

    def prepare_batch_data(self, cache_filename="", file_format="pickle", file_type="datap"):
        """
        先从指定文件获取batch_tokens与total_token_num数据
        对给出的batch_tokens进行mask覆盖及填充处理，并返回其他id数据
        """

        batch_tokens, total_token_num = self.get_tokens(cache_filename, file_format=file_format, file_type=file_type)

        self.logger.info("Start data-preprocessing before batching")

        if self.args["is_mask"]:
            batch_tokens, mask_label, mask_pos = self.mask(batch_tokens, self.max_seq_length, total_token_num)
            self.logger.info("  - Complete masking tokens")

        out = self.pad_batch_data(batch_tokens, self.max_seq_length,
                                  return_pos=True, return_sent=True, return_input_mask=True)
        src_ids, pos_ids, sent_ids, input_masks = out[0], out[1], out[2], out[3]
        qas_ids = []
        labels = []
        temp = {"Yes": 0, "No": 1, "Depends": 2}
        for example in self.examples:
            qas_ids.append(example.qas_id)
        if not self.for_prediction:
            for example in self.examples:
                try:
                    labels.append(temp[example.yes_or_no])
                except Exception:
                    raise KeyError("Error in labels of train-dataset") from Exception
                    # 训练集标签中出现Yes,No,Depends以外的值，返回错误信息
        else:
            labels = [3] * len(self.examples)
        self.logger.info("  - Complete filling the tokens to max_seq_length, and getting other ids")

        self.features = []
        for i in range(len(self.examples)):
            self.features.append(Feature(
                qas_ids[i], src_ids[i], pos_ids[i], sent_ids[i], input_masks[i], labels[i]
            ))
        self.logger.info("  - Complete constructing features object")
        self.logger.info("Finish data-preprocessing")

    def sample_generator(self):
        self.logger.info("Preprocessing a new round of data of {}".format(len(self.features)))
        if self.args["shuffle"]:
            random.shuffle(self.features)
        if not self.for_prediction:
            for feature in self.features:
                yield feature.qas_id, feature.src_id, feature.pos_id, feature.sent_id, feature.input_mask, feature.label
        else:
            for feature in self.features:
                yield feature.qas_id, feature.src_id, feature.pos_id, feature.sent_id, feature.input_mask

    def batch_generator(self):
        reader = fluid.io.batch(self.sample_generator, batch_size=self.args["batch_size"])
        return reader

    def reverse_qa(self):
        """
        将question和answer的位置互换
        """
        for example in self.examples:
            a = example.answer
            example.answer = example.question
            example.question = a

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


