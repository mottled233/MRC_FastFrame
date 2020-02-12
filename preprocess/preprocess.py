from util.util_filepath import *
from util.util_logging import UtilLogging as ULog
from preprocess.tokenizer_CHN import ChnTokenizer as CToken
from preprocess.batching import *


class Feature(object):

    def __init__(self, qas_id, src_id, pos_id, sent_id, input_mask, label):

        self.qas_id = qas_id
        self.src_id = src_id
        self.pos_id = pos_id
        self.sent_id = sent_id
        self.input_mask = input_mask
        self.label = label


class PreProcess:

    def __init__(self, logger, args, examples=None):

        self.logger = logger

        self.args = args
        self.max_seq_length = self.args["max_seq_length"]
        if examples is not None:
            self.examples = examples
        else:
            self.get_examples_from_file(
                self.args["examples_name"], self.args["examples_format"], self.args["examples_type"]
            )
        self.file_name = self.args["examples_name"]
        self.c_token = CToken(
            self.args["vocab_name"], self.args["vocab_format"], self.args["vocab_type"], self.args["do_lowercase"]
        )
        # 使用指定的字典，构建tokenizer

        self.features = []

    def get_examples_from_file(self, file_name, file_format="pickle", file_type="example"):
        """
        读取指定文件，返回list of Example结果
        """

        try:
            self.examples = read_file(file_type, file_name, file_format)
        except Exception:
            raise FileNotFoundError("未发现数据集文件")
            # 未发现数据集文件，返回错误信息

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
            q_ids = self.c_token.convert_tokens_to_ids(q_tokens)
            ques_ids.append(q_ids)
            a_tokens = self.c_token.tokenize(example.answer)
            ans_tokens.append(a_tokens)
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
        if special_char is None:
            special_char = {"CLS": 1, "SEP": 2, "MASK": 3}

        l1 = len(ques_ids)
        l2 = len(ans_ids)
        if l1 != l2:
            raise Exception("问题答案数量不匹配")
        batch_tokens = []
        max_len = 0
        total_token_num = 0
        for i in range(l1):
            sent = [special_char["CLS"]] + ques_ids[i] + [special_char["SEP"]] + ans_ids[i] + [special_char["SEP"]]
            if len(sent) > self.max_seq_length:
                sent = sent[:self.max_seq_length - 1] + [special_char["SEP"]]
            batch_tokens.append(sent)
            max_len = max(max_len, len(sent))
            total_token_num += len(sent)

        return batch_tokens, max_len, total_token_num

    def mask(self, batch_tokens, max_len, total_token_num, special_char=None):
        """
        进行mask覆盖，返回覆盖后的结果和覆盖信息
        """
        if special_char is None:
            special_char = {"CLS": 1, "SEP": 2, "MASK": 3}

        vocab_size = len(self.c_token.vocab)
        return mask(batch_tokens, max_len, total_token_num, vocab_size, special_char)

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

    def prepare_batch_data(self, is_mask=False, is_save=False, file_name=None):
        """
        完成数据tokenize操作与缓存，进行句子的填充并返回id数据
        """

        ques_ids, ans_ids = self.exams_tokenize(self.examples)
        batch_tokens, max_len, total_token_num = self.splice_ques_ans(ques_ids, ans_ids)
        if is_save:
            if file_name is None:
                file_name = self.file_name
            # self.save_tokens(ques_ids, file_name + "_ques_processed")
            # self.save_tokens(ans_ids, file_name + "_ans_processed")
            self.save_tokens(batch_tokens, file_name + "_processed")
        if is_mask:
            batch_tokens, mask_label, mask_pos = self.mask(batch_tokens, self.max_seq_length, total_token_num)

        out = self.pad_batch_data(batch_tokens, self.max_seq_length,
                                  return_pos=True, return_sent=True, return_input_mask=True)
        src_ids, pos_ids, sent_ids, input_masks = out[0], out[1], out[2], out[3]
        qas_ids = []
        labels = []
        temp = {"Yes": 0, "No": 1, "Depends": 2}
        for example in self.examples:
            qas_ids.append(example.qas_id)
            labels.append(temp[example.yes_or_no])

        self.features = []
        for i in range(len(self.examples)):
            self.features.append(Feature(
                qas_ids[i], src_ids[i], pos_ids[i], sent_ids[i], input_masks[i], labels[i]
            ))

    def data_generator(self, args):

        def feature_generator():
            if self.args["shuffle"]:
                if self.args["shuffle_seed"] is not None:
                    np.random.seed(self.args["shuffle_seed"])
                np.random.shuffle(self.features)
            for feature in self.features:
                yield feature

        def generate_batch_data(batch_data):
            qas_ids = [inst[0] for inst in batch_data]
            src_ids = [inst[1] for inst in batch_data]
            pos_ids = [inst[2] for inst in batch_data]
            sent_ids = [inst[3] for inst in batch_data]
            input_masks = [inst[4] for inst in batch_data]
            labels = [inst[5] for inst in batch_data]
            qas_ids = np.array(qas_ids).reshape([-1, 1])
            labels = np.array(labels).reshape([-1, 1])
            return [qas_ids, src_ids, pos_ids, sent_ids, input_masks, labels]

        def batch_generator():
            batch_data = []
            for feature in feature_generator():
                batch_data.append([
                    feature.qas_id, feature.src_id, feature.pos_id, feature.sent_id, feature.input_mask, feature.label
                ])
                if len(batch_data) == args["batch_size"]:
                    batch_data = generate_batch_data(batch_data)
                    yield batch_data
                    batch_data = []

        return batch_generator
