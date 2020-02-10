from util.util_filepath import *
from preprocess.tokenization import FullTokenizer as FToken
from preprocess.batching import *


class PreProcess():

    def __init__(self, max_seq_length=512, vocab_name="vocab", vocab_format="txt", file_type="vocab", do_lowercase=True):

        self.f_token = FToken(vocab_name, vocab_format, file_type, do_lowercase)
        # 使用指定的字典，构建tokenizer
        self.max_seq_length = max_seq_length

    def get_examples(self, file_name, file_format="pickle", file_type="example"):
        # 读取指定文件，返回list of Example结果

        exams = read_file(file_type, file_name, file_format)
        return exams

    def exams_tokenize(self, exams, token_id=1):
        # 完成对list of Example中question和answer的tokenize，并返回结果
        # token_id=1时返回tokens，token_id=2时返回ids

        ques_tokens = []
        ans_tokens = []
        ques_ids = []
        ans_ids = []
        for exam in exams:
            q_tokens = self.f_token.tokenize(exam.question)
            ques_tokens.append(q_tokens)
            q_ids = self.f_token.convert_tokens_to_ids(q_tokens)
            ques_ids.append(q_ids)
            a_tokens = self.f_token.tokenize(exam.answer)
            ans_tokens.append(a_tokens)
            a_ids = self.f_token.convert_tokens_to_ids(a_tokens)
            ans_ids.append(a_ids)
        if token_id == 1:
            return ques_tokens, ans_tokens
        elif token_id == 2:
            return ques_ids, ans_ids

    def save_tokens(self, tokens, file_name, file_format="pickle", file_type="datap"):
        # 将tokens储存在指定位置

        save_file(tokens, file_type,file_name, file_format)

    def splice_ques_ans(self, ques_ids, ans_ids, CLS=1, SEP=2):
        # 对问题张量和答案张量进行拼接，并返回句子最大长度与单个token总数的信息

        l1 = len(ques_ids)
        l2 = len(ans_ids)
        if l1 != l2:
            raise Exception("问题答案数量不匹配")
        batch_tokens = []
        max_len = 0
        total_token_num = 0
        for i in range(l1):
            sent = [CLS] + ques_ids[i] + [SEP] + ans_ids[i] + [SEP]
            if len(sent) > self.max_seq_length:
                sent = sent[:self.max_seq_length - 1] + [SEP]
            batch_tokens.append(sent)
            max_len = max(max_len, len(sent))
            total_token_num += len(sent)

        return batch_tokens, max_len, total_token_num

    def mask(self, batch_tokens, max_len, total_token_num, CLS=1, SEP=2, MASK=3):
        # 进行mask覆盖，返回覆盖后的结果和覆盖信息

        vocab_size = len(self.f_token.vocab)
        return mask(batch_tokens, max_len, total_token_num, vocab_size, CLS, SEP, MASK)

    def pad_batch_data(self,
                       insts,
                       pad_idx=0,
                       return_pos=False,
                       return_input_mask=False,
                       return_max_len=False,
                       return_num_token=False):
        # 将句子统一填充到最大句子长度，并生成相应的位置数据和输入覆盖

        return pad_batch_data(insts, pad_idx, return_pos, return_input_mask, return_max_len, return_num_token)

    def prepare_batch_data(self,
                           insts,
                           max_len,
                           total_token_num,
                           voc_size=0,
                           pad_id=None,
                           cls_id=None,
                           sep_id=None,
                           mask_id=None,
                           return_input_mask=True,
                           return_max_len=True,
                           return_num_token=False):
        # 创建数据张量、位置张量、自注意力覆盖（shape: batch_size*max_len*max_len）

        return prepare_batch_data(insts, max_len, total_token_num, voc_size, pad_id, cls_id, sep_id, mask_id,
                                  return_input_mask, return_max_len, return_num_token)

    def batch(self, file_name, file_format="pickle", file_type="example"):

        exams = self.get_examples(file_name, file_format, file_type)
        ques_ids, ans_ids = self.exams_tokenize(exams, token_id=2)
        batch_tokens, max_len, total_token_num = self.splice_ques_ans(ques_ids, ans_ids)
        batch_tokens, mask_label, mask_pos = self.mask(batch_tokens, self.max_seq_length, total_token_num)
        insts = self.pad_batch_data(batch_tokens, self.max_seq_length)
        return insts