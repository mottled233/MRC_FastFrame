from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import six

from util.util_filepath import *


def convert_to_unicode(text):
    """
    将文本转化为utf-8的统一编码，可输出到控制台或日志中的形式
    判断python版本以及文本格式并进行处理
    :param text: str
    :return: str
    """

    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    """
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")
    """


def load_vocab(vocab_name="vocab", vocab_format="txt", file_type="vocab"):
    """
    从文件中读入字典，并以有序字典的形式输出，格式为vocab[token]=num
    :param vocab_name: str
    :param vocab_format: str
    :param file_type: str
    :return: collections.OrderedDict
    """

    vocab = collections.OrderedDict()
    with open(get_fullurl(file_type, vocab_name, vocab_format), encoding="utf-8") as fin:
        for num, line in enumerate(fin):
            items = convert_to_unicode(line.strip()).split("\t")
            if len(items) > 2:
                break
            token = items[0].strip()
            index = items[1] if len(items) == 2 else num
            vocab[token] = int(index)
    return vocab


def convert_tokens_to_ids(vocab, tokens):
    """
    使用字典vocab将tokens转化为ids
    :param vocab: dict
    :param tokens: list of str
    :return: list of int
    """

    ids = []
    for token in tokens:
        ids.append(vocab[token])
    return ids


def convert_ids_to_tokens(inv_vocab, ids):
    """
    使用字典inv_vocab将ids转化为tokens
    :param inv_vocab: dict
    :param ids: list of int
    :return: list of str
    """

    tokens = []
    for ID in ids:
        tokens.append(inv_vocab[ID])
    return tokens


def whitespace_tokenize(text):
    """
    对一段字符串文本进行空白符的清理，并以此拆分子段
    :param text: str
    :return: list of str
    """

    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_whitespace(char):
    """
    判断字符是否为空白符，包括\t\n\r, 空格, 各式制表
    :param char: char
    :return: bool
    """

    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cate = unicodedata.category(char)  # Zs
    if cate == "Zs":
        return True
    return False


def _is_control(char):
    """
    判断字符是否为控制符，包括除\t\n\r以外的全部控制符
    :param char: char
    :return: bool
    """

    if char == "\t" or char == "\n" or char == "\r":  # Cc
        return False
    cate = unicodedata.category(char)  # Cf, Cn, Co, Cs without Cc
    if cate.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """
    判断字符是否为标点符号，这里包括其他除字母和数字以外的符号
    :param char: char
    :return: bool
    """

    cp = ord(char)
    if ((33 <= cp <= 47) or (58 <= cp <= 64) or
            (91 <= cp <= 96) or (123 <= cp <= 126)):
        return True
    cate = unicodedata.category(char)  # Pc, Pd, Pe, Pf, Pi, Po, Ps
    if cate.startswith("P"):
        return True
    return False


def _is_chinese_char(char):
    """
    判断字符是否为中日韩文字
    :param char: char
    :return: bool
    """

    cp = ord(char)
    if ((0x4E00 <= cp <= 0x9FFF) or
            (0x3400 <= cp <= 0x4DBF) or
            (0x20000 <= cp <= 0x2A6DF) or
            (0x2A700 <= cp <= 0x2B73F) or
            (0x2B740 <= cp <= 0x2B81F) or
            (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or
            (0x2F800 <= cp <= 0x2FA1F)):
        return True
    return False


def clean_text(text):
    """
    清除文本中的无效字符、控制符与空白符
    :param text: str
    :return: str
    """

    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def tokenize_chinese_chars(text):
    """
    在中日韩文字前后各加一个空格，完成对它们的拆分
    :param text: str
    :return: str
    """

    output = []
    for char in text:
        if _is_chinese_char(char):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def run_strip_accents(text):
    """
    统一字符串中的字母表示方法，并去除无法处理的特殊字符
    :param text: str
    :return: str
    """

    text = unicodedata.normalize("NFD", text)  # 文本标准化，统一相同字符的不同表示
    output = []
    for char in text:
        cate = unicodedata.category(char)  # Mark, Nonspacing
        if cate == "Mn":
            continue
        output.append(char)
    return "".join(output)


def run_split_on_punc(text):
    """
    将标点符号（定义如上）单独拆分出来
    :param text: str
    :return: list of str
    """

    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1
    return ["".join(x) for x in output]


class WordpieceTokenizer(object):
    """
    用以完成对若干字符组成的连续短语的tokenize
    """

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):

        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        # unk_token: 若短语过长或存在不在词典中的字符，返回未知值
        # max_input_chars_per_word: 允许的短语最大长度

    def tokenize(self, text):
        """
        使用该tokenizer的词典，完成对短语的处理，返回处理完成的tokens列表
        切割短语时，从首字母开始每次尽量匹配长度最长的语素
        :param text: str，文段，可包含多个短语，但使用时只有一个
        :return: list of str，token列表，若存在多个短语则依次排列
        """

        output_tokens = []

        text = convert_to_unicode(text)
        for token in whitespace_tokenize(text):
            chars = list(token)  # 将短语切分成单字符列表
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr  # 若不位于单词首部，则加上##前缀表示
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                # print("".join(chars))
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens
