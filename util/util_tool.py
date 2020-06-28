from data.Example import Example
import numpy as np


def trans_exam_list_to_colum(example_list, headers=None):
    """
    将example列表转换成以列表示的形式，用于适配输出附加信息
    :param example_list: example 列表
    :param headers: 需要的属性，默认为("question", "answer", "yes_or_no")
    :return: {header1:[...]，header2:[...],...}
    """
    if headers is None:
        headers = ("question", "answer", "yes_or_no")
    result = {}
    for header in headers:
        result[header] = []

    for example in example_list:
        for header in headers:
            result[header].append(getattr(example, header, ""))

    return result


def _is_chinese_char(cp):
    if ((0x4E00 <= cp <= 0x9FFF) or  #
            (0x3400 <= cp <= 0x4DBF) or  #
            (0x20000 <= cp <= 0x2A6DF) or  #
            (0x2A700 <= cp <= 0x2B73F) or  #
            (0x2B740 <= cp <= 0x2B81F) or  #
            (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or  #
            (0x2F800 <= cp <= 0x2FA1F)):  #
        return True

    return False


SPIECE_UNDERLINE = ''


def is_fuhao(c):
    if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
            or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
            or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
            or c == '‘' or c == '’':
        return True
    return False


def _tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or is_fuhao(char):
            if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                output.append(SPIECE_UNDERLINE)
            output.append(char)
            output.append(SPIECE_UNDERLINE)
        else:
            output.append(char)
    return "".join(output)


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
        return True
    return False
