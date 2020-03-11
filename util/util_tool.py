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
