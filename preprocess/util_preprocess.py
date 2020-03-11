import re
# ernie_chars = "。，、！？：；“”《》（）.,!?:;["


def half_width(str_):
    """
    替换所有符号中的全角字符为半角字符
    """

    rstring = ""
    for c in str_:
        inside_code = ord(c)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    str_ = rstring

    return str_


def lower(str_):

    return str_.lower()


def punctuation_replace_for_ernie(str_):
    """
    替换字符串中的部分标点符号，转化为ernie可用的
    """

    # 引号转化为中文标点
    rstring = ""
    flag = 0
    quotation = ["“", "”"]
    for c in re.sub(r"[〝〞“”‘’『』「」\'″′]", "\"", str_):
        if c == "\"":
            rstring += quotation[flag]
            flag = 1 - flag
        else:
            rstring += c
    str_ = rstring

    p = [r"[\(﹙〔【{]", r"[\)﹚〕】}]", r"[〈]", r"[〉]", r"[﹑]", r"[∶︰]", r"[﹐]"]
    new = ["（", "）", "《", "》", "、", "：", "，"]
    for i in range(len(p)):
        str_ = re.sub(p[i], new[i], str_)

    return str_


def translate_for_ernie(str_):
    """
    将部分含义确定的符号转化为中文表达
    """

    pattern = re.compile(r"(-?\d+)(\.\d+)?%")
    for s in pattern.findall(str_):
        s = "".join(s)
        str_ = re.sub(s + "%", "百分之" + s, str_)
    pattern = re.compile(r"(-?\d+)(\.\d+)?‰")
    for s in pattern.findall(str_):
        s = "".join(s)
        str_ = re.sub(s + "‰", "千分之" + s, str_)

    p = [r"℃"]
    new = ["摄氏度"]
    for i in range(len(p)):
        # pattern = re.compile(p[i])
        # print(pattern.findall(str_))
        str_ = re.sub(p[i], new[i], str_)

    return str_


def split_unk(str_, vocab):
    """
    将所有不在词表中的符号切分开，尽量保留在词表中的符号
    """

    rstring = ""
    for c in str_:
        if c in vocab.keys():
            rstring += c
        else:
            rstring += " " + c + " "
    str_ = rstring

    return str_
