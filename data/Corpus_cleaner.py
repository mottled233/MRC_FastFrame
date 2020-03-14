# -*- coding: utf-8 -*-

import json as js
import re


def clean_strip(text):
    """
    去除多符号结尾，改为以句号结尾,去掉连续的句号
    """
    text = text.strip(',..，,。？')
    text += '。'
    text = text.replace("。。", "。")
    return text


def clean_char(text):
    """
    去掉异常字符
    :type text: string
    """
    reg = re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.，,。？“”]+', re.UNICODE)
    text = reg.sub('', text)
    return text


def clean_pre(text):
    """
    1. 去除空格换行
    2. 去除html标签
    """
    text = text.replace("\t", "").replace(" ", "").replace("\n", "").replace("\r", "")
    text = re.sub('<.*?>', "", text)
    return text


def clean(text):
    text = clean_pre(text)
    text = clean_char(text)
    text = clean_strip(text)
    return text


class Corpus_cleaner:
    def __init__(self):
        self.docs = []

    def read_from_txt(self, txt_path):
        file = open(txt_path, 'r', encoding='utf-8')
        for line in file:
            self.docs.append(line[:-1])

    def read_from_src(self):
        with open("pretrain_corpus.txt", "w", encoding='utf-8') as f:
            files = []
            file1 = open("/home/aistudio/data/data23310/train.json", "r", encoding="utf-8")
            file2 = open("/home/aistudio/data/data23310/dev.json", "r", encoding="utf-8")
            file3 = open("/home/aistudio/data/data23310/test1.json", "r", encoding="utf-8")
            files.append(file1)
            files.append(file2)
            files.append(file3)
            self.docs = []
            count = 0
            for file in files:
                for line in file:
                    example = js.loads(line)
                    raw_docs = example['documents']
                    for raw_doc in raw_docs:
                        doc = ""
                        for paragraph in raw_doc['paragraphs']:
                            paragraph = clean(paragraph)
                            if len(paragraph) < 6:
                                continue
                            doc += paragraph
                        # 舍弃没有句号的文档
                        if "。" in doc[:-1]:
                            self.docs.append(doc)
                            f.write(doc + '\n')
                            count += 1
                            if count % 10000 == 0:
                                print("has read {} docs".format(count))
            print("docs_num == {}".format(len(self.docs)))
            f.close()

    def get_docs(self):
        return self.docs
