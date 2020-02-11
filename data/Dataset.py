from data.Example import Example
import json as js
from util.util_filepath import get_fullurl, save_file, read_file
import math


class Dataset:
    def __init__(self):
        self.examples = []
        self.train_examples = []
        self.dev_examples = []
        self.test_examples = []

    def __split(self, full_list, ratio):
        """
        私有方法，功能是将一个列表按ration切分成两个子列表
        :param full_list:
        :param ratio:
        :return:
        """
        n_total = len(full_list)
        offset = int(n_total * ratio)
        if n_total == 0 or offset < 1:
            return [], full_list
        sublist_1 = full_list[:offset]
        sublist_2 = full_list[offset:]
        return sublist_1, sublist_2

    def read_dataset(self, div_nums=None):
        """
        读取数据集和分割比例列表，默认分割比例[6，2，2], 要求分割比例为3个数且和为整数
        :param div_nums:
        :return:
        """
        # 合并所有的训练集
        files = []
        file1 = open(get_fullurl("data", "zhidao.train", "json"), "r", encoding='utf-8')
        file2 = open(get_fullurl("data", "search.train", "json"), "r", encoding='utf-8')
        files.append(file1)
        files.append(file2)

        # 分割比例判断
        if div_nums == [] or div_nums is None:
            div_nums = [6, 2, 2]
        assert len(div_nums) == 3, "div_ration need 3 int or float input"
        assert math.isclose(div_nums[0] + div_nums[1] + div_nums[2], 1) or \
               math.isclose(div_nums[0] + div_nums[1] + div_nums[2], 10) or \
               math.isclose(div_nums[0] + div_nums[1] + div_nums[2], 100), \
            "sum(div_ration) shoule close to 1 or 10 or 100"

        count = 0
        # 遍历所有文件的每一行，每一行是一个example
        for file in files:
            for line in file:
                example = js.loads(line)
                # 作筛选，只要是非观点型的问题
                if example["question_type"] == "YES_NO":
                    # 去除缺失答案，缺失问题，缺失文章，yesno答案是opinion，答案有矛盾的
                    if len(example["yesno_answers"]) == 0 or example["question"] == "" or \
                            len(example["answers"]) == 0 or len(example["documents"]) == 0 or \
                            len(example["answer_docs"]) == 0 or example["yesno_answers"][0] == "No_Opinion" or \
                            not all(x == example["yesno_answers"][0] for x in example["yesno_answers"]):
                        continue
                    yesno_example = example
                    raw_docs = yesno_example['documents']
                    # 两个文章列表，所有文章和所有选中文章
                    docs = []
                    docs_selected = []
                    for raw_doc in raw_docs:
                        doc = ""
                        # 用XXX作为段落分隔符
                        for paragraph in raw_doc['paragraphs']:
                            doc += paragraph.replace("\t", "") \
                                .replace(" ", "").replace("\n", "").replace("\r", "")
                            doc += "XXX"
                        docs.append(doc)
                        if raw_doc['is_selected']:
                            docs_selected.append(doc)
                    # 因为已经判断列表中所有答案一致，所以取第一个即可
                    yesno_answer = yesno_example["yesno_answers"][0]
                    question = yesno_example["question"]
                    answer = yesno_example["answers"][0].replace("\t", "") \
                        .replace(" ", "").replace("\n", "").replace("\r", "")
                    qas_id = yesno_example['question_id']
                    count += 1
                    if count % 1000 == 0:
                        print("has read {} examples".format(count))
                    # 获取好字段信息后生成example实例写入全局列表中
                    one_example = Example(
                        qas_id=qas_id,
                        question=question,
                        answer=answer,
                        yes_or_no=yesno_answer,
                        docs=docs,
                        docs_selected=docs_selected)
                    self.examples.append(one_example)

        # 根据输入的比例来分割examples为train，dev，test
        nums = div_nums
        ration1 = float(nums[0]) / (nums[0] + nums[1] + nums[2])
        self.train_examples, dev_test = self.__split(self.examples, ration1)
        ration2 = float(nums[1]) / (nums[1] + nums[2])
        self.dev_examples, self.test_examples = self.__split(dev_test, ration2)
        print("{len1}条训练example，{len2}条验证example，{len3}条测试example"
              .format(len1=len(self.train_examples), len2=len(self.dev_examples),
                      len3=len(self.test_examples)))

    def get_split(self):
        """
        获取example_list的接口
        :return:
        """
        return self.train_examples, self.dev_examples, self.test_examples

    def save_example(self):
        """
        保存example_list的缓存
        :return:
        """
        save_file(self.train_examples, "example", "train", "pickle")
        save_file(self.dev_examples, "example", "train", "pickle")
        save_file(self.test_examples, "example", "train", "pickle")
        return

    def load_examples(self):
        """
        读取example_list的缓存
        :return:
        """
        self.train_examples = read_file("example", "train", "pickle")
        self.dev_examples = read_file("example", "dev", "pickle")
        self.test_examples = read_file("example", "test", "pickle")
        return


if __name__ == "__main__":
    dataset = Dataset()
    dataset.read_dataset()
    train, dev, test = dataset.get_split()
    print(train)
