from data.Example import Example
import json as js
import os
import pickle


class Dataset:
    examples = []
    train_examples = []
    dev_examples = []
    test_examples = []

    # 列表按比例分割成两份
    def __split(self, full_list, ratio):
        n_total = len(full_list)
        offset = int(n_total * ratio)
        if n_total == 0 or offset < 1:
            return [], full_list
        sublist_1 = full_list[:offset]
        sublist_2 = full_list[offset:]
        return sublist_1, sublist_2

    def read_dataset(self, dataset_path, div_str):
        files = []
        file1 = open(os.getcwd() + "/data/zhidao.train.json", "r", encoding='utf-8')
        file2 = open(os.getcwd() + "/data/search.train.json", "r", encoding='utf-8')
        files.append(file1)
        files.append(file2)

        count = 0
        for file in files:
            for line in file:
                example = js.loads(line)
                # 作筛选，只要是非观点型的问题
                if example["question_type"] == "YES_NO":
                    # 去除缺失答案，缺失问题，缺失文章，yesno答案是opinion，答案有矛盾的，索引超出范围的错误数据（个例）
                    if len(example["yesno_answers"]) == 0 or example["question"] == "" or \
                            len(example["answers"]) == 0 or len(example["documents"]) == 0 or \
                            len(example["answer_docs"]) == 0 or example["yesno_answers"][0] == "No_Opinion" or \
                            not all(x == example["yesno_answers"][0] for x in example["yesno_answers"]):
                        continue
                    yesno_example = example
                    raw_docs = yesno_example['documents']
                    docs = []
                    docs_selected = []
                    for raw_doc in raw_docs:
                        doc = ""
                        for paragraph in raw_doc['paragraphs']:
                            doc += paragraph.replace("\t", "")\
                                .replace(" ", "").replace("\n", "").replace("\r", "")
                            doc += "XXX"
                        docs.append(doc)
                        if raw_doc['is_selected']:
                            docs_selected.append(doc)
                    yesno_answer = yesno_example["yesno_answers"][0]
                    question = yesno_example["question"]
                    answer = yesno_example["answers"][0].replace("\t", "")\
                        .replace(" ", "").replace("\n", "").replace("\r", "")
                    qas_id = yesno_example['question_id']
                    count += 1
                    print("第{}个是否类例子".format(count))
                    print("文本长度{}".format(len(doc)))
                    print(yesno_example["yesno_answers"])

                    one_example = Example(
                        qas_id=qas_id,
                        question=question,
                        answer=answer,
                        yes_or_no=yesno_answer,
                        docs=docs,
                        docs_selected=docs_selected)
                    self.examples.append(one_example)

        nums = div_str.split(':')
        nums = [int(x) for x in nums]
        print(nums)

        ration1 = float(nums[0])/(nums[0] + nums[1] + nums[2])
        self.train_examples, dev_test = self.__split(self.examples, ration1)
        ration2 = float(nums[1])/(nums[1] + nums[2])
        self.dev_examples, self.test_examples = self.__split(dev_test, ration2)
        print("{len1}条训练example，{len2}条验证example，{len3}条测试example"
              .format(len1=len(self.train_examples), len2=len(self.dev_examples),
                      len3=len(self.test_examples)))

    def get_split(self):
        return self.train_examples, self.dev_examples, self.test_examples

    def save_example(self):
        with open(os.getcwd() + '/data/train_examples', 'wb') as f_train:
            pickle.dump(self.train_examples, f_train)
        with open(os.getcwd() + '/data/dev_examples', 'wb') as f_dev:
            pickle.dump(self.dev_examples, f_dev)
        with open(os.getcwd() + '/data/test_examples', 'wb') as f_test:
            pickle.dump(self.test_examples, f_test)
        return

    def load_examples(self):
        with open(os.getcwd() + '/data/train_examples', 'rb') as f_train:
            self.train_examples = pickle.load(f_train)
        with open(os.getcwd() + '/data/dev_examples', 'rb') as f_dev:
            self.dev_examples = pickle.load(f_dev)
        with open(os.getcwd() + '/data/test_examples', 'rb') as f_test:
            self.test_examples = pickle.load(f_test)
        return
