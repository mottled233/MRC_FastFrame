import paddle
import paddle.fluid as fluid
import numpy as np
import time
import sys
import json
import pandas as pd
import re

from engine.train import TrainEngine as TrainEngine
from engine.predict import PredictEngine as PredictEngine
from data.Dataset import Dataset
from data.Example import Example
from preprocess.preprocess import PreProcess

import util.util_filepath as file_utils
from util.util_parameter import UtilParameter as UParam
from util.util_logging import UtilLogging as ULog
import util.util_tool as util_tool


def split_sent(paragraph):
    sents = re.split('(。|，|,|！|\!|\.|？|\?)', paragraph)
    res = []
    for sent in sents:
        if len(sent) != 0 and not re.match('(。|，|,|！|\!|\.|？|\?)', sent):
            res.append(sent)
    return res


if __name__ == "__main__":

    # 设置参数
    param = UParam()
    param.read_config_file("config_roberta_large")
    args = param.get_config(param.GLOBAL)
    # 初始化日志
    logger = ULog(param)

    app_name = args["app_name"]

    '''
    常数定义
    '''
    file_name = "File_Directory/results/{}.json".format(app_name)
    new_data_name = "{}_re_predict_data".format(app_name)
    new_result_name = "{}_re_predict_out".format(app_name)
    threshold = 0.8
    mix_rate = 0.6
    decay_rate = 15
    select_threshold = 0.4

    '''
    预测过程
    '''
    datasets = Dataset(logger=logger, args=param.get_config(param.DATASET))
    datasets.load_examples()
    trainset, validset, testset = datasets.get_split()

    predict_preprocess = PreProcess(logger=logger, args=param.get_config(param.DATASET), examples=validset,
                                    for_prediction=True)
    predict_preprocess.prepare_batch_data(cache_filename="")
    predict_vocab_size = predict_preprocess.get_vocab_size()
    predict_batch_reader = predict_preprocess.batch_generator()

    predict_engine = PredictEngine(param=param, logger=logger, vocab_size=predict_vocab_size)
    # predict_engine.init_model(vocab_size=predict_vocab_size)

    predict_engine.predict(predict_batch_reader)
    example_info = util_tool.trans_exam_list_to_colum(validset)
    predict_engine.write_full_info(attach_data=example_info)

    df_ori = pd.read_json(file_name, "records")
    logger.info("first stage acc:{}".format(len(df_ori[df_ori.yesno_answer == df_ori.yes_or_no]) / len(df_ori)))

    '''
    新数据集创建
    '''
    # 读取预测文件
    df = pd.read_json(file_name, "records")
    df_raw = df[(df.Yes <= threshold) & (df.No <= threshold) & (df.Depends <= threshold)]

    post_dataset = []
    for idx, row in df_raw.iterrows():
        for i, sent in enumerate(split_sent(row['answer'])):
            example = dict()
            example["qas_id"] = row['id']  # "{}_{}".format(row['id'], i),
            example["question"] = row["question"],
            example["question"] = example["question"][0]
            example["answer"] = sent
            example["yes_or_no"] = row["yes_or_no"]
            post_dataset.append(example)

    # 生成新文件
    file_utils.save_file(content=post_dataset, file_name=new_data_name, file_type="result", file_format="json")

    # 读取新数据
    data_json = post_dataset
    dataset = [
        Example(
            qas_id=data['qas_id'],
            question=data["question"],
            answer=data["answer"],
            yes_or_no=data["yes_or_no"],
            docs="",
            docs_selected=""
        ) for data in data_json

    ]

    '''
    重新预测
    '''
    predict_preprocess = PreProcess(logger=logger, args=param.get_config(param.DATASET), examples=dataset,
                                    for_prediction=True)
    predict_preprocess.prepare_batch_data(cache_filename="")
    predict_vocab_size = predict_preprocess.get_vocab_size()
    predict_batch_reader = predict_preprocess.batch_generator()

    predict_engine.predict(predict_batch_reader)
    example_info = util_tool.trans_exam_list_to_colum(dataset)
    predict_engine.write_full_info(name=new_result_name, attach_data=example_info)

    '''
    合并重预测结果
    '''
    def fuse(mix_rate=0.6, decay_rate=15, threshold=0.5, df_raw=None):
        with open("File_Directory/results/{}.json".format(new_result_name), "r") as f:
            re_predict = json.load(f)
        df_raw = df_raw.reset_index(drop=True)
        reduce_list = []
        max_length = 100
        count = 1
        for i in range(len(re_predict)):
            item = re_predict[i]
            if i < len(re_predict) - 1:
                next_item = re_predict[i + 1]
            else:
                next_item = None

            if next_item is not None and item['id'] == next_item['id']:
                max_prob = max([next_item["Yes"], next_item["No"], next_item["Depends"]])
                if count < max_length and max_prob >= threshold:
                    next_item["Yes"] = item["Yes"] + next_item["Yes"] / (count * decay_rate)
                    next_item["No"] = item["No"] + next_item["No"] / (count * decay_rate)
                    next_item["Depends"] = item["Depends"] + next_item["Depends"] / (count * decay_rate)
                    # next_item["Yes"] = item["Yes"] + next_item["Yes"] * decay_rate
                    # next_item["No"] = item["No"] + next_item["No"] * decay_rate
                    # next_item["Depends"] = item["Depends"] + next_item["Depends"] * decay_rate
                    count += 1
                else:
                    next_item["Yes"] = item["Yes"]
                    next_item["No"] = item["No"]
                    next_item["Depends"] = item["Depends"]
            else:
                count += 1
                item["Yes"] /= count
                item["No"] /= count
                item["Depends"] /= count
                item["Yes"] = (1 - mix_rate) * df_raw.at[len(reduce_list), "Yes"] + (mix_rate * item["Yes"])
                item["No"] = (1 - mix_rate) * df_raw.at[len(reduce_list), "No"] + (mix_rate * item["No"])
                item["Depends"] = (1 - mix_rate) * df_raw.at[len(reduce_list), "Depends"] + (mix_rate * item["Depends"])
                # item["Yes"] += df_raw.at[len(reduce_list), "Yes"]
                # item["No"] += df_raw.at[len(reduce_list), "No"]
                # item["Depends"] += df_raw.at[len(reduce_list), "Depends"]

                probs = [item["Yes"], item["No"], item["Depends"]]
                result = ["Yes", "No", "Depends"]
                item["yesno_answer"] = result[probs.index(max(probs))]
                reduce_list.append(item)
                count = 1

        df_mid = pd.read_json(json.dumps(reduce_list), 'records')
        print(len(df_mid[df_mid.yesno_answer == df_mid.yes_or_no]) / len(df_mid))
        print(len(df_raw[df_raw.yesno_answer == df_raw.yes_or_no]) / len(df_raw))
        pos = 0
        new_result = []
        for idx, row in df.iterrows():
            example = dict()
            example["id"] = row['id']  # "{}_{}".format(row['id'], i),
            if pos < len(reduce_list) and example["id"] == reduce_list[pos]["id"]:
                example["Yes"] = reduce_list[pos]["Yes"]
                example["No"] = reduce_list[pos]["No"]
                example["Depends"] = reduce_list[pos]["Depends"]
                example["yesno_answer"] = reduce_list[pos]["yesno_answer"]
                pos += 1
            else:
                example["Yes"] = row["Yes"]
                example["No"] = row["No"]
                example["Depends"] = row["Depends"]
                example["yesno_answer"] = row["yesno_answer"]
            if row["yesno_answer"] == "Depends":
                example["yesno_answer"] = "Depends"
            example["question"] = row["question"],
            example["question"] = example["question"][0]
            example["answer"] = row["answer"],
            example["yes_or_no"] = row["yes_or_no"]
            new_result.append(example)
        return new_result

    final_output = fuse(mix_rate, decay_rate, select_threshold, df_raw=df_raw)
    final_df = pd.read_json(json.dumps(final_output), 'records')
    f_c_df = final_df[final_df.yes_or_no == final_df.yesno_answer]
    logger.info("second stage acc: {}".format(len(f_c_df)/len(final_df)))

    file_utils.save_file(content=final_output, file_name=new_result_name, file_type="result", file_format="json")

