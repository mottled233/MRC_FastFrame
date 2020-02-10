import os,sys
import json
import csv
import pickle


root = os.path.abspath("")
folder = {"data": "dataset", "example": "examples", "datap": "dataset_processed",
                   "model": "models", "log": "logging", "config": "config"}
suffix = {"json": ".json", "csv": ".csv", "tsv": ".tsv", "txt": ".txt", "pickle": ""}
# root: 根目录
# folder: 文件类型对应保存文件名
# suffix: 文件格式对应的后缀

def get_fullurl(file_type, file_name, file_format="json"):
    # file_type文件类型，file_name文件名，file_format文件格式（默认json）
    # 生成完整文件路径

    url = root + "/File_Directory"
    try:
        url += '/' + folder[file_type]
    except Exception:
        raise KeyError("未知文件种类") from Exception
        # 出现未知文件种类，返回错误信息
    url += '/' + file_name
    try:
        url += suffix[file_format]
    except Exception:
        raise KeyError("不可处理的文件格式") from Exception
        # 出现不可处理的文件格式，返回错误信息
    return url


def read_file(file_type, file_name, file_format="json"):
    # 对指定文件进行读取操作，自动调用路径生成
    url = get_fullurl(file_type, file_name, file_format)
    content = []

    if file_format != "pickle":
        with open(url, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                if file_format == "json":
                    item = json.loads(line)
                elif file_format == "csv":
                    item = line.replace('\n', '').split(',')
                elif file_format == "tsv":
                    item = line.replace('\n', '').split('\t')
                elif file_format == "txt":
                    item = line
                # print(item)
                content.append(item)
    else:
        with open(url, 'rb') as f:
            content = pickle.load(f)

    return content

def save_file(content, file_type, file_name, file_format="json"):
    # 将存储内容写入指定位置，自动调用路径生成
    url = get_fullurl(file_type, file_name, file_format)

    if file_format != "pickle":
        with open(url, 'w', encoding='utf-8', newline='') as f:
            if file_format == "json":
                for line in content:
                    jsonstr = json.dumps(line)
                    f.write(jsonstr)
                    f.write('\n')
            elif file_format == "csv":
                writer = csv.writer(f)
                for line in content:
                    writer.writerow(line)
            elif file_format == "tsv":
                for line in content:
                    s = str(line[0])
                    for i in range(1, len(line)):
                        s += '\t' + str(line[i])
                    f.write(s)
                    f.write('\n')
            elif file_format == "txt":
                for line in content:
                    f.write(line)
                    f.write('\n')
    else:
        with open(url, 'wb') as f:
            pickle.dump(content, f)

