import os
import json
import csv
import pickle

root = os.path.abspath("")
folder = {"data": "/dataset", "example": "/examples", "datap": "/dataset_processed", "result": "/results",
          "model": "/models", "log": "/logging", "config": "/config", "vocab": "/vocab"}
suffix = {"json": ".json", "csv": ".csv", "tsv": ".tsv", "txt": ".txt", "pickle": "", "": ""}
# root: 根目录
# folder: 文件类型对应保存文件名
# suffix: 文件格式对应的后缀


def get_fullurl(file_type, file_name, file_format="json"):
    """
    生成完整文件路径
    :param file_type: 文件类型
    :param file_name: 文件名
    :param file_format: 文件格式（默认json）
    """

    url = root + "/File_Directory"
    try:
        url += folder[file_type]
    except Exception:
        raise KeyError("Unknown file-type '{}'".format(file_type)) from Exception
        # 出现未知文件种类，返回错误信息
    if not os.path.exists(url):
        os.mkdir(url)
    url += '/' + file_name
    try:
        url += suffix[file_format]
    except Exception:
        raise KeyError("Intractable file-format '{}'".format(file_format)) from Exception
        # 出现不可处理的文件格式，返回错误信息
    return url


def read_file(file_type, file_name, file_format="json"):
    """
    对指定文件进行读取操作，自动调用路径生成
    """
    url = get_fullurl(file_type, file_name, file_format)

    if file_format == "json":
        try:
            with open(url, 'r', encoding="utf-8") as f:
                content = json.load(f)
        except json.decoder.JSONDecodeError:
            with open(url, 'r', encoding="utf-8") as f:
                content = []
                for line in f.readlines():
                    content.append(json.loads(line))
    elif file_format == "csv":
        with open(url, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            content = list(reader)
    elif file_format == "tsv":
        with open(url, 'r', encoding="utf-8") as f:
            content = []
            for line in f.readlines():
                item = line.replace('\n', '').split('\t')
                content.append(item)
    elif file_format == "txt":
        with open(url, 'r', encoding="utf-8") as f:
            content = []
            for line in f.readlines():
                item = str(line)
                content.append(item)
    elif file_format == "pickle":
        with open(url, 'rb') as f:
            content = pickle.load(f)
    else:
        raise KeyError("Unknown file-type '{}'".format(file_type)) from Exception
        # 出现未知文件种类，返回错误信息

    return content


def save_file(content, file_type, file_name, file_format="json"):
    """
    将存储内容写入指定位置，自动调用路径生成
    """
    url = get_fullurl(file_type, file_name, file_format)

    if file_format == "json":
        with open(url, 'w', encoding='utf-8', newline='') as f:
            if type(content).__name__ == "list":
                for line in content:
                    f.write(json.dumps(line))
                    f.write('\n')
            else:
                json.dump(content, f)
    elif file_format == "csv":
        with open(url, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(content)
    elif file_format == "tsv":
        with open(url, 'w', encoding='utf-8', newline='') as f:
            for line in content:
                s = str(line[0])
                for i in range(1, len(line)):
                    s += '\t' + str(line[i])
                f.write(s)
                f.write('\n')
    elif file_format == "txt":
        with open(url, 'w', encoding='utf-8', newline='') as f:
            for line in content:
                f.write(str(line))
                f.write('\n')
    elif file_format == "pickle":
        with open(url, 'wb') as f:
            pickle.dump(content, f)
    else:
        raise KeyError("Unknown file-type '{}'".format(file_type)) from Exception
        # 出现未知文件种类，返回错误信息
