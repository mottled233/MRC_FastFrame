import os,sys
import datetime
import json
import csv
import pickle
import logging


'''
--------------------util_filepath--------------------
'''

# root: 根目录
# folder: 文件类型对应保存文件名
root = os.path.abspath('..')
folder = {"data":"dataset", "example":"examples", "datap":"dataset_processed",
          "model":"models", "log":"logging", "config":"config"}

def get_fullurl(file_type, file_name, file_format="json"):
    # file_type文件类型，file_name文件名，file_format文件格式（默认json）
    # 生成完整文件路径

    url = root + "/File_Directory"
    if file_type not in folder:
        return ""
        # 返回错误信息
    else:
        url += '/' + folder[file_type]
    url += '/' + file_name
    if file_format != "pickle":
        url += '.' + file_format
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
                    for i in range(1,len(line)):
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

# print(read_file("config","config_build","csv"))
# print(read_file("config","config_train","csv"))
# print(read_file("data", "zhidao.dev"))

# json1 = read_file("data", "zhidao.dev")
# save_file(json1, "data", "zhidao")
# json2 = read_file("data", "zhidao")
# print(json1)
# print(json2)


'''
--------------------util_parameter--------------------
'''

# config_menu: 变量目录
# config: 变量值
# part: 模块划分
config_menu = {}
config = {}
part = ["global", "build", "train", "predict"]
for k in part:
    config[k] = {}

def read_config_default(file_name="config_default", file_format="json"):
    # 读入变量定义文件，建立变量目录，并填充默认值
    file_type = "config"
    global config_menu, config

    config_menu = read_file(file_type, file_name, file_format)[0]
    for k in config_menu.keys():
        if k not in part:
            continue
        for kk in config_menu[k].keys():
            config[k][kk] = config_menu[k][kk]["default"]

read_config_default()
# print(config)

def read_config_file(file_name, file_format="json"):
    # 读入变量设置文件，管理新定义的变量，并读入其他设置值
    file_type = "config"

    config_new = read_file(file_type, file_name, file_format)[0]
    for k in config_new.keys():
        if k not in part:
            continue
        for kk in config_new[k].keys():
            if kk in config_menu[k].keys():
                config[k][kk] = config_new[k][kk]
            else:
                config_menu[k][kk] = config_new[k][kk]
                config[k][kk] = config_menu[k][kk]["default"]

# read_config_file("config_new")
# print(config)

def get_value(value_type, value):
    if value_type == "string":
        return value
    elif value_type == "bool":
        value = value.lower()
        if value == "true":
            return True
        elif value == "false":
            return False
        else:
            return ""
            # 返回错误信息
    elif value_type == "int":
        return int(value)
    elif value_type == "float":
        return float(value)
    else:
        return ""
        # 返回错误信息

def set_config():
    # 键盘录入需要的函数值
    # 以[part_name.]key=value的格式输入，输入"exit"结束
    s = input()
    while s != 'exit':

        s = s.replace(' ', '')
        temp, value = s.split('=')[0], s.split('=')[1]
        temp = temp.split('.')
        if len(temp) == 1:
            part_name = ""
            key = temp[0]
        else:
            part_name = temp[0]
            key = temp[1]

        if part_name != "":
            if part_name not in part:
                print("No such part!!")
            elif key not in config_menu[part_name].keys():
                print("No such parameter!!")
            else:
                config[part_name][key] = get_value(config_menu[part_name][key]["type"], value)
        else:
            num = 0
            for part_name in ["build", "train"]:
                if key in config_menu[part_name].keys():
                    num += 1
                    p = part_name
            if num == 0:
                print("No such parameter!!")
            elif num > 1:
                print("More than 1 parameter!!")
            else:
                config[p][key] = get_value(config_menu[p][key]["type"], value)

        # print(config)
        s = input()

# set_config()
# print(config)

def get_config(part_name):
    # 获取全局变量与对应部分的函数值

    if part_name not in part:
        return ""
        # 返回错误信息
    return config["global"], config[part_name]


'''
--------------------util_logging--------------------
'''

present_time = datetime.datetime.now()
log_name = datetime.datetime.strftime(present_time, '%Y-%m-%d %H-%M-%S')

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(get_fullurl("log", log_name, "txt"))
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')
# %(name)s：Logger的名字
# %(levelno)s：打印日志级别的数值
# %(levelname)s：打印日志级别的名称
# %(pathname)s：打印当前执行程序的路径，其实就是sys.argv[0]
# %(filename)s：打印当前执行程序名
# %(funcName)s：打印日志的当前函数
# %(lineno)d：打印日志的当前行号
# %(asctime)s：打印日志的时间
# %(thread)d：打印线程ID
# %(threadName)s：打印线程名称
# %(process)d：打印进程ID
# %(message)s：打印日志信息

def log_init():
    content = []
    for part_name in part:
        content.append(part_name)
        for k in config[part_name]:
            content.append("|---" + k + ':')
            content.append("    |---type: " + str(config_menu[part_name][k]["type"]))
            content.append("    |---description: " + str(config_menu[part_name][k]["description"]))
            content.append("    |---value: " + str(config[part_name][k]))
        content.append('')
    content.append('\n\n')
    save_file(content, "log", log_name, 'txt')

    global logger, file_handler, stream_handler
    logger.setLevel(level = logging.INFO)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

def log_input(level, message):
    if level == 'debug':
        logger.debug(message)
    elif level == 'info':
        logger.info(message)
    elif level == 'warning':
        logger.warning(message)
    elif level == 'error':
        logger.error(message)
    elif level == 'critical':
        logger.critical(message)

log_init()
