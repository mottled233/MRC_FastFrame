import os,sys
import datetime
import json
import logging

from util_filepath import *
from util_parameter import *


class UtilLogging(UtilParameter):

    def __init__(self):

        super().__init__()

        present_time = datetime.datetime.now()
        self.log_name = datetime.datetime.strftime(present_time, '%Y-%m-%d %H-%M-%S')
        # 用当前时间为本次日志命名

        self.logger = logging.getLogger(__name__)
        self.file_handler = logging.FileHandler(get_fullurl("log", self.log_name, "txt"))
        self.stream_handler = logging.StreamHandler()
        self.formatter = logging.Formatter('%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')
        # file_handler: 文件输出
        # stream_handler: 数据流输出
        # formatter: 统一的日志输出格式

    def log_init(self):
        # 初始化日志文件，并写入参数表

        content = []
        for part_name in self.part:
            content.append(part_name)
            for k in self.config[part_name]:
                content.append("|---" + k + ':')
                content.append("    |---type: " + str(self.config_menu[part_name][k]["type"]))
                content.append("    |---description: " + str(self.config_menu[part_name][k]["description"]))
                content.append("    |---value: " + str(self.config[part_name][k]))
            content.append('')
        # 输出全部参数信息
        jsonstr = json.dumps(self.config)
        content.append(jsonstr)
        # 输出可进行使用的json格式参数配置
        content.append('\n\n')
        save_file(content, "log", self.log_name, 'txt')

        self.logger.setLevel(level=logging.INFO)
        self.file_handler.setLevel(logging.INFO)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        self.stream_handler.setLevel(logging.INFO)
        self.stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)

    def log_input(self, level, message):
        # 记录日志信息，会同步输出到命令行和日志文件中
        # level为'debug', 'info', 'warning', 'error', 'critical'，分别用数值1-5表示

        if level == 1:
            self.logger.debug(message)
        elif level == 2:
            self.logger.info(message)
        elif level == 3:
            self.logger.warning(message)
        elif level == 4:
            self.logger.error(message)
        elif level == 5:
            self.logger.critical(message)



util = UtilLogging()
util.read_config_default()
util.log_init()
util.log_input(5, "success!")
