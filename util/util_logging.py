import os,sys
import datetime
import json
import logging
from util.util_filepath import *
from util.util_parameter import *


class UtilLogging():

    level_num = {1: logging.DEBUG, 2: logging.INFO, 3: logging.WARNING, 4: logging.ERROR, 5: logging.CRITICAL}

    def __init__(self, u_param, if_file=True, if_stream=True):
        # 选择是否写入文件与输出到控制台

        present_time = datetime.datetime.now()
        self.log_name = datetime.datetime.strftime(present_time, '%Y-%m-%d %H-%M-%S')
        # 用当前时间为本次日志命名

        self.logger = logging.getLogger(self.log_name)
        self.file_handler = logging.FileHandler(get_fullurl("log", self.log_name, "txt"))
        self.stream_handler = logging.StreamHandler()
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # file_handler: 文件输出
        # stream_handler: 数据流输出
        # formatter: 统一的日志输出格式

        self.logger.propagate = False # 不向root传播，防止重复输出
        # self.logger.setLevel(level = logging.INFO)
        self.file_handler.setLevel(logging.INFO)
        self.file_handler.setFormatter(self.formatter)
        if if_file:
            self.logger.addHandler(self.file_handler)
        self.stream_handler.setLevel(logging.INFO)
        self.stream_handler.setFormatter(self.formatter)
        if if_stream:
            self.logger.addHandler(self.stream_handler)

        self.log_config(u_param)

    def log_config(self, u_param):
        # 初始化日志文件，并写入参数表

        content = []
        for part_name in u_param.part:
            content.append(part_name)
            for k in u_param.config[part_name]:
                content.append("|---" + k + ':')
                content.append("    |---type: " + str(u_param.config_menu[part_name][k]["type"]))
                content.append("    |---description: " + str(u_param.config_menu[part_name][k]["description"]))
                content.append("    |---value: " + str(u_param.config[part_name][k]))
            content.append('')
        # 输出全部参数信息
        jsonstr = json.dumps(u_param.config)
        content.append(jsonstr)
        # 输出可进行使用的json格式参数配置
        content.append('\n\n')
        save_file(content, "log", self.log_name, 'txt')

    def log_input(self, level, message, pos=""):
        # 记录日志信息，会同步输出到命令行和日志文件中
        # level为'debug', 'info', 'warning', 'error', 'critical'，分别用数值1-5表示
        # u_log.log_input(level, message, sys._getframe().f_code)

        try:
            message = pos.co_filename + '/' + pos.co_name + ' - ' + message
        except AttributeError:
            pass
            # 未获取日志信息生成所在位置，不记录该信息
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

    def set_file_level(self, flevel):
        # 设置文件中日志的筛选等级

        self.file_handler.setLevel(UtilLogging.level_num[flevel])

    def set_stream_level(self, slevel):
        # 设置控制台中日志的筛选等级

        self.stream_handler.setLevel(UtilLogging.level_num[slevel])
