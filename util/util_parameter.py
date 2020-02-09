import os,sys
import getopt

from util_filepath import *


class UtilParameter():

    def __init__(self):

        self.config_menu = {}
        self.config = {}
        self.part = ["global", "build", "train", "predict"]
        for part_name in self.part:
            self.config[part_name] = {}
        # config_menu: 变量目录
        # config: 变量值
        # part: 模块划分

    def read_config_default(self, file_name="config_default", file_format="json"):
        # 读入变量定义文件，建立变量目录，并填充默认值
        file_type = "config"

        self.config_menu = read_file(file_type, file_name, file_format)[0]
        for part_name in self.part:
            self.config[part_name] = {}
        for part_name in self.config_menu.keys():
            try:
                for k in self.config_menu[part_name].keys():
                    self.config[part_name][k] = self.config_menu[part_name][k]["default"]
            except KeyError:
                return
                # 出现未知模块名，返回错误信息

    def read_config_file(self, file_name, file_format="json"):
        # 读入变量设置文件，管理新定义的变量，并读入其他设置值
        file_type = "config"

        config_new = read_file(file_type, file_name, file_format)[0]
        for part_name in config_new.keys():
            try:
                for k in config_new[part_name].keys():
                    if k not in self.config_menu[part_name].keys():
                        self.config_menu[part_name][k] = config_new[part_name][k]
                        self.config[part_name][k] = self.config_menu[part_name][k]["default"]
                    else:
                        self.config[part_name][k] = config_new[part_name][k]
            except KeyError:
                return
                # 出现未知模块名，返回错误信息

    def set_config(self, argv):
        # 录入从命令行获取的参数值，只处理长格式
        # set_config(sys.argv[1:])

        config_list = []
        for part_name in self.part:
            for k in self.config_menu[part_name].keys():
                config_list.append(k + '=')
        try:
            options, args = getopt.getopt(argv, "", config_list)
        except getopt.GetoptError:
            sys.exit()
            # argv格式错误，返回错误信息

        for option, value in options:
            for part_name in self.part:
                for k in self.config_menu[part_name].keys():
                    if option == '--' + k:
                        self.config[part_name][k] = value

    def get_config(self, part_name):
        # 获取全局变量与对应部分的函数值

        try:
            return self.config["global"], self.config[part_name]
        except KeyError:
            return {}, {}
            # 出现未知模块名，返回错误信息