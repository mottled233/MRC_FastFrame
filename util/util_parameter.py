import getopt
from util.util_filepath import *


class UtilParameter:

    GLOBAL = "global"
    DATASET = "dataset"
    MODEL_BUILD = "build"
    TRAIN = "train"
    PREDICT = "predict"
    part = [GLOBAL, DATASET, MODEL_BUILD, TRAIN, PREDICT]
    # part: 模块划分

    def __init__(self, file_name="config_default", file_format="json"):

        self.config_menu = {}
        self.config = {}
        for part_name in UtilParameter.part:
            self.config_menu[part_name] = {}
            self.config[part_name] = {}
        # config_menu: 变量目录
        # config: 变量值

        self.read_config_default(file_name, file_format)

    def read_config_default(self, file_name="config_default", file_format="json", file_type="config"):
        """
        读入变量定义文件，建立变量目录，并填充默认值
        """

        self.config_menu = read_file(file_type, file_name, file_format)
        for part_name in UtilParameter.part:
            self.config[part_name] = {}
        for part_name in self.config_menu.keys():
            try:
                for k in self.config_menu[part_name].keys():
                    self.config[part_name][k] = self.config_menu[part_name][k]["default"]
            except Exception:
                raise KeyError("未知模块名") from Exception
                # 出现未知模块名，返回错误信息

    def read_config_file(self, file_name, file_format="json", file_type="config"):
        """
        读入变量设置文件，管理新定义的变量，并读入其他设置值
        """

        config_new = read_file(file_type, file_name, file_format)
        for part_name in config_new.keys():
            try:
                for k in config_new[part_name].keys():
                    if k not in self.config_menu[part_name].keys():
                        self.config_menu[part_name][k] = config_new[part_name][k]
                        self.config[part_name][k] = self.config_menu[part_name][k]["default"]
                    else:
                        self.config[part_name][k] = config_new[part_name][k]
            except Exception:
                raise KeyError("未知模块名") from Exception
                # 出现未知模块名，返回错误信息

    def set_config(self, argv):
        """
        录入从命令行获取的参数值，只处理长格式
        :param argv: run as set_config(sys.argv[1:])
        """

        config_list = []
        for part_name in UtilParameter.part:
            if self.config_menu[part_name] is False:
                continue
            for k in self.config_menu[part_name].keys():
                config_list.append(k + '=')
                config_list.append(part_name + '.' + k + '=')
        try:
            options, args = getopt.getopt(argv, "", config_list)
        except Exception:
            raise getopt.GetoptError("argv格式错误") from Exception
            # sys.exit()
            # argv格式错误，返回错误信息

        for option, value in options:
            option = option.replace('-', '')
            opt = option.split('.')
            if len(opt) == 1:
                num = 0
                for part_name in UtilParameter.part:
                    if option in self.config_menu[part_name].keys():
                        num += 1
                        self.config[part_name][option] = value
                if num == 0:
                    raise Exception("不存在该参数")
                elif num > 1:
                    raise Exception("存在同名变量")
            else:
                part_name = opt[0]
                option = opt[1]
                try:
                    self.config[part_name][option] = value
                except Exception:
                    raise KeyError("不存在该参数") from Exception

    def get_config(self, part_name):
        """
        获取全局变量与对应部分的函数值，局部变量拥有更高的优先级
        """

        try:
            conf = self.config["global"]
            if part_name != "global":
                conf.update(self.config[part_name])
            return conf
        except Exception:
            raise KeyError("未知模块名") from Exception
            # 出现未知模块名，返回错误信息
