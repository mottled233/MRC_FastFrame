import datetime
import logging
# from colorama import Fore, Style
from util.util_parameter import *


class UtilLogging:

    lev = {1: logging.DEBUG, 2: logging.INFO, 3: logging.WARNING, 4: logging.ERROR, 5: logging.CRITICAL}

    def __init__(self, params, is_file=True, is_stream=True, output_config=True):
        """
        获取参数并记录到文件中，并选择是否写入文件与输出到控制台
        :param params: 参数数据
        :param is_file: 是否输出到文件
        :param is_stream: 是否输出到控制台
        :param output_config: 是否保存参数配置文件
        """

        present_time = datetime.datetime.now()
        self.log_name = datetime.datetime.strftime(present_time, '%Y-%m-%d %H-%M-%S')
        # 用当前时间为本次日志命名

        self.logger = logging.getLogger(self.log_name)
        self.logger.propagate = False  # 不向root传播，防止重复输出
        self.logger.setLevel(level=logging.DEBUG)  # 设置整体最低层级为debug

        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # formatter: 统一的日志输出格式
        # file_handler: 文件输出
        # stream_handler: 控制台输出

        if is_file:
            self.file_handler = logging.FileHandler(get_fullurl("log", self.log_name, "txt"))
            self.set_file_level(2)
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)
        if is_stream:
            self.stream_handler = logging.StreamHandler()
            self.set_stream_level(2)
            self.stream_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.stream_handler)

        if output_config:
            self.log_config(params)

    def log_config(self, params):
        """
        初始化日志文件，并写入参数表
        """

        content = []
        for part_name in params.part:
            content.append(part_name)
            for k in params.config[part_name]:
                content.append("|---" + k + ':')
                content.append("    |---type: " + str(params.config_menu[part_name][k]["type"]))
                content.append("    |---description: " + str(params.config_menu[part_name][k]["description"]))
                content.append("    |---value: " + str(params.config[part_name][k]))
            content.append('')
        # 输出全部参数信息
        jsonstr = json.dumps(params.config)
        content.append(jsonstr)
        # 输出可进行使用的json格式参数配置
        content.append('\n\n')
        save_file(content, "log", self.log_name + " config", 'txt')

    def log_input(self, level, message, pos=None):
        """
        记录日志信息，level为'debug', 'info', 'warning', 'error', 'critical'，分别用数值1-5表示
        # 颜色分别为白、绿、黄、红、红
        :param level: 信息等级
        :param message: 信息内容
        :param pos: 文件及函数所在位置，run as u_log.log_input(level, message, sys._getframe().f_code)
        """

        message = str(message)
        try:
            message = pos.co_filename + '/' + pos.co_name + ' - ' + message
        except AttributeError:
            pass
            # 未获取日志信息生成所在位置，不记录该信息
        if level == 1:
            self.logger.debug(message)
            # Fore.WHITE + message +Style.RESET_ALL
        elif level == 2:
            self.logger.info(message)
            # Fore.GREEN + message +Style.RESET_ALL
        elif level == 3:
            self.logger.warning(message)
            # Fore.YELLOW + message +Style.RESET_ALL
        elif level == 4:
            self.logger.error(message)
            # Fore.RED + message +Style.RESET_ALL
        elif level == 5:
            self.logger.critical(message)
            # Fore.RED + message +Style.RESET_ALL

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def set_file_level(self, flevel):
        """
        设置文件中日志的筛选等级
        """
        try:
            self.file_handler.setLevel(UtilLogging.lev[flevel])
        except Exception:
            raise AttributeError("Output is not set to file") from Exception

    def set_stream_level(self, slevel):
        """
        设置控制台中日志的筛选等级
        """

        try:
            self.stream_handler.setLevel(UtilLogging.lev[slevel])
        except Exception:
            raise AttributeError("Output is not set to console") from Exception
