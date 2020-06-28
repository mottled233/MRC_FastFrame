from dataset.example import Example
import json as js
import pickle
import os
import warnings
from util.util_logging import UtilLogging as ULog
from util.util_filepath import read_file, save_file, get_fullurl


class Dataset(object):
    def __init__(self, args):
        self.examples = []
        self.args = args
        self.logger = ULog(args, __name__)

    def read_from_srcfile(self, srcfile_path, cache="", **kwargs):
        """
        从数据集文件中提取examples列表，具体逻辑需在子类中重写_from_srcfile方法
        :param srcfile_path:  数据集源文件
        :param cache: 缓存文件，若为空串则不缓存
        :return: 无
        """
        if cache != "":
            if os.path.exists(get_fullurl(file_type='example', file_name=cache, file_format='json')):
                self.logger.info('load features from file')
                self.read_from_pickle(cache)
                self.logger.info('{} examples loaded'.format(len(self.examples)))
                return
        self.examples = self._from_srcfile(srcfile_path, **kwargs)
        self.save_to_pickle(cache)

    def _from_srcfile(self, srcfile_path, **kwargs):
        """
        从源文件读取数据集的集体逻辑实现
        :param srcfile_path: 源文件名/路径
        :param kwargs: 参数
        :return: example list
        """
        return []

    def read_from_elist(self, examples):
        """
        从传入的examplses列表中对对象的examples进行初始化
        :param examples:
        :return:
        """
        self.examples = examples

    def read_from_pickle(self, cachename):
        """
        读取examples_list的缓存
        :return:
        """
        name = cachename
        assert name != "", "if use read_from_pickle, must pass in example_file_name when initialize dataset!"
        self.examples = read_file(file_type='example', file_name=name, file_format='pickle')

    def save_to_pickle(self, cachename):
        """
        保存examples_list的缓存
        :return:
        """
        name = cachename
        # assert name != "", "if use save_to_pickle, cachename"
        if name != "":
            save_file(content=self.examples, file_type='example', file_name=name, file_format='pickle')

    def get_examples(self):
        return self.examples
