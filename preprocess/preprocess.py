import os
import paddle.fluid as fluid
from util.util_filepath import get_fullurl, read_file, save_file


class Preprocess(object):
    """
    抽象的预处理类,完成将examples转化为feature的工作，并返回data_generator和相关数据结果
    """
    def __init__(self, args, examples, is_prediction=False, cache=""):
        self.args = args
        self.examples = examples
        self.features = []
        self.is_prediction = is_prediction
        self.cache = cache
        self.logger = None

    def do_preprocess(self, **kwargs):
        """
        用户通过重写这部分代码完成数据的预处理工作，并返回data_generator和相关数据结果
        :return: data_generator
        """
        self.features = self.load_features()
        if self.features is None:
            self.features = self._process_feature(**kwargs)

            if self.cache != "":
                self.save_features(self.features)
        return self.get_generator()

    def _process_feature(self, **kwargs):
        """
        从examples转换为feature list的过程
        :param kwargs: 参数
        :return: feature list
        """
        return []

    def load_features(self):
        if self.cache != "":
            if os.path.exists(get_fullurl(file_type='datap', file_name=self.cache, file_format='json')):
                self.logger.info('load features from file')
                features = read_file('datap', self.cache, 'json')
                self.logger.info('{} features loaded'.format(len(features)))
                return features
        return None

    def save_features(self, features):
        save_file(features, 'datap', self.cache, 'json')

    def sample_generator(self):
        """
        返回一个单个样本，格式为tuple(data1, data2, ...)
        每个data可以是一个numpy的ndarray
        """
        pass

    def batch_generator(self):
        """
        成batch返回的data_generator，内部调用sample_generator，必须首先实现
        """
        reader = fluid.io.batch(self.sample_generator, batch_size=self.args["batch_size"])
        return reader

    def get_generator(self, **kwargs):
        return self.batch_generator()

    def get_features(self):
        return self.features
