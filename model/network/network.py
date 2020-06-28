class Network(object):
    def __init__(self, args):
        self.is_predict = False
        self.is_validate = False
        self.is_train = True
        self.args = args

    def create_model(self):
        """
        用于构建模型
        :return: reader: 数据入口
        """
        pass

    def train(self):
        """
        将模型设置为训练模式
        训练模式下，必须返回loss
        """
        self.is_predict = False
        self.is_validate = False
        self.is_train = True

    def validate(self):
        self.is_predict = False
        self.is_validate = True
        self.is_train = False

    def predict(self):
        self.is_predict = True
        self.is_validate = False
        self.is_train = False


