# 抽象的后处理类，其他后处理过程均继承自此
class Postprocess(object):
    def __init__(self, next_process=None):
        # 后处理工序为链表式
        self.next_process = next_process

    def run(self, param, data):
        """
        :param param: 字典，包含了后处理过程所需的控制信息
        :param data: 字典，包含了后处理过程所需的数据信息
        :return: 字典，包含了后处理的结果
        """
        result = self.do_postprocess(param, data)
        # 若存在下一个后处理过程，则调用其run方法，并合并得到的结果字典
        if self.get_next_process():
            result_next = self.get_next_process().run(param, data)
            if result_next is not None and result is not None:
                return dict(result, **result_next)
            elif result_next is not None:
                return result_next
        return result

    def get_next_process(self):
        """
        :return: 返回下一个后处理过程
        """
        return self.next_process

    def set_next_process(self, next):
        """
        :param next: 设置下一个后处理过程
        :return: 无
        """
        self.next_process = next

    def do_postprocess(self, param, data):
        """
        完成该后处理过程的工作，用户可通过重写这一部分实现多种后处理功能
        :param param: 字典，包含了后处理过程所需的控制信息
        :param data: 字典，包含了后处理过程所需的数据信息
        :return:
        """
        return {}
