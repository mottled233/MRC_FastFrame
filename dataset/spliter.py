from dataset.dataset import Dataset
import math


class Spliter(object):
    def __split(self, full_list, ratio):
        """
        私有方法，功能是将一个列表按ration切分成两个子列表
        :param full_list:
        :param ratio:
        :return:
        """
        n_total = len(full_list)
        offset = int(n_total * ratio)
        if n_total == 0 or offset < 1:
            return [], full_list
        sublist_1 = full_list[:offset]
        sublist_2 = full_list[offset:]
        return sublist_1, sublist_2

    def split_dataset(self, src_dataset, div_nums=None):
        """
        将一个dataset对象，按分割比例，拆分为3个dataset对象
        :param src_dataset: 待拆分数据集对象
        :param div_nums: 分割比例数组，数组中有3个数字代表比例
        :return: 拆分后的3个dataset对象
        """
        # 分割比例判断
        if div_nums == [] or div_nums is None:
            div_nums = [6, 2, 2]

        assert isinstance(src_dataset,Dataset), "Spliter only split Dataset object"
        assert len(div_nums) == 3, "div_ration need 3 int or float input"
        assert math.isclose(div_nums[0] + div_nums[1] + div_nums[2], 1) or \
               math.isclose(div_nums[0] + div_nums[1] + div_nums[2], 10) or \
               math.isclose(div_nums[0] + div_nums[1] + div_nums[2], 100), \
            "sum(div_ration) shoule close to 1 or 10 or 100"

        nums = div_nums
        dataset1 = Dataset(src_dataset.args)
        dataset2 = Dataset(src_dataset.args)
        dataset3 = Dataset(src_dataset.args)
        src_examples = src_dataset.get_examples()

        ration1 = float(nums[0]) / (nums[0] + nums[1] + nums[2])
        examples1, examples2 = self.__split(src_examples, ration1)
        dataset1.read_from_elist(examples1)

        ration2 = float(nums[1]) / (nums[1] + nums[2])
        examples2, examples3 = self.__split(examples2, ration2)
        dataset2.read_from_elist(examples2)
        dataset3.read_from_elist(examples3)

        return dataset1, dataset2, dataset3
