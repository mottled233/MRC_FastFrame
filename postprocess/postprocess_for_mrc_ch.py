from postprocess.evaluate import get_eval
from postprocess.output import write_predictions
from postprocess.postprocess import Postprocess


class PostprocessForMRCChineseWrite(Postprocess):
    """
    完成中文阅读理解任务中写预测结果工作的后处理类
    """
    def __init__(self, next_process=None):
        super().__init__(next_process)

    def do_postprocess(self, param, data):
        return write_predictions(data['examples'], data['features'], data['results'],
                                 n_best_size=param['n_best_size'], max_answer_length=param['max_answer_length'],
                                 do_lower_case=param['do_lower_case'],
                                 output_prediction_file=param['output_prediction_file'],
                                 output_nbest_file=param['output_nbest_file'])


class PostprocessForMRCChineseEval(Postprocess):
    """
    完成中文阅读理解任务中计算F1和EM的后处理类
    """
    def __init__(self, next_process=None):
        super().__init__(next_process)

    def do_postprocess(self, param, data):
        return get_eval(data['original_file'], data['prediction_file'])

