import collections

from engine.predict_engine import PredictEngine
from util.util_filepath import *
from postprocess.postprocess_for_mrc_ch import PostprocessForMRCChineseEval as EvalProcessor
from postprocess.postprocess_for_mrc_ch import PostprocessForMRCChineseWrite as WriteProcessor


class MRCPredictEngine(PredictEngine):

    def __init__(self, args, network):
        super().__init__(args, network)
        self.post_processor = WriteProcessor()

    def _predict(self, exe, **kwargs):
        """
            对验证过程的一个epoch的执行
        """
        predict_data = kwargs['predict_data']
        RawResult = collections.namedtuple("RawResult",
                                           ["unique_id", "start_logits", "end_logits"])
        output_file_name = kwargs['output_file']
        output_prediction_file = get_fullurl(file_type='result_predict', file_name=output_file_name, file_format='json')
        output_nbest_file = output_prediction_file.replace('predictions', 'nbest')
        output_full_info = output_prediction_file.replace('predictions', 'full_info')
        PRINT_PER_STEP = self.args["print_per_step"]
        USE_PARALLEL = self.args["use_parallel"]
        NUM_OF_DEVICE = self.args["num_of_device"]

        total_data = 0
        all_results = []
        for step, data in enumerate(self.predict_data_loader()):
            data_key = list(data[0].keys())[0]
            batch_size = data[0][data_key].shape()[0]
            if USE_PARALLEL and batch_size < NUM_OF_DEVICE:
                self.logger.warning("Batch size less than the number of devices. Batch aborted.")
                continue
            fetch_value = exe.run(program=self.predict_main_prog, feed=data,
                                  fetch_list=[self.predict_fetch_data['unique_ids'],
                                              self.predict_fetch_data['start_logits'],
                                              self.predict_fetch_data['end_logits']])
            unique_ids = [int(x[0]) for x in fetch_value[0]]
            start_logits = [x.tolist() for x in fetch_value[1]]
            end_logits = [x.tolist() for x in fetch_value[2]]
            for i in range(len(unique_ids)):
                all_results.append(RawResult(unique_id=unique_ids[i],
                                             start_logits=start_logits[i],
                                             end_logits=end_logits[i]))
            total_data += 1
            if PRINT_PER_STEP > 0:
                if step % PRINT_PER_STEP == 0:
                    self.logger.info("Predict batch {step} in epoch"
                                     .format(step=step))

        prediction_data = {'examples': predict_data['examples'], 'features': predict_data['features'],
                           'results': all_results, 'original_file': predict_data['test_file_path'],
                           'prediction_file': output_prediction_file}
        prediction_param = {'n_best_size': self.args['n_best_size'],
                            'max_answer_length': self.args['max_answer_length'],
                            'do_lower_case': True,
                            'output_prediction_file': output_prediction_file,
                            'output_nbest_file': output_nbest_file}

        results = self.post_processor.run(prediction_param, prediction_data)

