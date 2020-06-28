import collections

from postprocess.postprocess_for_mrc_ch import PostprocessForMRCChineseEval as EvalProcessor
from postprocess.postprocess_for_mrc_ch import PostprocessForMRCChineseWrite as WriteProcessor
from util.util_filepath import *
from engine.train_engine import TrainEngine


class MRCTrainEngine(TrainEngine):
    def __init__(self, args, network, train_data_generator, valid_data_generator=None, valid_data=None):
        """
        MRC任务训练引擎
        :param valid_data: MRC任务输出比较复杂，需要验证集原始数据信息。包含验证集信息的字典，应该含有features，examples，和dev_file_path
        """
        super().__init__(args, network, train_data_generator, valid_data_generator)
        self.valid_data = valid_data
        self.post_processor = WriteProcessor(EvalProcessor())

    def _run_train_iterable(self, executor, **kwargs):
        """
        对训练过程的一个epoch的执行
        """
        step_in_epoch = kwargs['step_in_epoch']
        total_step = kwargs['total_step']
        epoch =kwargs['epoch_id']

        PRINT_PER_STEP = self.args["print_per_step"]
        USE_PARALLEL = self.args["use_parallel"]
        NUM_OF_DEVICE = self.args["num_of_device"]
        APP_NAME = self.args["app_name"]
        SNAPSHOT_FREQUENCY = self.args["snapshot_frequency"]
        VALID_FREQUENCY = self.args["validate_frequency_step"]
        WHETHER_VALID = self.args["validate_in_epoch"]

        total_loss = 0
        total_data = 0
        valid_step_loss = 0
        batch_loss = 0
        for step, data in enumerate(self.train_data_loader()):
            # 为获取字段名，这里需要改
            if step_in_epoch != 0 and step <= step_in_epoch:
                continue
            data_key = list(data[0].keys())[0]
            batch_size = data[0][data_key].shape()[0]
            if USE_PARALLEL and batch_size < NUM_OF_DEVICE:
                self.logger.warning("Batch size less than the number of devices. Batch aborted.")
                continue
            fetch_value = executor.run(program=self.train_main_prog, feed=data, fetch_list=[self.train_loss, self.lr])
            total_loss += fetch_value[0]
            total_data += 1
            batch_loss += fetch_value[0]
            valid_step_loss += fetch_value[0]
            # 打印逻辑
            if PRINT_PER_STEP > 0:
                if step % PRINT_PER_STEP == 0 and step != 0:
                    self.logger.info("Step {step}: loss = {loss}, lr = {lr}".
                                     format(step=total_step + step, loss=batch_loss / PRINT_PER_STEP,
                                            lr=fetch_value[1]))
                    batch_loss = 0
            # 保存逻辑
            info = {
                "total_step": total_step,
                "step_in_epoch": step,
                "epoch": epoch
            }
            if step % SNAPSHOT_FREQUENCY == 0 and step != 0:
                file_path = self._save_model("{}_step{}".format(APP_NAME, total_step + step), info)
                self.logger.info("Snapshot of training process has been saved as folder {}".format(file_path))

            if WHETHER_VALID and step % VALID_FREQUENCY == 0 and step != 0:
                self._valid(executor, epoch_id="{}_step{}".format(epoch, total_step + step))
                self.logger.info("from {}-{} step: mean loss = {}"
                                 .format(total_step + step - VALID_FREQUENCY,
                                         total_step + step, valid_step_loss / VALID_FREQUENCY))
                valid_step_loss = 0
                if os.path.isfile("stop_signal"):
                    return False, False

        mean_loss = total_loss / total_data
        return total_step + step, mean_loss

    def _valid(self, exe, **kwargs):
        """
            对验证过程的一个epoch的执行
        """
        output_file_name = "{}_valid_out_epoch{}".format(self.app_name, kwargs['epoch_id'])
        RawResult = collections.namedtuple("RawResult",
                                           ["unique_id", "start_logits", "end_logits"])
        output_prediction_file = get_fullurl(file_type='result_valid', file_name=output_file_name, file_format='json')
        output_nbest_file = output_prediction_file.replace('predictions', 'nbest')
        PRINT_PER_STEP = self.args["print_per_step"]
        USE_PARALLEL = self.args["use_parallel"]
        NUM_OF_DEVICE = self.args["num_of_device"]

        total_loss = 0
        total_data = 0
        batch_loss = 0
        all_results = []
        for step, data in enumerate(self.valid_data_loader()):
            data_key = list(data[0].keys())[0]
            batch_size = data[0][data_key].shape()[0]
            if USE_PARALLEL and batch_size < NUM_OF_DEVICE:
                self.logger.warning("Batch size less than the number of devices. Batch aborted.")
                continue
            fetch_value = exe.run(program=self.valid_main_prog, feed=data,
                                  fetch_list=[self.valid_fetch_data['loss'],
                                              self.valid_fetch_data['unique_ids'],
                                              self.valid_fetch_data['start_logits'],
                                              self.valid_fetch_data['end_logits']])
            total_loss += fetch_value[0]
            batch_loss += fetch_value[0]
            unique_ids = [int(x[0]) for x in fetch_value[1]]
            start_logits = [x.tolist() for x in fetch_value[2]]
            end_logits = [x.tolist() for x in fetch_value[3]]
            for i in range(len(unique_ids)):
                all_results.append(RawResult(unique_id=unique_ids[i],
                                             start_logits=start_logits[i],
                                             end_logits=end_logits[i]))
            total_data += 1
            if PRINT_PER_STEP > 0:
                if step % PRINT_PER_STEP == 0:
                    self.logger.info("Valid batch {step} in epoch: loss = {loss}"
                                     .format(step=step, loss=batch_loss / PRINT_PER_STEP))
                    batch_loss = 0

        prediction_data = {'examples': self.valid_data['examples'], 'features': self.valid_data['features'],
                           'results': all_results, 'original_file': self.valid_data['dev_file_path'],
                           'prediction_file': output_prediction_file}
        prediction_param = {'n_best_size': self.args['n_best_size'],
                            'max_answer_length': self.args['max_answer_length'],
                            'do_lower_case': True,
                            'output_prediction_file': output_prediction_file,
                            'output_nbest_file': output_nbest_file}

        results = self.post_processor.run(prediction_param, prediction_data)

        mean_loss = total_loss / total_data

        self.logger.info('valid mean loss is {loss}, f1={f1}, em={em}'.format(loss=mean_loss, f1=results['F1'],
                                                                              em=results['EM']))
        return mean_loss
