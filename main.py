import os, sys
import paddle
import paddle.fluid as fluid
import numpy as np
import time
from engine.train import TrainEngine as TrainEngine
from data.Dataset import Dataset

from util.util_filepath import *
from util.util_parameter import UtilParameter as UParam
from util.util_logging import UtilLogging as ULog

ground_truth = np.random.random(size=(1, 4)).astype('int64')

def fake_sample_generator():
    for _ in range(1000):
        sample_x = np.random.random(size=(4,)).astype('float32')
        sample_y = np.dot(ground_truth, sample_x)
        yield sample_x, sample_y


if __name__ == "__main__":
    args = {
        "max_epoch": 100,
        "early_stopping": True,
        "warm_up": False,
        "continue_train": False,
        "model_path": "",
        "use_parallel": True,
        "use_gpu": False,
        "num_of_device": 2,
        "batch_size": 32,
        "base_learning_rate": 0.001,
        "learning_rate_strategy": "linear_warm_up_and_decay",
        "start_learning_rate": 1e-04,
        "warm_up_step": 200,
        "end_learning_rate": 1e-04,
        "decay_step": 500,
        "optimizer": "adagrad",
        "adagrad_epsilon": 1e-06,
        "adagrad_accumulator_value": 0,
        "early_stopping_threshold": 0.03,
        "early_stopping_times": 5
    }
    reader = fluid.io.batch(fake_sample_generator, batch_size=args["batch_size"])
    train_engine = TrainEngine(reader, reader, args)
    t1 = time.time()
    train_engine.train()
    t2 = time.time()
    print(t2-t1)

