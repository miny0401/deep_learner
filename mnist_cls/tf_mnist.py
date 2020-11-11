"""使用lenet训练mnist."""
import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layer
from tensorflow.keras import Model
from functools import partial
from abc import ABC

from tf_learner import Learner, set_seed, Metrics, LrScheduler, Callback


class MyLoss(object):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.loss = tf.losses.SparseCategoricalCrossentropy()

    def __call__(self, targets, outputs):
        return self.loss(targets, outputs[1])


class TrainCallback(Callback):
    def on_metric_begin(self, info: dict):
        # 由于输出了两个数据，但用于计算metric的，只是第2个元素
        info["orig_outputs"] = info["outputs"]
        info["outputs"] = info["outputs"][1]


class MyModel(Model, ABC):
    """LeNet."""
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = layer.Conv2D(filters=6, kernel_size=5)
        self.relu1 = layer.ReLU()
        self.pool1 = layer.MaxPool2D(2)
        self.conv2 = layer.Conv2D(filters=16, kernel_size=5)
        self.relu2 = layer.ReLU()
        self.pool2 = layer.MaxPool2D(2)
        self.fc1 = layer.Dense(120)
        self.relu3 = layer.ReLU()
        self.fc2 = layer.Dense(84)
        self.relu4 = layer.ReLU()
        self.fc3 = layer.Dense(10)
        self.relu5 = layer.ReLU()

    @tf.function
    def call(self, x, training=None, mask=None):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = tf.reshape(y, (y.shape[0], -1))
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        logit = self.relu5(y)
        return logit, tf.nn.softmax(logit, axis=1)


def main():
    """main."""
    set_seed(1)
    # 1. 数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype(np.float32)
    x_test = x_test[..., tf.newaxis].astype(np.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # 2. 训练模型
    logging.info(f"训练模型.")
    model = MyModel()
    learner = Learner(model=model,
                      train_ds=train_dataset,
                      valid_ds=valid_dataset,
                      metrics={'accuracy': partial(Metrics.accuracy_score, just_score=False),
                               'precision_macro': partial(Metrics.precision_score, average='macro', just_score=False),
                               'recall_macro': partial(Metrics.recall_score, average='macro', just_score=False),
                               'f1_macro': partial(Metrics.f1_score, average='macro', just_score=False)},
                      loss_func=MyLoss(),
                      optim_func=tf.optimizers.Adam,
                      batch_size=128,
                      lr=0.01,
                      lr_scheduler=LrScheduler.get_step_lr(step_size=5, gamma=0.2),
                      valid_batch=-1,
                      device=tf.device("GPU"),
                      train_callbacks=[TrainCallback()],
                      valid_callbacks=[TrainCallback()])
    learner.train(20)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s')
    main()
