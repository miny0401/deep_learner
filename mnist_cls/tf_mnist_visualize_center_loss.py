"""使用lenet训练mnist."""
import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layer
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from functools import partial
from abc import ABC

from tf_learner import Learner, set_seed, Metrics, LrScheduler, Callback


class CenterLoss(object):
    """使用parameter封装训练center.
    center_loss = ||feature_embedding - center_embedding||**2
    """
    def __init__(self, cls_num, feature_dim):
        super().__init__()
        self.cls_num = cls_num
        self.featur_num = feature_dim
        self.center = tf.Variable(tf.random.normal((cls_num, feature_dim), 0.0, 1.0))
        self.is_first = True

    def __call__(self, ys, xs):
        """
        if self.is_first:
            self.center = self.center.to(xs.device)
            self.is_first = False
        """
        center_exp = tf.gather(self.center, tf.cast(ys, tf.int32))
        count = tf.histogram_fixed_width(tf.cast(ys, tf.float32), value_range=(0, self.cls_num-1), nbins=self.cls_num)
        count_dis = tf.gather(count, tf.cast(ys, tf.int32)) + 1
        loss = tf.reduce_sum(tf.reduce_sum((xs - center_exp) ** 2, axis=1) / 2.0 / tf.cast(count_dis, tf.float32))
        return loss


class MyLoss(object):
    def __init__(self, cls_num: int = 10, feature_dim: int = 2):
        super(MyLoss, self).__init__()
        self.loss = tf.losses.SparseCategoricalCrossentropy()
        self.center_loss = CenterLoss(cls_num, feature_dim)

    def __call__(self, targets, outputs):
        return self.loss(targets, outputs[1]) + self.center_loss(targets, outputs[0])


class TrainCallback(Callback):
    def __init__(self):
        self.embeddings = []    # 2维的embedding，用于可视化
        self.labels = []

    def on_epoch_begin(self, info: dict):
        # 每次epoch开始时，清空之前的数据
        self.embeddings.clear()
        self.labels.clear()

    def on_loss_begin(self, info: dict):
        # 获取每个batch得到的向量
        self.embeddings.append(info["outputs"][0])
        self.labels.append(tf.cast(info["y"][0], tf.int32))

    def on_metric_begin(self, info: dict):
        info["orig_outputs"] = info["outputs"]
        info["outputs"] = info["outputs"][1]

    def on_epoch_end(self, info: dict):
        # 每个epoch结束时，对向量进行可视化
        embeddings = tf.concat(self.embeddings, axis=0)
        labels = tf.concat(self.labels, axis=0)
        self.visualize(embeddings.numpy(), labels.numpy(), info["epoch"])

    @staticmethod
    def visualize(feat, labels, epoch):
        plt.ion()
        c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        plt.clf()
        for i in range(10):
            plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
        plt.xlim(xmin=-2.5, xmax=2.5)
        plt.ylim(ymin=-2.5, ymax=2.5)
        plt.text(-7.8, 7.3, "epoch=%d" % epoch)
        plt.savefig('/tmp/epoch=%d.jpg' % epoch)
        plt.draw()
        plt.pause(0.001)


class ValidCallback(Callback):
    def on_metric_begin(self, info: dict):
        # 由于输出了两个数据，但用于计算metric的，只是第2个元素
        info["orig_outputs"] = info["outputs"]
        info["outputs"] = info["outputs"][1]


class MyModel(Model, ABC):
    """LeNet."""
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = layer.Conv2D(filters=6, kernel_size=5)
        self.relu1 = layer.PReLU(alpha_initializer=tf.constant_initializer(0.25))
        self.pool1 = layer.MaxPool2D(2)
        self.conv2 = layer.Conv2D(filters=16, kernel_size=5)
        self.relu2 = layer.PReLU(alpha_initializer=tf.constant_initializer(0.25))
        self.pool2 = layer.MaxPool2D(2)
        self.fc1 = layer.Dense(120)
        self.relu3 = layer.PReLU(alpha_initializer=tf.constant_initializer(0.25))
        self.fc2 = layer.Dense(84)
        self.relu4 = layer.PReLU(alpha_initializer=tf.constant_initializer(0.25))
        self.fc3 = layer.Dense(2)
        self.relu5 = layer.PReLU(alpha_initializer=tf.constant_initializer(0.25))
        self.fc = layer.Dense(10)

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
        y = self.fc(logit)
        return logit, tf.nn.softmax(y, axis=1)


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
                      valid_callbacks=[ValidCallback()])
    learner.train(20)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s')
    main()
