"""pytorch trainer.

1. 测试时，model.eval()一定要调用，不然如果model中有BatchNorm及Dropout等，会按训练的情况（如Dropout会生效)
2. 确保metric计算正常
3. 注意lr的设置
4. 多观察训练集与验证集的指标，以确定是过拟合还是欠拟合

@author: huangwm
"""
import time
import random
import logging
import numpy as np
import tensorflow as tf
import sklearn.metrics as skm
from collections import defaultdict
from tensorflow.keras import Model, regularizers
from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
from tensorflow.python.keras import backend as keras_backend
from tensorflow.python.framework.ops import EagerTensor

from progress import master_bar, progress_bar


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
tf.debugging.set_log_device_placement(False)


def set_seed(seed: int = 1):
    """设置随机种子."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed=seed)


# noinspection DuplicatedCode
class Learner(object):
    """tensorflow trainer."""

    def __init__(self,
                 model: Model,
                 train_ds: Dataset,
                 valid_ds: Dataset = None,
                 valid_batch: int = -1,
                 collate_fn=None,
                 loss_func: tf.keras.losses.Loss = None,
                 optim_func: type = None,
                 device: tf.device = tf.device("/gpu:0"),
                 batch_size: int = 128,
                 wd: float = 1e-5,
                 lr: float = 0.01,
                 lr_scheduler=None,
                 metrics: dict = None,
                 train_callbacks: list = None,
                 valid_callbacks: list = None):
        self.model = model
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.valid_batch = valid_batch
        self.collate_fn = collate_fn
        self.loss_func = loss_func
        self.optim_class = optim_func
        self.batch_size = batch_size
        self.device = device
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.wd = wd
        self.metrics = metrics
        self.train_callbacks = train_callbacks if train_callbacks else []
        self.valid_callbacks = valid_callbacks if valid_callbacks else []
        self.train_metric_vals = defaultdict(float)
        self.valid_metric_vals = defaultdict(float)
        self.train_dl_len = self.train_ds_len = self.valid_dl_len = self.valid_ds_len = 0
        self._init()

    def _init(self):
        """初始化"""
        # 1. 获取可用的gpu，限制使用第一块gpu，并打开内存增长
        #   需在最开始设置
        # 2. 构建DataLoader
        self.train_dl = self.train_ds \
            .shuffle(2*self.batch_size) \
            .batch(self.batch_size) \
            .prefetch(self.batch_size)
        if hasattr(self.train_dl, '__len__'):
            self.train_dl_len = len(self.train_dl)
            self.train_ds_len = len(self.train_ds)
        if self.valid_ds:
            self.valid_dl = self.valid_ds \
                .batch(3*self.batch_size) \
                .prefetch(self.batch_size)
            if hasattr(self.valid_dl, '__len__'):
                self.valid_dl_len = len(self.valid_dl)
                self.valid_ds_len = len(self.valid_ds)
        # 3. 模型
        self.model = self.model
        # 4. 设置优化器(如果没有设置策略，则表示使用默认的AdamW优化器)
        if not self.optim_class:
            self.optim_class = tf.keras.optimizers.Adam
        self.optim_func: tf.optimizers.Optimizer = self.optim_class(learning_rate=self.lr)
        # TODO self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        if self.wd > 0:
            for layer in self.model.layers:
                layer.kernel_regularizer = regularizers.l2(self.wd)
        # 5. 设置损失函数(如果没有设置，则默认使用交叉熵损失)
        if not self.loss_func:
            self.loss_func = tf.keras.losses.BinaryCrossentropy()
        else:
            self.loss_func = self.loss_func
        # 6. 设置metrics
        if not self.metrics:
            self.metrics = dict()
        self.metric_names = []
        self.metric_keys = list(self.metrics.keys())
        for name in ["loss"] + self.metric_keys:
            self.metric_names.append(f"t_{name}")
            self.metric_names.append(f"v_{name}")
        # 7. 设置学习率策略(如果没有设置策略，则表示不改变学习率，则gamma值为1)
        if not self.lr_scheduler:
            self.lr_scheduler = LrScheduler.get_step_lr(step_size=1, gamma=1)
        else:
            self.lr_scheduler = self.lr_scheduler
        # 8. 显示信息
        logging.info("=" * 80)
        logging.info(f"learner info: ")
        logging.info(f"train ds: {self.train_ds_len} samples, "
                     f"{self.train_dl_len} batches.")
        if self.valid_ds:
            logging.info(f"valid ds: {self.valid_ds} samples, "
                         f"{self.valid_dl} batches.")
        logging.info(f"lr: {self.lr}, lr scheduler: {vars(self.lr_scheduler)}")
        logging.info(f"weight decay: {self.wd}")
        logging.info(f"loss: {self.loss_func}")
        logging.info(f"optim: {self.optim_func}")
        logging.info(f"batch size: {self.batch_size}")
        logging.info(f"metrics: {self.metrics}")
        logging.info(f"train callbacks: {self.train_callbacks}")
        logging.info(f"valid callbacks: {self.valid_callbacks}")
        logging.info(f"collate_fn: {self.collate_fn}")
        logging.info("=" * 80)

    # noinspection DuplicatedCode
    def train(self, epochs):
        """训练"""
        # 打印展示的指标名
        mb = master_bar(range(epochs))
        mb.write(["epoch"] + self.metric_names + ["lr", "time"], table=True)
        # 开始训练
        total_batch = 0
        info = dict()
        for callback in self.train_callbacks:
            callback.on_train_begin(info)
        for epoch in mb:
            info["epoch"] = epoch
            epoch_start_time = time.time()
            # 开始第epoch个训练
            for callback in self.train_callbacks:
                callback.on_epoch_begin(info)
            train_loss = 0
            valid_loss = 0
            batch_idx = 0
            self.train_metric_vals.clear()
            for (x, y) in progress_bar(self.train_dl,
                                       total=self.train_dl_len if self.train_dl_len else 0,
                                       parent=mb):
                info["x"], info["y"] = x, y
                # 开始第batch_idx批次的训练
                for callback in self.train_callbacks:
                    callback.on_batch_begin(info)
                # 数据listy
                if not isinstance(info["x"], (tuple, list)):
                    info["x"] = [info["x"]]
                if not isinstance(info["y"], (tuple, list)):
                    info["y"] = [info["y"]]
                with tf.GradientTape() as tape:
                    # 模型计算
                    info["outputs"] = self.model(*info["x"])
                    # 计算损失
                    for callback in self.train_callbacks:
                        callback.on_loss_begin(info)
                    loss = self.loss_func(*info["y"], info["outputs"])
                    train_loss += loss.numpy()
                # 记录当前的训练的损失
                mb.child.comment = f"train loss: {train_loss / (batch_idx + 1):.4f}, " \
                                   f"valid loss: {valid_loss:.4f}"
                # 梯度回传
                for callback in self.train_callbacks:
                    callback.on_backward_begin(info)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                # 梯度更新
                for callback in self.train_callbacks:
                    callback.on_step_begin(info)
                self.optim_func.apply_gradients(zip(gradients, self.model.trainable_variables))
                # metric
                for callback in self.train_callbacks:
                    callback.on_metric_begin(info)
                for metric_name, metric in self.metrics.items():
                    self.train_metric_vals[metric_name] += metric(info["outputs"], *info["y"])
                # writer valid
                total_batch += 1
                if self.valid_batch > 0 and total_batch % self.valid_batch == 0:
                    valid_loss = self._valid(mb)
                    mb.child.comment = f"train loss: {train_loss / (batch_idx + 1):.4f}, " \
                                       f"valid loss: {valid_loss:.4f}"
                for callback in self.train_callbacks:
                    callback.on_batch_end(info)
                batch_idx += 1
            for callback in self.train_callbacks:
                callback.on_epoch_end(info)
            # 更新指标
            if not hasattr(self.train_dl, '__len__') or len(self.train_dl) == 0:
                self.train_dl_len = batch_idx + 1
            train_loss = train_loss / self.train_dl_len
            for metric_name in self.train_metric_vals.keys():
                self.train_metric_vals[metric_name] /= self.train_dl_len
            valid_loss = self._valid(mb)
            # logging
            epoch_end_time = time.time()
            log_info = [str(epoch), f"{train_loss:.4f}", f"{valid_loss:.4f}"]
            for key in self.metric_keys:
                if isinstance(self.train_metric_vals[key], float):
                    log_info.append(f"{self.train_metric_vals[key]:.4f}")
                    log_info.append(f"{self.valid_metric_vals[key]:.4f}")
                else:
                    log_info.append(str(self.train_metric_vals[key]))
                    log_info.append(str(self.valid_metric_vals[key]))
            log_info.append(f"{self.optim_func.lr.numpy():.6f}")
            log_info.append(f"{epoch_end_time - epoch_start_time:.4f}")
            mb.write(log_info, table=True)
            # 更新lr策略
            lr = float(keras_backend.get_value(self.optim_func.lr))
            lr = self.lr_scheduler(epoch, lr)
            keras_backend.set_value(self.optim_func.lr, keras_backend.get_value(lr))
        for callback in self.train_callbacks:
            callback.on_train_end(info)

    def _valid(self, mb):
        """验证."""
        if not self.valid_ds:
            return 0
        valid_loss = 0
        self.valid_metric_vals.clear()
        outputs_list, ys = [], []
        info = dict()
        batch_idx = 0
        for (x, y) in progress_bar(self.valid_dl,
                                   total=self.valid_dl_len if self.valid_dl_len else 0,
                                   parent=mb):
            info["x"], info["y"], info["batch_idx"] = x, y, batch_idx
            if not isinstance(x, (tuple, list)):
                info["x"] = [info["x"]]
            if not isinstance(info["y"], (tuple, list)):
                info["y"] = [info["y"]]
            for callback in self.valid_callbacks:
                callback.on_batch_begin(info)
            info["outputs"] = self.model(*info["x"])
            """
            if len(info["outputs"].shape) == 0:
                # 有时候最后一个batch的大小只有1，而如果模型返回时，直接squeeze()，则其shape为[]
                #   而我们期待的是[batch_ize]，所以此时需要reshape(按理来说应该由模型保证)
                info["outputs"] = tf.reshape(info["outputs"], (-1,))
            """
            for callback in self.valid_callbacks:
                callback.on_loss_begin(info)
            info["loss"] = self.loss_func(*info["y"], info["outputs"])
            for callback in self.valid_callbacks:
                callback.on_metric_begin(info)
            valid_loss += info["loss"].numpy()
            outputs_list.append(info["outputs"])
            ys.append(*info["y"])
            batch_idx += 1
        if not hasattr(self.valid_dl, '__len__') or len(self.valid_dl) == 0:
            self.valid_dl_len = batch_idx + 1
        for callback in self.valid_callbacks:
            callback.on_epoch_end(info)
        outputs_list = tf.concat(outputs_list, axis=0)
        ys = tf.concat(ys, axis=0)
        for metric_name in self.metrics.keys():
            self.valid_metric_vals[metric_name] = self.metrics[metric_name](outputs_list, ys)
        valid_loss /= self.valid_dl_len
        return valid_loss


class LrScheduler(object):
    @staticmethod
    def get_step_lr(step_size, gamma=0.1):
        """
        Decays the learning rate of each parameter.

        example:
            StepLR(optimizer, step_size=30, gamma=0.1)
            # lr = 0.05     if epoch < 30
            # lr = 0.005    if 30 <= epoch < 60
            # lr = 0.0005   if 60 <= epoch < 90

        :param step_size: 每step_size个epoch对学习率进行衰减
        :param gamma: 衰减因子 default: 0.1
        :return:
        """
        def scheduler(epoch, lr):
            if (epoch+1) % step_size == 0:
                return lr * gamma
            else:
                return lr
        return scheduler


class Metrics(object):
    """metrics."""

    @staticmethod
    def accuracy_score(inputs: EagerTensor,
                       targs: EagerTensor,
                       axis: int = -1,
                       just_score: bool = False):
        """Compute accuracy with `targ` when `pred` is bs * n_classes"""
        inputs, targs = inputs.numpy(), targs.numpy()
        if just_score:
            # 说明inputs为1的score
            inputs = np.stack([1 - inputs, inputs], axis=1)
            preds = inputs.argmax(axis=axis)
        else:
            # 说明inputs为0和1的score
            preds = inputs.argmax(axis=axis)
        acc = skm.accuracy_score(preds.reshape(-1,), targs.reshape(-1,))
        return acc

    @staticmethod
    def recall_score(inputs: EagerTensor,
                     targs: EagerTensor,
                     axis: int = -1,
                     average: str = 'binary',
                     just_score: bool = False):
        """Compute recall with `targ` when `pred` is bs * n_classes"""
        inputs, targs = inputs.numpy(), targs.numpy()
        if just_score:
            # 说明inputs为1的score
            inputs = np.stack([1 - inputs, inputs], axis=1)
            preds = inputs.argmax(axis=axis)
        else:
            # 说明inputs为0和1的score
            preds = inputs.argmax(axis=axis)
        recall = skm.recall_score(preds.reshape(-1,),
                                  targs.reshape(-1,),
                                  average=average,
                                  zero_division=0)
        return recall

    @staticmethod
    def precision_score(inputs: EagerTensor,
                        targs: EagerTensor,
                        axis: int = -1,
                        average: str = 'binary',
                        just_score: bool = False):
        """Compute precision with `targ` when `pred` is bs * n_classes"""
        inputs, targs = inputs.numpy(), targs.numpy()
        if just_score:
            # 说明inputs为1的score
            inputs = np.stack([1 - inputs, inputs], axis=1)
            preds = inputs.argmax(axis=axis)
        else:
            # 说明inputs为0和1的score
            preds = inputs.argmax(axis=axis)
        precision = skm.precision_score(preds.reshape(-1,),
                                        targs.reshape(-1,),
                                        average=average,
                                        zero_division=0)
        return precision

    @staticmethod
    def f1_score(inputs: EagerTensor,
                 targs: EagerTensor,
                 axis: int = -1,
                 average: str = 'binary',
                 just_score: bool = False):
        """Compute f1 score with `targ` when `pred` is bs * n_classes"""
        inputs, targs = inputs.numpy(), targs.numpy()
        if just_score:
            # 说明inputs为1的score
            inputs = np.stack([1 - inputs, inputs], axis=1)
            preds = inputs.argmax(axis=axis)
        else:
            # 说明inputs为0和1的score
            preds = inputs.argmax(axis=axis)
        f1 = skm.f1_score(preds.reshape(-1,),
                          targs.reshape(-1,),
                          average=average,
                          zero_division=0)
        return f1

    @staticmethod
    def auc_roc_score(outputs: EagerTensor,
                      targs: EagerTensor):
        """计算auc(area under the curve)(只适用于二分类).

        :param outputs: (np.ndarray)预测概率值(batchsize,)
        :param targs: (np.ndarray)标签(batchsize,)
        """

        def roc_curve(predicts: np.ndarray,
                      targets: np.ndarray):
            """计算receiver operator characteristic (ROC)曲线. 先得到不同阈值下的TPR和FPR
            (针对sigmoid的输出).

            :param predicts: (np.ndarray)预测概率值(batchsize,)
            :param targets: (np.ndarray)标签(batchsize,)
            """
            # 设outputs和targs的格式分别为[0.1, 0.8, 0.6, 0.3]和[1, 1, 1, 0]
            # 1. 根据input的概率值对input和targ进行从高到低重新排序
            desc_score_indices = np.argsort(-predicts)
            predicts = predicts[desc_score_indices]
            targets = targets[desc_score_indices]
            # 2. roc曲线不是每个点都要记录，只需记录有值的点即可，以下threshold_idxs为有值点下标
            diffs = predicts[1:] - predicts[:-1]
            distinct_indices = np.nonzero(diffs)[0]
            threshold_idxs = np.concatenate(
                (distinct_indices, [len(targets) - 1]))
            # 3. 计算tps(true positives sum)/fps(false positives sum)
            #   fps的计算：threshold_idxs的值ele，表示只有前ele+1个元素被认为是正样本(下标从0开始)
            #       所以fps = threshold_idxs + 1 - tps (ele+1个元素不是正样本就是负样本)
            tps = np.cumsum(targets)[threshold_idxs]
            fps = threshold_idxs + 1 - tps
            if tps[0] != 0 or fps[0] != 0:
                tps = np.concatenate(([0], tps))
                fps = np.concatenate(([0], fps))
            # 4. 计算tpr(true positive rate)/fpr(false positive rate)
            fpr_ = fps.astype(np.float) / (fps[-1] + 1e-8)
            tpr_ = tps.astype(np.float) / (tps[-1] + 1e-8)
            return fpr_, tpr_

        inputs, targs = outputs.numpy(), targs.numpy()
        # 1. 计算fpr和tpr
        fpr, tpr = roc_curve(outputs, targs)
        # 2. 计算auc
        #   fpr为横坐标，tpr为纵坐标，通过计算每一小块矩形的面积(xi*yi)，再相加得到auc
        # diffs为一系列小矩形的宽:[x1, x2, ...., xn]
        widths = fpr[1:] - fpr[:-1]
        heights = (tpr[:-1] + tpr[1:]) / 2
        auc = (widths * heights).sum()
        return auc


class Callback(object):
    """Base class for callbacks that want to record values, dynamically change learner params, etc."""

    def on_train_begin(self, info: dict):
        pass

    def on_epoch_begin(self, info: dict):
        pass

    def on_batch_begin(self, info: dict):
        pass

    def on_loss_begin(self, info: dict):
        pass

    def on_backward_begin(self, info: dict):
        pass

    def on_step_begin(self, info: dict):
        pass

    def on_metric_begin(self, info: dict):
        pass

    def on_batch_end(self, info: dict):
        pass

    def on_epoch_end(self, info: dict):
        pass

    def on_train_end(self, info: dict):
        pass
