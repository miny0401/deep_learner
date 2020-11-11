"""pytorch trainer.

1. 测试时，model.eval()一定要调用，不然如果model中有BatchNorm及Dropout等，会按训练的情况（如Dropout会生效)
2. 确保metric计算正常
3. 注意lr的设置
4. 多观察训练集与验证集的指标，以确定是过拟合还是欠拟合

@author: huangwm
"""
import time
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as skm
from torch.utils.data import Dataset, DataLoader
from functools import partial
from collections import defaultdict
from fastprogress import master_bar, progress_bar


def set_seed(seed: int = 1):
    """设置随机种子."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


# noinspection DuplicatedCode
class Learner(object):
    """pytorch trainer."""

    def __init__(self,
                 model: nn.Module,
                 train_ds: Dataset,
                 valid_ds: Dataset = None,
                 valid_batch: int = -1,
                 collate_fn=None,
                 loss_func: nn.Module = None,
                 optim_func: type = None,
                 device: torch.device = torch.device('cpu'),
                 batch_size: int = 128,
                 wd: float = 1e-5,
                 lr: float = 0.1,
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
        self._init()

    def _init(self):
        """初始化"""
        # 1. device(如果使用gpu，但不支持gpu，则改为cpu)
        if 'cuda' in self.device.type and not torch.cuda.is_available():
            self.device = torch.device('cpu')
        # 2. 构建DataLoader
        if self.collate_fn is None:
            self.collate_fn = torch.utils.data.dataloader.default_collate
        self.train_dl = DataLoader(self.train_ds,
                                   batch_size=self.batch_size,
                                   shuffle=True,
                                   collate_fn=self.collate_fn)
        self.valid_dl = DataLoader(self.valid_ds,
                                   batch_size=self.batch_size,
                                   collate_fn=self.collate_fn) if self.valid_ds else None
        # 3. 模型
        self.model = self.model.to(self.device)
        # 4. 设置优化器(如果没有设置策略，则表示使用默认的AdamW优化器)
        if not self.optim_class:
            self.optim_class = optim.AdamW
        self.optim_func: optim.Optimizer = self.optim_class(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        # 5. 设置损失函数(如果没有设置，则默认使用交叉熵损失)
        if not self.loss_func:
            self.loss_func = nn.CrossEntropyLoss().to(self.device)
        else:
            self.loss_func = self.loss_func.to(self.device)
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
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optim_func, step_size=1, gamma=1, last_epoch=-1)
        else:
            self.lr_scheduler = self.lr_scheduler(self.optim_func)
        # 8. 显示信息
        logging.info("=" * 80)
        logging.info(f"learner info: ")
        logging.info(f"train ds: {len(self.train_ds)} samples, "
                     f"{len(self.train_dl)} batches.")
        if self.valid_ds:
            logging.info(f"valid ds: {len(self.valid_ds)} samples, "
                         f"{len(self.valid_dl)} batches.")
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
            self.train_metric_vals.clear()
            for batch_idx, (x, y) in progress_bar(enumerate(self.train_dl),
                                                  total=len(self.train_dl),
                                                  parent=mb):
                info["x"], info["y"] = x, y
                # 开始第batch_idx批次的训练
                for callback in self.train_callbacks:
                    callback.on_batch_begin(info)
                self.model.train()
                # 梯度清零
                self.optim_func.zero_grad()
                # 数据移至正确的设备
                if isinstance(info["x"], (tuple, list)):
                    info["x"] = [e.to(self.device) for e in info["x"]]
                else:
                    info["x"] = [info["x"].to(self.device)]
                if isinstance(info["y"], (tuple, list)):
                    info["y"] = [e.to(self.device) for e in info["y"]]
                else:
                    info["y"] = [info["y"].to(self.device)]
                # 模型计算
                info["outputs"] = self.model(*info["x"])
                # 计算损失
                for callback in self.train_callbacks:
                    callback.on_loss_begin(info)
                loss = self.loss_func(info["outputs"], *info["y"])
                train_loss += loss.item()
                # 记录当前的训练的损失
                mb.child.comment = f"train loss: {train_loss / (batch_idx + 1):.4f}, " \
                                   f"valid loss: {valid_loss:.4f}"
                # 梯度回传
                for callback in self.train_callbacks:
                    callback.on_backward_begin(info)
                loss.backward()
                # 梯度更新
                for callback in self.train_callbacks:
                    callback.on_step_begin(info)
                self.optim_func.step()
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
            for callback in self.train_callbacks:
                callback.on_epoch_end(info)
            # 更新指标
            train_loss = train_loss / len(self.train_dl)
            for metric_name in self.train_metric_vals.keys():
                self.train_metric_vals[metric_name] /= len(self.train_dl)
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
            log_info.append(f"{self.optim_func.param_groups[0]['lr']:.6f}")
            log_info.append(f"{epoch_end_time - epoch_start_time:.4f}")
            mb.write(log_info, table=True)
            # 更新LR策略
            self.lr_scheduler.step()
        for callback in self.train_callbacks:
            callback.on_train_end(info)

    def _valid(self, mb):
        """验证."""
        if not self.valid_ds or len(self.valid_ds) == 0:
            return 0
        valid_loss = 0
        self.valid_metric_vals.clear()
        self.model.eval()
        outputs_list, ys = [], []
        info = dict()
        for batch_idx, (x, y) in progress_bar(enumerate(self.valid_dl), total=len(self.valid_dl), parent=mb):
            info["x"], info["y"], info["batch_idx"] = x, y, batch_idx
            if isinstance(x, (tuple, list)):
                info["x"] = [e.to(self.device) for e in info["x"]]
            else:
                info["x"] = [info["x"].to(self.device)]
            if isinstance(info["y"], (tuple, list)):
                info["y"] = [e.to(self.device) for e in info["y"]]
            else:
                info["y"] = [info["y"].to(self.device)]
            with torch.no_grad():
                for callback in self.valid_callbacks:
                    callback.on_batch_begin(info)
                info["outputs"] = self.model(*info["x"])
                """
                if len(info["outputs"].shape) == 0:
                    # 有时候最后一个batch的大小只有1，而如果模型返回时，直接squeeze()，则其shape为[]
                    #   而我们期待的是[batch_ize]，所以此时需要reshape(按理来说应该由模型保证)
                    info["outputs"] = info["outputs"].view(-1)
                """
                for callback in self.valid_callbacks:
                    callback.on_loss_begin(info)
                info["loss"] = self.loss_func(info["outputs"], *info["y"])
                for callback in self.valid_callbacks:
                    callback.on_metric_begin(info)
                valid_loss += info["loss"].item()
                outputs_list.append(info["outputs"])
                ys.append(*info["y"])
        for callback in self.valid_callbacks:
            callback.on_epoch_end(info)
        outputs_list = torch.cat(outputs_list, dim=0)
        ys = torch.cat(ys)
        for metric_name in self.metrics.keys():
            self.valid_metric_vals[metric_name] = self.metrics[metric_name](outputs_list, ys)
        valid_loss /= len(self.valid_dl)
        return valid_loss


class LrScheduler(object):
    @staticmethod
    def get_step_lr(step_size, gamma=0.1, last_epoch=-1):
        """
        Decays the learning rate of each parameter.

        example:
            StepLR(optimizer, step_size=30, gamma=0.1)
            # lr = 0.05     if epoch < 30
            # lr = 0.005    if 30 <= epoch < 60
            # lr = 0.0005   if 60 <= epoch < 90

        :param step_size: 每step_size个epoch对学习率进行衰减
        :param gamma: 衰减因子 default: 0.1
        :param last_epoch: 在last_epoch之后将不再进行衰减 default: 0.1
        :return:
        """
        return partial(optim.lr_scheduler.StepLR,
                       step_size=step_size,
                       gamma=gamma,
                       last_epoch=last_epoch)


class CollateFnFactory(object):
    """定义如何取样本.
    1. 如何组合多个样本
    2. 转换成torch.tensor格式
    """

    @staticmethod
    def text_collate_fn(batches):
        """定义如何取样本.
        1. 如何组合多个样本
        2. 转换成torch.tensor格式

        batches格式: [sample1, sample2, ...]
        如果dataset__getitem__返回的值为 feature, label，则batches的值为
            [(sample1_feature, sample1_label), (sample2_feature, sample2_label), ...]
        """
        features, labels = [], []
        for batch in batches:
            if isinstance(batch[0], torch.Tensor):
                features.append(batch[0].clone().detach().unsqueeze(0))
            else:
                features.append(torch.tensor(batch[0]).unsqueeze(0))
            if isinstance(batch[1], torch.Tensor):
                labels.append(batch[1].clone().detach().unsqueeze(0))
            else:
                labels.append(torch.tensor(batch[1]).unsqueeze(0))
        return torch.cat(features, dim=0), torch.cat(labels, dim=0)


class Metrics(object):
    """metrics."""

    @staticmethod
    def accuracy_score(inputs: torch.tensor,
                       targs: torch.tensor,
                       axis: int = -1,
                       just_score: bool = False):
        """Compute accuracy with `targ` when `pred` is bs * n_classes"""
        if just_score:
            # 说明inputs为1的score
            inputs = torch.stack([1 - inputs, inputs], dim=1)
            preds = inputs.argmax(dim=axis)
        else:
            # 说明inputs为0和1的score
            preds = inputs.argmax(dim=axis)
        acc = skm.accuracy_score(preds.cpu().data.numpy().reshape(-1, ),
                                 targs.cpu().data.numpy().reshape(-1, ))
        return acc

    @staticmethod
    def recall_score(inputs: torch.tensor,
                     targs: torch.tensor,
                     axis: int = -1,
                     average: str = 'binary',
                     just_score: bool = False):
        """Compute recall with `targ` when `pred` is bs * n_classes"""
        if just_score:
            # 说明inputs为1的score
            inputs = torch.stack([1 - inputs, inputs], dim=1)
            preds = inputs.argmax(dim=axis)
        else:
            # 说明inputs为0和1的score
            preds = inputs.argmax(dim=axis)
        recall = skm.recall_score(preds.cpu().data.numpy().reshape(-1, ),
                                  targs.cpu().data.numpy().reshape(-1, ),
                                  average=average, zero_division=0)
        return recall

    @staticmethod
    def precision_score(inputs: torch.tensor,
                        targs: torch.tensor,
                        axis: int = -1,
                        average: str = 'binary',
                        just_score: bool = False):
        """Compute precision with `targ` when `pred` is bs * n_classes"""
        if just_score:
            # 说明inputs为1的score
            inputs = torch.stack([1 - inputs, inputs], dim=1)
            preds = inputs.argmax(dim=axis)
        else:
            # 说明inputs为0和1的score
            preds = inputs.argmax(dim=axis)
        precision = skm.precision_score(preds.cpu().data.numpy().reshape(-1, ),
                                        targs.cpu().data.numpy().reshape(-1, ),
                                        average=average, zero_division=0)
        return precision

    @staticmethod
    def f1_score(inputs: torch.tensor,
                 targs: torch.tensor,
                 axis: int = -1,
                 average: str = 'binary',
                 just_score: bool = False):
        """Compute f1 score with `targ` when `pred` is bs * n_classes"""
        if just_score:
            # 说明inputs为1的score
            inputs = torch.stack([1 - inputs, inputs], dim=1)
            preds = inputs.argmax(dim=axis)
        else:
            # 说明inputs为0和1的score
            preds = inputs.argmax(dim=axis)
        f1 = skm.f1_score(preds.cpu().data.numpy().reshape(-1, ),
                          targs.cpu().data.numpy().reshape(-1, ),
                          average=average, zero_division=0)
        return f1

    @staticmethod
    def auc_roc_score(outputs: torch.tensor,
                      targs: torch.tensor):
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

        outputs = outputs.detach().cpu().numpy()
        targs = targs.detach().cpu().numpy()
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
