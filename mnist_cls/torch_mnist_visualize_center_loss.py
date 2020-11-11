"""使用lenet训练mnist."""
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from abc import ABC
from functools import partial
from torchvision import datasets, transforms
from torch.autograd.function import Function

from torch_learner import Learner, set_seed, Metrics, LrScheduler, Callback


class CenterLoss(nn.Module, ABC):
    """使用parameter封装训练center.
    center_loss = ||feature_embedding - center_embedding||**2
    """
    def __init__(self, cls_num, feature_dim):
        super().__init__()
        self.cls_num = cls_num
        self.featur_num = feature_dim
        self.center = nn.Parameter(torch.randn(cls_num, feature_dim))
        self.is_first = True

    def forward(self, xs, ys):
        """
        if self.is_first:
            self.center = self.center.to(xs.device)
            self.is_first = False
        """
        center_exp = self.center.index_select(dim=0, index=ys.long())
        count = torch.histc(ys.float(), bins=self.cls_num, min=0, max=self.cls_num - 1)
        count_dis = count.index_select(dim=0, index=ys.long()) + 1
        loss = torch.sum(torch.sum((xs - center_exp) ** 2, dim=1) / 2.0 / count_dis.float())
        return loss


# noinspection PyMethodOverriding
class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None


class CenterLoss2(nn.Module, ABC):
    """使用torch的Function计算center loss，需要自己计算梯度，大部分时候没必要."""
    def __init__(self, cls_num, feature_dim, size_average=True):
        super(CenterLoss2, self).__init__()
        self.centers = nn.Parameter(torch.randn(cls_num, feature_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feature_dim
        self.size_average = size_average

    def forward(self, feat, label):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss


class MyLoss(nn.Module, ABC):
    def __init__(self, center_loss_weight: float = 1.0, cls_num: int = 10, feature_dim: int = 2):
        super(MyLoss, self).__init__()
        self.center_loss_func = CenterLoss(cls_num, feature_dim)
        self.nllloss_func = nn.NLLLoss()
        self.center_loss_weight = center_loss_weight

    def forward(self, outputs, targets):
        return self.nllloss_func(outputs[1], targets) + \
            self.center_loss_weight * self.center_loss_func(outputs[0], targets)


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
        self.labels.append(info["y"][0].int())

    def on_metric_begin(self, info: dict):
        info["orig_outputs"] = info["outputs"]
        info["outputs"] = info["outputs"][1]

    def on_epoch_end(self, info: dict):
        # 每个epoch结束时，对向量进行可视化
        embeddings = torch.cat(self.embeddings, dim=0)
        labels = torch.cat(self.labels, dim=0)
        self.visualize(embeddings.detach().cpu().numpy(), labels.detach().cpu().numpy(), info["epoch"])

    @staticmethod
    def visualize(feat, labels, epoch):
        plt.ion()
        c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        plt.clf()
        for i in range(10):
            plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
        plt.xlim(xmin=-10, xmax=10)
        plt.ylim(ymin=-10, ymax=10)
        plt.text(-7.8, 7.3, "epoch=%d" % epoch)
        plt.savefig('/tmp/epoch=%d.jpg' % epoch)
        plt.draw()
        plt.pause(0.001)


class ValidCallback(Callback):
    def on_metric_begin(self, info: dict):
        info["orig_outputs"] = info["outputs"]
        info["outputs"] = info["outputs"][1]


class Model(nn.Module, ABC):
    """LeNet with prelu."""
    def __init__(self, cls_num: int = 10):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.PReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.PReLU()
        self.fc3 = nn.Linear(84, 2)
        self.relu5 = nn.PReLU()
        self.fc = nn.Linear(2, cls_num)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        logit = self.relu5(y)
        y = self.fc(logit)
        return logit, self.log_softmax(y)


def main():
    """main."""
    set_seed(1)
    # 1. 数据集
    train_dataset = datasets.MNIST(
        '~/.cache/datasets/mnist',
        download=True,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))
    valid_dataset = datasets.MNIST(
        '~/.cache/datasets/mnist',
        download=True,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))
    # 2. 训练模型
    # 注：如果使用CrossEntropyLoss则最后一层不应使用Softmax，因为CrossEntropyLoss会为我们做Softmax，
    #   而相反，如果使用NLLLoss，则需要使用LogSoftmax
    logging.info(f"训练模型.")
    model = Model()
    learner = Learner(model=model,
                      train_ds=train_dataset,
                      valid_ds=valid_dataset,
                      metrics={'accuracy': partial(Metrics.accuracy_score, just_score=False),
                               'precision_macro': partial(Metrics.precision_score, average='macro', just_score=False),
                               'recall_macro': partial(Metrics.recall_score, average='macro', just_score=False),
                               'f1_macro': partial(Metrics.f1_score, average='macro', just_score=False)},
                      loss_func=MyLoss(),
                      optim_func=torch.optim.Adam,
                      batch_size=128,
                      lr=0.001,
                      lr_scheduler=LrScheduler.get_step_lr(step_size=5, gamma=0.2),
                      valid_batch=-1,
                      device=torch.device('cuda'),
                      train_callbacks=[TrainCallback()],
                      valid_callbacks=[ValidCallback()])
    learner.train(20)
    torch.save(learner.model.state_dict(), "/tmp/lenet_mnist.pth")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s')
    main()
