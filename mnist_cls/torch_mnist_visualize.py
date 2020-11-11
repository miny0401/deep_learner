"""使用lenet训练mnist."""
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from abc import ABC
from functools import partial
from torchvision import datasets, transforms

from torch_learner import Learner, set_seed, Metrics, LrScheduler, Callback


class MyLoss(nn.Module, ABC):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.nllloss = nn.NLLLoss()

    def forward(self, outputs, targets):
        return self.nllloss(outputs[1], targets)


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
        plt.xlim(xmin=-75, xmax=75)
        plt.ylim(ymin=-75, ymax=75)
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
    def __init__(self):
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
        self.fc = nn.Linear(2, 10)
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
    logging.info(f"{len(train_dataset)}个训练样本，{len(valid_dataset)}个测试样本.")
    # 2. 训练模型
    # 注：如果使用CrossEntropyLoss则最后一层不应使用Softmax，因为CrossEntropyLoss会为我们做Softmax，
    #   而相反，如果使用NLLLoss，则需要使用LogSoftmax
    logging.info(f"训练模型.")
    model = Model()
    print(model.parameters)
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
                      lr=0.01,
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
