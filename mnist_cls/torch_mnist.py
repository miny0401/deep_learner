"""使用lenet训练mnist."""
import logging
import torch
import torch.nn as nn
from abc import ABC
from functools import partial
from torchvision import datasets, transforms

from torch_learner import Learner, set_seed, Metrics, LrScheduler


class Model(nn.Module, ABC):
    """LeNet."""
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()
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
        y = self.relu5(y)
        return self.log_softmax(y)


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
                      loss_func=nn.NLLLoss(),
                      optim_func=torch.optim.Adam,
                      batch_size=128,
                      lr=0.01,
                      lr_scheduler=LrScheduler.get_step_lr(step_size=5, gamma=0.2),
                      valid_batch=-1,
                      device=torch.device('cuda'))
    learner.train(20)
    torch.save(learner.model.state_dict(), "/tmp/lenet_mnist.pth")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s')
    main()
