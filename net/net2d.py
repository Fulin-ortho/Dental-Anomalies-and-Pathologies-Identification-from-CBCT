import torch
import torch.nn as nn
from torchvision import models


def conv_block_2d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        activation
    )


def max_pooling_2d():
    return nn.MaxPool2d(kernel_size=2)


class BoxNet2D(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(BoxNet2D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(64)

        self.down_1 = conv_block_2d(
            self.in_dim, self.num_filters, self.activation)
        self.pool1 = max_pooling_2d()

        self.down_2 = conv_block_2d(
            self.num_filters, self.num_filters * 4, self.activation)
        self.pool2 = max_pooling_2d()

        self.down_3 = conv_block_2d(
            self.num_filters * 4, self.num_filters * 16, self.activation)
        self.pool3 = max_pooling_2d()

        self.down_4 = conv_block_2d(
            self.num_filters * 16, self.num_filters * 64, self.activation)
        self.pool4 = max_pooling_2d()

        self.lin1 = torch.nn.Linear(
            self.num_filters * 64 * 14 * 14, 64, bias=False)
        self.lin2 = torch.nn.Linear(64, self.out_dim)

    def forward(self, x):
        down_1 = self.down_1(x)
        pool_1 = self.pool1(down_1)

        down_2 = self.down_2(pool_1)
        pool_2 = self.pool2(down_2)

        down_3 = self.down_3(pool_2)
        pool_3 = self.pool3(down_3)

        down_4 = self.down_4(pool_3)
        pool_4 = self.pool3(down_4)

        pool_4 = pool_4.view(-1, self.num_filters * 64 * 14 * 14)
        out = self.lin1(pool_4)
        out = self.bn(out)
        out = self.activation(out)
        out = self.lin2(out)
        out = torch.sigmoid(out)
        return out


class MyResBoxNet(nn.Module):
    def __init__(self, out):
        super(MyResBoxNet, self).__init__()
        model = models.resnet18()
        out_channel = model.conv1.out_channels
        model.conv1 = nn.Conv2d(1, out_channel, 3, 2, 1)  # 改变网络结构
        fc_in = model.fc.in_features
        model.fc = nn.Linear(fc_in, out, bias=True)
        self.net = model

    def forward(self, x):
        out = self.net(x)

        return out
