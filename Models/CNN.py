import torch
import torch.nn as nn


class Swish(nn.Module):  # 一种激活函数
    def forward(self, input):
        return (input * torch.sigmoid(input))

    def __repr__(self):  # 返回类名称Swish()
        return self.__class__.__name__ + ' ()'


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()  # 首先找到CNNModel的父类（比如是类NNet），然后把类CNNModel的对象self转换为类NNet的对象，然后“被转换”的类NNet
        # 对象调用自己的init函数,这是对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # 输入1通道。输出16通道。卷积核3*3,卷积核在图像窗口上每次平移的间隔，即所谓的步长1,图像扩展1
        self.bn1 = nn.BatchNorm2d(16)
        # 归一化处理。16为通道数，即为特征数
        self.swish1 = Swish()
        # 激活函数
        nn.init.xavier_normal(self.cnn1.weight)
        # Xavier初始化
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        # 池化层

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.swish2 = Swish()
        nn.init.xavier_normal(self.cnn2.weight)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.swish3 = Swish()
        nn.init.xavier_normal(self.cnn3.weight)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 6 * 6, 10)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.bn1(out)
        out = self.swish1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.bn2(out)
        out = self.swish2(out)
        out = self.maxpool2(out)
        out = self.cnn3(out)
        out = self.bn3(out)
        out = self.swish3(out)
        out = self.maxpool3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.softmax(out)

        return out
