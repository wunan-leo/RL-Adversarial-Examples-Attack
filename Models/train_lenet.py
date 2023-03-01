# -*- encoding: utf-8 -*-
'''
@File    :   train_lenet.py
@Modify time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/11/5 19:48   leoy         1.0         none
'''

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from lenet import LeNet
import numpy as np
import math

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-path', type=str, default='./models/mnist.pth', metavar='N',
                    help='save model path')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.is_variational = False

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

cifar10_transform = transforms.Compose([
    transforms.ToTensor(),  # numpy -> Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化 ，范围[-1,1]
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/', train=False, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

cifar10_train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data/', train=True, download=True,
                     transform=cifar10_transform),
    batch_size=args.batch_size, shuffle=True, **kwargs)

cifar10_test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data/', train=False, download=True,
                     transform=cifar10_transform),
    batch_size=args.batch_size, shuffle=True, **kwargs)

model = LeNet(in_channels=3, out_size=2048)

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)


def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(cifar10_train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            centers = centers.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        batch_norm = torch.nn.utils.clip_grad_norm(model.parameters(), 20, norm_type=2)
        optimizer.step()
        # torch.save(model.state_dict(), args.model_path)
        torch.save(model.state_dict(), './models/cifar10.pth')
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} norm: {:.4f}'.format(
                epoch, batch_idx * len(data), len(cifar10_train_loader.dataset),
                       100. * batch_idx / len(cifar10_train_loader), loss.item(), batch_norm))


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in cifar10_test_loader:

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.cross_entropy(output, target,
                                     size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(cifar10_test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(cifar10_test_loader.dataset),
        100. * correct / len(cifar10_test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    scheduler.step()
    train(epoch)
    test(epoch)
