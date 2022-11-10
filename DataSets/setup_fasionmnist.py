import numpy as np
import torch
import os
import codecs
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
from PIL import Image
import gc

class FashionMNISTDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=transforms.ToTensor()):
        self.root = os.path.join(root_dir)  # 数据集根目录
        self.transform = transform  # 图像变换，一般是用于丰富数据集
        self.train = train

        if self.train:
            with open(os.path.join(self.root, 'train-images-idx3-ubyte'), 'rb') as f:
                data = f.read()
                length = int(codecs.encode(data[4:8], 'hex'), 16)
                num_rows = int(codecs.encode(data[8:12], 'hex'), 16)
                num_cols = int(codecs.encode(data[12:16], 'hex'), 16)
                parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
                self.train_data = torch.from_numpy(parsed).view(length, num_rows, num_cols)

            with open(os.path.join(self.root, 'train-labels-idx1-ubyte'), 'rb') as f:
                data = f.read()
                parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
                self.train_labels = torch.from_numpy(
                    np.array([[int(parsed[i] == j) for j in range(10)] for i in range(len(parsed))]))

        else:
            with open(os.path.join(self.root, 't10k-images-idx3-ubyte'), 'rb') as f:
                data = f.read()
                length = int(codecs.encode(data[4:8], 'hex'), 16)
                num_rows = int(codecs.encode(data[8:12], 'hex'), 16)
                num_cols = int(codecs.encode(data[12:16], 'hex'), 16)
                parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
                self.test_data = torch.from_numpy(parsed).view(length, num_rows, num_cols)

            with open(os.path.join(self.root, 't10k-labels-idx1-ubyte'), 'rb') as f:
                data = f.read()
                parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
                self.test_labels = torch.from_numpy(
                    np.array([[int(parsed[i] == j) for j in range(10)] for i in range(len(parsed))]))

    '''
    if...else...是对于fashion-mnist数据集提取数据，直接从编码格式里面提取
    '''

    def __len__(self):  # 返回数据长度
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):  # 返回指定位置的图像数据
        if self.train:
            img, target = self.train_data[idx], self.train_labels[idx]
        else:
            img, target = self.test_data[idx], self.test_labels[idx]
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)  # 图像变换
        return img, target
