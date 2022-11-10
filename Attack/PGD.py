import torch
import torch.nn as nn
import numpy as np

ITERATIONS = 40   # number of iterations to perform gradient descent
EPSILON = 0.3       # number of Perturbation range
STEPALPHA = 0.01   # number of Perturbation for every step

class PGD:
    def __init__(self, model, device, eps=EPSILON, alpha=STEPALPHA, iters=ITERATIONS):
        """
        :description: PGD算法的初始化函数
        :param {
            model:所要攻击的已经加载好的模型
            device:模型运行设备”GPU“or“CPU”
            eps:图像最大扰动范围
            alpha:单次扰动参数
            iters:循环迭代次数
        }
        :return: None
        "epsilon": 0.3,
        "k": 40,
        "a": 0.01官方数据
        """
        self.model = model
        self.device = device
        self.init_model(self.device)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters

    def init_model(self, device):
        """
        :description:设置为非训练模式
        :param device: device:模型运行设备”GPU“or“CPU”
        :return: None
        """
        self.model.eval().to(device)

    def generate_image(self, image, label):
        """
        :description:使用的损失函数为交叉熵CrossEntropyLoss（）
        :param image: dataloader的image
        :param label:dataloader的label
        :return:attack_image
        """
        image = image.to(self.device)
        label = label.to(self.device)
        loss_func = nn.CrossEntropyLoss()

        ori_image = image.data

        image = image + torch.empty(image.shape, dtype=torch.float32).uniform_(-self.eps, self.eps)  # 添加随机的均匀扰动

        for _ in range(self.iters):
            image.requires_grad = True
            outputs = self.model(image)

            self.model.zero_grad()
            cost = loss_func(outputs, label).to(self.device)
            cost.backward()

            adv_image = image + self.alpha * image.grad.sign()
            eta = torch.clamp(adv_image - ori_image, min=-self.eps, max=self.eps)  # 将input的值限制在[min, max]之间，并返回结果
            image = torch.clamp(ori_image + eta, min=0,max=1).detach_()  # 返回一个新的tensor，新的tensor和原来的tensor共享数据内存，但不涉及梯度计算
            # 从计算中脱离。
        return image
