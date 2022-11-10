import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
from CNN import CNNModel

from DataSets import setup_fasionmnist

'''
自行下载fashion-mnist数据集，放在/Test/data/fashion-mnist/data/fashion中
'''
train_dataset = setup_fasionmnist.FashionMNISTDataset(root_dir="../Test/data/fashion-mnist/data/fashion/")
test_dataset = setup_fasionmnist.FashionMNISTDataset(root_dir="../Test/data/fashion-mnist/data/fashion/", train=False)
'''
import torchvision
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                             download=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                            download=False)  # 第一次ture后面就不用下载，改为false
testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                         shuffle=False, num_workers=2)
可以利用pytorch提供的dataset里面下载，不用上面那么多
'''

batch_size = 100
n_iters = 18000
num_epochs = (n_iters * batch_size) / len(train_dataset)
num_epochs = int(num_epochs)
'''
（1）batchsize：批大小。在深度学习中，一般采用SGD(随机梯度下降)训练，即每次训练在训练集中取batchsize个样本训练；
（2）iteration：1个iteration等于使用batchsize个样本训练一次；
（3）epoch：1个epoch等于使用训练集中的全部样本训练一次；
'''

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)  # shuffle将训练模型的数据集进行打乱的操作
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



model = CNNModel()
#from torchsummary import summary
#summary(model,(1,28,28))
#  train()
if torch.cuda.is_available():  # 判断是否可以再GPU运行
    model.cuda()

criterion = nn.BCELoss()  # 二分类交叉熵

learning_rate = 0.005

optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)  # 优化器

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,  # 调度器，也是调整学习率。但是是基于epoch
                                            gamma=0.1)  # this will decrease the learning rate by factor of 0.1 every 10 epochs
iter = 0


train_loss=0.0
running_loss = 0.0
total = 0
correct = 0
totalT = 0
correctT = 0

Train_Loss_list = []  # 存储训练损失值
Valid_Loss_list = []  # 存储测试损失值
Valid_Accuracy_list = []  # 存储测试准确率
Train_Accuracy_list = []  # 存储训练准确率

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        model.train()
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs.float(), labels.float())

        loss.backward()

        optimizer.step()
        iter += 1

        running_loss += loss.item()
        train_loss += loss.item()

        if iter % 500 == 0:
            print('[epoch:%d, iter:%5d] loss: %.4f' %
                  (epoch, iter, running_loss / 500))
            running_loss = 0.0
            _, predicted = torch.max(outputs.data, 1)
            _, labelsI = torch.max(labels, 1)
            total += labelsI.size(0)
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labelsI.cpu()).sum()
            else:
                correct += (predicted == labelsI).sum()
            print('Accuracy of the network on the %d tran images: %.3f %%' % (total, 100.0 * float(correct) / total))

            total = 0
            correct = 0

    # 存储数据 4.25!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Train_Loss_list.append((train_loss / len(train_dataset)))
    train_loss =0.0# 归零进行下一次计算

    _, predictedT = torch.max(outputs.data, 1)
    _, labelsT = torch.max(labels, 1)
    totalT += labels.size(0)
    if torch.cuda.is_available():
        correctT += (predictedT.cpu() == labelsT.cpu()).sum()
    else:
        correctT += (predictedT == labelsT).sum()

    Train_Accuracy_list.append(100.0 * float(correctT) / totalT)
    totalT=0
    correctT=0
    torch.save(model.state_dict(), r'./model%s.pth' % epoch)  # 保存模型

    scheduler.step()
