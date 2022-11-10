import os
import shutil
import numpy as np

'''
用于处理kaggle上下载的dogvscat数据集，放在Test/data/dog-breed-identification目录下，在PGD测试中可以用到
'''

# kaggle原始数据集在本地电脑的文件路径，自行更改
original_dataset_dir = 'D:\\download\kaggle\\train\\train'
total_num = int(len(os.listdir(original_dataset_dir)) / 2)
random_idx = np.array(range(total_num))
np.random.shuffle(random_idx)

# 待处理的数据集地址，自行更改
base_dir = 'D:\\PycharmProjects\\Fashion-mnist_cnn\\dogsVScats'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# 训练集、验证集的划分
sub_dirs = ['train', 'validate']
animals = ['cats', 'dogs']
train_idx = random_idx[:int(total_num * 0.9)]
validate_idx = random_idx[int(total_num * 0.9):]
numbers = [train_idx, validate_idx]
for idx, sub_dir in enumerate(sub_dirs):
    dir = os.path.join(base_dir, sub_dir)
    if not os.path.exists(dir):
        os.mkdir(dir)
    for animal in animals:
        animal_dir = os.path.join(dir, animal)
        if not os.path.exists(animal_dir):
            os.mkdir(animal_dir)
        fnames = [animal[:-1] + '.{}.jpg'.format(i) for i in numbers[idx]]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(animal_dir, fname)
            shutil.copyfile(src, dst)

        # 训练集、验证集的图片数目
        print(animal_dir + ' total images : %d' % (len(os.listdir(animal_dir))))
