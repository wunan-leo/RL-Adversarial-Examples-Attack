import torch
import numpy as np
from Attack import PGD

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ''''''
    import json  # 导入ImageNet的标签
    class_idx = json.load(open("./data/ImageNet/imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

    import torchvision
    from torchvision import transforms, datasets
    import os
    ''''''

    '''
    下面是ImageNet的dogs子集的读取
    '''

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        # 从图像中心裁切224x224大小的图片
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])

    test_ds = torchvision.datasets.ImageFolder(os.path.join('./data/dog-breed-identification', 'train_valid_test', 'valid'),
        transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_ds, 1, shuffle=False,drop_last = False)
    ''''''''''''''
    #使用torch自带的公共模型alexnet
    model = torchvision.models.alexnet(pretrained=True)

    model.to(device)
    model.eval()  # 测试

    pgd = PGD.PGD(model, device)

    accuracy=0
    attack_suss=0

    for i, data in enumerate(test_loader, 0):
        images, labels = data
        #print(images)
        attack_images = pgd.generate_image(images, labels)

        outputs = model(attack_images)
        outputs_p = model(images)

        from matplotlib import pyplot as plt
        img = torchvision.utils.make_grid(images).numpy()
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.show() #展示原图片
        img = torchvision.utils.make_grid(attack_images).numpy()
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.show() #展示原图片


        _, predicted = torch.max(outputs_p.data, 1)
        _, predictedT = torch.max(outputs.data, 1)

        if idx2label[predictedT].lower() == test_ds.classes[labels.item()]:
            attack_suss+=1
        if idx2label[predicted].lower() == test_ds.classes[labels.item()]:
            accuracy+=1

        print("Attack:{} , Predicted:{} , True:{}".format(idx2label[predictedT],idx2label[predicted],test_ds.classes[labels.item()]))

    print('\naccuracy:{}%\nattack_acc:{}%'.format((accuracy/(i+1)*100),(attack_suss/(i+1)*100)))
