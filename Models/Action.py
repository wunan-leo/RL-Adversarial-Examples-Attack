import numpy as np
import torch
import torchvision


class Aaction():
    def __init__(self, image, vector, size=3):
        """
        使用ChooseAction（）选择操作
        :param image: 图片参数，格式为C,H,W,(tensor)
        :param vector:[r,g,b,对比度，亮度，饱和度，x坐标，y坐标]，以x，y为【1,1】位置截取sie*size的大小方框
        :param size:[min,max]，必须要大于0，建议0~1之间
        :param action_num: 动作要求
        """
        self.action_num = 1
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        self.un_mean = [-1, -1, -1]
        self.un_std = [2, 2, 2]

        self.dtype = image.dtype
        self.device = image.device

        un_normalize = torchvision.transforms.normalize(mean=self.un_mean, std=self.un_std)
        image1 = un_normalize(image)
        print(image1.numpy().shape)
        image_arr = image1.numpy().transpose(1, 2, 0)

        self.image_original = image_arr * 255

        self.vector = vector
        self.size = size

        h, w, c = self.image_original.shape

        # 填充0
        pad = self.size // 2
        image_out = np.zeros((h + 2 * pad, w + 2 * pad, c), dtype=float)
        image_out[pad:pad + h, pad:pad + w] = self.image_original.copy()

        self.image_extend = image_out

        x = vector[6] * h
        y = vector[7] * w
        self.x = int(x)
        self.y = int(y)

        print(x, y)

        self.image = image_out[self.x:self.x + self.size - 1, self.y:  self.y + self.size - 1]

    def saturation(self, increment):
        """
        饱和度计算
        :param increment:饱和度增量倍率
        :param rgb_img:图片参数
        :return:
        """
        img_out = self.image * 1.0
        img_min = img_out.min(axis=2)
        img_max = img_out.max(axis=2)

        # 获取HSL空间的饱和度和亮度
        delta = (img_max - img_min) / 255.0
        value = (img_max + img_min) / 255.0

        L = value / 2.0

        # s = L<0.5 ? s1 : s2
        mask_1 = L < 0.5
        s1 = delta / value
        s2 = delta / (2 - value)
        s = s1 * mask_1 + s2 * (1 - mask_1)

        # 增量倍率大于1，饱和度指数增强
        if increment > 1:
            # alpha = increment+s > 1 ? alpha_1 : alpha_2
            temp = increment * s
            mask_2 = temp > 1
            alpha_1 = s
            alpha_2 = s * 0 + 1 - (increment - 1) * s
            alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)

            alpha = 1 / alpha - 1
            img_out[:, :, 0] = img_out[:, :, 0] + (img_out[:, :, 0] - L * 255.0) * alpha
            img_out[:, :, 1] = img_out[:, :, 1] + (img_out[:, :, 1] - L * 255.0) * alpha
            img_out[:, :, 2] = img_out[:, :, 2] + (img_out[:, :, 2] - L * 255.0) * alpha
        # 增量倍率小于1，饱和度线性衰减
        else:
            alpha = (increment - 1) * s
            img_out[:, :, 0] = img_out[:, :, 0] + (img_out[:, :, 0] - L * 255.0) * alpha
            img_out[:, :, 1] = img_out[:, :, 1] + (img_out[:, :, 1] - L * 255.0) * alpha
            img_out[:, :, 2] = img_out[:, :, 2] + (img_out[:, :, 2] - L * 255.0) * alpha

        img_out = np.clip(img_out, 0, 255)

        return img_out.astype(int)

    def contrast(self, increment, threshold=0.5):
        """
        对比度计算
        :param increment: 增量倍率（对比度增量contrast等于255*倍率，使用时转化到0~1）
        :param threshold:阀值0.5
        :return:
        """
        # 对比度增量
        contrast = (increment - 1)

        img_out = self.image * 1.0
        # 增量等于1，按灰度阈值最多调整成八种颜色：
        # 黑、红、绿、蓝、黄(255,255,0)、品红(255,0,255)、青(0,255,255)、白
        if contrast == 1:
            # newRGB = RGB >= Threshold? 255 : 0
            mask_1 = img_out >= threshold * 255.0
            rgb1 = 255.0
            rgb2 = 0
            img_out = rgb1 * mask_1 + rgb2 * (1 - mask_1)

        # 增量大于0小于1
        elif contrast >= 0:
            alpha = 1 - contrast
            alpha = 1 / alpha - 1
            img_out[:, :, 0] = img_out[:, :, 0] + (img_out[:, :, 0] - threshold * 255.0) * alpha
            img_out[:, :, 1] = img_out[:, :, 1] + (img_out[:, :, 1] - threshold * 255.0) * alpha
            img_out[:, :, 2] = img_out[:, :, 2] + (img_out[:, :, 2] - threshold * 255.0) * alpha

        # 增量小于0
        else:
            alpha = contrast
            img_out[:, :, 0] = img_out[:, :, 0] + (img_out[:, :, 0] - threshold * 255.0) * alpha
            img_out[:, :, 1] = img_out[:, :, 1] + (img_out[:, :, 1] - threshold * 255.0) * alpha
            img_out[:, :, 2] = img_out[:, :, 2] + (img_out[:, :, 2] - threshold * 255.0) * alpha

        img_out = np.clip(img_out, 0, 255)

        return img_out.astype(int)

    def brightness(self, increment):
        """
        亮度改变
        :param increment: 增量倍率
        :return:
        """
        img_out = self.image * 1.0

        bright = increment - 1

        # 增量大于0，指数调整
        if bright >= 0:
            alpha = 1 - bright
            alpha = 1 / alpha

        # 增量小于0，线性调整
        else:
            alpha = bright + 1

        img_out[:, :, 0] = img_out[:, :, 0] * alpha
        img_out[:, :, 1] = img_out[:, :, 1] * alpha
        img_out[:, :, 2] = img_out[:, :, 2] * alpha

        img_out = np.clip(img_out, 0, 255)

        return img_out.astype(int)

    def r(self, increment):
        img_out = self.image * 1.0
        img_out[:, :, 0] = img_out[:, :, 0] * increment

        return img_out.astype(int)

    def g(self, increment):
        img_out = self.image * 1.0
        img_out[:, :, 1] = img_out[:, :, 1] * increment

        return img_out.astype(int)

    def b(self, increment):
        img_out = self.image * 1.0
        img_out[:, :, 2] = img_out[:, :, 2] * increment

        return img_out.astype(int)

    def weighted_mean_filter(self):
        h, w, c = self.image.shape
        size = 3  # 默认为3
        kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]  # 高斯滤波模板

        image = self.image * 1.0

        # 填充0
        pad = size // 2
        image_out = np.zeros((h + 2 * pad, w + 2 * pad, c), dtype=float)
        image_out[pad:pad + h, pad:pad + w] = image.copy()

        tmp = image_out.copy()
        for x in range(h):
            for y in range(w):
                for z in range(c):
                    image_out[pad + x, pad + y, z] = np.mean(np.multiply(tmp[x:x + size, y:y + size, z], kernel))

        image_out = image_out[pad:pad + h, pad:pad + w]

        image_out = np.clip(image_out, 0, 255)

        return image_out.astype(int)

    def projection_to_sphere(self, img):
        un_normalize = torchvision.transforms.normalize(mean=self.un_mean, std=self.un_std)
        image1 = un_normalize(img)
        image_arr = image1.numpy().transpose(1, 2, 0)

        rows = image_arr.shape[0]
        cols = image_arr.shape[1]
        blank = np.zeros_like(image_arr)
        from matplotlib import pyplot as plt
        plt.imshow(image_arr)
        plt.show()

        # 圆心定为图片中心
        center_x = int(rows / 2)
        center_y = int(cols / 2)
        # 假设球的半径
        r = int(((rows ** 2 + cols ** 2) ** 0.5) / 2) + 20
        # 假设映射平面位于 z = r 处
        pz = r
        for x in range(rows):
            ox = x
            x = x - center_x
            for y in range(cols):
                oy = y
                y = y - center_y
                z = (r * r - x * x - y * y) ** 0.5
                # 假设光源点为(0,0,2r)
                k = (pz - 2 * r) / (z - 2 * r)
                px = int(k * x)
                py = int(k * y)
                px = px + center_x
                py = py + center_y
                blank[px, py, :] = image_arr[ox, oy, :]

        plt.imshow(blank)
        plt.show()

        blank = np.array(blank).transpose(2, 0, 1)  # 表示C x H x W
        img1 = torch.tensor(blank / 255, dtype=self.dtype, device=self.device)
        normalize = torchvision.transforms.normalize(mean=self.mean, std=self.std)  # 正常归一化
        img_out = normalize(img1)

        return img_out

    def do_action(self):
        self.image = self.r(self.vector[0] * 2 - 1)
        self.image = self.g(self.vector[1] * 2 - 1)
        self.image = self.b(self.vector[2] * 2 - 1)
        self.image = self.contrast(self.vector[3] * 2 - 1)
        self.image = self.brightness(self.vector[4] * 2 - 1)
        self.image = self.saturation(self.vector[5] * 2 - 1)
        self.image = self.weighted_mean_filter()

        h, w, _ = self.image_original.shape
        self.image_extend[self.x:self.x + self.size - 1, self.y:  self.y + self.size - 1] = self.image[0:self.size - 1,
                                                                                            0:self.size - 1]
        pad = self.size // 2
        image2 = self.image_extend[pad:pad + h, pad:pad + w]

        img = np.array(image2).transpose(2, 0, 1)  # 表示C x H x W
        img1 = torch.tensor(img / 255, dtype=self.dtype, device=self.device)
        normalize = torchvision.transforms.normalize(mean=self.mean, std=self.std)  # 正常归一化
        img_out = normalize(img1)

        return img_out

