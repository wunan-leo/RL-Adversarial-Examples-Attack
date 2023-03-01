import numpy as np


class Action:
    def __init__(self, image):
        """
        :param image: 图片参数，格式为H,W,C
        :param action_num: 动作要求
        """
        self.image = image
        self.action_num = 1

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

    def R(self, increment):
        img_out = self.image * 1.0
        img_out[:, :, 0] = img_out[:, :, 0] * increment

        return img_out.astype(int)

    def G(self, increment):
        img_out = self.image * 1.0
        img_out[:, :, 1] = img_out[:, :, 1] * increment

        return img_out.astype(int)

    def B(self, increment):
        img_out = self.image * 1.0
        img_out[:, :, 2] = img_out[:, :, 2] * increment

        return img_out.astype(int)

    def MeanFilter(self, size=3):
        h, w, c = self.image.shape

        image = self.image * 1.0

        # 填充0
        pad = size // 2
        image_out = np.zeros((h + 2 * pad, w + 2 * pad, c), dtype=float)
        image_out[pad:pad + h, pad:pad + w] = image.copy()

        tmp = image_out.copy()
        for x in range(h):
            for y in range(w):
                for z in range(c):
                    image_out[pad + x, pad + y, z] = np.mean(tmp[x:x + size, y:y + size, z])

        image_out = image_out[pad:pad + h, pad:pad + w]

        image_out = np.clip(image_out, 0, 255)

        return image_out.astype(int)

    def WeightedMeanFilter(self):
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

    def chooseAction(self, action_num):
        self.action_num = action_num
        if self.action_num == 0:
            image = self.contrast(1.1)
        elif self.action_num == 1:
            image = self.contrast(0.9)
        elif self.action_num == 2:
            image = self.saturation(1.1)
        elif self.action_num == 3:
            image = self.saturation(0.9)
        elif self.action_num == 4:
            image = self.brightness(1.1)
        elif self.action_num == 5:
            image = self.brightness(0.9)
        elif self.action_num == 6:
            image = self.R(1.1)
        elif self.action_num == 7:
            image = self.R(0.9)
        elif self.action_num == 8:
            image = self.G(1.1)
        elif self.action_num == 9:
            image = self.G(0.9)
        elif self.action_num == 10:
            image = self.B(1.1)
        elif self.action_num == 11:
            image = self.B(0.9)
        elif self.action_num == 12:
            image = self.MeanFilter()
        elif self.action_num == 13:
            image = self.WeightedMeanFilter()
        else:
            print("No such action!\n")
            return None

        return image.astype(int)
