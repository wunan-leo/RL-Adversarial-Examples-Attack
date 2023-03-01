# -*- encoding: utf-8 -*-
'''
@File    :   evaluation.py  
@Modify time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/11/9 23:27   leoy         1.0         none
'''

import numpy as np
import pandas as pd
import torch
# from skimage.metrics import structural_similarity as SSIM

class Evaluation():

    def __init__(self, origin_input, adv_input, origin_label,
                 target_label, adv_prediction, is_targeted = True):
        self.origin_input = origin_input
        self.adv_input = adv_input
        self.origin_label = origin_label
        self.target_label = target_label
        self.adv_prediction = adv_prediction
        self.is_targeted = is_targeted
        self.MIN_COMPENSATION = 1

    def acac(self):
        total = len(self.origin_input)
        number = 0
        prob = 0.0
        outputs = torch.from_numpy(self.adv_prediction)
        outputs_softmax = torch.nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, 1)
        outputs_softmax = outputs_softmax.data.numpy()
        preds = preds.data.numpy()

        if not self.is_targeted:
            for i in range(total):
                if preds[i] != np.argmax(self.origin_label[i]):
                    number += 1
                    prob += np.max(outputs_softmax[i])
        else:
            for i in range(total):
                if preds[i] == np.argmax(self.target_label[i]):
                    number += 1
                    prob += np.max(outputs_softmax[i])

        if not number == 0:
            acac = prob/number
        else:
            acac = prob / (number + self.MIN_COMPENSATION)

        return acac

    def actc(self):

        total = len(self.origin_input)
        outputs = torch.from_numpy(self.adv_prediction)
        outputs_softmax = torch.nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, 1)
        outputs_softmax = outputs_softmax.data.numpy()
        preds = preds.data.numpy()
        number = 0
        prob = 0.0
        if not self.is_targeted:
            for i in range(total):
                if preds[i] != np.argmax(self.origin_label[i]):
                    number += 1
                    prob += outputs_softmax[i, np.argmax(self.origin_label[i])]
        else:
            for i in range(total):
                if preds[i] == np.argmax(self.target_label[i]):
                    number += 1
                    prob += outputs_softmax[i, np.argmax(self.origin_label[i])]

        if not number == 0:
            actc = prob / number
        else:
            actc = prob / (number + self.MIN_COMPENSATION)

        return actc

    def acc(self):
        total = len(self.origin_input)
        number = 0
        outputs = torch.from_numpy(self.adv_prediction)
        preds = torch.argmax(outputs, 1)
        preds = preds.data.numpy()
        if not self.is_targeted:
            for i in range(total):
                if preds[i] != np.argmax(self.origin_label[i]):
                    number += 1
        else:
            for i in range(total):
                if preds[i] == np.argmax(self.target_label[i]):
                    number += 1

        if not total == 0:
            acc = number / total
        else:
            acc = number / (total + self.MIN_COMPENSATION)

        return acc

    def aldp2(self):
        total = len(self.origin_input)
        number = 0
        dist_l2 = 0.0
        if not self.is_targeted:
            for i in range(total):
                if np.argmax(self.adv_prediction[i]) != np.argmax(self.origin_label[i]):
                    number += 1
                    dist_l2 +=(np.linalg.norm(
                        np.reshape(self.adv_input[i] - self.origin_input[i], -1), ord=2) /
                               np.linalg.norm(np.reshape(self.origin_input[i], -1), ord=2))
        else:
            for i in range(total):
                if np.argmax(self.adv_prediction[i]) == np.argmax(self.target_label[i]):
                    number += 1
                    dist_l2 +=(np.linalg.norm(
                        np.reshape(self.adv_input[i] - self.origin_input[i], -1), ord=2) /
                               np.linalg.norm(np.reshape(self.origin_input[i], -1), ord=2))

        if not total == 0:
            aldp2 = dist_l2 / number
        else:
            aldp2 = dist_l2 / (number + self.MIN_COMPENSATION)

        return aldp2

    def ass(self):
        total = len(self.adv_input)
        print("total", total)
        ori_r_channel = np.transpose(np.round(self.origin_input.numpy() * 255), (0, 2, 3, 1)).astype(dtype=np.float32)
        adv_r_channel = np.transpose(np.round(self.adv_input.numpy() * 255), (0, 2, 3, 1)).astype(dtype=np.float32)
        totalSSIM = 0
        number = 0
        outputs = torch.from_numpy(self.adv_prediction)
        preds = torch.argmax(outputs, 1)
        preds = preds.data.numpy()
        for i in range(len(preds)):
            if preds[i] != np.argmax(self.origin_label[i]):
                number += 1
                totalSSIM += SSIM(X=ori_r_channel[i], Y=adv_r_channel[i], multichannel=True)
                print(SSIM(X=ori_r_channel[i], Y=ori_r_channel[i], multichannel=True))
        if not number==0:
            ass = totalSSIM / number
        else:
            ass = totalSSIM / (number + self.MIN_COMPENSATION)
        return ass

    def nte(self):
        total = len(self.adv_input)
        print("total", total)
        outputs = torch.from_numpy(self.adv_prediction)
        number = 0
        diff = 0
        outputs_softmax=torch.nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, 1)
        outputs_softmax = outputs_softmax.data.numpy()
        preds = preds.data.numpy()

        if not self.is_targeted:
            for i in range(preds.size):
                if preds[i] != np.argmax(self.origin_label[i]):
                    number += 1
                    sort_preds = np.sort(outputs_softmax[i])
                    diff += sort_preds[-1] - sort_preds[-2]
        else:
            for i in range(preds.size):
                if preds[i] == np.argmax(self.origin_label[i]):
                    number += 1
                    sort_preds = np.sort(outputs_softmax[i])
                    diff += sort_preds[-1] - sort_preds[-2]
        if not number==0:
            nte = diff/number
        else:
            nte = diff / (number+self.MIN_COMPENSATION)
        return nte

    def evaluation(self):
        indicator = {}
        indicator['acc'] = self.acc()
        indicator['actc'] = self.actc()
        indicator['acac'] = self.acac()
        indicator['aldp2'] = self.aldp2()

        indicator = pd.DataFrame(indicator, index=[0])
        return indicator
