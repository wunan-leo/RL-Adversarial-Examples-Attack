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

    def evaluation(self):
        indicator = {}
        indicator['acc'] = self.acc()
        indicator['actc'] = self.actc()
        indicator['acac'] = self.acac()
        indicator['aldp2'] = self.aldp2()

        indicator = pd.DataFrame(indicator, index=[0])
        return indicator
