#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   CarliniL2.py
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/5/18 2:17    leoy        1.0        C&W L2 algorithm
"""

import sys
import torch
from torch.autograd import Variable
import numpy as np

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-3     # the initial constant c to pick as a first guess


class CarliniL2:
    def __init__(self, model, shape, class_num=10, confidence=CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS,
                 abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST, is_cuda=False):

        """
        The L_2 optimized attack.

        This attack is the most efficient and should be used as the primary
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targeted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        """
        self.model = model
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = shape[0]

        self.repeat = binary_search_steps >= 10

        self.shape = shape
        self.is_cuda = is_cuda

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to', len(imgs))
        # transpose the image.
        imgs = np.transpose(imgs, [0, 3, 1, 2])
        for i in range(0, len(imgs), self.batch_size):
            print('tick', i)
            r.extend(self.attack_batch(
                imgs[i:i + self.batch_size], targets[i:i + self.batch_size]))
        return np.transpose(np.array(r), [0, 2, 3, 1])

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                x[y] -= self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        # convert to tanh-space
        # ? 为什么是这样的转换方式
        imgs = np.arctanh(imgs * 1.999999)

        # 为batch中的每一个都设立了CONST,二分法进行搜索
        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # 预先定义了最佳的L2，最佳分数，最佳的攻击样本
        # the best l2, score, and image attack
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [np.zeros(imgs[0].shape)] * batch_size

        # 在有限次数内进行二分搜索最佳的CONST
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print(o_bestl2)
            # completely reset adam's internal state.

            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size

            # The last iteration (if we run many steps) repeat the search once.
            # 最后一次了，直接把Const设置为upper_bound
            # 论文中说它通过二分法找一下，满足要求的最小的CONST，如果此刻还不行，直接CONST = upper_bound了
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            # the variable we're going to optimize over

            ############### pytroch #################
            # these are the variables to initialize when we run
            if self.is_cuda:
                modifier = Variable(torch.zeros(
                    *self.shape).cuda(), requires_grad=True)
                self.timg = Variable(torch.FloatTensor(
                    batch).cuda(), requires_grad=True)
                self.tlab = Variable(torch.FloatTensor(
                    batchlab).cuda(), requires_grad=True)
                self.const = Variable(torch.FloatTensor(
                    CONST).cuda(), requires_grad=True)
                self.zero = Variable(torch.zeros(1).cuda(), requires_grad=True)
            else:
                modifier = Variable(torch.zeros(*self.shape), requires_grad=True)
                self.timg = Variable(
                    torch.FloatTensor(batch), requires_grad=True)
                self.tlab = Variable(torch.FloatTensor(
                    batchlab), requires_grad=True)
                self.const = Variable(
                    torch.FloatTensor(CONST), requires_grad=True)
                self.zero = Variable(torch.zeros(1), requires_grad=True)

            # Setup the adam optimizer and keep track of variables we're creating
            optimizer = torch.optim.Adam(
                [{'params': modifier}], lr=self.LEARNING_RATE)

            ##########################################################

            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                ################## pytorch ###################
                optimizer.zero_grad()
                # the resulting image, tanh'd to keep bounded from -0.5 to 0.5

                self.newimg = torch.tanh(modifier + self.timg) / 2.0
                # print(self.newimg)
                # print(self.shape)

                # prediction BEFORE-SOFTMAX of the model
                self.output = self.model(self.newimg)

                # distance to the input data
                self.l2dist = torch.sum(
                    torch.pow(self.newimg - torch.tanh(self.timg) / 2.0, 2), dim=1)

                self.l2dist = torch.sum(torch.sum(self.l2dist, dim=1), dim=1)

                # compute the probability of the label class versus the maximum other
                real = torch.sum((self.tlab) * self.output, 1)
                other = torch.max((1 - self.tlab) *
                                  self.output - (self.tlab * 10000), dim=1)[0]

                if self.TARGETED:
                    # if targetted, optimize for making the other class most likely
                    loss1 = torch.max(
                        other - real + self.CONFIDENCE, self.zero)
                else:
                    # if untargeted, optimize for making this class least likely.
                    loss1 = torch.max(
                        real - other + self.CONFIDENCE, self.zero)

                # sum up the losses
                self.loss2 = torch.sum(self.l2dist)
                self.loss1 = torch.sum(loss1 * self.const)
                self.loss = self.loss1 + self.loss2

                self.loss.backward()
                optimizer.step()
                l = self.loss.cpu().item()
                l2s = self.l2dist.cpu().data.numpy()
                scores = self.output.cpu().data.numpy()
                nimg = self.newimg.cpu().data.numpy()
                ###################################################

                # print out the losses every 10%
                if iteration % (self.MAX_ITERATIONS // 10) == 0:
                    print('iter: {} l: {} l1: {} l2: {}'.format(iteration,
                                                                l, self.loss1.cpu().item(), self.loss2.item()))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS // 10) == 0:
                    if l > prev * .9999:
                        break
                    prev = l

                # adjust the best result found so far
                for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
                    if l2 < bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack
