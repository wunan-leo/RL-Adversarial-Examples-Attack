#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   L2_Attack.py    
@Contact :   1421877537@qq.com
@License :   (C)Copyright 2017-2018
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/5/18 2:17    leoy        1.0        C&W L2 algorithm
"""

# import lib
import torch
from torch.autograd import Variable
import numpy as np

# define the argument variable.
BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-3     # the initial constant c to pick as a first guess

class CarliniL2:

    def __init__(self, model, shape,
                 batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS,
                 max_iterations=MAX_ITERATIONS, abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST, is_cuda=True):
        """
        The L_2 optimized attack.
        This attack is the most efficient and should be used as the primary attack to evaluate potential defenses.
        Returns adversarial examples for the supplied model.

        :param model: Input the attacked model.
        :param shape: The shape of model. accept the shape like [Channel, Height, Weight]
        :param batch_size: Number of attacks to run simultaneously.
        :param confidence: Confidence of adversarial examples: higher produces examples
           that are farther away, but more strongly classified as adversarial.
        :param targeted: True if we should perform a targeted attack, False otherwise.
        :param learning_rate: The learning rate for the attack algorithm. Smaller values
           produce better results but are slower to converge.
        :param binary_search_steps: The number of times we perform binary search to
           find the optimal tradeoff-constant between distance and confidence.
        :param max_iterations: The maximum number of iterations. Larger values are more
           accurate; setting too small will require a large learning rate and will
           produce poor results.
        :param abort_early: If true, allows early aborts if gradient descent gets stuck.
        :param initial_const: The initial tradeoff-constant to use to tune the relative
           importance of distance and confidence. If binary_search_steps is large,
           the initial constant is not important.
        :return: adversarial examples for the supplied model.
        """
        self.model = model
        self.shape = [batch_size]
        self.shape.extend(shape)
        self.CONFIDENCE = confidence
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const
        self.is_cuda = is_cuda
        self.repeat = binary_search_steps >= 10
        self.batch_size = batch_size

    def attack(self, images, targets) -> np.array:
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.

        :param images: input images information.
        :param targets: the target class for images. list of "0" or "1"  "1" response the target.
        :return: adv example for images.
        """
        result = []
        # Pytorch input image 2D: N × C × H × W, we need transpose the original images.
        images = np.transpose(images, [0, 3, 1, 2])
        for i in range(0, len(images), self.batch_size):
            batch_image = images[i: i + self.batch_size]
            batch_target = targets[i: i + self.batch_size]
            print("Run attack_batch: {0}".format(i))
            result.extend(self.attack_batch(batch_image, batch_target))
        return np.transpose(np.array(result), [0, 2, 3, 1])

    def attack_batch(self, images, targets) -> list:
        """
        Run the attack on a batch of images and labels.

        :param images: input images information of one batch.
        :param targets:  the target class for images of one batch.
        :return: adv example for images of one batch.
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

        # prepare the hyper parameters "W" for the tanh space transformation.
        # image The original range was divided to [-0.5, 0.5].
        hyper_factor = 1.99999
        images = np.arctanh(images * hyper_factor)

        # Prepare the upper and lower bounds of the dichotomous search in advance.
        lower_bound = np.zeros(self.batch_size)
        upper_bound = np.ones(self.batch_size) * 1e10
        CONST = np.ones(self.batch_size) * self.INITIAL_CONST

        # Prepare in advance the best data(L2, score, attack) information obtained during the iterative process
        outer_best_l2_distance = [1e10] * self.batch_size
        outer_best_score = [-1] * self.batch_size
        outer_best_attack = [np.zeros(images[0].shape)] * self.batch_size

        # Search for the best results under the restricted dichotomous search CONST.
        for binary_search_step in range(self.BINARY_SEARCH_STEPS):

            # completely reset adam's internal state.
            batch_image = images[: self.batch_size]
            batch_target = targets[: self.batch_size]
            best_l2_distance = [1e10] * self.batch_size
            best_score = [-1] * self.batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and binary_search_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            # the variable we're going to optimize over in pytorch
            if self.is_cuda:
                modifier = Variable(torch.zeros(*self.shape).cuda(), requires_grad=True)
                self.tensor_images = Variable(torch.FloatTensor(batch_image).cuda(), requires_grad=True)
                self.tensor_target = Variable(torch.FloatTensor(batch_target).cuda(), requires_grad=True)
                self.tensor_const = Variable(torch.FloatTensor(CONST).cuda(), requires_grad=True)
                self.zero = Variable(torch.zeros(1).cuda(), requires_grad=True)
            else:
                modifier = Variable(torch.zeros(*self.shape), requires_grad=True)
                self.tensor_images = Variable(torch.FloatTensor(batch_image), requires_grad=True)
                self.tensor_target = Variable(torch.FloatTensor(batch_target), requires_grad=True)
                self.tensor_const = Variable(torch.FloatTensor(CONST), requires_grad=True)
                self.zero = Variable(torch.zeros(1), requires_grad=True)

            # Setup the adam optimizer and keep track of variables we're creating
            optimizer = torch.optim.Adam(
                [{'params': modifier}], lr=self.LEARNING_RATE)

            # Find the best result each time within the limit of the maximum number of iterations.

            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                optimizer.zero_grad()

                # use the hyper parameters image to generate new image which in tanh space (-0.5,0.5)
                self.newImages = torch.tanh(modifier + self.tensor_images) / 2.0
                # prediction BEFORE-SOFTMAX of the model
                self.output = self.model(self.newImages)

                # compute the loss of attack.
                # First, we could compute the l2 distance.
                # different channel  -> dim = 1
                self.l2_distance = torch.sum(torch.pow(self.newImages - torch.tanh(self.tensor_images) / 2.0, 2), dim=1)
                # different images  -> height × weight
                self.l2_distance = torch.sum(torch.sum(self.l2_distance, dim=1), dim=1)

                # Second, we could compute f(x')
                real = torch.sum(self.tensor_target * self.output, 1)
                other = torch.max((1 - self.tensor_target) * self.output - (self.tensor_target * 10000), dim=1)

                # whether the attack is targeted.
                if self.TARGETED:
                    loss_target = torch.max(real - other + self.CONFIDENCE, self.zero)
                else:
                    loss_target = torch.max(other - real + self.CONFIDENCE, self.zero)

                self.loss2 = torch.sum(self.l2_distance)
                self.loss1 = torch.sum(self.tensor_const * loss_target)
                self.loss = self.loss1 + self.loss2

                self.loss.backward()
                optimizer.step()

                current_loss = self.loss.cpu().data[0]
                current_l2_distance = self.l2_distance.cpu().data[0]
                current_score = self.output.cpu().data.numpy()
                current_adv_image = self.newImages.cpu().data.numpy()

                # print out the losses every 10%
                if iteration % (self.MAX_ITERATIONS // 10) == 0:
                    print('iter: {} l: {} l1: {} l2: {}'.format(iteration, current_loss,
                                                                self.loss1.cpu().data[0], self.loss2.data[0]))

                # adjust the best result found so far
                for e, (l2, sc, adv_img) in enumerate(zip(current_l2_distance, current_score, current_adv_image)):
                    # if current l2 distance less than best distance in this iteration and the attack is successful
                    if l2 < best_l2_distance[e] and compare(sc, np.argmax(batch_target[e])):
                        best_l2_distance[e] = l2
                        best_score[e] = np.argmax(sc)
                    if l2 < outer_best_l2_distance[e] and compare(sc, np.argmax(batch_target[e])):
                        outer_best_l2_distance[e] = l2
                        outer_best_score[e] = np.argmax(sc)
                        outer_best_attack[e] = adv_img

                # adjust the constant as needed
                for e in range(self.batch_size):
                    if compare(best_score[e], np.argmax(batch_target[e])) and best_score[e] != -1:
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
        outer_best_l2_distance = np.array(outer_best_l2_distance)
        return outer_best_attack
