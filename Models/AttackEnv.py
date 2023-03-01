import gym
import numpy as np
from gym import spaces
import torch
from skimage.metrics import structural_similarity as SSIM

import Utils.tool as tool
from DRLBAaction import DRLBAaction
from torchvision import transforms


class AttackEnv(gym.Env):
    def __init__(self, action_num, state_dim, attacked_model, max_modify_num, alpha):

        self.state = None
        self.action_space = spaces.Discrete(action_num)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(state_dim,))
        self.origin_image = None
        self.attacked_model = attacked_model
        self.origin_label = None
        self.max_modify_num = max_modify_num
        self.current_num = 0
        self.alpha = alpha

    def resetEnv(self, origin_image, origin_label):
        # 重置state 拼接两张原始图像作为init状态
        self.origin_label = origin_label.numpy()[0]
        self.origin_image = origin_image
        self.state = np.concatenate([self.origin_image, self.origin_image], axis=0).flatten()
        self.current_num = 0
        return self.state

    def reset(self):
        pass

    def step(self, action):

        reword = 0
        done = False
        # convert state to attack image
        attack_image = self.state.reshape(1, 6, 32, 32)
        attack_image = attack_image[:, 3:, :, :]
        # base on the action to change the attack image.
        attack_action = DRLBAaction(torch.tensor(attack_image[0]))
        attack_image = attack_action.chooseAction(action)
        attack_image = attack_image.numpy().reshape(1, 3, 32, 32)
        # predict base on the model.
        #transform

        outputs = self.attacked_model(torch.tensor(attack_image))
        predict = torch.max(outputs, dim=1)[1].data.numpy()[0]

        if self.current_num > self.max_modify_num:
            reword = -1
            done = True
        elif predict != self.origin_label and \
                SSIM(self.origin_image.numpy()[0].transpose(1, 2, 0),
                     attack_image[0].transpose(1, 2, 0), multichannel=True) >= self.alpha:
            # attack works
            reword = SSIM(self.origin_image.numpy()[0].transpose(1, 2, 0), attack_image[0].transpose(1, 2, 0), multichannel=True)
            if reword > 0.9:
                tool.show_img(torch.tensor(self.origin_image.numpy()[0]), torch.tensor(attack_image[0]))
            self.current_num += 1
            done = True
        else:
            reword = 0
            self.current_num += 1
            done = False
        self.state = np.concatenate([self.origin_image, attack_image], axis=0).flatten()
        return self.state, reword, done, {}

    def seed(self, seed=None):
        pass

    def render(self, mode='human'):
        pass
