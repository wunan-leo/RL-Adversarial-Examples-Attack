import numpy as np
from torchvision import datasets, transforms
import math
import torch
from AttackEnv import AttackEnv
from DQN import DQNAgent

from Util.DQNConfig import Config


class Trainer:
    def __init__(self, agent: DQNAgent, env: AttackEnv, config: Config):
        self.epsilon_by_frame = None
        self.agent = agent
        self.env = env
        self.config = config
        self.output_dir = self.config.output
        # non-Linear epsilon decay

    def reset_frame(self):
        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

    def train(self, pre_fr=0):

        train_data, _ = self.load_data()
        attack_success_num = 0
        data_iter = iter(train_data)
        data, target = next(data_iter)
        state = self.env.resetEnv(data, target)
        losses = []
        all_rewards = []
        episode_reward = 0
        ep_num = 0
        self.reset_frame()
        for fr in range(pre_fr + 1, self.config.frames + 1):

            epsilon = self.epsilon_by_frame(fr)
            action = self.agent.act(state, epsilon)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            loss = 0
            if self.agent.buffer.size() > self.config.batch_size:
                loss = self.agent.learning(fr)
                losses.append(loss)

            if done:
                all_rewards.append(episode_reward)
                data, target = next(data_iter)

                # attack success.
                if episode_reward > 0:
                    attack_success_num += 1
                state = self.env.resetEnv(data, target)
                episode_reward = 0
                ep_num += 1

            if ep_num != 0 and ep_num % 250 == 0:
                print('Ran %d episodes attack, successfully attack %d times, the accuracy of it is %.4f, ran fr %d' % (
                    ep_num, attack_success_num, attack_success_num / ep_num, fr
                ))

        print('Ran %d attack episodes, and the accuracy is %.4f' % (ep_num, attack_success_num / ep_num))
        self.agent.save_model(self.output_dir, 'dqn-attack')

    def load_data(self):
        cifar10_transform = transforms.Compose([
            transforms.ToTensor(),  # numpy -> Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化 ，范围[-1,1]
        ])
        cifar10_train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data/', train=True, download=True,
                             transform=cifar10_transform))

        cifar10_test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data/', train=False, download=True,
                             transform=cifar10_transform))
        return cifar10_train_loader, cifar10_test_loader
