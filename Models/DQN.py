import torch
import argparse
import random
from torch import nn
from Utils.ReplyBuffer import ReplayBuffer
from Utils.DQNConfig import Config
from torch.optim import Adam
import Utils.tool as tool
import train_DQN
from AttackEnv import AttackEnv
from lenet import LeNet


class DQN(nn.Module):
    def __init__(self, num_inputs, actions_dim):
        super(DQN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, actions_dim)
        )

    def forward(self, x):
        return self.nn(x)


class DQNAgent:
    def __init__(self, config: Config):
        self.config = config
        self.is_training = True
        self.buffer = ReplayBuffer(self.config.max_buff)
        self.model = DQN(self.config.state_dim, self.config.action_dim)
        self.model_optim = Adam(self.model.parameters(), lr=self.config.learning_rate)

        if self.config.use_cuda:
            self.cuda()

    def cuda(self):
        self.model.cuda()

    def act(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.config.epsilon_min
        if random.random() > epsilon or not self.is_training:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            if self.config.use_cuda:
                state = state.cuda()
            q_value = self.model.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.config.action_dim)
        return action

    def learning(self, fr):
        state, action, reward, next_state, done = self.buffer.sample(self.config.batch_size)

        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        if self.config.use_cuda:
            state = state.cuda()
            next_state = next_state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            done = done.cuda()

        q_values = self.model(state)
        next_q_values = self.model(state)
        next_q_value = next_q_values.max(1)[0]

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.config.gamma * next_q_value * (1 - done)
        # Notice that detach the expected_q_value
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()

        return loss.item()

    def load_weights(self, model_path):
        if model_path is None:
            return
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, output, tag=''):
        torch.save(self.model.state_dict(), '%s/DQNAgentModel_%s.pkl' % (output, tag))

    def save_config(self, output):
        with open(output + '/config.txt', 'w') as f:
            attr_val = tool.get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model', default=True)
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--model_path', type=str, help='if test, import the model')
    args = parser.parse_args()
    # dqn.py --train --env CartPole-v0

    config = Config()
    config.gamma = 0.99
    config.epsilon = 1
    config.epsilon_min = 0.01
    config.eps_decay = 500
    config.frames = 160000
    config.use_cuda = False
    config.learning_rate = 1e-3
    config.max_buff = 1000
    config.update_tar_interval = 100
    config.batch_size = 128
    config.print_interval = 200
    config.log_interval = 200
    config.win_break = True
    config.alpha = 0.5
    config.max_modify_num = 100
    config.action_dim = 14
    config.state_dim = 6 * 32 * 32
    config.output = r'./models/'

    attacked_model = LeNet(in_channels=3, out_size=2048)
    attacked_model_path = r'./models/cifar10.pth'
    attacked_model.load_state_dict(torch.load(attacked_model_path))
    env = AttackEnv(config.action_dim, config.state_dim, attacked_model, config.max_modify_num, config.alpha)
    config.action_dim = env.action_space.n
    config.state_dim = env.observation_space.shape[0]
    agent = DQNAgent(config)

    if args.train:
        trainer = train_DQN.Trainer(agent, env, config)
        trainer.train()

    # elif args.test:
    #     if args.model_path is None:
    #         print('please add the model path:', '--model_path xxxx')
    #         exit(0)
    #     tester = Tester(agent, env, args.model_path)
    #     tester.test()
