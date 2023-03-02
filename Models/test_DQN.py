from torchvision import datasets, transforms
import torch

class Tester(object):

    def __init__(self, agent, env, model_path, test_image_num = 5000):

        self.agent = agent
        self.env = env
        self.agent.is_training = False
        self.agent.load_weights(model_path)
        self.policy = lambda x: agent.act(x)

        self.test_image_num = test_image_num


    def test(self, debug=False, visualize=False):

        attack_num = 0
        attack_success_num = 0
        total_reward = 0
        dataLoader = self.load_data()
        iter_test_data = iter(dataLoader)
        data, target = next(iter_test_data)
        state = self.env.resetEnv(data, target)
        while True:
            if visualize:
                self.env.render()
            action = self.policy(state)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done:
                attack_num += 1
                if total_reward > 0 :
                    attack_success_num += 1    
                data, target = next(iter_test_data)
                state = self.env.resetEnv(data, target)
                total_reward = 0
                if attack_num == self.test_image_num:
                    print("ran %d times attack, successfully attack %d times, accuracy of it is %.4f" % (
                        attack_num, attack_success_num, attack_success_num / attack_num
                    ))
                    break
    
            
    def load_data(self):
        cifar10_transform = transforms.Compose([
            transforms.ToTensor(),  # numpy -> Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化 ，范围[-1,1]
        ])
        cifar10_test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data/', train=False, download=True,
                             transform=cifar10_transform))
        return cifar10_test_loader
