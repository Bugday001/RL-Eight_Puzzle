# coding = utf-8

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from EightPuzzleEnv import EightPuzzleEnv
from test_net12_13 import DuelingDqn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
Batch_size = 64
# Lr = 0.002
episodes = 5000
Epsilon = 0.4  # greedy policy
delta_ep = 3.5e-4  # 1000个episode，每次给贪婪加delta_ep
Gamma = 0.9  # reward discount
Target_replace_iter = 300  # target update frequency
Memory_capacity = 5000
env = EightPuzzleEnv(3, 3)  # (2, 3)
# env = env.unwrapped
N_actions = env.ActionDim
N_states = env.m * env.n
difficulty_steps = 12
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
last_action = 9


class DQN(object):
    def __init__(self, lr=1e-4):
        self.eval_net, self.target_net = DuelingDqn(N_states, N_actions).to(device) \
            , DuelingDqn(N_states, N_actions).to(device)
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((Memory_capacity, N_states * 2 + 2))  # initialize memory
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0)).to(device)
        if np.random.uniform() < Epsilon:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.cpu().numpy()
            action = action[0]  # if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:
            action = np.random.randint(0, N_actions)
            action = action  # if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))  # 竖向堆叠, shape: (10,)
        index = self.memory_counter % Memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net update
        if self.learn_step_counter % Target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(Memory_capacity, Batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_states])).to(device)
        b_a = Variable(torch.LongTensor(b_memory[:, N_states:N_states + 1].astype(int))).to(device)
        b_r = Variable(torch.FloatTensor(b_memory[:, N_states + 1:N_states + 2])).to(device)
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_states:])).to(device)

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + Gamma * q_next.max(1)[0].view(Batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, section):
        torch.save({'model': self.eval_net.state_dict()}, r'torch_models\3last_step{}.pth'.format(section))
        # torch.save({'model': self.target_net.state_dict()}, r'torch_models\dueling_target2x3{}.pth')

    def load_model(self, section=1):
        state_dict = torch.load(r'torch_models\3last_step{}.pth'.format(section))
        self.eval_net.load_state_dict(state_dict['model'])

    def continue_train(self, section):
        state_dict = torch.load(r'torch_models\3last_step11.pth')
        self.eval_net.load_state_dict(state_dict['model'])
        self.target_net.load_state_dict(state_dict['model'])

    def predict(self, x):
        global last_action
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0)).to(device)
        action_value = self.eval_net.forward(x).detach()
        action = torch.max(action_value, 1)[1].data.cpu().numpy()
        action = action[0]

        # if np.random.uniform() < Epsilon:
        #     action_value = self.eval_net.forward(x)
        #     action = torch.max(action_value, 1)[1].data.cpu().numpy()
        #     action = action[0]  # if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        # else:
        #     action = np.random.randint(0, N_actions)
        #     action = action

        return action


def each_train(dqn, env, episode, section="0"):
    global Epsilon# , delta_ep
    print('\nCollecting experience...')
    delta_ep = (0.8 - Epsilon) / episodes  # 1000个episode，每次给贪婪加delta_ep
    count = 0
    for i_episode in range(episode):
        s = env.reset3(difficulty_steps)
        s = s.flatten()  # 拉平
        step_inner = 0
        while True:
            # env.render()
            ep_r = 0
            a = dqn.choose_action(s)

            # take action
            s_, r, done, info = env.step(a)
            s_ = s_.flatten()  # 拉平
            # 存记忆, state, action, reward, next_state
            dqn.store_transition(s, a, r, s_)
            ep_r += r
            if dqn.memory_counter > Memory_capacity:
                dqn.learn()
                # print("train")
                if done: #and step_inner % 50 == 0:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2), "Epsilon:", Epsilon, end=",,")
                    print(dqn.optimizer.param_groups[0]['lr'])
                step_inner += 1
            if done:
                count += 1
            if done or env.max_episode_steps < step_inner:
                break
            s = s_
        Epsilon += delta_ep
    print("success:", count/episode)
    dqn.save_model(section)


def train():
    global Epsilon
    dqn = DQN(1e-6)  # 0.0005 warm up????
    dqn.continue_train(0)
    Epsilon = 0.2
    each_train(dqn, env, episodes, "-2")
    test(-2)
    dqn.optimizer.param_groups[0]['lr'] = 0.00005
    Epsilon = 0.3
    each_train(dqn, env, episodes, "-1")
    test(-1)
    dqn.optimizer.param_groups[0]['lr'] = 1e-5
    Epsilon = 0.4
    each_train(dqn, env, episodes, "0")
    test(0)
    dqn.optimizer.param_groups[0]['lr'] = 1e-6
    Epsilon = 0.5
    each_train(dqn, env, episodes, "111")
    env.close()
    test(111)


def test(section=1):
    print("start test!")
    agent = DQN()
    agent.load_model(section)
    count = 0
    N = 500
    for i in range(N):
        step = 0
        s = env.reset3(difficulty_steps)
        s = s.flatten()  # 拉平
        while True:
            # env.render()
            a = agent.predict(s)
            # take action
            s_, r, done, info = env.step(a)
            step += 1
            s_ = s_.flatten()  # 拉平
            # 存记忆, state, action, reward, next_state
            if done:
                s = s_
                count += 1
                # env.render()
                break
            elif step > difficulty_steps+1:
                break
            s = s_
    print("success:", count/N)


if __name__ == "__main__":
    difficulty_steps = 12  # best 1111 10
    print("train or test:")
    mode = input()
    if mode == "train":
        train()
    elif mode == "test":
        test(111)  # 8是11
    else:
        print("error input")
