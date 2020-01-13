# coding: utf-8

import copy
import time
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import wrappers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from collections import namedtuple, deque

from game2048 import Game2048Env

class ReplayMemory(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = [] #deque()
        self.td_errors = []

    def append(self, data, td_error):
        self.memory.append(data)

        self.td_errors.append(1)
        # self.td_errors.append(td_error)

        if len(self.memory) > self.memory_size:
            self.memory = self.memory[1:]
            self.td_errors = self.td_errors[1:]
            #self.memory.popleft()

    def sample(self, batch_size):
        #return np.random.choice(self,memory, batch_size, p=[i / sum(self.td_errors) for i in self.td_errors], repeat=False)
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

    def __call__(self):
        return np.random.permutation(self.memory)

class QNetwork(nn.Module):
    def __init__(self, obs_num, acts_num):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_num, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, acts_num)

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        y = F.relu(self.fc4(h))
        return y

class Agent(object):
    def __init__(self, env=gym.make('CartPole-v0'),
                 episode_num=10000, step_num=10000, memory_size=10**6, batch_size=1000,   # TODO cannot change batch_size
                 epsilon=1.0, epsilon_decrease=(1.0 - 0.2)/3000/300, epsilon_min=0.2, start_reduce_epsilon_step=200,
                 train_freq=10, update_target_q_freq=20,
                 gamma=0.97, log_freq=100):
        # variable
        self.env = env

        self.episode_num = episode_num
        self.step_num = step_num   # number of step
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.epsilon = epsilon
        self.epsilon_decrease = epsilon_decrease
        self.epsilon_min = epsilon_min
        self.start_reduce_epsilon_step = start_reduce_epsilon_step

        self.train_freq = train_freq   # frequency of training
        self.update_target_q_freq = update_target_q_freq   # frequency of updating target network

        self.gamma = gamma   # discount rate
        self.log_freq = log_freq  # logging rate for print


        # hoge
        self.obs_num = self.env.observation_space.shape[0]
        self.acts_num = self.env.action_space.n


        # model
        self.Q = QNetwork(self.obs_num, self.acts_num)   # QNetwork
        self.Q_target = copy.deepcopy(self.Q)

        # optimizer
        self.optimizer = optim.RMSprop(self.Q.parameters(), lr=0.00015, alpha=0.95, eps=0.01)

        self.memory = ReplayMemory(memory_size)
        self.total_rewards = []   # for recording

        self.log_dir = "log/test1"

    def train(self):
        print("\t".join(map(str,["episode", "epsilon", "reward", "max_score", "max_score_episode", "total_step", "time"]))) # print in stdcout
        start_time = time.time()
        total_step = 0

        max_score = 0
        max_score_episode = 0
        for episode in range(self.episode_num):
            pobs = self.env.reset() # init environment
            step = 0 # current step
            done = False # flag of finishing current step
            total_reward = 0 # accumulated reward

            while not done: # and step < self.step_num:
                self.env.render()

                # select action with epsilon greedy method
                if np.random.rand() > self.epsilon:   # calculate optimize action
                    pobs_ = np.array(pobs, dtype="float32").reshape((1, self.obs_num))
                    pobs_ = Variable(torch.from_numpy(pobs_))
                    pact = self.Q(pobs_)
                    maxs, indices = torch.max(pact.data, 1)
                    pact = indices.numpy()[0]
                else:   # random sample
                    pact = self.env.action_space.sample()

                # act in environment
                obs, reward, done, _ = self.env.step(pact)
                #if done:
                #reward = -1


                # calculate TD error
                pobss = np.array(obs.tolist(), dtype="float32").reshape((1, self.obs_num))
                q = self.Q(Variable(torch.from_numpy(pobss)))
                maxs, indices = torch.max(q.data, 1)
                obss = np.array(obs.tolist(), dtype="float32").reshape((1, self.obs_num))
                maxqs = self.Q_target(Variable(torch.from_numpy(obss))).data.numpy() # ここからindiciesの行動の評価値で更新する
                target = copy.deepcopy(q.data.numpy())
                pacts = np.array(pact, dtype="int32")
                td_error = abs(q[0][pacts].data.numpy() - self.gamma * maxqs[0, indices.numpy()[0]])
                #print(td_error)
                #print(q[0][pacts].data.numpy()[0] - reward + self.gamma * maxqs[indices.numpy()[0]] * (not done))

                # memorize
                #self.memory.append((pobs, pact, reward, obs, done)) # state_{t}, action_{t}, reward_{t}, state_{t+1}, done flag
                self.memory.append((pobs, pact, reward, obs, done), td_error) # state_{t}, action_{t}, state_{t+1}, reward_{t}

                #print(q[0][pacts], indices.numpy()[0])
                #print(reward + self.gamma * maxqs[indices.numpy()[0]] * (not done))




                # train Q network if memory size is enough
                if len(self.memory) == self.memory_size:
                    # 経験リプレイ
                    if total_step % self.train_freq == 0:
                        #memory_ = np.random.permutation(self.memory)
                        memory_ = self.memory.memory   # self.memory()
                        memory_idxs = range(len(memory_))
                        for i in memory_idxs[::self.batch_size]:
                            batch = np.array(memory_[i:i + self.batch_size]) # 経験ミニバッチ
                            pobss = np.array(batch[:,0].tolist(), dtype="float32").reshape((self.batch_size, self.obs_num))
                            pacts = np.array(batch[:,1].tolist(), dtype="int32")
                            rewards = np.array(batch[:,2].tolist(), dtype="int32")
                            obss = np.array(batch[:,3].tolist(), dtype="float32").reshape((self.batch_size, self.obs_num))
                            dones = np.array(batch[:,4].tolist(), dtype="bool")

                            # set y_doubleq
                            pobss_ = Variable(torch.from_numpy(pobss))
                            q = self.Q(pobss_)
                            maxs, indices = torch.max(q.data, 1)
                            indices = indices.numpy()
                            obss_ = Variable(torch.from_numpy(obss))

                            maxqs = self.Q_target(obss_).data.numpy() # ここからindiciesの行動の評価値で更新する
                            target = copy.deepcopy(q.data.numpy())
                            for j in range(self.batch_size):
                                target[j, pacts[j]] = rewards[j] + self.gamma * maxqs[j, indices[j]] * (not dones[j]) # 教師信号

                            # Perform a gradient descent step
                            self.optimizer.zero_grad()
                            loss = nn.MSELoss()(q, Variable(torch.from_numpy(target)))
                            loss.backward()
                            self.optimizer.step()
                    # update Q network
                    if total_step % self.update_target_q_freq == 0:
                        self.Q_target = copy.deepcopy(self.Q)
                # decrease epsilon
                if self.epsilon > self.epsilon_min and total_step > self.start_reduce_epsilon_step:
                    self.epsilon -= self.epsilon_decrease
                # next action
                total_reward += reward
                step += 1
                total_step += 1
                pobs = obs

            # memorize accumulated reward
            self.total_rewards.append(total_reward)

            # calculate max score
            if max_score < self.env.game.score:
                max_score = self.env.game.score
                max_score_episode = episode
                self.env.game.save(self.log_dir + "/board/" + str(episode) + "_" + str(max_score) + ".jpg")

            if (episode + 1) % self.log_freq == 0:
                reward = sum(self.total_rewards[((episode + 1) - self.log_freq):(episode + 1)]) / self.log_freq
                elapsed_time = time.time() - start_time
                #print("\t".join(map(str,["episode", "epsilon", "reward", "max score", "total_step", "time"]))) # print in stdcout
                print("\t".join(map(str,[episode + 1, self.epsilon, reward, max_score, max_score_episode, total_step, str(elapsed_time) + "[sec]"]))) # print in stdcout
                start_time = time.time()

if __name__ == '__main__':
    agent = Agent(Game2048Env(render=False, debug_print=False))
    agent.train()

    plt.figure(figsize=(15,7))
    resize = (len(agent.total_rewards) // 10, 10)
    tmp = np.array(agent.total_rewards, dtype="float32").reshape(resize)
    tmp = np.average(tmp, axis=1)
    plt.plot(tmp, color="cyan")
    plt.show()
