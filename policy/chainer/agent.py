# coding: utf-8

import chainer
import chainerrl
import numpy as np
import random

from game2048 import Game2048Env

from dqn import QFunction

class Agent(object):
    def __init__(self):
        self.env = Game2048Env(False)

        self.n_episodes = 3000

        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        q_func = QFunction(obs_size, n_actions)

        optimizer = chainer.optimizers.Adam(eps=1e-1)
        optimizer.setup(q_func) #設計したq関数の最適化にAdamを使う
        gamma = 0.95
        # explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        #     epsilon=0.5, random_action_func=env.action_space.sample)
        explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=0.6, end_epsilon=0.1, decay_steps=self.n_episodes * 100, random_action_func=self.env.action_space.sample)
        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10**6)
        phi = lambda x:x.astype(np.float32, copy=False)##型の変換(chainerはfloat32型。float64は駄目)

        self.agent = chainerrl.agents.DoubleDQN(
            q_func, optimizer, replay_buffer, gamma, explorer,
            replay_start_size=500, update_interval=1,
            target_update_interval=1000, phi=phi)


    def train(self, model_path='sample'):
        print("[Train]")

        import time
        start = time.time()
        reward_sum = 0  # return (sum of rewards)

        max_score = 0
        max_score_episode = 0
        for i in range(1, self.n_episodes + 1):
            obs = self.env.reset()
            reward = 0
            done = False
            t = 0  # time step
            while not done:
                self.env.render()
                action = self.agent.act_and_train(obs, reward)
                obs, reward, done, _ = self.env.step(action)
                reward_sum += reward
                t += 1
            if max_score < self.env.game.score:
                max_score = self.env.game.score
                max_score_episode = i

            if i % 10 == 0:
                print('episode:', i,
                      'reward_sum:', reward_sum / 10.0,
                      'statistics:', self.agent.get_statistics(),
                      'epsilon:', self.agent.explorer.epsilon,
                      'max_score:', max_score, max_score_episode)
                reward_sum = 0
            if i % 100 == 0:
                self.agent.save(model_path + '/epoch' + str(i))
            self.agent.stop_episode_and_train(obs, reward, done)
        print('Finished, elapsed time : {}'.format(time.time()-start))

        #self.agent.save(model_path)   # save model

    def test(self, model_path='sample/epoch4100'):
        print("[Test]")
        self.agent.load(model_path)   # load model

        #self.agent.explorer.epsilon = 0   # TODO is itTrue?

        self.env.game.make_window()
        while True:
            obs = self.env.reset()
            reward = 0
            done = False
            while not done:
                self.env.render()
                #action = self.agent.act_and_train(obs, reward)
                if random.random() < 0.05:
                    action = random.randint(0, 3)
                else:
                    action = self.agent.act(obs)
                obs, reward, done, _ = self.env.step(action)
            self.env.close()
