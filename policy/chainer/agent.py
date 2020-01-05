# coding: utf-8

import chainer
import chainerrl
import numpy as np

from game2048 import Game2048Env

from dqn import QFunction

class Agent(object):
    def __init__(self):
        self.env = Game2048Env(False)

        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        q_func = QFunction(obs_size, n_actions)

        optimizer = chainer.optimizers.Adam(eps=1e-1)
        optimizer.setup(q_func) #設計したq関数の最適化にAdamを使う
        gamma = 0.95
        # explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        #     epsilon=0.5, random_action_func=env.action_space.sample)
        explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon=0.9, end_epsilon=0.2, decay_steps=1000 * 100, random_action_func=self.env.action_space.sample)
        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10**6)
        phi = lambda x:x.astype(np.float32, copy=False)##型の変換(chainerはfloat32型。float64は駄目)

        self.agent = chainerrl.agents.DoubleDQN(
            q_func, optimizer, replay_buffer, gamma, explorer,
            replay_start_size=500, update_interval=1,
            target_update_interval=1000, phi=phi)

    def train(self, model_path='sample'):
        print("[Train]")
        import time
        n_episodes = 1000
        start = time.time()
        R = 0  # return (sum of rewards)
        for i in range(1, n_episodes + 1):
            obs = self.env.reset()
            reward = 0
            done = False
            t = 0  # time step
            while not done:
                # 動きを見たければここのコメントを外す
                self.env.render()
                action = self.agent.act_and_train(obs, reward)
                obs, reward, done, _ = self.env.step(action)
                R += reward
                t += 1
            if i % 10 == 0:
                print('episode:', i,
                      'R:', R / 10.0,
                      'statistics:', self.agent.get_statistics(),
                      'epsilon:', self.agent.explorer.epsilon)
                R = 0  # return (sum of rewards)
            self.agent.stop_episode_and_train(obs, reward, done)
        print('Finished, elapsed time : {}'.format(time.time()-start))

        self.agent.save(model_path)   # save model

    def test(self, model_path='sample'):
        print("[Test]")
        self.agent.load(model_path)   # load model

        self.agent.explorer.epsilon = 0   # TODO is itTrue?

        self.env.game.make_window()
        while True:
            obs = self.env.reset()
            reward = 0
            done = False
            while not done:
                self.env.render()
                action = self.agent.act_and_train(obs, reward)
                obs, reward, done, _ = self.env.step(action)
            self.env.close()
