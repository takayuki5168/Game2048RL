# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np

from game2048 import Game2048Env

env = Game2048Env(False)


class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions):
        super(QFunction, self).__init__(##python2.x用
        #super().__init__(#python3.x用
            conv11 = L.Convolution2D(16, 64, ksize=(1, 2)),
            conv12 = L.Convolution2D(None, 128, ksize=(1, 2)),
            conv13 = L.Convolution2D(None, 256, ksize=(4, 2)),

            conv21 = L.Convolution2D(16, 64, ksize=(2, 1)),
            conv22 = L.Convolution2D(None, 128, ksize=(2, 1)),
            conv23 = L.Convolution2D(None, 256, ksize=(2, 4)),

            l1=L.Linear(None, 256),
            l2=L.Linear(None, 128),
            l3=L.Linear(None, n_actions))

    def __call__(self, x, test=False):
        """
        x ; 観測#ここの観測って、stateとaction両方？
        test : テストモードかどうかのフラグ
        """

        debug_print = False
        if debug_print: print(x.shape)
        h1 = F.relu(self.conv11(x))
        if debug_print: print(h1.shape)
        h1 = F.relu(self.conv12(h1))
        if debug_print: print(h1.shape)
        h1 = F.relu(self.conv13(h1))
        if debug_print: print(h1.shape)

        if debug_print: print(x.shape)
        h2 = F.relu(self.conv21(x))
        if debug_print: print(h2.shape)
        h2 = F.relu(self.conv22(h2))
        if debug_print: print(h2.shape)
        h2 = F.relu(self.conv23(h2))
        if debug_print: print(h2.shape)

        if debug_print: print("PO")

        h = F.concat((h1, h2))
        if debug_print: print(h.shape)
        h = h.reshape((-1, 512))
        if debug_print: print(h.shape)

        if debug_print: print(h.shape)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))

        return chainerrl.action_value.DiscreteActionValue(self.l3(h))

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)

optimizer = chainer.optimizers.Adam(eps=1e-1)
optimizer.setup(q_func) #設計したq関数の最適化にAdamを使う
gamma = 0.95
# explorer = chainerrl.explorers.ConstantEpsilonGreedy(
#     epsilon=0.5, random_action_func=env.action_space.sample)
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=0.9, end_epsilon=0.2, decay_steps=1000 * 100, random_action_func=env.action_space.sample)
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10**6)
phi = lambda x:x.astype(np.float32, copy=False)##型の変換(chainerはfloat32型。float64は駄目)

agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1,
    target_update_interval=1000, phi=phi)


import time
n_episodes = 1000
start = time.time()
R = 0  # return (sum of rewards)
for i in range(1, n_episodes + 1):
    obs = env.reset()
    reward = 0
    done = False
    t = 0  # time step
    while not done:
        # 動きを見たければここのコメントを外す
        env.render()
        action = agent.act_and_train(obs, reward)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
    if i % 10 == 0:
        print('episode:', i,
              'R:', R / 10.0,
              'statistics:', agent.get_statistics())
        R = 0  # return (sum of rewards)
    agent.stop_episode_and_train(obs, reward, done)
print('Finished, elapsed time : {}'.format(time.time()-start))
agent.save("hoge")


## test
print("TEST")
env.game.make_window()
while True:
    obs = env.reset()
    reward = 0
    done = False
    while not done:
        env.render()
        action = agent.act_and_train(obs, reward)
        obs, reward, done, _ = env.step(action)
    env.close()
