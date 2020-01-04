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
            conv1 = L.Convolution2D(None, 128, ksize=(1, 2)),
            conv2 = L.Convolution2D(None, 128, ksize=(2, 1)),
            conv3 = L.Convolution2D(None, 128, ksize=(2, 2)),

            l1=L.Linear(None, 100),
            l2=L.Linear(None, 100),
            l3=L.Linear(None, n_actions))

    def __call__(self, x, test=False):
        """
        x ; 観測#ここの観測って、stateとaction両方？
        test : テストモードかどうかのフラグ
        """
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(x))
        #pool1 = F.max_pooling_2d(conv1_2, ksize=2, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))

        return chainerrl.action_value.DiscreteActionValue(self.l3(h))

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)

optimizer = chainer.optimizers.Adam(eps=1e-1)
optimizer.setup(q_func) #設計したq関数の最適化にAdamを使う
gamma = 0.95
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.5, random_action_func=env.action_space.sample)
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity = 6000) #10**6)
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
