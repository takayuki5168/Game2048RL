# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np

from game2048 import Game2048Env

env = Game2048Env(False)


class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=100):
        super(QFunction, self).__init__(##python2.x用
        #super().__init__(#python3.x用
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels,n_hidden_channels),
            l2=L.Linear(n_hidden_channels,n_hidden_channels),
            l3=L.Linear(n_hidden_channels,n_hidden_channels),
            l4=L.Linear(n_hidden_channels, n_actions))

    def __call__(self, x, test=False):
        """
        x ; 観測#ここの観測って、stateとaction両方？
        test : テストモードかどうかのフラグ
        """
        h = F.tanh(self.l0(x)) #活性化関数は自分で書くの？
        h = F.tanh(self.l1(h))
        h = F.tanh(self.l2(h))
        h = F.tanh(self.l3(h))
        return chainerrl.action_value.DiscreteActionValue(self.l4(h))

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)

optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func) #設計したq関数の最適化にAdamを使う
gamma = 0.95
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity = 10**6)
phi = lambda x:x.astype(np.float32, copy=False)##型の変換(chainerはfloat32型。float64は駄目)

agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1,
    target_update_interval=1000, phi=phi)


import time
n_episodes = 2000
start = time.time()
for i in range(1, n_episodes + 1):
    obs = env.reset()
    reward = 0
    done = False
    R = 0  # return (sum of rewards)
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
              'R:', R,
              'statistics:', agent.get_statistics())
    agent.stop_episode_and_train(obs, reward, done)
print('Finished, elapsed time : {}'.format(time.time()-start))



## test
env.make_window()
obs = env.reset()
reward = 0
done = False
while not done:
    env.render()
    action = agent.act_and_train(obs, reward)
    obs, reward, done, _ = env.step(action)
