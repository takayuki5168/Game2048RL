# coding: utf-8

import gym
from gym import spaces
import random
from game2048 import Game2048
import numpy as np

class Game2048Env(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num    Observation                 Min         Max
        0      Cart Position             -4.8            4.8
        1      Cart Velocity             -Inf            Inf
        2      Pole Angle                 -24 deg        24 deg
        3      Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
         Num   Action
         0     move up
         1     move right
         2     move down
         4     move left

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    def __init__(self, render=True, debug_print=True):
        self.game = Game2048(render, debug_print)
        self.debug_print = debug_print

        self.action_space = spaces.Discrete(4)

        low = np.array([0 for i in range(16)])
        high = np.array([np.finfo(np.float32).max for i in range(16)])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def manual(self):
        self.game.loop()

    def render(self):
        self.game.draw()

    def step(self, action):
        if not action in [0, 1, 2, 3]:
            raise Exception
        ret = self.game.step(action)

        # calculate observation and done
        observation = self.transform_board(self.game.board_number)
        done = self.game.finish_flag


        # calclate indices for reward
        previous_empty = self.game.calc_empty(self.game.pre_board_number)
        current_empty = self.game.calc_empty(self.game.board_number)
        diff_empty = current_empty - previous_empty + 1   # +1 because one cell is added per step

        previous_max = self.game.calc_max(self.game.pre_board_number)
        current_max = self.game.calc_max(self.game.board_number)
        diff_max = current_max if current_max > previous_max else 0

        # calculate reward
        # TODO 1stepの平均が正になる必要あり，負だとstepが増えるにつれrewardが下がる
        # diff_maxの係数をあげたい, 64と128で報酬の差の合計があまり変わらない...

        # if ret == -1:
        #     reward = -0.00001
        # elif not done:
        #     #reward = 0.1
        #     reward = (diff_max + diff_empty) * 0.01
        # else:
        #     reward = -1.0
        if not done:
            #reward = (diff_max + diff_empty) * 0.01
            #reward = diff_max * 0.05 + diff_empty * 0.0001
            reward = self.game.score - self.game.pre_score
        else:
            #reward = -1.0
            reward = 0

        #print(reward)
        self.reward_sum += reward
        if self.debug_print and done:
            print("score: {}, max_number: {}, reward_sum: {}".format(self.game.score, 2**max([max(ns) for ns in self.game.board_number]), self.reward_sum))


        return observation, reward, done, {}

    def reset(self):
        self.game.reset()
        self.reward_sum = 0

        return self.transform_board(self.game.board_number)

    def transform_board(self, board):
        return np.array(board)
        # res_board = [[[0 for j in range(4)] for k in range(4)] for i in range(16)]
        # for i in range(4):
        #     for j in range(4):
        #         n = board[i][j]
        #         res_board[n][i][j] = 1
        # return np.array(res_board)

    def close(self):
        self.game.close()

def main():
    env = Game2048Env(render=True)
    while not env.game.finish_flag:
        a = random.randint(0, 3)
        env.render()
        env.step(a)

    env.close()

if __name__ == "__main__":
    main()
