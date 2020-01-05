# coding: utf-8

import random

from game2048 import Game2048Env

env = Game2048Env(render=True)
env.reset()
while not env.game.finish_flag:
    a = random.randint(0, 3)
    env.render()
    env.step(a)

env.close()
