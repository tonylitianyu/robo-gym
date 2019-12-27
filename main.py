import gym
import gym_quadrotor
import numpy as np

import os
import gc
import matplotlib.pyplot as plt





env = gym.make('quad-v0')

#initialization
run = True
curr_reward = 0
episode = 0
reward_arr = []
line, = plt.plot(reward_arr)
x_axis_min = 0



while run:
    episode += 1
    #input z row pitch yaw
    state, reward, done = env.step(np.array([39.2,0.001,0,0]))
    env.render()
