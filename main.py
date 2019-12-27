import gym
import gym_quadrotor
import numpy as np

import os
import gc
import matplotlib.pyplot as plt

env = gym.make('quad-v0')

#set RL agent property
from Agent import Agent, Memory
memory_size = 1000000
single_episode_time = 300
memory = Memory(memory_size)
agent = Agent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0], memory)
#########


#initialization
run = True
curr_reward = 0
episode = 0
reward_arr = []
line, = plt.plot(reward_arr)
x_axis_min = 0



while run:
    episode += 1
    curr_state = np.float32(env.reset())
    #input z row pitch yaw
    for r in range(single_episode_time):
        env.render()
        curr_state = np.float32(curr_state)

        #try the model each 10 episode
        if episode % 10 == 0:
            action = agent.use_action(curr_state)
        else:
            action = agent.get_action(curr_state)


        n_state, reward, done = env.step(action)
        #reward in this episode
        curr_reward += reward
        if done:
            n_state = None
            curr_state = None
            break
        else:
            memory.add(curr_state,action,reward,np.float32(n_state))
            curr_state = n_state
            agent.train()

    #record reward for each episode
    reward_arr.append(curr_reward)
    #save model each 50 episode
    if episode % 50 == 0:
        agent.saveNetwork()
        memory.save()
    #reward plot
    if episode % 1000 == 0:
        x_axis_min = episode

    plt.axis([x_axis_min, episode, min(reward_arr), max(reward_arr)])
    line.set_xdata(np.arange(len(reward_arr)))
    line.set_ydata(reward_arr)
    plt.draw()
    plt.pause(0.01)
    curr_reward = 0

    gc.collect()
