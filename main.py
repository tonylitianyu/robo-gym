import gym
import gym_quadrotor
import numpy as np

import os
import gc
import time
import matplotlib.pyplot as plt

env = gym.make('quad-v0')

start_new = True
if start_new:
    if os.path.exists("model/noise.p"):
        os.remove("model/memory_list.p")
        os.remove("model/noise.p")
        os.remove("model/quadrotor_actor.pkl")
        os.remove("model/quadrotor_critic.pkl")
        os.remove("model/quadrotor_target_actor.pkl")
        os.remove("model/quadrotor_target_critic.pkl")
        print('deleted to start new')

#set RL agent property
from Agent import Agent, Memory
memory_size = 1000000
single_episode_time = 300
memory = Memory(memory_size)
agent  = Agent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0], memory)
#########


# Vars Init
run = True
render = False
curr_reward = 0
episode = 0
reward_arr = []
x_axis_min = 0
goodResult = False
maxreward = -10000000

env.setrender(render)
env.setdt(0.02)

# Plot Init
fig = plt.figure(figsize=(13,6))
ax1 = fig.add_subplot(111)
line, = ax1.plot(reward_arr)
plt.xlabel("Number of episodes");plt.ylabel("Reward");
plt.grid();plt.ion();plt.show()

# while run:
while episode < 10:
    episode += 1
    curr_state = np.float32(env.reset())

    #run each episode

    for r in range(single_episode_time):
        
        if render:
            env.render()
        curr_state = np.float32(curr_state)


        if goodResult == True: #stop learning
            agent.learning_rate = 0.0
            action = agent.use_action(curr_state)
        else:
            #try the model each 10 episodes
            if episode % 10 == 0:
                action = agent.use_action(curr_state)
            else:
                action = agent.get_action(curr_state)

        n_state, done = env.step(action)#input [thrust row pitch yaw]
        reward = agent.rewardFunc(n_state)
        #reward in this episode
        curr_reward += reward

        if done: #if out of bound
            n_state = None
            curr_state = None
            break
        else:
            memory.add(curr_state,action,reward,np.float32(n_state))
            curr_state = n_state
            if (episode+1) % 1 == 0:
                agent.train()

        # print('cycle')

    #record reward for each episode
    reward_arr.append(curr_reward)

    #stop learnng condition
    if episode > 1000 and max(reward_arr) > 10000:
        goodResult = True

    # Save model every 50 episode
    # isbest = curr_reward > maxreward
    if episode % 50 == 0:
        agent.saveNetwork(True)
        memory.save()

    # Update reward plot
    if episode % 1000 == 0:
        x_axis_min = episode
    
    if episode % 1 == 0:
        plt.axis([x_axis_min, max(episode,x_axis_min+5), min(reward_arr)-100, max(reward_arr)+100])
        line.set_xdata(np.arange(len(reward_arr)))
        line.set_ydata(reward_arr)
        fig.canvas.flush_events()

    curr_reward = 0
    gc.collect()
