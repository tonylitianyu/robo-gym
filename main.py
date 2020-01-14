import gym
import gym_quadrotor
import numpy as np

import os
import gc
import time
import math
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

def compute_td_err(agent, s, a, r, s1):
		a  = Variable(torch.from_numpy(a))
		s  = Variable(torch.from_numpy(s))
		s1 = Variable(torch.from_numpy(s1))

		if agent.gpuid >= 0:
			s  = s.cuda()
			a  = a.cuda()
			s1 = s1.cuda()
		
		a1 = agent.target_actor.forward(s1).detach()
		next_val = torch.squeeze(agent.target_critic.forward(s1, a1).detach())
		td_err = r + agent.gamma*next_val.cpu().numpy() - torch.squeeze(agent.critic.forward(s, a).detach()).cpu().numpy()

		return td_err

env = gym.make('quad-v0')
gpuid = -1
if gpuid >= 0:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

start_new = True
if start_new:
    if os.path.exists("model/memory_list.p"):
        os.remove("model/memory_list.p")
		# os.remove("model/noise.p")
        os.remove("model/quadrotor_actor.pkl")
        os.remove("model/quadrotor_critic.pkl")
        os.remove("model/quadrotor_target_actor.pkl")
        os.remove("model/quadrotor_target_critic.pkl")
        print('deleted to start new')

#set RL agent property
from Agent import Agent, Memory
memory_size = 100000 # 200 for each episode, 400k = 2k episodes
single_episode_time = 300
memory = Memory(memory_size)
agent  = Agent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0],\
               memory, gpuid)
#########


# Vars Init
run = True
render = True
curr_reward = 0
episode = 0
reward_arr,baseline_arr = [0.0],[0.0]
x_axis_min = 0
goodResult = False
nonoise = False
maxreward = -10000000

env.setrender(render)
env.setdt(0.02)

# Plot Init
fig = plt.figure(figsize=(18,5))
ax1 = fig.add_subplot(111)
line1, = ax1.plot(reward_arr)
line2, = ax1.plot(baseline_arr)
plt.xlabel("Number of episodes");plt.ylabel("Step Average Reward")
plt.grid();plt.ion();plt.show()

while run:
# while episode < 3000:
    episode += 1
    curr_state = np.float32(env.reset())
    agent.noiseMachine.resetNoise()

	# Set Learning rate by cosine annealing
    lr_min = 0.00001
    lr_max = 0.0001
    agent.learning_rate_a = lr_min + 0.5*(lr_max-lr_min)*(1+math.cos((episode%25)/25*math.pi))
    agent.learning_rate_c = 10*agent.learning_rate_a

    #run each episode
    for r in range(single_episode_time):
        
        if render:
            env.render()
        curr_state = np.float32(curr_state)

        
        if goodResult == True: #stop learning
            agent.learning_rate_a = 0.0
            action = agent.use_action(curr_state)
        else:
            #try the model each 10 episodes
            if episode % 10 == 0:
                action = agent.use_action(curr_state)
                nonoise = True
            else:
                action = agent.get_action(curr_state)
                nonoise = False

        n_state, done = env.step(action)#input [thrust row pitch yaw]
        reward = agent.rewardFunc(n_state,action)
        #reward in this episode
        curr_reward += reward

        if done: #if out of bound
            n_state = None
            curr_state = None
            break
        else:
            # td_err = compute_td_err(agent, curr_state,action,reward,np.float32(n_state))
            # print(td_err)
            memory.add(curr_state,action,reward,np.float32(n_state))
            curr_state = n_state

        agent.train(((r+1) % 3 == 0))

    # if (episode+1) % 1 == 0:
        

        # print('cycle')

    #record reward for each episode
    reward_arr.append(curr_reward/r)
    if nonoise:
        baseline_arr.append(curr_reward/r)
        
    #stop learnng condition
    if episode > 1000 and max(reward_arr) > 10000:
        goodResult = True

    # Save model every 50 episode
    # isbest = curr_reward > maxreward
    if episode % 50 == 0:
        # agent.learning_rate_a *= 0.9
        agent.saveNetwork(True)
        memory.save()

    # Update reward plot
    if episode % 1000 == 0:
        x_axis_min = episode

    if episode % 250 == 0:
        plt.savefig('overnight_res.png')
    # print("Current Episode:%4.0d Learning Actor Rate: %7.5f " %(episode,agent.learning_rate_a))
    if episode % 1 == 0:
        N = len(reward_arr)
        sample_sz = 1000.0
        samplestep = math.ceil(N/sample_sz)
        xdata1 = np.arange(1,N,samplestep)
        xdata2 = np.arange(1,N+1,10)-1
        ydata1 = reward_arr[1::samplestep]
        baseline_arr[0] = reward_arr[1]

        plt.axis([1, N, min(ydata1)-10, max(ydata1)+10])
        line1.set_xdata(xdata1)
        line1.set_ydata(ydata1)
        line2.set_xdata(xdata2)
        line2.set_ydata(baseline_arr)

        # plt.axis([x_axis_min, max(episode,x_axis_min+5), min(reward_arr)-100, max(reward_arr)+100])
        # line.set_xdata(np.arange(len(reward_arr)),episode)
        # line.set_ydata(reward_arr)
        
        fig.canvas.flush_events()
        if os.name == 'posix':
            plt.pause(0.000001) # to make the plot visiable on OSX

    curr_reward = 0
    gc.collect()