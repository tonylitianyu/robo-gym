import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import random
from collections import deque
import pickle
import math
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def normalize_weight(size):
	v = 1. / np.sqrt(size[0])
	return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):

	def __init__(self, state_dim, action_dim):

		super(Critic, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.EPS = 0.003

		self.fcs1 = nn.Linear(state_dim,256)
		self.fcs1.weight.data = normalize_weight(self.fcs1.weight.data.size())

		self.fcs2 = nn.Linear(256,128)
		self.fcs2.weight.data = normalize_weight(self.fcs2.weight.data.size())

		self.fca1 = nn.Linear(action_dim,128)
		self.fca1.weight.data = normalize_weight(self.fca1.weight.data.size())

		self.fc2 = nn.Linear(256,128)
		self.fc2.weight.data = normalize_weight(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(128,1)
		self.fc3.weight.data.uniform_(-self.EPS,self.EPS)


		self.fcs1.to(device)
		self.fcs2.to(device)
		self.fca1.to(device)
		self.fc2.to(device)
		self.fc3.to(device)

	def forward(self, state, action):
		state = state.to(device)
		action = action.to(device)
		s1 = F.relu(self.fcs1(state))
		s2 = F.relu(self.fcs2(s1))
		a1 = F.relu(self.fca1(action))
		x = torch.cat((s2,a1),dim=1)

		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x


class Actor(nn.Module):

	def __init__(self, state_dim, action_dim, max_action):

		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.max_action = max_action
		self.EPS = 0.003

		self.fc1 = nn.Linear(state_dim,256)
		self.fc1.weight.data = normalize_weight(self.fc1.weight.data.size())

		self.fc2 = nn.Linear(256,128)
		self.fc2.weight.data = normalize_weight(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(128,64)
		self.fc3.weight.data = normalize_weight(self.fc3.weight.data.size())

		self.fc4 = nn.Linear(64,action_dim)
		self.fc4.weight.data.uniform_(-self.EPS,self.EPS)

		self.fc1.to(device)
		self.fc2.to(device)
		self.fc3.to(device)
		self.fc4.to(device)

	def forward(self, state):
		state = state.to(device)
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		action = torch.tanh(self.fc4(x))

		action = action * self.max_action

		return action

#ornstein
class NoiseGenerator:
	def __init__(self,action_size,max_action):
		self.sigma = 0.2
		self.mu = 0
		self.tau = 1
		self.dt = 0.15
		if os.path.exists('model/noise.p'):
			self.x = pickle.load(open('model/noise.p','rb'))
		else:
			self.x = np.ones(action_size)*self.mu
		self.max_action = max_action

	def randomNoise(self):
		self.x += self.dt * (-(self.x - self.mu) / self.tau) + self.sigma*np.sqrt(2./self.tau)*np.sqrt(self.dt)*np.random.randn(len(self.x))
		return self.x*self.max_action
		#*np.sqrt(2./self.tau)*np.sqrt(self.dt)

class Memory:
	def __init__(self, size):
		if os.path.exists('model/memory_list.p'):

		    self.memory_list = pickle.load(open('model/memory_list.p','rb'))
		else:
			self.memory_list = deque(maxlen=size)

	def sample(self, count):

		batch = []
		count = min(count, len(self.memory_list))
		batch = random.sample(self.memory_list, count)

		s_arr = np.float32([arr[0] for arr in batch])
		a_arr = np.float32([arr[1] for arr in batch])
		r_arr = np.float32([arr[2] for arr in batch])
		next_s_arr = np.float32([arr[3] for arr in batch])

		return s_arr, a_arr, r_arr, next_s_arr

	def add(self, s, a, r, s1):

		transition = (s,a,r,s1)
		self.memory_list.append(transition)

	def save(self):
		pickle.dump(self.memory_list, open('model/memory_list.p','wb'))



class Agent:
	def __init__(self, state_size, action_size, max_action, memory):
		print(device)
		self.state_size = state_size
		self.action_size = action_size
		self.max_action = max_action
		self.gamma = 0.99
		self.batch_size = 128
		self.learning_rate = 0.001

		self.noiseMachine = NoiseGenerator(self.action_size,self.max_action)
		self.memory = memory

		self.loadAllNetwork()
		self.optim_a = torch.optim.Adam(self.actor.parameters(),self.learning_rate)
		self.optim_c = torch.optim.Adam(self.critic.parameters(),self.learning_rate)
		self.copy(self.target_actor, self.actor)
		self.copy(self.target_critic, self.critic)

	def loadAllNetwork(self):
		if os.path.exists('model/quadrotor_actor.pkl'):
		    self.actor = torch.load('model/quadrotor_actor.pkl')
		    print('Actor Model loaded')
		else:
		    self.actor = Actor(self.state_size, self.action_size, self.max_action)

		if os.path.exists('model/quadrotor_target_actor.pkl'):
		    self.target_actor = torch.load('model/quadrotor_target_actor.pkl')
		    print('Target Actor Model loaded')
		else:
		    self.target_actor = Actor(self.state_size, self.action_size, self.max_action)

		if os.path.exists('model/quadrotor_critic.pkl'):
		    self.critic = torch.load('model/quadrotor_critic.pkl')
		    print('Critic Model loaded')
		else:
		    self.critic = Critic(self.state_size, self.action_size)

		if os.path.exists('model/quadrotor_target_critic.pkl'):
		    self.target_critic = torch.load('model/quadrotor_target_critic.pkl')
		    print('Target Critic Model loaded')
		else:
		    self.target_critic = Critic(self.state_size, self.action_size)

	def saveNetwork(self):
		torch.save(self.actor, 'model/quadrotor_actor.pkl')
		torch.save(self.target_actor, 'model/quadrotor_target_actor.pkl')
		torch.save(self.critic, 'model/quadrotor_critic.pkl')
		torch.save(self.target_critic, 'model/quadrotor_target_critic.pkl')
		pickle.dump(self.noiseMachine.x, open('model/noise.p','wb'))


	def use_action(self,state):
		state = Variable(torch.from_numpy(state))
		action = self.target_actor.forward(state).detach()
		return action.cpu().numpy()#action.data.numpy()


	def get_action(self, state):
		state = Variable(torch.from_numpy(state))
		action = self.actor.forward(state).detach()
		return action.cpu().numpy() + self.noiseMachine.randomNoise()

	def copy(self,target, source):
		for target_param, source_param in zip(target.parameters(), source.parameters()):
				target_param.data.copy_(source_param.data)

	def updateTarget(self,target, source):
		#update target network from source
		for target_param, source_param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(
				target_param.data * (1.0 - self.learning_rate) + source_param.data * self.learning_rate
			)

	def rewardFunc(self,state):
		x = state[0]
		y = state[2]
		z = state[4]
		phi_   = state[6]
		theta_ = state[7]
		psi_   = state[8]
		reward = 0
		#tranable
		distance = math.sqrt((x**2)+(y**2)+(z**2))

		if distance > 10:
			reward = -100
		elif distance < 10 and distance > 5:
			reward = -50
		elif distance < 5 and distance > 1:
			reward = -20
			reward -= abs(phi_*50)+abs(theta_*50)+abs(psi_*50)

		elif distance < 1:
			reward = 200
			reward -= abs(phi_*150)+abs(theta_*150)+abs(psi_*150)

		return reward



	def train(self):
		s,a,r,ns = self.memory.sample(self.batch_size)

		s = Variable(torch.from_numpy(s))
		s = s.to(device)
		a = Variable(torch.from_numpy(a))
		a = a.to(device)
		r = Variable(torch.from_numpy(r))
		r = r.to(device)
		ns = Variable(torch.from_numpy(ns))
		ns = ns.to(device)

		a2 = self.target_actor.forward(ns).detach()
		next_val = torch.squeeze(self.target_critic.forward(ns, a2).detach())

		tic = time.time()

		y_expected = r + self.gamma*next_val

		y_predicted = torch.squeeze(self.critic.forward(s, a))
		# compute critic loss, and update the critic
		loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
		self.optim_c.zero_grad()
		loss_critic.backward()
		self.optim_c.step()

		toc1 = time.time()-tic

		pred_a = self.actor.forward(s)
		loss_actor = -1*torch.sum(self.critic.forward(s, pred_a))
		self.optim_a.zero_grad()
		loss_actor.backward()
		self.optim_a.step()

		self.updateTarget(self.target_actor, self.actor)
		self.updateTarget(self.target_critic, self.critic)

		toc = time.time()-tic
		print(toc1,toc)
