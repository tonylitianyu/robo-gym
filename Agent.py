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
import copy
from scipy.stats import rankdata

def normalize_weight(size):
	v = 1. / np.sqrt(size[0])
	return torch.Tensor(size).uniform_(-v, v)

def  gelu_approx(x):
	return torch.sigmoid(1.702*x)*x

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

	def forward(self, state, action):

		s1 =  F.relu(self.fcs1(state))
		s2 =  F.relu(self.fcs2(s1))
		a1 =  F.relu(self.fca1(action))

		if s2.dim() == 1:
			s2 = s2.unsqueeze(0)
			a1 = a1.unsqueeze(0)

		x = torch.cat((s2,a1),dim=1)

		x =  F.relu(self.fc2(x))
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

	def forward(self, state):

		x =  F.relu(self.fc1(state))
		x =  F.relu(self.fc2(x))
		x =  F.relu(self.fc3(x))
		action = torch.tanh(self.fc4(x))

		action = action * self.max_action

		return action

# ornstein
# https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/
class NoiseGenerator:
	def __init__(self,action_size,max_action):
		self.sigma = 0.75 # std
		self.mu = 0      # mean
		self.tau = 1 	 # time const
		self.dt = 0.02   # time step
		self.resetNoise()
		if os.path.exists('model/noise.p'):
			self.x = pickle.load(open('model/noise.p','rb'))
		else:
			self.x = np.ones(action_size)*self.mu
		self.max_action = max_action

	def randomNoise(self):
		self.x += self.dt * ((self.mu - self.x) / self.tau) + self.sigma*np.sqrt(2./self.tau)*np.sqrt(self.dt)*np.random.randn(len(self.x))
		return self.x*self.max_action
		#*np.sqrt(2./self.tau)*np.sqrt(self.dt)

	def resetNoise(self):
		pass

class OUNoise:
	"""Ornstein-Uhlenbeck process."""

	def __init__(self, action_size, mu, theta, sigma,max_action):
		"""Initialize parameters and noise process."""
		self.mu = mu * np.ones(action_size)
		self.theta = theta
		self.sigma = sigma
		self.resetNoise()

	def resetNoise(self):
		"""Reset the internal state (= noise) to mean (mu)."""
		self.state = copy.copy(self.mu)

	def randomNoise(self):
		"""Update internal state and return it as a noise sample."""
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
		self.state = x + dx
		return np.float32(self.state)

class Memory1:
	def __init__(self, size):
		if os.path.exists('model/memory_list.p'):
		    self.memory_list = pickle.load(open('model/memory_list.p','rb'))
		else:
			self.memory_list = deque(maxlen=size)
		
	def sample(self, count):

		batch = []
		count = min(count, len(self.memory_list))
		# print(len(self.memory_list))
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

# Calculate TD error at sample
class Memory:
	def __init__(self, size):
		if os.path.exists('model/memory_list.p'):
		    [self.memory_list,self.write_idx,self.size,self.full_replay] = pickle.load(open('model/memory_list.p','rb'))
		else:
			self.memory_list = np.zeros((size,4),dtype=object)
			self.write_idx = 0
			self.size = size
			self.full_replay = False
		
	def sample(self, count):

		if self.full_replay:
			batch = np.random.choice(self.size, size=count, replace=False)
		else:
			if self.write_idx <= count:
				batch = np.arange(0,self.write_idx)
			else:
				batch = np.random.choice(self.write_idx, size=count, replace=False)

		s_arr = np.float32(np.stack(self.memory_list[batch,0],axis=0))
		a_arr = np.float32(np.stack(self.memory_list[batch,1],axis=0))
		r_arr = np.float32(np.stack(self.memory_list[batch,2],axis=0))
		next_s_arr = np.float32(np.stack(self.memory_list[batch,3],axis=0))

		return s_arr, a_arr, r_arr, next_s_arr

	def PER_sample(self, count, agent):

		if self.full_replay:
			s  = np.float32(np.stack(self.memory_list[:,0],axis=0))
			a  = np.float32(np.stack(self.memory_list[:,1],axis=0))
			r  = np.float32(np.stack(self.memory_list[:,2],axis=0))
			s1 = np.float32(np.stack(self.memory_list[:,3],axis=0))
			td_err = abs(compute_td_err(agent, s, a, r, s1))
			D = 1.0/(td_err.size-rankdata(td_err)+1) # rankdata start from min
			P = D/sum(D)
			batch = np.random.choice(self.size, size=count, replace=False,p = P)
		else:
			if self.write_idx < count:
				batch = np.arange(0,self.write_idx)
			else:
				batch = np.random.choice(self.write_idx, size=count, replace=False)

		s_arr = np.float32(np.stack(self.memory_list[batch,0],axis=0))
		a_arr = np.float32(np.stack(self.memory_list[batch,1],axis=0))
		r_arr = np.float32(np.stack(self.memory_list[batch,2],axis=0))
		next_s_arr = np.float32(np.stack(self.memory_list[batch,3],axis=0))

		return s_arr, a_arr, r_arr, next_s_arr


	def add(self, s, a, r, s1):

		transition = np.array([s,a,r,s1],dtype=object)
		self.memory_list[self.write_idx,:] = transition
		self.write_idx += 1
		self.write_idx = self.write_idx % self.size
		if self.write_idx == 0:
			self.full_replay = self.full_replay or True

	def save(self):
		pickle.dump([self.memory_list,self.write_idx,self.size,self.full_replay], open('model/memory_list.p','wb'))

# Calculate TD Error at add
class Memory3:
	def __init__(self, size):
		if os.path.exists('model/memory_list.p'):
		    [self.memory_list,self.write_idx,self.size,self.full_replay] = pickle.load(open('model/memory_list.p','rb'))
		else:
			self.memory_list = np.zeros((size,5),dtype=object)
			self.write_idx = 0
			self.size = size
			self.full_replay = False
		
	def sample(self, count):

		if self.full_replay:
			batch = np.random.choice(self.size, size=count, replace=False)
		else:
			if self.write_idx < count:
				batch = np.arange(0,self.write_idx)
			else:
				batch = np.random.choice(self.write_idx, size=count, replace=False)

		s_arr = np.float32(np.stack(self.memory_list[batch,0],axis=0))
		a_arr = np.float32(np.stack(self.memory_list[batch,1],axis=0))
		r_arr = np.float32(np.stack(self.memory_list[batch,2],axis=0))
		next_s_arr = np.float32(np.stack(self.memory_list[batch,3],axis=0))

		return s_arr, a_arr, r_arr, next_s_arr

	def PER_sample(self, count, agent):

		td_arr = np.float32(np.stack(self.memory_list[:,4],axis=0))
		if self.full_replay:
			td_err = abs(td_arr)
			D = 1.0/(td_err.size-rankdata(td_err)+1) # rankdata start from min
			P = D/sum(D)
			batch = np.random.choice(self.size, size=count, replace=False,p = P)
		else:
			if self.write_idx < count:
				batch = np.arange(0,self.write_idx)
			else:
				batch = np.random.choice(self.write_idx, size=count, replace=False)

		s_arr = np.float32(np.stack(self.memory_list[batch,0],axis=0))
		a_arr = np.float32(np.stack(self.memory_list[batch,1],axis=0))
		r_arr = np.float32(np.stack(self.memory_list[batch,2],axis=0))
		next_s_arr = np.float32(np.stack(self.memory_list[batch,3],axis=0))

		return s_arr, a_arr, r_arr, next_s_arr


	def add(self, s, a, r, s1, td_err):

		transition = np.array([s,a,r,s1,td_err],dtype=object)
		self.memory_list[self.write_idx,:] = transition
		self.write_idx += 1
		self.write_idx = self.write_idx % self.size
		if self.write_idx == 0:
			self.full_replay = self.full_replay or True

	def save(self):
		pickle.dump([self.memory_list,self.write_idx,self.size,self.full_replay], open('model/memory_list.p','wb'))

class Agent:
	def __init__(self, state_size, action_size, max_action, memory, gpuid):
		self.gpuid = gpuid

		self.state_size = state_size
		self.action_size = action_size
		self.max_action = max_action
		self.gamma = 0.99
		self.batch_size = 128
		self.learning_rate_a = 0.0001
		self.learning_rate_c = 10*self.learning_rate_a

		# self.noiseMachine = NoiseGenerator(self.action_size,self.max_action)
		self.noiseMachine = OUNoise(self.action_size,0,0.15,0.2,self.max_action)
		self.memory = memory
		self.gpuid = gpuid

		self.loadAllNetwork()
		self.optim_a = torch.optim.Adam(self.actor.parameters(),self.learning_rate_a)
		self.optim_c = torch.optim.Adam(self.critic.parameters(),self.learning_rate_c)
		self.copy(self.target_actor, self.actor)
		self.copy(self.target_critic, self.critic)

	def loadAllNetwork(self):
		# Either load network if aviliable or creat new network to train
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
		# https://github.com/dgriff777/rl_a3c_pytorch Cuda checked against this repo
		if self.gpuid >= 0:
				self.actor  = self.actor.cuda()
				self.critic = self.critic.cuda()
				self.target_actor  = self.target_actor.cuda()
				self.target_critic = self.target_critic.cuda()

	def saveNetwork(self,isbest):
		if isbest:
			torch.save(self.actor, 'model/quadrotor_actor.pkl')
			torch.save(self.target_actor, 'model/quadrotor_target_actor.pkl')
			torch.save(self.critic, 'model/quadrotor_critic.pkl')
			torch.save(self.target_critic, 'model/quadrotor_target_critic.pkl')
			# pickle.dump(self.noiseMachine.x, open('model/noise.p','wb'))


	def use_action(self,state):
		state = Variable(torch.from_numpy(state))
		if self.gpuid >= 0:
				state = state.cuda()
		
		action = self.target_actor.forward(state).detach()
		return np.float32(action.cpu().numpy())#action.data.numpy()


	def get_action(self, state):
		state = Variable(torch.from_numpy(state))
		if self.gpuid >= 0:
			state = state.cuda()
		
		action = self.actor.forward(state).detach()
		return (action.cpu().numpy()) + self.noiseMachine.randomNoise()

	def copy(self,target, source):
		for target_param, source_param in zip(target.parameters(), source.parameters()):
				target_param.data.copy_(source_param.data)

	def updateTarget(self,target, source, learning_rate):
		#update target network from source
		for target_param, source_param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(
				target_param.data * (1.0 - learning_rate) + source_param.data * learning_rate
			)

	def rewardFunc(self,state,action):
		x = state[0]
		y = state[2]
		z = state[4]
		phi_   = state[6]
		theta_ = state[7]
		psi_   = state[8]
		reward = 0

		### You Nei Weier Reward
		distance = math.sqrt((x**2)+(y**2)+(z**2))
		o_error = abs(phi_)+abs(theta_)+abs(psi_)
		if distance > 3:
			reward -= (distance + 100*o_error)

		else:
			reward -= (0.01*distance + 10*o_error)

		# if distance < 1:
		# 	reward = 2000

		### Ken reward
		# agent_pos = np.array([x,y,z],dtype=float)
		# des_pos   = np.array([0,0,0],dtype=float)
		# distance  = np.linalg.norm(agent_pos-des_pos)

		# weight_mat = np.array([-3,-3,-10,-0.2,-0.2,-0.2],dtype=float)
		# xdes 	   = np.array([0,0,0,0,0,0],dtype=float)
		# xcurr 	   = np.array([x,y,z,phi_,theta_,psi_],dtype=float)
		# reward     = np.inner(weight_mat,abs(xdes-xcurr))
		if distance < 10:
			reward = reward - 0.5*min(max(x,-10),10)**2 + 30
			reward = reward - 0.5*min(max(y,-10),10)**2 + 30

		reward -= abs(np.sum(action[1:]))*0.2
		# if distance < 2:
		# 	reward += 30.0/distance
		# else:
			# reward = -distance*10
		# 	reward = 30.0/distance
		

		# if distance > 10:
		# 	reward = -100
		# elif distance < 10 and distance > 5:
		# 	reward = -50
		# elif distance < 5 and distance > 1:
		# 	reward = -20
		# 	reward -= abs(phi_*50)+abs(theta_*50)+abs(psi_*50)

		# elif distance < 1:
		# 	reward = 200
		# 	reward -= abs(phi_*150)+abs(theta_*150)+abs(psi_*150)

		return reward



	def train(self,train_all):
		
		s,a,r,ns = self.memory.sample(self.batch_size)
		# s,a,r,ns = self.memory.PER_sample(self.batch_size,self)

		s = Variable(torch.from_numpy(s))
		a = Variable(torch.from_numpy(a))
		r = Variable(torch.from_numpy(r))
		ns = Variable(torch.from_numpy(ns))

		if self.gpuid >= 0:
			s = s.cuda()
			a = a.cuda()
			r = r.cuda()
			ns = ns.cuda()

		a2 = self.target_actor.forward(ns).detach()
		next_val = torch.squeeze(self.target_critic.forward(ns, a2).detach())

		y_expected = r + self.gamma*next_val
		y_predicted = torch.squeeze(self.critic.forward(s, a)) # 0.001

		# td_err = y_expected-y_predicted
		# np.matmul(td_err*w_arr)/self.batch_size

		# compute critic loss, update the critic
		loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
		self.optim_c.zero_grad()
		loss_critic.backward() # 0.0015
		torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
		self.optim_c.step()# 0.003

		# Addressing Function Approximation Error in Actor-Critic Algorithms (Fujimoto et al. ‘18)
		if train_all:
			# Actor loss, update actor
			pred_a = self.actor.forward(s) # 0.001
			loss_actor = -1*torch.sum(self.critic.forward(s, pred_a)) # 0.001
			self.optim_a.zero_grad() # 0.001
			loss_actor.backward() # 0.003
			torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
			self.optim_a.step() # 0.003

			# Update target networks
			self.updateTarget(self.target_actor, self.actor, self.learning_rate_a) # 0.0025
			self.updateTarget(self.target_critic, self.critic, self.learning_rate_c) # 0.0025
			# print("%5.3f %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f " %(toc1,toc2,toc3,toc4,toc5,toc6,toc7))
