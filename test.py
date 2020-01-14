import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# a = np.ones([10,10])
# b = 1.5
# c = 2

# pickle.dump([a,b,c],open('test.p','wb'))

[a,b,c] = pickle.load(open('test.p','rb'))

print(a,b,c)


# # Callgrind profiling
# python -m cProfile -o gymprof.cprof main.py
# pyprof2calltree -i gymprof.cprof -o callgrind.gymtxt


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
			if self.write_idx < count:
				batch = np.arange(0,self.write_idx)
			else:
				batch = np.random.choice(self.write_idx, size=count, replace=False)

		s_arr = np.float32(self.memory_list[batch,0])
		a_arr = np.float32(self.memory_list[batch,1])
		r_arr = np.float32(self.memory_list[batch,2])
		next_s_arr = np.float32(self.memory_list[batch,3])

		return s_arr, a_arr, r_arr, next_s_arr

	def add(self, s, a, r, s1):

		transition = np.array([s,a,r,s1],dtype=object)
		print(transition,self.memory_list)
		self.memory_list[self.write_idx,:] = transition
		self.write_idx += 1
		self.write_idx = self.write_idx % self.size
		print(self.write_idx,self.memory_list)
		if self.write_idx == 0:
			self.full_replay = self.full_replay or True

	def save(self):
		pickle.dump([self.memory_list,self.write_idx,self.size,self.full_replay], open('model/memory_list.p','wb'))