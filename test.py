import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# # a = np.ones([10,10])
# # b = 1.5
# # c = 2

# # pickle.dump([a,b,c],open('test.p','wb'))

# [a,b,c] = pickle.load(open('test.p','rb'))

# print(a,b,c)

a = np.array([1.0,2.0,3.0])
b = np.array([6.0,5.0,4.0])

print(a/b)

# # Callgrind profiling
# python -m cProfile -o gymprof.cprof main.py
# pyprof2calltree -i gymprof.cprof -o callgrind.gymtxt