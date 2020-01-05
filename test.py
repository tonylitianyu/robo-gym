import numpy as np

# a = 1.0200002
# b = 2.5000005

# print("%2.3f %2.3f" %(a,b))

a = np.array([1.0,2.0,3.0])
b = np.array([1.0,2.0,3.0])
print(np.inner(a,b))

# # Callgrind profiling
# python -m cProfile -o gymprof.cprof main.py
# pyprof2calltree -i gymprof.cprof -o callgrind.gymtxt