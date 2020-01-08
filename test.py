import numpy as np
import matplotlib.pyplot as plt
import os

plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
# plt.show()

# a = 1.0200002
# b = 2.5000005

# print("%2.3f %2.3f" %(a,b))

a = np.array([1.0,2.0,3.0])
b = np.array([1.0,2.0,3.0])
print(np.inner(a,b))

print(os.name)

# # Callgrind profiling
# python -m cProfile -o gymprof.cprof main.py
# pyprof2calltree -i gymprof.cprof -o callgrind.gymtxt