import numpy as np


labels = np.genfromtxt('labels-1_-1')
a = [2, 2, 19, 2]

for i in range(4):
    for j in range(a[i]):
        c = np.genfromtxt('labels{0}_{1}'.format(i+1, j+1))
        print((c==labels).sum())
        



