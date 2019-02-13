import numpy as np


labels = np.genfromtxt('batch_32/labels-1_-1')
a = [3, 3, 23, 3]

for i in range(4):
    for j in range(a[i]):
        #print(i+1, j)
        c = np.genfromtxt('batch_64/labels{0}_{1}'.format(i+1, j))
        print((c==labels).sum())
        #        print()
        #print("labels{0}_{1}_{2}".format(i+1, j, (c==labels).sum()))
        



