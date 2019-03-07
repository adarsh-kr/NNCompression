import numpy as np

labels = np.genfromtxt('coral_labels')
a = [3, 3, 23, 3]

for i in range(4):
    for j in range(a[i]):
        try:
            c = np.genfromtxt('Coral_H264_topk_slower/labels/labels{0}_{1}'.format(i+1, j))
            print((c==labels).sum())
        except Exception:
            print("no")
        #        print()
        #print("labels{0}_{1}_{2}".format(i+1, j, (c==labels).sum()))
        



