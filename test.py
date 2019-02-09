#import matplotlib.pyplot as plt
import wrap
import numpy as np

file = "IntraFrameMNIST"





data = np.loadtxt(file, delimiter=",")
print(data.shape)
data = data.reshape(1,144*10)

q_min = data.min()
q_max = data.max()

a = wrap.compress(data, q_min, q_max, 10, 12, 12, "random")

# print("size pring maadi")
# print(a.size)
# print(data.size)
comp = []
init = []
diff = []
for i in range(a.size):
#    print("{0}, {1}".format(a[i], data[0][i]))
    comp = comp + [a[i]]
    init = init + [data[0][i]]
    diff = diff + [a[i] - data[0][i]]
# plt.plot(comp)
#plt.plot(diff)
# plt.plot(x, 2 * x)
# plt.plot(x, 3 * x)
# plt.plot(x, 4 * x)
#plt.show()
