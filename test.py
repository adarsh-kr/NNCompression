import wrap
import numpy as np

file = "IntraFrameMNIST"

data = np.loadtxt(file, delimiter=",")
print(data.shape)
data = data.reshape(1,144*10)

q_min = data.min()
q_max = data.max()

a = wrap.compress(data, q_min, q_max, 10, 12, 12, "random")
print(a.size)
