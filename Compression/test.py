import torch
import numpy as np 
import CompressionLayer as CL 

# a = np.random.randint(1000, size=(10, 16, 5, 5))
# data = torch.from_numpy(a)


# inv_bhw = CL.Inverse_BHW_Format(bhw, batch, final_h, final_w, data.shape[1], data.shape[2], data.shape[3], img_per_row)

# print(a.shape)
# print(inv_bhw.shape)
# print((a == inv_bhw).sum())

#import matplotlib.pyplot as plt
import wrap
import numpy as np

data = torch.from_numpy(np.random.uniform(10, size=(10, 4, 5, 5))).type(torch.DoubleTensor)
bhw, batch, final_h, final_w, img_per_row = CL.Convert_BHW_Format(data)
q_min = data.min()
q_max = data.max()
a = wrap.compress(bhw, q_min, q_max, 10, final_h, final_w, "random")
inv_bhw = torch.from_numpy(CL.Inverse_BHW_Format(a, batch, final_h, final_w, data.shape[1], data.shape[2], data.shape[3], img_per_row))

print(inv_bhw.type())
print(data.type())

print((inv_bhw))
print((data))