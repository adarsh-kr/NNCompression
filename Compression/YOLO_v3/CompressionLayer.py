import os
import torch 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import random
import argparse, math
import numpy as np
import time
import wrap

# preset : ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, placebo
PRESET_PARAMETER = 'fast'


def Inverse_BHW_Format(comp_data, b, h, w, init_c, init_h, init_w, img_per_row):
    comp_data  = comp_data.reshape((b, h, w))
    final_data = np.zeros((b, init_c, init_h, init_w))
    
    # for each i in range(b)
    for i in range(b):
        # for each batch 
        for j in range(init_c):
            top_left_corner_row   = int((j)/img_per_row)*init_h
            top_left_corner_col = int(((j)%img_per_row)*init_w)
            channel_j = comp_data[i, top_left_corner_row:(top_left_corner_row+init_h), top_left_corner_col:(top_left_corner_col+init_w)]
            final_data[i, j, :, : ] = channel_j
    return final_data

def Convert_BHW_Format(layerData):
    
    # get the shape 
    batch, channel, height, width = layerData.shape
    # nearestSqr = getNearestPerfectSqr(channel)
    
    buf = (math.sqrt(channel))
    if buf%2==0:
        nearestSqr = buf**2
    else:
        nearestSqr = (buf+1)**2
    
    final_h = int(math.sqrt(nearestSqr))*height
    final_w = int(math.sqrt(nearestSqr))*width

    data = layerData.numpy()

    finalData = np.zeros((batch, final_h, final_w))

    img_per_row = math.sqrt(nearestSqr)

    for j in range(batch):
        for i in range(channel):
            img = data[j, i, :, :]
            top_left_corner_row   = int((i)/img_per_row)*height
            top_left_corner_col = int(((i)%img_per_row)*width)
            finalData[j, top_left_corner_row:(top_left_corner_row+height), top_left_corner_col:(top_left_corner_col+width)] = img

    return finalData, batch, final_h, final_w, img_per_row   

class CompressionLayer(nn.Module):
    def __init__(self, fileName, returnCompressedTensor=False, compress=True):
        super(CompressionLayer, self).__init__()
        self.fileName = fileName
        print(self.fileName)
        # pooling for downsample
        self.returnCompressedTensor = returnCompressedTensor
        self.compress = compress

    def forward(self, x):
        # print(self.fileName)
        if not self.training:
            
            if self.compress:
                init_b, init_c, init_h, init_w = x.shape
                data, b, h, w, img_per_row = Convert_BHW_Format(x)
                start = time.time()
                print("New height : {0}, new width {1}".format(h, w))
                try:
                    # comp_data is going to be one dimensional b*h
                    comp_data = wrap.compress(data, data.min(), data.max(), b, h, w, "random", PRESET_PARAMETER)
                except:
                    print("Going to Exception")
                    time.sleep(5)
                    comp_data = wrap.compress(data, data.min(), data.max(), b, h, w, "random", PRESET_PARAMETER)

                end = time.time()
                
                elapsedTime = end - start
                # get file size
                fsize = os.path.getsize("random")

                comp_x = Inverse_BHW_Format(comp_data, b, h, w, init_c, init_h, init_w, img_per_row)
                comp_x = torch.from_numpy(comp_x).type(torch.FloatTensor)        

                rmse_I   = 0 #torch.mean(abs(comp_x - x)/abs(x+1))
                rmse_II  = 1 #torch.mean(abs(comp_x - x)/abs(x+1))
                rmse_III = 2 #torch.mean(abs(comp_x - x)/abs(x+10^-6)) 
                rmse_IV  = 3 #torch.mean(2*abs(comp_x - x)/(abs(x) + abs(y))) 

                with open(self.fileName, "a") as f:
                    f.write("{0},{1},{2},{3},{4},{5}\n".format(fsize, elapsedTime, rmse_I, rmse_II, rmse_III, rmse_IV))

            if self.returnCompressedTensor:
                return comp_x
            else:
                return x
