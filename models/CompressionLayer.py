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


def Convert_BHW_Format(layerData):
    # get the shape 
    batch, channel, height, width = layerData.shape
    # nearestSqr = getNearestPerfectSqr(channel)
    nearestSqr = math.ceil(math.sqrt(channel))*math.ceil(math.sqrt(channel))

    
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

    return finalData, batch, final_h, final_w   


class CompressionLayer(nn.Module):

    def __init__(self, fileName):
        super(CompressionLayer, self).__init__()
        self.fileName = fileName
        # pooling for downsample
        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x):
        
        if not self.training:
            # y = self.avg_pool(x)
            data, b, h, w = Convert_BHW_Format(x)
            start = time.time()
            a = wrap.compress(data, data.min(), data.max(), b, h, w, "random")
            end = time.time()
            elapsedTime = end - start
            # get file size
            fsize = os.path.getsize("random")
            with open(self.fileName, "a") as f:
                f.write("{0},{1}\n".format(fsize, elapsedTime))
        
        return x