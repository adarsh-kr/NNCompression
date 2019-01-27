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
import CompressionLayer
import wrap


# get nearest perfect square
# not needed btw, as we always want perfect square greater than than the present
def getNearestPerfectSqr(n):
    l = int(math.sqrt(n))
    h = math.ceil(math.sqrt(n))

    if abs(n-l)>abs(n-h):
        return l*l
    else:
        return h*h

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


class EmptyShortcutLayer(nn.Module):
    def __init__(self):
        super(EmptyShortcutLayer, self).__init__()
    def forward(self, x):
        return x

class ResNet18(nn.Module):
    # do all layer manually 
    # DO NOT CREATE sequential for blocks
    def __init__(self, input_size, block="Basic", num_classes=10):
        super(ResNet18, self).__init__()
        # blocks 2,2,2,2
        # introduce checkpoints for dumping outputs
        self.checkpoints = {
                                "block_1":False,
                                "block_2":False,
                                "block_3":False,
                                "block_4":False,
                                "block_5":False,
                                "block_6":False,
                                "block_7":False, 
                                "block_8":False
                           } 

        self.layerDumps = {}

        self.in_planes = 64
    
        self.input_size = input_size
        self.output_size = self.input_size
        self.num_classes = num_classes
        
        # introduce non-linearity
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # batch normalization takes C as argument, i.e. the channels 
        self.bn1 = nn.BatchNorm2d(64)
        
        # segment 1
        # block 1 
        self.block_1_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block_1_bn1   = nn.BatchNorm2d(64)

        self.block_1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block_1_bn2 = nn.BatchNorm2d(64)

        self.block_1_shortcut = EmptyShortcutLayer()
        # self.block_1_compression = CompressionLayer.CompressionLayer("CompressionDataLayer", 1)

        # block 2
        self.block_2_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block_2_bn1   = nn.BatchNorm2d(64)

        self.block_2_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block_2_bn2 = nn.BatchNorm2d(64)

        self.block_2_shortcut = EmptyShortcutLayer()

        # segment 2 
        # block 3
        self.block_3_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.block_3_bn1 = nn.BatchNorm2d(128)

        # stride 2, so half it
        self.output_size = self.output_size/2

        self.block_3_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.block_3_bn2 = nn.BatchNorm2d(128)

        self.block_3_shortcut = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128))

        # block 4 
        self.block_4_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.block_4_bn1 = nn.BatchNorm2d(128)

        self.block_4_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.block_4_bn2 = nn.BatchNorm2d(128)

        self.block_4_shortcut = EmptyShortcutLayer()

        # segment 3
        # block 5
        self.block_5_conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.block_5_bn1    = nn.BatchNorm2d(256)
        
        # stride =2, output size is halved 
        self.output_size = self.output_size/2

        self.block_5_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.block_5_bn2    = nn.BatchNorm2d(256)

        self.block_5_shortcut = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256))
        
        # block 6
        self.block_6_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.block_6_bn1    = nn.BatchNorm2d(256)

        self.block_6_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.block_6_bn2    = nn.BatchNorm2d(256)

        self.block_6_shortcut = EmptyShortcutLayer()

        # segment 4 
        # block 7
        self.block_7_conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.block_7_bn1    = nn.BatchNorm2d(512)

        # stride=2, output size is halved 
        self.output_size = self.output_size/2

        self.block_7_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.block_7_bn2    = nn.BatchNorm2d(512)

        self.block_7_shortcut = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(512))
        
        # block 8
        self.block_8_conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.block_8_bn1    = nn.BatchNorm2d(512)

        self.block_8_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.block_8_bn2    = nn.BatchNorm2d(512)

        self.block_8_shortcut = EmptyShortcutLayer()

        # avg pooling 
        self.avg_pool = nn.AvgPool2d(int(self.output_size))
        self.output_size = self.output_size/4 
        
        # final layer
        self.linear = nn.Linear(512, self.num_classes) 
    
    def addLayerDumps(self, block_name, y):
         if self.checkpoints[block_name]==True:
            self.layerDumps[block_name] = y

    def forward(self, x):
        
        y = x
        # first conv
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        
        # first block
        out = self.block_1_conv1(y)
        out = self.block_1_bn1(out)
        out = self.relu(out)

        # add compression 
        if not self.training:
            print("Nothing")
            # out = self.block_1_compression(out)
            # data, b, h, w = Convert_BHW_Format(out)
            # a = wrap.compress(data, data.min(), data.max(), b, h, w, "random")
            # # get file size
            # fsize = os.path.getsize("random")
            # print("FSize: {}".format(fsize))
            # time.sleep(10)

            # to update the going vector 

        out = self.block_1_conv2(out)
        out = self.block_1_bn2(out)
        y = self.block_1_shortcut(y) + out
        y = self.relu(y)
        # add to layer dumps
        if self.training==False:
            self.addLayerDumps("block_1", y)

        # second block 
        out = self.block_2_conv1(y)
        out = self.block_2_bn1(out)
        out = self.relu(out)


        out = self.block_2_conv2(out)
        out = self.block_2_bn2(out)

        y = self.block_2_shortcut(y) + out
        y = self.relu(y)

        if self.training==False:
            self.addLayerDumps("block_2", y)

        # third block
        out = self.block_3_conv1(y)
        out = self.block_3_bn1(out)
        out = self.relu(out)

        out = self.block_3_conv2(out)
        out = self.block_3_bn2(out)

        y   = self.block_3_shortcut(y) + out
        y = self.relu(y)

        if self.training==False:
            self.addLayerDumps("block_3", y)


        # fourth block
        out = self.block_4_conv1(y)
        out = self.block_4_bn1(out)
        out = self.relu(out)
        
        out = self.block_4_conv2(out)
        out = self.block_4_bn2(out)

        y = self.block_4_shortcut(y) + out
        y = self.relu(y)

        if self.training==False:
            self.addLayerDumps("block_4", y)

        # fifth block
        out = self.block_5_conv1(y)
        out = self.block_5_bn1(out)
        out = self.relu(out)
        
        out = self.block_5_conv2(out)
        out = self.block_5_bn2(out)

        y = self.block_5_shortcut(y) + out
        y = self.relu(y)

        if self.training==False:
            self.addLayerDumps("block_5", y)
        

        # sixth block 
        out = self.block_6_conv1(y)
        out = self.block_6_bn1(out)
        out = self.relu(out)
        
        out = self.block_6_conv2(out)
        out = self.block_6_bn2(out)

        y = self.block_6_shortcut(y) + out
        y = self.relu(y)

        if self.training==False:
            self.addLayerDumps("block_6", y)

        # seveth block
        out = self.block_7_conv1(y)
        out = self.block_7_bn1(out)
        out = self.relu(out)
        
        out = self.block_7_conv2(out)
        out = self.block_7_bn2(out)

        y = self.block_7_shortcut(y) + out
        y = self.relu(y)

        if self.training==False:
            self.addLayerDumps("block_7", y)

        # eigth block
        out = self.block_8_conv1(y)
        out = self.block_8_bn1(out)
        out = self.relu(out)
        
        out = self.block_8_conv2(out)
        out = self.block_8_bn2(out)

        y = self.block_8_shortcut(y) + out
        y = self.relu(y)

        if self.training==False:
            self.addLayerDumps("block_8", y)

        # avg pool
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.linear(y)

        return y

