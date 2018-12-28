from MobileNetV2 import MobileNetV2
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from MobileNetV2 import MobileNetV2



model = torch.nn.DataParallel(MobileNetV2(n_class=1000))
state_dict = torch.load('mobilenetv2_Top1_71.806_Top2_90.410.pth.tar', map_location='cpu')# if no gpu
trainedModel = {}
for key,val in state_dict.items():
    trainedModel[key.replace("module.","")]=val

print(trainedModel.keys())
model.load_state_dict(state_dict)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

input_size = 224

traindir = "../data/CaliforniaI_600/"
valdir = "../data/CaliforniaI_600/"

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        # transforms.Resize(int(input_size/0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=10, shuffle=False, pin_memory=False)



model.eval()
for i, (input, target) in enumerate(val_loader):
        # compute output
        output = model(input)
        print("Iteration {0}/{1}\n".format(i, len(val_loader)))
  
