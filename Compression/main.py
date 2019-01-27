import os
import torch 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data.sampler as Sampler
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import random
import argparse
import PIL, time
import numpy as np
from ResNet18 import *
from Resnet import *
from collections import Counter

# arguments
parser=argparse.ArgumentParser(description='Custom PyTorch ImageNet Training')
parser.add_argument('--seed', default=None, type=int, help='seed for inititalizing training')
parser.add_argument('--arch', default="resnet18", type=str, help='seed for inititalizing training')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64    , type=int, help="batch size")
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--decay', '--decay-factor', default=0.1, type=float,
                    metavar='LR', help='leraning rate decay factor')
parser.add_argument('--decay_after_n', type=int, default=30)

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--saveModel', action='store_true', default=False)
# train or eval mode
parser.add_argument("--trainMode", action='store_true', default=False)
parser.add_argument("--evalMode", action='store_true', default=False)
parser.add_argument("--filterMode", action='store_true', default=False)
parser.add_argument("--filterOnTrainData", action ='store_true', default=False)

parser.add_argument("--checkpointDir", type=str, default="checkpoint"  )
parser.add_argument("--checkpointInterval", type=int, default=30  )
parser.add_argument("--layerDump", type=str, default="layerDump")
parser.add_argument("--evalOnTrainData", action='store_true')
parser.add_argument("--evalOnVideoData", action='store_true')
parser.add_argument("--video_name", type=str, default="Lausanne", help='name of video dataset')

LAYER_DUMP_DIR = "layerDump"
# load the imageNet class 
ImageNet1000 = {}
with open("ImageNet1000") as f:
    data = f.readlines()
    for line in data:
        key, val = line.strip().split(':')
        ImageNet1000[int(key)] = val.split(",")[0].replace("'","").strip()
    
# train 
def train(dataGen, model, criterion, optimizer, epoch, device):
    model.train()
    # select some batches  
    for batchNum, (input, targets) in enumerate(dataGen):
        # just training on small dataset to check the architecture
        input, targets = input.to(device), targets.to(device)
        output = model(input)
        # print(layer_dump)
        loss = criterion(output, targets)
        # make the grads zero for each var
        optimizer.zero_grad()
        # compute the gradients
        loss.backward()
        # optimize step 
        optimizer.step()
        # print status every 100 batchNum
        if batchNum%100==0:
            print("Epoch:{0}, BatchNum:{1}, Loss:{2}".format(epoch, batchNum, loss))
        return



# filter topK
def filterTopK(dataGen, model, criterion, epoch, device, k, classId):
    print("Boss, Filter Mode On!")
    time.sleep(100)
    model.eval()
    total = 0
    correct = 0
    files = {}
    # name of the class
    className = ImageNet1000[classId]
    # create a output file 
    fileName = className + "_index"
    classesFound = []
    with open(fileName, "w") as f:
        with torch.no_grad():
            for batchNum, (input, targets) in enumerate(dataGen):
                print("Batch {} Done".format(batchNum))
                input, targets = input.to(device), targets.to(device)
                output = model(input)
                batchSize = output.size(0)
                # get the ones
                for i in range(output.size(0)):
                    # print(output[i,:].topk(3)[1])
                    if classId in output[i,:].topk(3)[1]:
                        f.write(str(batchNum*batchSize + i) + "\n")
            # print(Counter(classesFound))

# eval
def test(dataGen, model, criterion, epoch, device, dumpLayerOutput=False, trainData=False):
    print("Boss, Eval Mode On!")
    model.eval()
    total = 0
    correct = 0
    files = {}
    # just to get names
    def getFileName(template, key=""):
        if trainData==False:
            return os.path.join(LAYER_DUMP_DIR, os.path.join("TestData/", template+key))
        else:
            return os.path.join(LAYER_DUMP_DIR, os.path.join("TrainData/", template+key))
            
    # open required files
    if dumpLayerOutput:
        for key in model.checkpoints.keys():
            if model.checkpoints[key]:
                files[key] = open(getFileName("LayerOutput_", key), "w")
                print("file created {}".format(key))    
        files["labels"] = open(getFileName("labels"), "w")
    with torch.no_grad():
        for batchNum, (input, targets) in enumerate(dataGen):
            input, targets = input.to(device), targets.to(device)
            output = model(input)
            loss = criterion(output, targets)
            _, predicted = output.max(1)
            batchSize = output.size(0)
            correctBatch = (predicted==targets).sum().item()
            total = total + batchSize
            correct = correct + correctBatch

            if dumpLayerOutput:
                # save the labels once
                if device.type == "cpu":
                    np.savetxt(files["labels"], predicted.view(batchSize, -1).numpy(), fmt='%1.1f', delimiter=',')
                else:
                    np.savetxt(files["labels"], predicted.view(batchSize, -1).cpu().numpy(), fmt='%1.1f', delimiter=',')

                for key in model.layerDumps.keys():
                    if device.type == "cpu":
                        print(model.layerDumps[key].shape)
                        x = model.layerDumps[key].view(batchSize, -1).numpy()
                        print("{0} {1}".format(key, x.shape))
                        # save the layer
                        np.savetxt(files[key], x, fmt='%1.9f', delimiter=',')
                    else:
                        x = model.layerDumps[key].view(batchSize, -1).cpu().numpy()
                        # save the layer
                        np.savetxt(files[key], x, fmt='%1.9f', delimiter=',')

            print("Epoch: {0} Accuracy: {1}, TotalEx: {2}, CorrectEx: {3} ".format(epoch, correct*1.0/total, total, correct))
        #print(Counter(classesFound))
        return correct*1.0/total 


def adjustLearningRate(optimizer, decay_factor):
    for param_group in optimizer.param_groups:
        prev = param_group['lr']
        param_group['lr'] = param_group['lr']*decay_factor
        print("Learning Rate Decayed {0} --> {1}".format(prev, param_group['lr']))

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # define transform for train data
    # mean and std calculated from utils/GetMeanNdStd
    transform = transforms.Compose(
                                   [
                                    # transforms.Resize(224),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                                   ])
    # load datasets
    trainData = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    # sampler = list(range(64*10))
    trainDataGen = DataLoader(trainData, batch_size=args.batch_size, num_workers=1, shuffle=False)#, sampler=Sampler.SubsetRandomSampler(sampler))


    # change transform for test dataset
    if args.evalOnVideoData:
        # for video data make sure to resize to 32
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

        data = torchvision.datasets.ImageFolder(args.video_name, transform = transform)
        testDataGen = DataLoader(data, batch_size=args.batch_size, num_workers=1, shuffle=False)
    else:
        transform = transforms.Compose([
                                        transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        testData = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        testDataGen = DataLoader(testData, batch_size=args.batch_size, num_workers=1, shuffle=False)

    # define network
    print("Creating model {}".format(args.arch))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.arch == "resnet18":
        # model = ResNet18(32)
        model = resnet18(pretrained=True)
        model.to(device)
    elif args.arch == "resnet34":
        model = resnet34(pretrained = args.pretrained)
    elif args.arch == "resnet50":
        model = resnet50(pretrained = args.pretrained)
    elif args.arch == "resnet101":
        model = resnet101(pretrained = args.pretrained)
    elif args.arch == "resnet152":
        model = resnet152(pretrained = args.pretrained)

    # layer_dump = {}
    # def hook_block_(module, input, output):
    #     layer_dump[""].append(output)
    
    # model.block_3_shortcut.register_forward_hook(hook)

    # decide loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 

    # initial epoch
    epoch = 0
    best_acc = 0
    if args.resume:
        if device.type =="cpu": 
            checkpoint_ = torch.load(args.resume, map_location = device.type)
        else:
            checkpoint_ = torch.load(args.resume, map_location = device.type + ":" + str(device.index))

        best_acc = checkpoint_["best_acc"]
        model.load_state_dict(checkpoint_['state_dict'])
        epoch = checkpoint_['epoch']
        optimizer.load_state_dict(checkpoint_['optimizer'])

  
    if args.trainMode:
        while epoch < args.epochs:
            epoch = epoch + 1
            # adjust learning rate after even `n` epochhs
            if epoch%args.decay_after_n==0:
                adjustLearningRate(optimizer, args.decay) 
            
            # train for an epoch
            train(trainDataGen, model, criterion, optimizer, epoch, device)
            # test on the validation dataset
            # acc = test(testDataGen, model, criterion, epoch, device)
            acc = test(testDataGen, model, criterion, epoch, device, dumpLayerOutput=False)

            
            # if acc better that best or regular interval, then save model
            if args.saveModel:
                filename = os.path.join(args.checkpointDir, 'checkpoint_{}.pth.tar'.format(epoch))
                best_acc_filename = os.path.join(args.checkpointDir, 'checkpoint_best_acc.pth.tar')
                print("Best Acc Till Now : {0}, Acc this epoch : {1}".format(best_acc, acc))
                
                state = {
                        'epoch': epoch+1,
                        'acc': acc,
                        'best_acc': best_acc,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }
                
                # every regular interval save
                if epoch%args.checkpointInterval==0:
                    torch.save(state, filename)
                # if best till now
                if acc>best_acc:
                    best_acc = acc
                    torch.save(state, best_acc_filename)
            
    if args.evalMode:
        model.checkpoints = {
                                "block_1":False,
                                "block_2":False,
                                "block_3":False,
                                "block_4":False,
                                "block_5":False,
                                "block_6":False,
                                "block_7":False, 
                                "block_8":False
                           }   

        if args.evalOnTrainData:
            test(trainDataGen, model, criterion, epoch, device, dumpLayerOutput=False, trainData=True)
        else:
            test(testDataGen, model, criterion, epoch, device,  dumpLayerOutput=False, trainData=False)
    
    # if args.filterMode:
    #     if args.filterOnTrainData:
    #         filterTopK(trainDataGen, model, criterion, epoch, device,  k=3, classId=404)
    #     else:
    #         filterTopK(testDataGen, model, criterion,  epoch, device,  k=3, classId=404)

if __name__ == "__main__":
    IMAGENETCLass = "ImageNet1000"
    main()
    


