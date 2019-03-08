import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from CompressionLayer import CompressionLayer
import time

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes,
                 planes,
                 fileName,
                 layerNum,
                 blockNum,
                 stride=1,
                 downsample=None,
                 compress=False,
                 returnCompressedTensor=False):

        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.fileName = fileName + "_layerNum_"+ str(layerNum) + "_blockNum_" + str(blockNum)
        self.compressionLayer = CompressionLayer(self.fileName, returnCompressedTensor, compress)


    def forward(self, x):
        start = time.time()
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        end = time.time()
        a = end-start
        start = time.time()

        out = self.compressionLayer(out)
        end = time.time()
        #with open(self.fileName, "a") as f:
        #    f.write("{0},{1}\n".format(a, end-start))
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, fileName, layerNum, blockNum, stride=1, downsample=None, 
                 returnCompressedTensor=False,
                 compress=False):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.fileName = fileName + "_layerNum_"+ str(layerNum) + "_blockNum_" + str(blockNum)
        self.compressionLayer = CompressionLayer(self.fileName, returnCompressedTensor, compress)

    def forward(self, x):
        start = time.time()
        print("Bottlenect Forward")
        print(x.shape)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        end = time.time()
        a = end-start
        out = self.compressionLayer(out)
        start = time.time()
        end   = time.time()

       # with open(self.fileName, "a") as f:
       #      f.write("{0},{1}\n".format(a, end-start))
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, compressAtLayer=1, compressAtBlock=1):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.compressAtLayer = compressAtLayer
        self.compressAtBlock = compressAtBlock

        self.layer1 = self._make_layer(block, 64,  layers[0],  layerNum=1)
        self.layer2 = self._make_layer(block, 128, layers[1],  layerNum=2, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2],  layerNum=3, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3],  layerNum=4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

        self.checkpoints = {
                                "block_1":False,
                                "block_2":False,
                                "block_3":False,
                                "block_4":False,
                                "block_5":False
                           }
        
        self.layerDumps = {}

    def addLayerDumps(self, block_name, y):
            if self.checkpoints[block_name]==True:
                self.layerDumps[block_name] = y


    def _make_layer(self, block, planes, blocks, layerNum, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if self.compressAtLayer == layerNum and self.compressAtBlock==0:
            layers.append(block(self.inplanes, planes, "LayerData", layerNum, 0, stride, downsample, compress=True, returnCompressedTensor=True))
        else: 
            layers.append(block(self.inplanes, planes, "LayerData", layerNum, 0, stride, downsample, compress=False, returnCompressedTensor=False))

        self.inplanes = planes * block.expansion
        blockNum=0
        for _ in range(1, blocks):
            blockNum+=1
            if self.compressAtLayer == layerNum and self.compressAtBlock == blockNum: 
                layers.append(block(self.inplanes, planes, "LayerData", layerNum, blockNum, compress=True, returnCompressedTensor=True))
            else:
                layers.append(block(self.inplanes, planes, "LayerData", layerNum, blockNum, compress=False, returnCompressedTensor=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.training==False:
            self.addLayerDumps("block_1", x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if self.training==False:
            self.addLayerDumps("block_2", x)

        x = self.layer2(x)
        if self.training==False:
            self.addLayerDumps("block_3", x)

        x = self.layer3(x)
        if self.training==False:
            self.addLayerDumps("block_4", x)

        x = self.layer4(x)
        if self.training==False:
            self.addLayerDumps("block_5", x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
