from collections import OrderedDict

import numpy as np

import torch; 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from nnfc.modules.nnfc import CompressionLayer
from PIL import Image

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class LinearBottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, t=6, activation=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes * t)
        self.conv2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=3, stride=stride, padding=1, bias=False,
                               groups=inplanes * t)
        self.bn2 = nn.BatchNorm2d(inplanes * t)
        self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.activation = activation(inplace=True)
        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        return out

    
class Compressor(nn.Module):
    def __init__(self, inplanes):
        super(Compressor, self).__init__()
        # self.compression_layer = CompressionLayer(encoder_name='jpeg_encoder',
        #                                           encoder_params_dict={'quantizer' : 36},
        #                                           decoder_name='jpeg_decoder',
        #                                           decoder_params_dict={})

        # self.compression_layer = CompressionLayer(encoder_name='avc_encoder',
        #                                           encoder_params_dict={'quantizer' : 42},
        #                                           decoder_name='avc_decoder',
        #                                           decoder_params_dict={})

        self.compression_layer = CompressionLayer(encoder_name='nnfc2_encoder',
                                                  encoder_params_dict={},
                                                  decoder_name='nnfc2_decoder',
                                                  decoder_params_dict={})

        # a = torch.arange(0,128).reshape((1, 1, 16, 8)).float()
        # print(a.shape, a)
        # b = self.compression_layer(a)
        # print(b.shape, b)
        # print(a == b)
        
        self.sizes = []
        self.pad = nn.ReplicationPad2d(2)
        
        # define the bottleneck layers
        # expland to 6x the size with 3x3
        # mix with 1x1
        # bottleneck to same spatial dims, but less channels        

        #planes = int(inplanes / 2)
        
        # encoder
        #t = 12
        #self.encoder = LinearBottleneck(inplanes, planes, t=t)
        
        # decoder
        #self.decoder = LinearBottleneck(planes, inplanes, t=t)

        print('inPlanes', inplanes)
        # print('compressPlanes', planes)
        # print('t =', t)

    def get_compressed_sizes(self):
        return self.compression_layer.get_compressed_sizes()        
        
    def forward(self, x):
        # x = self.encoder(x)
        
        # x_min = float(x.min())
        # x_max = float(x.max())
        # print(x.max(), x.min())
        # thres = 0.5

        # x_top = x.clone()
        # x_top[x_top <= thres] = 0

        # x_bot = x.clone()
        # x_bot[x_bot >= -thres] = 0

        # x = x_top + x_bot
        
        # density = np.histogram(x.cpu().detach().numpy(), bins=10)
        # print(density)        
        # print(x.max(), x.min())

        # visualization code
        # print(x.shape)
        # d = x[0,:,:,:].cpu().detach().numpy()
        # dmin = np.min(d)
        # dmax = np.max(d)
        # for i in range(x.shape[1]):
        #     d = x[0,i,:,:].cpu().detach().numpy()
        #     print(d.shape)
        #     print(dmin, dmax)
            
        #     img = ((255 * (d - dmin)) / (dmax - dmin)).astype(np.uint8)
        #     imgmin = np.min(img)
        #     imgmax = np.max(img)
            
        #     img = Image.fromarray(img)
        #     img.save('/home/jemmons/intermediates/intermediate{}.png'.format(i))
            
        # print(x.shape)
        x = self.pad(x)
        x = self.compression_layer(x)
        x = x[:,:,1:-1,1:-1]
        self.sizes += self.compression_layer.get_compressed_sizes()

        # x = self.decoder(x)

        # print(np.mean(np.asarray(self.sizes)))
        # print(np.median(np.asarray(self.sizes)))

        return x


class MobileNet2(nn.Module):
    """MobileNet2 implementation.
    """

    def __init__(self, scale=1.0, input_size=224, t=6, in_channels=3, num_classes=1000, activation=nn.ReLU6):
        """
        MobileNet2 constructor.
        :param in_channels: (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
        :param input_size:
        :param num_classes: number of classes to predict. Default
                is 1000 for ImageNet.
        :param scale:
        :param t:
        :param activation:
        """

        super(MobileNet2, self).__init__()

        self.scale = scale
        self.t = t
        self.activation_type = activation
        self.activation = activation(inplace=True)
        self.num_classes = num_classes

        self.num_of_channels = [32, 16, 24, 32, 64, 96, 160, 320]
        assert (input_size % 32 == 0)

        self.compression_layer = Compressor(32)
        
        self.c = [_make_divisible(ch * self.scale, 8) for ch in self.num_of_channels]
        self.n = [1, 1, 2, 3, 4, 3, 3, 1]
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]
        self.conv1 = nn.Conv2d(in_channels, self.c[0], kernel_size=3, bias=False, stride=self.s[0], padding=1)
        self.bn1 = nn.BatchNorm2d(self.c[0])
        self.bottlenecks = self._make_bottlenecks()

        # Last convolution has 1280 output channels for scale <= 1
        self.last_conv_out_ch = 1280 if self.scale <= 1 else _make_divisible(1280 * self.scale, 8)
        self.conv_last = nn.Conv2d(self.c[-1], self.last_conv_out_ch, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(self.last_conv_out_ch)
        self.avgpool = nn.AvgPool2d(int(input_size // 32))
        self.dropout = nn.Dropout(p=0.2, inplace=True)  # confirmed by paper authors
        self.fc = nn.Linear(self.last_conv_out_ch, self.num_classes)
        self.init_params()

        # freeze all parameters
        # print('freezing all parameters')
        # for param in self.parameters():
        #     param.requires_grad = False

        # # unfreeze the parameters 
        # print('unfreezing compressor parameters')
        # for param in self.compression_layer.parameters():
        #     param.requires_grad = True               
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_stage(self, inplanes, outplanes, n, stride, t, stage):
        modules = OrderedDict()
        stage_name = "LinearBottleneck{}".format(stage)

        # First module is the only one utilizing stride
        first_module = LinearBottleneck(inplanes=inplanes, outplanes=outplanes, stride=stride, t=t,
                                        activation=self.activation_type)
        modules[stage_name + "_0"] = first_module

        # add more LinearBottleneck depending on number of repeats
        for i in range(n - 1):
            name = stage_name + "_{}".format(i + 1)
            module = LinearBottleneck(inplanes=outplanes, outplanes=outplanes, stride=1, t=6,
                                      activation=self.activation_type)
            modules[name] = module

        return nn.Sequential(modules)

    def _make_bottlenecks(self):
        modules = OrderedDict()
        stage_name = "Bottlenecks"

        # First module is the only one with t=1
        bottleneck1 = self._make_stage(inplanes=self.c[0], outplanes=self.c[1], n=self.n[1], stride=self.s[1], t=1,
                                       stage=0)
        modules[stage_name + "_0"] = bottleneck1

        # add more LinearBottleneck depending on number of repeats
        for i in range(1, len(self.c) - 1):
            name = stage_name + "_{}".format(i)
            module = self._make_stage(inplanes=self.c[i], outplanes=self.c[i + 1], n=self.n[i + 1],
                                      stride=self.s[i + 1],
                                      t=self.t, stage=i)
            modules[name] = module
            if i == 2:
                modules['compressor'] = self.compression_layer
                
        return nn.Sequential(modules)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.bottlenecks(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)

        # average pooling layer
        x = self.avgpool(x)
        x = self.dropout(x)

        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    """Testing
    """
    model1 = MobileNet2()
    print(model1)
    model2 = MobileNet2(scale=0.35)
    print(model2)
    model3 = MobileNet2(in_channels=2, num_classes=10)
    print(model3)
    x = torch.randn(1, 2, 224, 224)
    print(model3(x))
    model4_size = 32 * 10
    model4 = MobileNet2(input_size=model4_size, num_classes=10)
    print(model4)
    x2 = torch.randn(1, 3, model4_size, model4_size)
    print(model4(x2))
    model5 = MobileNet2(input_size=196, num_classes=10)
    x3 = torch.randn(1, 3, 196, 196)
    print(model5(x3))  # fail
