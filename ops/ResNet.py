"""
An example combining `Temporal Shift Module` with `ResNet`. This implementation
is based on `Temporal Segment Networks`, which merges temporal dimension into
batch, i.e. inputs [N*T, C, H, W]. Here we show the case with residual connections
and zero padding with 8 frames as input.
"""
import numpy as np
import numbers
import copy
from math import sqrt
from collections import OrderedDict
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import ops.logging as logging
from ops.rsa import RSA

logger = logging.get_logger(__name__)

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3x3_CS(in_planes, out_planes, stride=1):
    """3x3x3 channel-separated convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding = 1, bias=False, groups=in_planes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_segments, kernel=(3,3,3), stride=1, groups=1, downsample=None, remainder=0, conv_mode='conv', config=None):
        super(BasicBlock, self).__init__()
        
        self.conv_mode = conv_mode # 'conv' or 'dyconv'
        
        
        if self.conv_mode == 'RSA':
            self.conv2 = RSA(
                planes, planes,
                dk=config['dk'],
                du=config['du'],
                nh=config['nh'],
                ng=config['mg'],
                kernel=config['kernel'],
                stride=(1, stride, stride),
                transpose=False,
                kernel_type=config['kernel_type'], 
                feat_type=config['feat_type'],
                qkv_bn=config['qkv_bn']
            )
        else:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=kernel, stride=(1, stride, stride), padding=tuple(k//2 for k in kernel), groups=groups, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel, stride=(1, stride, stride), padding=tuple(k//2 for k in kernel), groups=groups, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.downsample = downsample
        self.stride = stride
        self.remainder =remainder
        self.num_segments = num_segments        

    def forward(self, x):
        identity = x  
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out          
    
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, num_segments, kernel=(3,3,3), stride=1, groups=1, downsample=None, remainder=0, conv_mode='conv', config=None, stochastic_depth_prob=1):
        super(Bottleneck, self).__init__()
        
        self.conv_mode = conv_mode # 'conv' or 'RSA'
        
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        
        if self.conv_mode == 'RSA':
            self.conv2 = RSA(
                planes, planes,
                nh=config['nh'],
                dk=config['dk'],
                dv=config['dv'],
                dd=config['dd'],
                kernel_size=config['kernel_size'],
                stride=(1, stride, stride),
                kernel_type=config['kernel_type'], 
                feat_type=config['feat_type']
            )
        elif self.conv_mode == 'conv':
            self.conv2 = nn.Conv3d(
                planes, planes, 
                kernel_size=kernel, 
                stride=(1, stride, stride), 
                padding=tuple(k//2 for k in kernel), 
                groups=1, 
                bias=False
            )
        else:
            self.conv2 = nn.Conv3d(
                planes, planes, 
                kernel_size=kernel, 
                stride=(1, stride, stride), 
                padding=tuple(k//2 for k in kernel), 
                groups=groups, 
                bias=False
            )
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride
        self.remainder= remainder        
        self.num_segments = num_segments  
        
        self.st_depth = stochastic_depth_prob
        
        
    def st_depth_func(self, x, st_depth_prob):
        mask = tr.empty([x.shape[0],1,1,1,1], dtype=x.dtype, device=x.device)
        mask.bernoulli_(st_depth_prob)
        x.div_(st_depth_prob)
        x.mul_(mask)
        return x

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # stochastic depth
        if self.training and self.st_depth<1.0:
            out = self.st_depth_func(out, self.st_depth)            
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 block2,
                 layers,
                 num_segments,
                 kernels=[(3,3,3),(3,3,3),(3,3,3),(3,3,3)],
                 conv_modes=['conv','conv','conv','conv'],
                 transform=None,
                 groups=[1,1,1,1],
                 num_classes=1000,
                 zero_init_residual=True,
                 stochastic_depth=[1,0.8]
                 ):
        
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.num_segments = num_segments     
        
        ### stochastic depth
        self.st_depth_prob_now = stochastic_depth[0]
        self.st_depth_prob_delta =  stochastic_depth[0] -  stochastic_depth[1]
        self.st_depth_prob_step = self.st_depth_prob_delta/(sum(layers)-1)
                                                              
        self.layer1 = self._make_layer(block2, 64, layers[0], position=transform['position'][0], num_segments=num_segments, kernel=kernels[0], conv_mode=conv_modes[0], groups=groups[0], transform=transform)
        self.layer2 = self._make_layer(block2, 128, layers[1], position=transform['position'][1],  num_segments=num_segments, kernel=kernels[1], conv_mode=conv_modes[1], groups=groups[1], stride=2, transform=transform)
        self.layer3 = self._make_layer(block2, 256, layers[2], position=transform['position'][2],  num_segments=num_segments, kernel=kernels[2], conv_mode=conv_modes[2], groups=groups[2], stride=2, transform=transform)
        self.layer4 = self._make_layer(block2, 512, layers[3], position=transform['position'][3],  num_segments=num_segments, kernel=kernels[3], conv_mode=conv_modes[3], groups=groups[3], stride=2, transform=transform)
        
            
        self.avgpool = nn.AdaptiveAvgPool3d((num_segments, 1, 1))
        self.fc1 = nn.Conv1d(512*block.expansion, num_classes, kernel_size=1, stride=1, padding=0,bias=True)                   
        
        
        for m in self.modules():       
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
        
    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    position,
                    num_segments,
                    kernel=(3,3,3),
                    conv_mode='conv',
                    transform=None,
                    stride=1,
                    groups=1
                   ):       
        downsample = None        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, (1,stride,stride)),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                num_segments,
                kernel,
                stride,
                groups,
                downsample,
                conv_mode=conv_mode,
                stochastic_depth_prob=self.st_depth_prob_now
            )
        )
        self.st_depth_prob_now = self.st_depth_prob_now - self.st_depth_prob_step
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            remainder =int( i % 3)
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    num_segments,
                    kernel if i not in position else transform['kernel_size'],
                    groups,
                    remainder=remainder,
                    conv_mode=conv_mode if i not in position else transform['transform'],
                    config= None if i not in position else transform,
                    stochastic_depth_prob=self.st_depth_prob_now
                )
            )
            self.st_depth_prob_now = self.st_depth_prob_now - self.st_depth_prob_step
            
        return nn.Sequential(*layers)            

    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
           
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)    
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(x.size(0)*self.num_segments, -1,1)
        x = self.fc1(x)
        
        return x



def resnet50(pretrained=False, num_segments = 8, transform=None,  stochastic_depth=0.0, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    
    model = ResNet(Bottleneck, Bottleneck, [3, 4, 6, 3], num_segments=num_segments, transform=transform, stochastic_depth=[1,1-stochastic_depth],  **kwargs)
        
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        new_state_dict =  model.state_dict()

        for k, v in pretrained_dict.items():    
            if 'layer' in k:
                k_stage = k.split('.')[0]
                k_stage = int(k.split('.')[0].split('layer')[-1]) -1
                k_layer = int(k.split('.')[1])
                
                if (k_layer in transform['position'][k_stage]):
                    pass           
                else:
                    if 'conv' in k or 'downsample.0.weight' in k:
                        new_state_dict.update({k:v.unsqueeze(2)})      
                        print ("%s layer has pretrained weights" % k)            
                    else:
                        new_state_dict.update({k:v})      
                        print ("%s layer has pretrained weights" % k)
            elif 'fc' in k:
                pass        
            else:
                if 'conv' in k or 'downsample.0.weight' in k:
                    new_state_dict.update({k:v.unsqueeze(2)})      
                    print ("%s layer has pretrained weights" % k)            
                else:
                    new_state_dict.update({k:v})      
                    print ("%s layer has pretrained weights" % k)        
        model.load_state_dict(new_state_dict)
        
    return model