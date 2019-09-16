import torch
import sys
import torch.nn as nn
from config import *
import torchvision.models as models
from gelu import GELU

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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

class PoseProposalNet(nn.Module):
    def __init__(self, backbone, insize=(384,384), outsize=(24,24), keypoint_names = KEYPOINT_NAMES , local_grid_size= (21,21), edges = EDGES ):
        super(PoseProposalNet, self).__init__()
        self.insize = insize
        self.keypoint_names = keypoint_names
        self.edges = edges
        self.local_grid_size = local_grid_size

        self.outsize = outsize
        inW, inH = self.insize
        outW, outH = self.outsize
        sW, sH = self.local_grid_size
        self.gridsize = (int(inW / outW), int(inH / outH))
        self.lastsize = 6*(len(self.keypoint_names))+(sW)*(sH)*len(self.edges)

        downsample = nn.Sequential(
                conv1x1(256, 512, 1),
                nn.BatchNorm2d(512),
            )

        #ResNet w/o avgpool&fc
        self.backbone = backbone

        self.basicblock1 = BasicBlock(256, 512, 1, downsample)
        self.basicblock2 = BasicBlock(512, 512, 1, None)

        # modified cnn layer
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=3//2, bias=False)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=3//2, bias=True)
        self.conv3 = nn.Conv2d(512, self.lastsize, kernel_size=1, stride=1)

        #self.linear = nn.Linear(144,1024)
        self.lRelu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.Relu = nn.ReLU()
#        self.Gelu = GELU()
#        self.dropout = nn.Dropout2d(p=0.2)
#        self.dropout5 = nn.Dropout2d(p=0.5)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):

        #print("input type:", input.dtype)
        # load resnet 
        resnet_out = self.backbone(input)

        # Add deleted basicblocks
        resnet_out = self.basicblock1(resnet_out)
        resnet_out = self.basicblock2(resnet_out)

        conv1_out = self.conv1(resnet_out)
        bn1 = self.bn1(conv1_out)
        lRelu1 = self.lRelu(bn1)

        conv2_out = self.conv2(lRelu1)
        bn2 = self.bn2(conv2_out)
        lRelu2 = self.lRelu(bn2)

        conv3_out = self.conv3(lRelu2)
        out = self.sigmoid(conv3_out)

        return out

if __name__ == '__main__':
    # create model
    arch = 'resnet18'

    print("=> creating model '{}'".format(arch))
    model = models.__dict__[arch]()

    # Detach under avgpoll layer in Resnet
    #modules = list(model.children())[:-4]
    modules = list(model.children())[:-3]
    model = nn.Sequential(*modules)
    model = PoseProposalNet(model, local_grid_size=(21,21))
    print(model)

    inputs = torch.randn(1,3, 384, 384)
    #inputs = torch.randn(1,3, 497, 497)
    #inputs = torch.randn(1,3, 512, 512)
    y = model(inputs)
    _, _, outH, outW =y.shape
    outsize = (outW, outH)
    print("outsize:",outsize)
    print("y:",y.shape)

