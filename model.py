import torch
from torchsummary import summary
import sys
import torch.nn as nn
from config import *
import torchvision.models as models
from gelu import GELU

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1 

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x): 
        residual = x 

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        #out = self.relu(out)

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
            nn.Conv2d(512, 512,
                      kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
        )

        # Dilated Residual Network without last 2 layer
        self.backbone = backbone

        # shrink size
        self.basicblock1 = BasicBlock(512, 512, 2, downsample)
        self.basicblock2 = BasicBlock(512, 512, 1, None)

        # modified cnn layer
        self.conv1x1_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=1//2, bias=False)
        self.conv1x1_2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=1//2, bias=False)

        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=3//2, bias=False)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=3//2, bias=True)
        self.conv3 = nn.Conv2d(512, self.lastsize, kernel_size=1, stride=1)

        self.lRelu = nn.LeakyReLU(0.1)

        self.bn0_1 = nn.BatchNorm2d(512)
        self.bn0_2 = nn.BatchNorm2d(128)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(512)

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):

        # load resnet 
        resnet_out = self.backbone(input)

        # Add deleted basicblocks
        resnet_out = self.basicblock1(resnet_out)
        resnet_out = self.basicblock2(resnet_out)

        # Add last residual module + full pre activation
        conv1_out = self.bn0_1(resnet_out)
        conv1_out = self.lRelu(conv1_out)
        conv1_out = self.conv1x1_1(conv1_out)
        
        conv1_out = self.bn1(conv1_out)
        conv1_out = self.lRelu(conv1_out)
        conv1_out = self.conv1(conv1_out)
        

        conv1_out = self.bn0_2(conv1_out)
        conv1_out = self.lRelu(conv1_out)
        conv1_out = self.conv1x1_2(conv1_out)

        conv1_out += resnet_out

        conv2_out = self.conv2(conv1_out)
        bn2 = self.bn2(conv2_out)
        lRelu2 = self.lRelu(bn2)

        conv3_out = self.conv3(lRelu2)
        out = self.sigmoid(conv3_out)

        return out

if __name__ == '__main__':
    # create model
    import drn
    model = drn.drn_d_22()

    # Detach under avgpoll layer in Resnet
    modules = list(model.children())[:-2]
    model = nn.Sequential(*modules)
    model = PoseProposalNet(model, local_grid_size=(21,21)).cuda()
    for idx, ((name, value), value2) in enumerate(zip(model.named_parameters(),model.parameters())):
        print(idx, name, value.shape, value2.shape)

    summary(model, (3,384,384))
