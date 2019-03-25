import torch.nn as nn
from config import *
import torchvision.models as models


class PoseProposalNet(nn.Module):
    def __init__(self, backbone, insize=(384,384), outsize=(12,12), keypoint_names = KEYPOINT_NAMES , local_grid_size= (9,9), edges = EDGES ):
        super(PoseProposalNet, self).__init__()
        self.insize = insize
        self.keypoint_names = keypoint_names
        self.edges = edges
        self.local_grid_size = local_grid_size
        #self.dtype = dtype

        #self.instance_scale = np.array(instance_scale)

        #self.outsize = self.get_outsize()
        self.outsize = outsize
        inW, inH = self.insize
        outW, outH = self.outsize
        sW, sH = self.local_grid_size
        self.gridsize = (int(inW / outW), int(inH / outH))
        self.lastsize = 6*(len(self.keypoint_names))+(sW)*(sH)*len(self.edges)

        #ResNet w/o avgpool&fc
        self.backbone = backbone

        # modified cnn layer
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=3//2, bias=False)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=3//2, bias=True)
        self.conv3 = nn.Conv2d(512, self.lastsize, kernel_size=1, stride=1)
        #self.conv3 = nn.Conv2d(512, 1311, kernel_size=1, stride=1)

        self.linear = nn.Linear(144,1024)
        self.lRelu = nn.LeakyReLU(0.1)
        self.Relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(512)

    def forward(self, input):

        #print("input type:", input.dtype)
        # load resnet 
        resnet_out = self.backbone(input)
        conv1_out = self.conv1(resnet_out)
        bn1 = self.bn(conv1_out)
        lRelu1 = self.lRelu(bn1)

        conv2_out = self.conv2(lRelu1)
        lRelu2 = self.lRelu(conv2_out)

        conv3_out = self.conv3(lRelu2)
        #out = self.sigmoid(conv3_out)
        out = self.Relu(conv3_out)

        return out

'''

    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    print(model_names)


parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
'''

if __name__ == '__main__':
    # create model
    arch = 'resnet18'

    print("=> creating model '{}'".format(arch))
    model = models.__dict__[arch]()

    # Detach under avgpoll layer in Resnet
    modules = list(model.children())[:-2]
    model = nn.Sequential(*modules)
 
    print(PoseProposalNet(model))
