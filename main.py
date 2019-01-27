import argparse
import os
import random
import shutil
import time
import warnings
import sys
import json
import math
import cv2

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision import utils
import torchvision.datasets as datasets
import torchvision.models as models

from skimage import io, transform
from skimage.color import gray2rgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import pairwise
from PIL import Image

import imgaug as ia
from imgaug import augmenters as iaa

from torchsummary import summary
from visdom import Visdom

name_list = [ 
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "thorax",
        "pelvis",
        "neck",
        "top"
]

COLOR_MAP = {
    'instance': (225, 225, 225),
    'nose': (255, 0, 0),
    'right_shoulder': (255, 170, 0),
    'right_elbow': (255, 255, 0),
    'right_wrist': (170, 255, 0),
    'left_shoulder': (85, 255, 0),
    'left_elbow': (0, 127, 0),
    'left_wrist': (0, 255, 85),
    'right_hip': (0, 170, 170),
    'right_knee': (0, 255, 255),
    'right_ankle': (0, 170, 255),
    'left_hip': (0, 85, 255),
    'left_knee': (0, 0, 255),
    'left_ankle': (85, 0, 255),
    'right_eye': (170, 0, 255),
    'left_eye': (255, 0, 255),
    'right_ear': (255, 0, 170),
    'left_ear': (255, 0, 85),
    'thorax': (128, 0, 80),
    'pelvis': (128, 80, 170),
    'neck': (255, 85, 0),
    'top': (200, 80, 0)
}

EDGES_BY_NAME = [
    ['instance', 'thorax'],
    ['thorax','neck'],
    ['neck', 'nose'],
    ['nose', 'top'],
    ['nose', 'left_eye'],
    ['left_eye', 'left_ear'],
    ['nose', 'right_eye'],
    ['right_eye', 'right_ear'],
    ['instance', 'left_shoulder'],
    ['left_shoulder', 'left_elbow'],
    ['left_elbow', 'left_wrist'],
    ['instance', 'right_shoulder'],
    ['right_shoulder', 'right_elbow'],
    ['right_elbow', 'right_wrist'],
    ['instance', 'pelvis'],
    ['pelvis', 'left_hip'],
    ['left_hip', 'left_knee'],
    ['left_knee', 'left_ankle'],
    ['pelvis', 'right_hip'],
    ['right_hip', 'right_knee'],
    ['right_knee', 'right_ankle'],
]

KEYPOINT_NAMES = ['instance'] + name_list
EDGES = [[KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(d)] for s, d in EDGES_BY_NAME]

TRACK_ORDER_0 = ['instance', 'thorax', 'neck', 'nose', 'left_eye', 'left_ear']
TRACK_ORDER_1 = ['instance', 'thorax', 'neck', 'nose', 'right_eye', 'right_ear']
TRACK_ORDER_2 = ['instance', 'thorax', 'neck', 'nose', 'top']
TRACK_ORDER_3 = ['instance', 'left_shoulder', 'left_elbow', 'left_wrist']
TRACK_ORDER_4 = ['instance', 'right_shoulder', 'right_elbow', 'right_wrist']
TRACK_ORDER_5 = ['instance', 'pelvis', 'left_hip', 'left_knee', 'left_ankle']
TRACK_ORDER_6 = ['instance', 'pelvis', 'right_hip', 'right_knee', 'right_ankle']

TRACK_ORDERS = [TRACK_ORDER_0, TRACK_ORDER_1, TRACK_ORDER_2, TRACK_ORDER_3, TRACK_ORDER_4, TRACK_ORDER_5, TRACK_ORDER_6]

DIRECTED_GRAPHS = []

for keypoints in TRACK_ORDERS:
    es = [EDGES_BY_NAME.index([a, b]) for a, b in pairwise(keypoints)]
    ts = [KEYPOINT_NAMES.index(b) for a, b in pairwise(keypoints)]
    DIRECTED_GRAPHS.append([es, ts])

EPSILON = 1e-6


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch PoseProposalNet Training')
parser.add_argument("-train", "--train_file", help="json file path")
parser.add_argument("-val", "--val_file", help="json file path")
parser.add_argument("-img", "--root_dir", help="path to train2017 or val2018")

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


# Network implementation
def parse_size(text):
    w, h = text.split('x')
    w = float(w)
    h = float(h)
    if w.is_integer():
        w = int(w)
    if h.is_integer():
        h = int(h)
    return w, h


def area(bbox):
    _, _, w, h = bbox
    return w * h 


def intersection(bbox0, bbox1):
    x0, y0, w0, h0 = bbox0
    x1, y1, w1, h1 = bbox1

    w = F.relu(torch.min(x0 + w0 / 2, x1 + w1 / 2) - torch.max(x0 - w0 / 2, x1 - w1 / 2))
    h = F.relu(torch.min(y0 + h0 / 2, y1 + h1 / 2) - torch.max(y0 - h0 / 2, y1 - h1 / 2))

    return w * h 


def iou(bbox0, bbox1):
    area0 = area(bbox0)
    area1 = area(bbox1)
    intersect = intersection(bbox0, bbox1)

    return intersect / (area0 + area1 - intersect + EPSILON)


def show_landmarks(image, keypoints, bbox):
    """Show image with keypoints"""
    print("show_landmarks:", type(image), image.dtype)
    plt.imshow(image)

    # change 0 to nan
    x = keypoints[:,0]
    x[x==0] = np.nan

    y = keypoints[:,1]
    y[y==0] = np.nan
    cx1,cy1,w,h = bbox
    rect = patches.Rectangle((cx1-w//2,cy1-h//2),w,h,linewidth=2,edgecolor='b',facecolor='none')
    plt.gca().add_patch(rect)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=30, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


class IAA(object):
    def __init__(self, output_size, mode):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.mode = mode

    def __call__(self, sample):
        image, keypoints, bbox = sample['image'], sample['keypoints'][:,[0,1]], sample['bbox']
        h, w = image.shape[:2]

        #filter existed keypoints , aka exclude zero value
        kps_coords = []
        kps = []
        keypoints = keypoints.tolist()
        for temp in keypoints:
            if temp[0] >0 and temp[1] >0: 
                kps_coords.append((temp[0],temp[1]))

        for kp_x, kp_y in kps_coords:
            kps.append(ia.Keypoint(kp_x, kp_y))

        bbs = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1 = bbox[0]-bbox[2]//2, 
                        y1=bbox[1]-bbox[3]//2,
                        x2=bbox[0]+bbox[2]//2, 
                        y2=bbox[1]+bbox[3]//2)
            ], shape=image.shape[:2])
        kps_oi = ia.KeypointsOnImage(kps, shape= image.shape[:2])
        if self.mode =='train':
            seq = iaa.Sequential([
                iaa.Affine(
                    rotate=(-40, 40),
                    scale=(0.35, 2.5)
                ), # random rotate by -40-40deg and scale to 35-250%, affects keypoints
                iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect keypoints
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.CropAndPad(
                    percent=(-0.2, 0.2),
                    pad_mode=["constant", "edge"],
                    pad_cval=(0, 128)
                ),
                iaa.Scale({"height": self.output_size[0], "width": self.output_size[1]})
            ])
        else:
            seq = iaa.Sequential([
                iaa.Affine(
                    rotate=(-40, 40),
                    scale=(0.35, 2.5)
                ), # random rotate by -40-40deg and scale to 35-250%, affects keypoints
                iaa.Multiply((0.5, 1.5)), # change brightness, doesn't affect keypoints
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Scale({"height": self.output_size[0], "width": self.output_size[1]})
            ])

        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_images([image])[0]
        keypoints_aug = seq_det.augment_keypoints([kps_oi])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    
        # update keypoints and bbox
        cnt = 0 
        for temp in keypoints:
            if temp[0] >0 and temp[1] >0: 
                temp[0] = keypoints_aug.keypoints[cnt].x
                temp[1] = keypoints_aug.keypoints[cnt].y
                cnt +=1

        keypoints = np.asarray(keypoints, dtype= np.float32)
        new_bbox = []
        for i in range(len(bbs_aug.bounding_boxes)):
            temp = bbs_aug.bounding_boxes[i]
            new_bbox.append((temp.x2+temp.x1)/2)
            new_bbox.append((temp.y2+temp.y1)/2)
            new_bbox.append((temp.x2-temp.x1))
            new_bbox.append((temp.y2-temp.y1))

        img = transform.resize(image_aug, (384, 384))
        sample['keypoints'][:,[0,1]] = keypoints 
        
        return {'image': img, 'keypoints': sample['keypoints'], 'bbox': new_bbox}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, keypoints, bbox = sample['image'], sample['keypoints'], sample['bbox']

        # swap color axis because
        # PIL  image 
        # numpy image: H x W x C
        # torch image: C X H X W
        #print("totensor img shape:", image.shape)
        #sys.stdout.flush()
        image = np.array(image).transpose((2, 0, 1))
        image = torch.from_numpy(image)
        image = image.float()
        return {'image': image,
                'keypoints': torch.from_numpy(keypoints),
                'bbox': torch.from_numpy(np.asarray(bbox))}

 
class KeypointsDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        """
        Args:
            json_file (string): path to the json file with annotations
            root_dir(string): path to Directory for images
            transform (optional):  transform to be applied on a sample.
        """
        json_data = open(json_file)
        self.annotations = json.load(json_data)["annotations"]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations[idx]['file_name'])
        image = gray2rgb(io.imread(img_name).astype(np.uint8))
        
        # center_x, center_y, visible_coco, width, height
        keypoints = np.array(self.annotations[idx]["keypoints"], dtype='float32').reshape(-1, 5)
        bbox = self.annotations[idx]['bbox']    # [center_x, center_y , width, height]
        sample = {'image': image, 'keypoints': keypoints, 'bbox': bbox}

        if self.transform:
            sample = self.transform(sample)

        return sample

#network
class PoseProposalNet(nn.Module):
    def __init__(self, backbone, insize=(384,384), outsize=(32,32), keypoint_names = KEYPOINT_NAMES , local_grid_size= (9,9), edges = EDGES ):
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
        #TODO batch norm
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=3//2, bias=False)
        self.conv3 = nn.Conv2d(512, self.lastsize, kernel_size=1, stride=1)
        #self.conv3 = nn.Conv2d(512, 1311, kernel_size=1, stride=1)

        self.linear = nn.Linear(144,1024)
        self.lRelu = nn.LeakyReLU(0.1)
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
        #print("conv3_out:", conv3_out[0][0][0].shape)
        #print("conv3_out:", conv3_out[0][0][0])

        out = self.linear(conv3_out.reshape(-1,self.lastsize, 144)).reshape(-1,self.lastsize, 32,32)
        #print("network out type:", out.dtype)

        return out


#loss function
class PPNLoss(nn.Module):
    def __init__(self, batch_size=4,
                insize=(384,384),
                outsize=(32,32),
                keypoint_names = KEYPOINT_NAMES , local_grid_size= (9,9), edges = EDGES,
                width_multiplier=1.0,
                lambda_resp=0.25,
                lambda_iou=1.0,
                lambda_coor=5.0,
                lambda_size=5.0,
                lambda_limb=0.5
                ):
        super(PPNLoss, self).__init__()
        #print("PPLloss init")
        self.batch_size = batch_size
        self.insize = insize
        self.keypoint_names = keypoint_names
        self.edges = edges
        self.local_grid_size = local_grid_size

        self.dtype = np.float32

        #TODO implement get_outsize
        #self.outsize = self.get_outsize()
        self.outsize = outsize
        inW, inH = self.insize
        outW, outH = self.outsize
        sW, sH = self.local_grid_size
        self.gridsize = (int(inW / outW), int(inH / outH))

        #set loss parameters
        self.lambda_resp = lambda_resp
        self.lambda_iou = lambda_iou
        self.lambda_coor = lambda_coor
        self.lambda_size = lambda_size
        self.lambda_limb = lambda_limb
        #print("PPLloss init done")

    def restore_xy(self, x, y):
        gridW, gridH = self.gridsize
        outW, outH = self.outsize
        X, Y = torch.Tensor(np.meshgrid(np.arange(outW, dtype=np.float32), np.arange(outH, dtype=np.float32))).cuda()
        return (x + X) * gridW, (y + Y) * gridH
    def restore_size(self, w, h):
        inW, inH = self.insize
        return inW * w, inH * h

    # forward input
    def forward(self, feature_map, bbox, target):
        #encoding target
        delta, max_delta_ij, tx, ty, tw, th, te = self.encode(bbox, target)

        K = len(self.keypoint_names)
        B = self.batch_size
        #B, _, _, _ = image.shape
        outW, outH = self.outsize

        #feature_map = self.forward(image)      
        #print("feature_map shape:", feature_map.shape)
        #loss function with torch
        resp = feature_map[:, 0 * K:1 * K, :, :]
        conf = feature_map[:, 1 * K:2 * K, :, :]
        x = feature_map[:, 2 * K:3 * K, :, :]
        y = feature_map[:, 3 * K:4 * K, :, :]
        w = feature_map[:, 4 * K:5 * K, :, :]
        h = feature_map[:, 5 * K:6 * K, :, :]
        e = feature_map[:, 6 * K:, :, :].reshape((
            B,
            len(self.edges),
            self.local_grid_size[1], self.local_grid_size[0],
            outH, outW
        ))
#        print("resp", resp.dtype, resp.shape)
#        print("conf", conf.dtype, conf.shape)
#        print("sliced x", x.dtype, x.shape)
#        print("sliced y", y.dtype, y.shape)
#        print("sliced w", w.dtype, w.shape)
#        print("sliced h", h.dtype, h.shape)
#        print("sliced e", e.dtype, e.shape)
#        print("feature_map", feature_map.dtype)
        (rx, ry), (rw, rh) = self.restore_xy(x, y), self.restore_size(w, h)
        (rtx, rty), (rtw, rth) = self.restore_xy(tx, ty), self.restore_size(tw, th)
        #torch.set_printoptions(threshold=10000)
        #print("tx:", tx.shape, tx[0][0])
        #print("rtx:", rtx.shape, rtx[0][0])
        ious = iou((rx, ry, rw, rh), (rtx, rty, rtw, rth))

        # add weight where can't find keypoint
        #xp = get_array_module(max_delta_ij)
        zero_place = torch.zeros(max_delta_ij.shape).cuda()
        zero_place[max_delta_ij < 0.5] = 0.0005
        weight_ij = torch.min(max_delta_ij + zero_place,
                                torch.ones(zero_place.shape, dtype=torch.float32).cuda())

        # add weight where can't find keypoint
        #zero_place = np.zeros(delta.shape).astype(self.dtype)
        zero_place = torch.zeros(delta.shape).cuda()
        zero_place[delta < 0.5] = 0.0005
        #weight = np.minimum(delta + zero_place, 1.0)
        weight = torch.min(delta + zero_place,
                                torch.ones(zero_place.shape, dtype=torch.float32).cuda())

        half = torch.zeros(delta.shape).cuda()
        half[delta < 0.5] = 0.5

        loss_resp = torch.sum((resp - delta)**2, (2,3))
        loss_iou = torch.sum(delta * (conf - ious)**2, (2,3))
        loss_coor = torch.sum(weight * ((x - tx - half)**2 + (y - ty - half)**2), (2,3))
        loss_size = torch.sum(weight * ((torch.sqrt(torch.abs(w + EPSILON)) - torch.sqrt(torch.abs(tw + EPSILON)))**2 +
                                    (torch.sqrt(torch.abs(h + EPSILON)) - torch.sqrt(torch.abs(th + EPSILON)))**2 ), (2,3))
        loss_limb = torch.sum(weight_ij * (e - te)**2, (2,3,4,5))

        loss_resp = torch.mean(loss_resp)
        loss_iou = torch.mean(loss_iou)
        loss_coor = torch.mean(loss_coor)
        loss_size = torch.mean(loss_size)
        loss_limb = torch.mean(loss_limb)

        loss = self.lambda_resp * loss_resp + \
            self.lambda_iou * loss_iou + \
            self.lambda_coor * loss_coor + \
            self.lambda_size * loss_size + \
            self.lambda_limb * loss_limb
        #print("loss:", loss)
        #print("loss_resp:", loss_resp)
        #print("loss_iou:", loss_iou)
        #print("loss_coor:", loss_coor)
        #print("loss_size:", loss_size)
        #print("loss_limb:", loss_limb)
        return loss


    def encode(self, bbox, keypoints):
        #print("keypoints size:", keypoints.size())
        keypoints_xy = torch.narrow(keypoints, 2,0,2)
        #print("keypoints xy size:", keypoints_xy.size())
        keypoints_wh = torch.narrow(keypoints, 2,3,2)
        #print("keypoints wh size:", keypoints_wh.size())
        visible = torch.narrow(keypoints, 2,2,1)

        B = self.batch_size

        #bbox = in_data['bbox'] # [x1, y1, w, h]

        #is_labeled = in_data['is_labeled']
        #dataset_type = in_data['dataset_type']
        #is_visible = in_data['is_visible']

        inW, inH = self.insize
        outW, outH = self.outsize
        gridW, gridH = self.gridsize
        K = len(self.keypoint_names)

        delta = torch.zeros((B, K, outH, outW), dtype=torch.float32).cuda()
        tx = torch.zeros((B, K, outH, outW), dtype=torch.float32).cuda()
        ty = torch.zeros((B, K, outH, outW), dtype=torch.float32).cuda()
        tw = torch.zeros((B, K, outH, outW), dtype=torch.float32).cuda()
        th = torch.zeros((B, K, outH, outW), dtype=torch.float32).cuda()
        te = torch.zeros((B,
            len(self.edges),
            self.local_grid_size[1], self.local_grid_size[0],
            outH, outW),
            dtype=torch.float32).cuda()
        # Set delta^i_k
        for idx, ((box_x, box_y, box_w, box_h), points, pointswh, v) in enumerate(zip(bbox, keypoints_xy, keypoints_wh, visible)):
            # mpi bbox instance converted to coco bbox instance format (total human)

            # parts already defined by segmentation data and number of keypoints which encoded in keypoints objects
            # process each idx-th batch
            points =  list([torch.tensor([box_x, box_y], dtype=torch.float32).cuda()]) + list(points)
            pointswh =  list([torch.tensor([box_w, box_h], dtype=torch.float32).cuda()]) + list(pointswh)
            visibled = list(torch.tensor([[2.]]).cuda()) + list(v)

            #print("visibled:", visibled) 
            #print("points:", points)
            #print("pointswh:", pointswh)

            for k, (xy, pwh, v) in enumerate(zip(points, pointswh, visibled)):
                #print(k, ", v:", v)
                if v <= 0:  # filtered unlabeled 
                    continue
                cx = xy[0] / gridW
                cy = xy[1] / gridH

                #find appropriate grid which include center point
                ix, iy = int(cx), int(cy)
                sizeW, sizeH = pwh
                #print("ix,iy:", ix, iy, outW, outH) 
                if 0 <= iy < outH and 0 <= ix < outW:
                    #print("k,iy,ix:", k, iy, ix)
                    delta[idx, k, iy, ix] = 1
                    tx[idx, k, iy, ix] = cx - ix
                    ty[idx, k, iy, ix] = cy - iy
                    tw[idx, k, iy, ix] = sizeW / inW
                    th[idx, k, iy, ix] = sizeH / inH

            #np.set_printoptions(threshold=np.nan)
            for ei, (s, t) in enumerate(self.edges):
                if visibled[s] <= 0:
                    continue
                if visibled[t] <= 0:
                    continue

                src_xy = points[s]
                tar_xy = points[t]
                ixy = (int(src_xy[0] / gridW), int(src_xy[1] / gridH))
                jxy = (int(tar_xy[0] / gridW) - ixy[0] + self.local_grid_size[0] // 2,
                       int(tar_xy[1] / gridH) - ixy[1] + self.local_grid_size[1] // 2)

                if ixy[0] < 0 or ixy[1] < 0 or ixy[0] >= outW or ixy[1] >= outH:
                    continue
                if jxy[0] < 0 or jxy[1] < 0 or jxy[0] >= self.local_grid_size[0] or jxy[1] >= self.local_grid_size[1]:
                    continue

                # edge, tar_y, tar_x, start_y, start_x
                te[idx, ei, jxy[1], jxy[0], ixy[1], ixy[0]] = 1

        # define max(delta^i_k1, delta^j_k2) which is used for loss_limb
        or_delta = torch.zeros((B, len(self.edges), outH, outW), dtype=torch.float32).cuda()
        zeropad = nn.ZeroPad2d((self.local_grid_size[0]//2, self.local_grid_size[0]//2, self.local_grid_size[1]//2, self.local_grid_size[1]//2))
        padded_delta = zeropad(delta)

        for idx in range(self.batch_size):
            for ei, (s, t) in enumerate(self.edges):
                or_delta[idx][ei] = torch.min(delta[idx][s] + delta[idx][t],
                                torch.ones(delta[idx][s].shape, dtype=torch.float32).cuda())

        mask = nn.MaxPool2d((self.local_grid_size[1], self.local_grid_size[0]), #kernel_size
                                stride=1,
                                padding=(self.local_grid_size[1] // 2, self.local_grid_size[0] // 2)).cuda()
        m = mask(or_delta)
        max_delta_ij = m.unsqueeze_(-1).expand(-1,-1,-1,-1,self.local_grid_size[0]).unsqueeze_(-1).expand(-1,-1,-1,-1,-1,self.local_grid_size[1])

        max_delta_ij = max_delta_ij.permute(0,1,4,5,2,3)

        return delta, max_delta_ij, tx, ty, tw, th, te



best_loss = float("inf")

def main():
    args = parser.parse_args()


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_loss
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Data loading code
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    train_dataset = KeypointsDataset(json_file = args.train_file, root_dir = args.root_dir+"/train2017/",
                    transform=transforms.Compose([
                        IAA((384,384),'train'),
                        ToTensor(),
                    ]))

    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None


    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = KeypointsDataset(json_file = args.val_file, root_dir = args.root_dir+"/val2017/",
                transform=transforms.Compose([
                    IAA((384,384),'val'),
                    ToTensor(),
                ]))


    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # Detach under avgpoll layer in Resnet
    modules = list(model.children())[:-2]
    model = nn.Sequential(*modules)
    model = PoseProposalNet(model)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    # define custom loss 
    criterion = PPNLoss(batch_size = args.batch_size).cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return


    # Start trainin iterations
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        epoch_loss = validate(val_loader, model, criterion, epoch, args)
        plotter.plot('loss', 'val', 'PPN Loss', epoch*len(train_loader), epoch_loss) 

        # remember best acc@1 and save checkpoint
        is_best = epoch_loss < best_loss
        best_loss = min(epoch_loss, best_loss)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            print("checkpoints is saved!!!")
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, target in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        adjust_learning_rate(optimizer, i, args)

#        if args.gpu is not None:
#            img = target['image'].cuda(args.gpu, non_blocking=True)
#            bbox = target['bbox'].cuda(args.gpu, non_blocking=True)
#            label = target['keypoints'].cuda(args.gpu, non_blocking=True)

        img = target['image'].cuda()
        bbox = target['bbox'].cuda()
        label = target['keypoints'].cuda()

        # compute output
        output = model(img)
        loss = criterion(output, bbox, label)

        # measure accuracy and record loss
        losses.update(loss.item(), img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            plotter.plot('loss', 'train', 'PPN Loss', epoch*len(train_loader)+i, losses.avg) 

#TODO function for evaluation like OKS

def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, target in enumerate(val_loader):
            img = target['image'].cuda()
            bbox = target['bbox'].cuda()
            label = target['keypoints'].cuda()

            # compute output
            output = model(img)
            loss = criterion(output, target)

            # measure and record loss
            losses.update(loss.item(), img.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                       epoch, i, len(val_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses))
    return losses.avg


def save_checkpoint(state, is_best, filename='PPN_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'PPN_model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Iteration',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

def adjust_learning_rate(optimizer, iters, args):
    """Sets the learning rate to the initial LR decayed linearly each iteration"""
    lr = 0.007 * (1  - iters/260000)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    import logging
    global plotter
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    plotter = VisdomLinePlotter(env_name="PoseProposalNet")
    main()

#b^i_k 
#1<i< hxw - grid cell
#k = 0... K(number of parts)
#k=0 person instances
#
#bik - confidence of the bounding box
#    - the coordinates, width, height of bounding box
#bik = {p(R|k,i), p(I|R,k,i), oixk, oiyk, wik, hik}
#
#R, I  - binary random variable
#
#p(R|k,i) probability of the grid cell i responsible for detection of k
# - center of Gt box of k fall into a grid cell
#
#p(I|R,k,i) how well the bounding box predicted in i fits k
# - IoU between predicted bbox and GT bbox
#
#(oixk, oiyk) center of the bounding box (realtive to bounds of the grid cell with scale normalized by length of the cells)
#
#wik, hik  normalized by length of image width and height
#
#bounding boxes of person instances - rect of entire body or head
#
#parts are grid-wise detected in out method and the box sizes are supervised proportional to the person scale
#eg. one fifth of the length of the upper body or half the head segment length
#question(how to figure out upper body and head) - ?
#
#C k1, k2 encodes a set of probability presence of each limb
#C_k1,k2 = {p(C|k1,k2,x,x+dx)} dx in X
#C is binary random variable
#
#each prediction corresponds to each channel in the depth of output 3d tensor
#size HxWx{6(K+1) + H`W`|L|}
#
## NMS for these RPs
#probabilistic, greedy parsing algorithm
#Dnk : Confidence score for detection of the n-th RP of k
# = p(R|k,i) p(I|R,k,n)
#En1n2k1k2 :Confidence score of the limb 
# = p(C|k1,k2,x,x+dx)
#
#Part association uses pairwise part association scores
#Z = {Zn1n2k1k2|(k1,k2) in L, n1 in N1, n2 in N2}
#optimal assignment problem for the set of all the possible

# Parse the merged RPs into individual people and generate pose proposals

