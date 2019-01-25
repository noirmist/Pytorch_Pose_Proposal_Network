from __future__ import print_function, division
import os
import sys
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import pairwise

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.datasets as datasets
import torchvision.models as models
from torchsummary import summary

import json
import argparse
import math
import cv2

#from augment import *
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--file", help="json file path")
parser.add_argument("-img", "--root_dir", help="path to train2017 or val2018")

args = parser.parse_args()
root_dir = args.root_dir

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
    print("box0:", bbox0[0].shape)
    print("box1:", bbox1[0].shape)
    #print('area0:', area0.shape)
    #print('area1:', area1.shape)
    
    intersect = intersection(bbox0, bbox1)
    print('intersect:', intersect.shape)

    return intersect / (area0 + area1 - intersect + EPSILON)

def max_delta(delta1, delta2):
    batch = delta.shape[0]
    
    max_val = torch.max(delta1, delta2)

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


class KeypointsDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        """
        Args:
            file (string): path to the json file with annotations
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
        image = io.imread(img_name).astype(np.uint8)
        # center_x, center_y, visible_coco, width, height
        keypoints = np.array(self.annotations[idx]["keypoints"], dtype='float32').reshape(-1, 5)
        bbox = self.annotations[idx]['bbox']    # [center_x, center_y , width, height]
        sample = {'image': image, 'keypoints': keypoints, 'bbox': bbox}

        if self.transform:
            sample = self.transform(sample)

        return sample
   
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, bbox = sample['image'], sample['keypoints'][:,[0,1]], sample['bbox']
        h, w = image.shape[:2]

#        if isinstance(self.output_size, int):
#            if h > w:
#                new_h, new_w = self.output_size * h / w, self.output_size
#            else:
#                new_h, new_w = self.output_size, self.output_size * w / h
#        else:
#            new_h, new_w = self.output_size
#
#        new_h, new_w = int(new_h), int(new_w)

        new_h, new_w = self.output_size
        #scikit-learn resize transform
        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        diff_w = new_w/w
        diff_h = new_h/h
        
        landmarks = landmarks * [diff_w, diff_h]
        sample['keypoints'][:,[0,1]] = landmarks
        bbox = [ bbox[0] * diff_w , bbox[1] *diff_h, bbox[2] *diff_w, bbox[3]*diff_h]
        #print('rescale image', landmarks)
        
        return {'image': img, 'keypoints': sample['keypoints'], 'bbox': bbox}

#transform class
class AugmentColor(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        method = np.random.choice(
            ['random_distort', 'pil'],
            p=[self.prob, 1-self.prob])

        if method == 'random_distort':
            image = random_distort(image, contrast_low=0.3, contrast_high=2)
        if method == 'pil':
            image, _ = random_process_by_PIL(image)

        sample['image'] = image
        return sample

def rotate_point(point_xy, angle, center_xy):
    offset_x, offset_y = center_xy
    shift = point_xy - center_xy
    shift_x, shift_y = shift[0], shift[1]
    cos_rad = math.cos(math.radians(angle)) #np.cos(np.deg2rad(angle))
    sin_rad = math.sin(math.radians(angle)) #np.sin(np.deg2rad(angle))
    qx = offset_x + cos_rad * shift_x + sin_rad * shift_y
    qy = offset_y - sin_rad * shift_x + cos_rad * shift_y

    return np.array([qx, qy])


def rotate_image(image, angle, center_xy):
    print(center_xy, type(image), image.shape)
    cx,cy = center_xy
    rot = transforms.functional.rotate(Image.fromarray(image), angle, center=(cx, cy))
    # disable image collapse
    #rot = np.clip(rot, 0, 255)
    return rot 


def random_rotate(image, keypoints, bbox):
    #angle = np.random.uniform(-40, 40)
    angle=40.0
    param = {}
    param['angle'] = angle
    new_keypoints = keypoints.copy()
    #center_xy = np.array(image.shape[:2]) / 2 
    x, y, w, h  = bbox
    center_xy = np.array([x+w/2, y+h/2])

    for points in new_keypoints:
        if points[0] == 0.0 and points[1] == 0.0:
            continue
        rot_points = rotate_point(points[:2],
                                  angle,
                                  center_xy)
        points[:2] = rot_points

    new_bbox = []
    bbox_points = np.array(
        [
            [x, y], 
            [x + w, y], 
            [x, y+h], 
            [x + w, y + h]
        ]   
    )
    
    for idx, points in enumerate(bbox_points):
        if idx == 0:
            rot_points = rotate_point(
                points,
                angle,
                center_xy
            )
        else:
            temp = rotate_point(
                points,
                angle,
                center_xy
            )
            rot_points = np.vstack((rot_points,temp))

    xmax = np.max(rot_points[:, 0]) 
    ymax = np.max(rot_points[:, 1]) 
    xmin = np.min(rot_points[:, 0]) 
    ymin = np.min(rot_points[:, 1]) 
    # x,y,w,h
    new_bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

    image = rotate_image(image, angle, center_xy)
    return image, new_keypoints, new_bbox, param



class RandomRotate(object):
    def __call__(self, sample):
        image, keypoints, bbox = sample['image'], sample['keypoints'], sample['bbox']

        image, keypoints, bbox, _ = random_rotate(image, keypoints, bbox)

        sample['image'] = image
        sample['keypoints'] = keypoints
        sample['bbox'] = bbox
        return sample 

class Origin(object):
    def __call__(self, sample):
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, keypoints, bbox = sample['image'], sample['keypoints'], sample['bbox']

        # swap color axis because
        # PIL  image 
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.array(image).transpose((2, 0, 1))
        image = torch.from_numpy(image)
        image = image.float()
        print("ToTensor image dtype:", image.dtype)
        return {'image': image,
                'keypoints': torch.from_numpy(keypoints),
		'bbox': torch.from_numpy(np.asarray(bbox))}

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


    def forward(self, input):

        print("input type:", input.dtype)
        # load resnet 
        resnet_out = self.backbone(input)
        conv1_out = self.conv1(resnet_out)
        #TODO add batchnorm
        lRelu1 = self.lRelu(conv1_out)

        conv2_out = self.conv2(lRelu1)
        lRelu2 = self.lRelu(conv2_out)

        conv3_out = self.conv3(lRelu2)
        print("conv3_out:", conv3_out[0][0][0].shape)
        print("conv3_out:", conv3_out[0][0][0])

        out = self.linear(conv3_out.reshape(-1,self.lastsize, 144)).reshape(-1,self.lastsize, 32,32)
        print("network out type:", out.dtype)

        return out


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
        print("PPLloss init")
        self.batch_size = batch_size
        self.insize = insize
        self.keypoint_names = keypoint_names
        self.edges = edges
        self.local_grid_size = local_grid_size

        #TODO datatype [mpi, coco]
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
        print("PPLloss init done")

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
        print("feature_map shape:", feature_map.shape)
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
        print("resp", resp.dtype, resp.shape)
        print("conf", conf.dtype, conf.shape)
        print("sliced x", x.dtype, x.shape)
        print("sliced y", y.dtype, y.shape)
        print("sliced w", w.dtype, w.shape)
        print("sliced h", h.dtype, h.shape)
        print("sliced e", e.dtype, e.shape)
        print("feature_map", feature_map.dtype)

        
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

#        reporter.report({
#            'loss': loss,
#            'loss_resp': loss_resp,
#            'loss_iou': loss_iou,
#            'loss_coor': loss_coor,
#            'loss_size': loss_size,
#            'loss_limb': loss_limb
#        }, self)
        
        print("loss:", loss)
        print("loss_resp:", loss_resp)
        print("loss_iou:", loss_iou)
        print("loss_coor:", loss_coor)
        print("loss_size:", loss_size)
        print("loss_limb:", loss_limb)
        return loss


    def encode(self, bbox, keypoints):
        #in_data is Tensor
        #image = in_data['image'] # batch x channel x width x height
        #keypoints = in_data['keypoints'] # batch x keypoints x [ x, y, v, w, h ]

        #print("keypoints size:", keypoints.size())
        keypoints_xy = torch.narrow(keypoints, 2,0,2)
        print("keypoints xy size:", keypoints_xy.size())
        keypoints_wh = torch.narrow(keypoints, 2,3,2)
        print("keypoints wh size:", keypoints_wh.size())
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
#        max_delta_ij = torch.tensor(np.zeros((B, len(self.edges),
#                                outH, outW,
#                                self.local_grid_size[1], self.local_grid_size[0]), 
#                                dtype=np.float32)).cuda()

        or_delta = torch.zeros((B, len(self.edges), outH, outW), dtype=torch.float32).cuda()
        zeropad = nn.ZeroPad2d((self.local_grid_size[0]//2, self.local_grid_size[0]//2, self.local_grid_size[1]//2, self.local_grid_size[1]//2))
        padded_delta = zeropad(delta)
        
#        for idx in range(self.batch_size):
#            for ei, (s, t) in enumerate(self.edges):
#                #delta[idx][s], delta[idx][t]
#                start = delta[idx][s]
#                target = delta[idx][t]
#                for p in range(self.local_grid_size[1]//2 + self.outsize[1]):
#                    for q in range(self.local_grid_size[0]//2 + self.outsize[0]):
#                        kkkkkkk
#
#                        start[p][q]
                        
                         

        for idx in range(self.batch_size):
            for ei, (s, t) in enumerate(self.edges):
                or_delta[idx][ei] = torch.min(delta[idx][s] + delta[idx][t],
                                torch.ones(delta[idx][s].shape, dtype=torch.float32).cuda())

        mask = nn.MaxPool2d((self.local_grid_size[1], self.local_grid_size[0]), #kernel_size
                                stride=1,
                                padding=(self.local_grid_size[1] // 2, self.local_grid_size[0] // 2)).cuda()
        m = mask(or_delta)
         
        #for index, _ in np.ndenumerate(m):
        #    max_delta_ij[index] *= m[index]
        print("or_delta.shape:", or_delta.shape)
        print("m.shape:", m.shape)
        max_delta_ij = m.unsqueeze_(-1).expand(-1,-1,-1,-1,self.local_grid_size[0]).unsqueeze_(-1).expand(-1,-1,-1,-1,-1,self.local_grid_size[1])

        print("0.max_delta_ij:", max_delta_ij.shape)
#        for p in range(32):
#            for q in range(32):
#                print("(x,y): (",p,q,") : \n", max_delta_ij[0][0][p][q])
        max_delta_ij = max_delta_ij.permute(0,1,4,5,2,3)

        print("1.max_delta_ij:", max_delta_ij.shape, max_delta_ij.device)

        #return image, delta, max_delta_ij, tx, ty, tw, th, te
        return delta, max_delta_ij, tx, ty, tw, th, te
 
    

#face_dataset = KeypointsDataset(json_file = args.file, root_dir = args.root_dir)
#scale = Rescale((256,256))
##crop = RandomCrop(128)
##composed = transforms.Compose([Rescale(256),
##                               RandomCrop(224)])
#
## Apply each of the above transforms on sample.
#fig = plt.figure()
#sample = face_dataset[0]
#for i, tsfrm in enumerate([scale]):
#    transformed_sample = tsfrm(sample)
#
#    ax = plt.subplot(1, 1, i + 1)
#    plt.tight_layout()
#    ax.set_title(type(tsfrm).__name__)
#    show_landmarks(**transformed_sample)
#
#plt.show()

# Load training set
# Augmentation data

train_set = KeypointsDataset(json_file = args.file, root_dir = args.root_dir,
        transform=transforms.Compose([
            Rescale((384,384)),
            #RandomRotate(),
            ToTensor()
        ]))

for i in range(len(train_set)):
    sample = train_set[i]

    print(i, sample['image'].size(), sample['keypoints'].size(), sample['bbox'].size())

    if i == 3:
        break


dataloader = DataLoader(train_set, batch_size=4,
                        shuffle=False, num_workers=1)

#class PoseProposalNet(nn.Module):
#    def __init__(self, backbone, insize=(384,384), outsize=(32,32), keypoint_names = KEYPOINT_NAMES , local_grid_size= (9,9), edges = EDGES ):
       
# create model for resnet18
#device = torch.device('cuda')

model = models.__dict__['resnet18'](pretrained=True)

# Detach under avgpull layer in Resnet
modules = list(model.children())[:-2]
model = nn.Sequential(*modules)

model = PoseProposalNet(model).cuda()
loss = PPNLoss().cuda()
summary(model, (3,384,384))


#summary(model, (3,224,224))

for i_batch, x in enumerate(dataloader):

    img = x['image'].cuda()
    bbox = x['bbox'].cuda()
    label = x['keypoints'].cuda()
    
    out = model(img)
    #out cuda tensor
    loss(out, bbox, label)
    sys.exit(1)
    

## Helper function to show a batch
#def show_landmarks_batch(sample_batched):
#    """Show image with landmarks for a batch of samples."""
#    images_batch, landmarks_batch = \
#            sample_batched['image'], sample_batched['keypoints']
#    batch_size = len(images_batch)
#    im_size = images_batch.size(2)
#
#    grid = utils.make_grid(images_batch)
#    plt.imshow(grid.numpy().transpose((1,2,0)))
#
#    #print(landmarks_batch.shape, grid.shape)
#    for i in range(batch_size):
#        #print(i, ", x:", landmarks_batch[i, :, 0].numpy(), ", y:", landmarks_batch[i, :, 1].numpy())
#        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
#                    landmarks_batch[i, :, 1].numpy(),
#                    s=10, marker='.', c='r')
#
#        plt.title('Batch from dataloader')
#
#for i_batch, sample_batched in enumerate(dataloader):
#    print(i_batch, sample_batched['image'].size(),
#          sample_batched['keypoints'].size())
#
#    # observe 4th batch and stop.
#    if i_batch == 1:
#        plt.figure()
#        show_landmarks_batch(sample_batched)
#        plt.axis('off')
#        plt.ioff()
#        plt.show()
#        break
#
##Load model
##encode data
#
##apply transformation
#
##show examples
