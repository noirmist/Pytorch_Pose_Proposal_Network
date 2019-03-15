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
import functools

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

#from utils import pairwise
from PIL import Image

import imgaug as ia
from imgaug import augmenters as iaa

from torchsummary import summary
from visdom import Visdom

#Custom models
from model import *
from config import *
from dataset import *


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch PoseProposalNet Training')
parser.add_argument("-train", "--train_file", help="json file path")
parser.add_argument("-val", "--val_file", help="json file path")
parser.add_argument("-img", "--root_dir", help="path to train2017 or val2018")
parser.add_argument("-imsize", "--image_size", default=384, help="set input image size")

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
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
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


# Aument implementation
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


def show_landmarks(image, keypoints, bbox, fname):
    """Show image with keypoints"""
    #print("show_landmarks:", type(image), image.dtype)

    image = image.numpy().astype(np.uint8)
    image = np.array(image).transpose((1, 2, 0))
    bboxes = bbox.numpy()
    keypoints = keypoints.reshape(-1,2).numpy()
    
    fig = plt.figure()
    plt.imshow(image)

    # change 0 to nan
    x = keypoints[:,0]
    x[x==0] = np.nan

    y = keypoints[:,1]
    y[y==0] = np.nan

    for (cx1,cy1,w,h) in bboxes:
        rect = patches.Rectangle((cx1-w//2,cy1-h//2),w,h,linewidth=2,edgecolor='b',facecolor='none')
        plt.gca().add_patch(rect)

    plt.scatter(keypoints[:, 0], keypoints[:, 1], marker='.', c='r')
    plt.pause(0.0001)  # pause a bit so that plots are updated
    plt.savefig("aug_img/"+fname) 
    plt.close()

class IAA(object):
    def __init__(self, output_size, mode):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.mode = mode

    def __call__(self, sample):
        #sample = {'image': image, 'keypoints': keypoints, 'bbox': bboxes, 'is_visible':is_visible, 'size': size}
        image, keypoints, bboxes, is_visible, size = sample['image'], sample['keypoints'], sample['bbox'], sample['is_visible'], sample['size']

        
        h, w = image.shape[:2]

        #filter existed keypoints , aka exclude zero value
        kps_coords = []
        kps = []
        #keypoints = keypoints.reshape(-1,2).tolist()
        for temp in keypoints:
            if temp[0] >0 and temp[1] >0: 
                kps_coords.append((temp[0],temp[1]))

        for kp_x, kp_y in kps_coords:
            kps.append(ia.Keypoint(x=kp_x, y=kp_y))

        box_list = []
        for bbox in bboxes:
            box_list.append(ia.BoundingBox(x1 = bbox[0]-bbox[2]//2, 
                                y1=bbox[1]-bbox[3]//2,
                                x2=bbox[0]+bbox[2]//2, 
                                y2=bbox[1]+bbox[3]//2))

        bbs = ia.BoundingBoxesOnImage(box_list, shape=image.shape)
        kps_oi = ia.KeypointsOnImage(kps, shape=image.shape)
        if self.mode =='train':
            seq = iaa.Sequential([
                iaa.Affine(
                    rotate=(-40, 40),
                    scale=(0.25, 2.5),
                    fit_output=True
                ), # random rotate by -40-40deg and scale to 35-250%, affects keypoints
                iaa.Multiply((0.5, 1.5)), # change brightness, doesn't affect keypoints
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
#                iaa.CropAndPad(
#                    percent=(-0.2, 0.2),
#                    pad_mode=["constant", "edge"],
#                    pad_cval=(0, 128)
#                ),
                iaa.Scale({"height": self.output_size[0], "width": self.output_size[1]})
            ])

        else:
            seq = iaa.Sequential([
                iaa.Affine(
                    rotate=(-40, 40),
                    scale=(0.25, 2.5)
                ), # random rotate by -40-40deg and scale to 35-250%, affects keypoints
                iaa.Multiply((0.8, 1.5)), # change brightness, doesn't affect keypoints
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Scale({"height": self.output_size[0], "width": self.output_size[1]})
            ])

        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_images([image])[0]
        keypoints_aug = seq_det.augment_keypoints([kps_oi])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image() 


        # update keypoints and bbox
        cnt = 0
        for temp in keypoints:
            if temp[0] >0 and temp[1] >0:
                # ignore outside keypoints
                if keypoints_aug.keypoints[cnt].x >0 and keypoints_aug.keypoints[cnt].x < image_aug.shape[1] and \
                    keypoints_aug.keypoints[cnt].y >0 and keypoints_aug.keypoints[cnt].y < image_aug.shape[0]:
                    temp[0] = keypoints_aug.keypoints[cnt].x
                    temp[1] = keypoints_aug.keypoints[cnt].y
                else:
                    temp[0] = 0.0 
                    temp[1] = 0.0 
                cnt +=1 

        keypoints = np.asarray(keypoints, dtype= np.float32)
        new_bboxes = []
        if len(bbs_aug.bounding_boxes) > 0:
            for i in range(len(bbs_aug.bounding_boxes)):
                new_bbox = []
                temp = bbs_aug.bounding_boxes[i]
                new_bbox.append((temp.x2+temp.x1)/2)    #center x
                new_bbox.append((temp.y2+temp.y1)/2)    #center y
                new_bbox.append((temp.x2-temp.x1))      #width
                new_bbox.append((temp.y2-temp.y1))      #height
                new_bboxes.append(new_bbox)
        else:
            new_bbox = [0.0,0.0,0.0,0.0]

        #img = transform.resize(image_aug, (self.output_size[0], self.output_size[1]))
        sample['keypoints'][:,[0,1]] = keypoints 
        #sample = {'image': image, 'keypoints': keypoints, 'bbox': bboxes, 'is_visible':is_visible, 'size': size}
        return {'image': image_aug, 'keypoints': sample['keypoints'], 'bbox': new_bboxes, 'is_visible':is_visible, 'size': size}


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
                'bbox': torch.from_numpy(np.asarray(bbox)),
                'size': sample['size'],
                'is_visible': sample['is_visible']}


#loss function
class PPNLoss(nn.Module):
    def __init__(self,
                insize=(384,384),
                outsize=(12,12),
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
    #loss = criterion(output, delta, max_delta_ij, tx, ty, tw, th, te)
    def forward(self, image, feature_map, delta, max_delta_ij, tx, ty, tw, th, te):
        #encoding target
        #delta, max_delta_ij, tx, ty, tw, th, te = self.encode(bboxes, targets, size, is_visible)

        ## TODO
        K = len(self.keypoint_names)
        B, _, _, _ = image.shape
        outW, outH = self.outsize

        #loss function with torch
        resp = feature_map[:, 0 * K:1 * K, :, :]
        conf = feature_map[:, 1 * K:2 * K, :, :]
        x = feature_map[:, 2 * K:3 * K, :, :]
        y = feature_map[:, 3 * K:4 * K, :, :]
        w = feature_map[:, 4 * K:5 * K, :, :]
        h = feature_map[:, 5 * K:6 * K, :, :]
        e = feature_map[:, 6 * K:, :, :].reshape(
            B,
            len(self.edges),
            self.local_grid_size[1], self.local_grid_size[0],
            outH, outW
        )

#        print("feature_map shape:", feature_map.shape)
#        print("resp", resp.dtype, resp.shape)
#        print("conf", conf.dtype, conf.shape)
#        print("sliced x", x.dtype, x.shape)
#        print("sliced y", y.dtype, y.shape)
#        print("sliced w", w.dtype, w.shape)
#        print("sliced h", h.dtype, h.shape)
#        print("sliced e", e.dtype, e.shape)
#        print("te shape :", te.shape)
#        sys.stdout.flush()

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
        loss_resp = torch.sum((resp - delta)**2, tuple(range(1, resp.dim())) )
        loss_iou = torch.sum(delta * (conf - ious)**2, tuple(range(1, conf.dim())))
        loss_coor = torch.sum(weight * ((x - tx - half)**2 + (y - ty - half)**2), tuple(range(1, x.dim())))
        loss_size = torch.sum(weight * ((torch.sqrt(torch.abs(w + EPSILON)) - torch.sqrt(torch.abs(tw + EPSILON)))**2 + (torch.sqrt(torch.abs(h + EPSILON)) - torch.sqrt(torch.abs(th + EPSILON)))**2 ), tuple(range(1, w.dim())))
        loss_limb = torch.sum(weight_ij * (e - te)**2, tuple(range(1, e.dim())))

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

#        print("loss:", loss.item())
#        print("loss_resp:", loss_resp.item())
#        print("loss_iou:", loss_iou.item())
#        print("loss_coor:", loss_coor.item())
#        print("loss_size:", loss_size.item())
#        print("loss_limb:", loss_limb.item())
    
        return loss, loss_resp, loss_iou, loss_coor, loss_size, loss_limb


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

    # Test model to get outsize
    inputs = torch.randn(1,3, args.image_size, args.image_size)
    y =  model(inputs)
    _, _, outH, outW =y.shape
    outsize = (outW, outH)
    insize = (args.image_size, args.image_size)

    # Data loading code
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    train_dataset = KeypointsDataset(json_file = args.train_file, root_dir = args.root_dir+"/train2017/",
                    transform=transforms.Compose([
                        IAA(insize,'train'),
                        ToTensor()
                    ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    collate_fn = functools.partial(custom_collate_fn, 
                                insize=insize,
                                outsize = outsize, 
                                keypoint_names = KEYPOINT_NAMES ,
                                local_grid_size = (9,9),
                                edges = EDGES)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, collate_fn = collate_fn, pin_memory=True, sampler=train_sampler)

    val_dataset = KeypointsDataset(json_file = args.val_file, root_dir = args.root_dir+"/val2017/",
                transform=transforms.Compose([
                    IAA(insize,'val'),
                    ToTensor()
                ]))


    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn = collate_fn, pin_memory=True)


    # define loss function (criterion) and optimizer
    # define custom loss 
    criterion = PPNLoss(
                insize=insize,
                outsize=outsize,
                keypoint_names = KEYPOINT_NAMES,
                local_grid_size= (9,9), 
                edges = EDGES,
                width_multiplier=1.0,
                lambda_resp=0.25,
                lambda_iou=1.0,
                lambda_coor=5.0,
                lambda_size=5.0,
                lambda_limb=0.5
            ).cuda()

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
        validate(val_loader, model, criterion, 1, args)
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
            print("checkpoints checking")
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best, epoch+1, args.save)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_resp = AverageMeter()
    losses_iou = AverageMeter()
    losses_coor = AverageMeter()
    losses_size = AverageMeter()
    losses_limb = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (target_img, delta, max_delta_ij, tx, ty, tw, th, te) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        adjust_learning_rate(optimizer, i, args)

        img = target_img.cuda()
        delta = delta.cuda()
        max_delta_ij = max_delta_ij.cuda()
        tx = tx.cuda()
        ty = ty.cuda()
        tw = tw.cuda()
        th = th.cuda()
        te = te.cuda()

        # compute output
        output = model(img)
        loss, loss_resp, loss_iou, loss_coor, loss_size, loss_limb = criterion(img, output, delta, max_delta_ij, tx, ty, tw, th, te)

        # measure accuracy and record loss
        losses.update(loss.item(), img.size(0))
        losses_resp.update(loss_resp.item(), img.size(0))
        losses_iou.update(loss_iou.item(), img.size(0))
        losses_coor.update(loss_coor.item(), img.size(0))
        losses_size.update(loss_size.item(), img.size(0))
        losses_limb.update(loss_limb.item(), img.size(0))


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
                  'Loss {loss.val:.4f} ({loss.avg:.4f}: {loss_resp.avg:.4f} + '
                  '{loss_iou.avg:.4f} + {loss_coor.avg:.4f} + '
                  '{loss_size.avg:.4f} + {loss_limb.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, loss_resp=losses_resp, loss_iou=losses_iou, loss_coor=losses_coor, loss_size=losses_size, loss_limb=losses_limb))
            sys.stdout.flush()
            plotter.plot('loss', 'train', 'PPN Loss', epoch*len(train_loader)+i, losses.avg) 

#TODO function for evaluation like OKS

def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    losses_resp = AverageMeter()
    losses_iou = AverageMeter()
    losses_coor = AverageMeter()
    losses_size = AverageMeter()
    losses_limb = AverageMeter()

    end = time.time()


    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, (target_img, delta, max_delta_ij, tx, ty, tw, th, te) in enumerate(val_loader):
            
            data_time.update(time.time() - end)

            img = target_img.cuda()
            delta = delta.cuda()
            max_delta_ij = max_delta_ij.cuda()
            tx = tx.cuda()
            ty = ty.cuda()
            tw = tw.cuda()
            th = th.cuda()
            te = te.cuda()

            # compute output
            output = model(img)
            loss, loss_resp, loss_iou, loss_coor, loss_size, loss_limb = criterion(img, output, delta, max_delta_ij, tx, ty, tw, th, te)

            # measure and record loss
            losses.update(loss.item(), img.size(0))
            losses_resp.update(loss_resp.item(), img.size(0))
            losses_iou.update(loss_iou.item(), img.size(0))
            losses_coor.update(loss_coor.item(), img.size(0))
            losses_size.update(loss_size.item(), img.size(0))
            losses_limb.update(loss_limb.item(), img.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f}: {loss_resp.avg:.4f} + '
                      '{loss_iou.avg:.4f} + {loss_coor.avg:.4f} + '
                      '{loss_size.avg:.4f} + {loss_limb.avg:.4f})'.format(
                       epoch, i, len(val_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, loss_resp=losses_resp, loss_iou=losses_iou, loss_coor=losses_coor, loss_size=losses_size, loss_limb=losses_limb))
                sys.stdout.flush()
    return losses.avg


def save_checkpoint(state, is_best, epochs, save_folder):
    filename =  save_folder+'/PPN_model_'+str(epochs)+'.pth.tar'
    torch.save(state, filename)

    if is_best:
        print("best checkpoints is saved!!!")
        shutil.copyfile(filename, save_folder+'/PPN_model_best.pth.tar')



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

