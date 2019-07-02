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
from functools import partial

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
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


from PIL import Image
from visdom import Visdom

#Custom models
from model import *
from config import *
from dataset import *
from aug import *

import pdb

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch PoseProposalNet Training')
parser.add_argument("-train", "--train_file", help="json file path")
parser.add_argument("-val", "--val_file", help="json file path")
parser.add_argument("-img", "--root_dir", help="path to train2017 or val2018")
parser.add_argument("-imsize", "--image_size", default=384, type=int, help="set input image size")

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
parser.add_argument('--reset_lr', action='store_true')
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
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
#parser.add_argument('--gpu', default=None, type=int,
#                    help='GPU id to use.')
parser.add_argument('--parallel_ckpt', action='store_true')
parser.add_argument('--deterministic', action='store_true')

parser.add_argument('--distributed', action='store_true')

parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--sync_bn', action='store_true',
                    help='enabling apex sync BN.')

parser.add_argument('--opt-level', type=str, default='O0')
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
#parser.add_argument('--rank', default=-1, type=int,
#                    help='node rank for distributed training')
#parser.add_argument('--multiprocessing-distributed', action='store_true',
#                    help='Use multi-processing distributed training to launch '
#                         'N processes per node, which has N GPUs. This is the '
#                         'fastest way to use PyTorch for either single node or '
#                         'multi node data parallel training')


# Augment implementation
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

#loss function
class PPNLoss(nn.Module):
    def __init__(self, 
                insize=(384,384),
                outsize=(24,24),
                keypoint_names = KEYPOINT_NAMES , local_grid_size= (21,21), edges = EDGES,
                lambda_resp=0.25,
                lambda_iou=1.0,
                lambda_coor=5.0,
                lambda_size=5.0,
                lambda_limb=0.5
                ):
        super(PPNLoss, self).__init__()
        self.insize = insize
        self.keypoint_names = keypoint_names
        self.edges = edges
        self.local_grid_size = local_grid_size

        self.dtype = np.float32

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

        #self.X, self.Y = torch.Tensor(np.meshgrid(np.arange(outW, dtype=np.float32), np.arange(outH, dtype=np.float32))).cuda()

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
    #loss, loss_resp, loss_iou, loss_coor, loss_size, loss_limb = criterion(
    #        img, output, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te)
    def forward(self, image, feature_map, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te):
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

        (rx, ry), (rw, rh) = self.restore_xy(x, y), self.restore_size(w, h)
        (rtx, rty), (rtw, rth) = self.restore_xy(tx, ty), self.restore_size(tw, th)
        ious = iou((rx, ry, rw, rh), (rtx, rty, rtw, rth))
        
        # Original loss function
        loss_resp = torch.sum((resp - delta)**2, tuple(range(1, resp.dim())) )

        loss_iou = torch.sum(delta * (conf - ious)**2, tuple(range(1, conf.dim())))
        loss_coor = torch.sum(weight * ((x - tx_half)**2 + (y - ty_half)**2), tuple(range(1, x.dim())))
        loss_size = torch.sum(weight * ((torch.sqrt(w + EPSILON) - torch.sqrt(tw + EPSILON))**2 + (torch.sqrt(h + EPSILON) - torch.sqrt(th + EPSILON))**2 ), tuple(range(1, w.dim())))
        loss_limb = torch.sum(weight_ij * (e - te)**2, tuple(range(1, e.dim())))

        loss_resp = torch.mean(loss_resp)
        loss_iou = torch.mean(loss_iou)
        loss_coor = torch.mean(loss_coor)
        loss_size = torch.mean(loss_size)
        loss_limb = torch.mean(loss_limb)


        # remake Binary cross entrophy function
#        loss_bce = nn.BCEWithLogitsLoss(pos_weight= 4*torch.ones([K, outH, outW]).cuda())
##        loss_bce_delta = nn.BCELoss(weight = delta)
##        loss_bce_weight = nn.BCELoss(weight = weight)
#        loss_bce_weight_ij = nn.BCEWithLogitsLoss(weight = weight_ij, pos_weight= 4*torch.ones([len(self.edges), self.local_grid_size[1], self.local_grid_size[0], outH, outW]).cuda())
#
#        loss_resp = loss_bce(resp, delta)
#
#        # ious derivative mission
#        #loss_iou = loss_bce_delta(conf, ious)
#
#        loss_iou = torch.sum(delta * (conf - ious)**2, tuple(range(1, conf.dim())))
#
##        loss_coor_x = loss_bce_weight(x, tx)
##        loss_coor_y = loss_bce_weight(y, ty)
##        loss_coor = loss_coor_x + loss_coor_y 
##
##        loss_size_w = loss_bce_weight(w, tw)
##        loss_size_h = loss_bce_weight(h, th)
##        loss_size = loss_size_w + loss_size_h 
#
#        loss_coor = torch.sum(weight * ((x - tx_half)**2 + (y - ty_half)**2), tuple(range(1, x.dim())))
#        loss_size = torch.sum(weight * ((torch.sqrt(torch.abs(w + EPSILON)) - torch.sqrt(torch.abs(tw + EPSILON)))**2 + (torch.sqrt(torch.abs(h + EPSILON)) - torch.sqrt(torch.abs(th + EPSILON)))**2 ), tuple(range(1, w.dim())))
#
#        loss_limb = loss_bce_weight_ij(e, te)
#
#        loss_resp = torch.mean(loss_resp)
#        loss_iou = torch.mean(loss_iou)
#        loss_coor = torch.mean(loss_coor)
#        loss_size = torch.mean(loss_size)
#        loss_limb = torch.mean(loss_limb)

        loss = self.lambda_resp * loss_resp + \
            self.lambda_iou * loss_iou + \
            self.lambda_coor * loss_coor + \
            self.lambda_size * loss_size + \
            self.lambda_limb * loss_limb

        return loss, loss_resp, loss_iou, loss_coor, loss_size, loss_limb


best_loss = 1e30000
args = parser.parse_args()

def main():
    global best_loss
    global args
    
    print("opt_level = {}".format(args.opt_level))
    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    #print("world_size:", args.world_size)
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        
#        #Fix weight from 0 to 3 layers
#        child_counter = 0
#        for child in model.children():
#            if child_counter < 4:
#                print("child ",child_counter," was frozen")
#                for param in child.parameters():
#                    param.requires_grad = False
#            else:
#                break
#            child_counter += 1
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
   

    #TODO Hardcoding local grid size
    #local_grid_size=(29, 29)
    local_grid_size=(21, 21)

    # Detach under avgpoll layer in Resnet
    modules = list(model.children())[:-3]
    model = nn.Sequential(*modules)
    model = PoseProposalNet(model, local_grid_size=local_grid_size)

    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model) 

    model = model.cuda()

    # Test model to get outsize
    inputs = torch.randn(1,3, args.image_size, args.image_size).cuda()
    y =  model(inputs)
    _, _, outH, outW =y.shape
    outsize = (outW, outH)
    print("outsize:",outsize)
    print("y:",y.shape)
    sys.stdout.flush()
    # 384 to 512
    insize = (args.image_size, args.image_size)

#    optimizer = torch.optim.SGD(model.parameters(), args.lr,
#                                momentum=args.momentum,
#                                weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    if args.distributed:
        model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32= None,
                                      loss_scale=args.loss_scale
                                      )

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with 
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) - custom loss
    criterion = PPNLoss(
                insize=insize,
                outsize=outsize,
                keypoint_names = KEYPOINT_NAMES,
                local_grid_size = local_grid_size,
                edges = EDGES,
                lambda_resp=0.25,
                lambda_iou=1.0,
                lambda_coor=5.0,
                lambda_size=5.0,
                lambda_limb=0.5
            ).cuda()
    '''
                lambda_resp=0.25,
                lambda_iou=1.0,
                lambda_coor=5.0,
                lambda_size=5.0,
                lambda_limb=0.5
    '''

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                best_loss = checkpoint['best_loss']

                if args.parallel_ckpt:
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint['state_dict'].items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v

                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(checkpoint['state_dict'])

                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    if args.reset_lr:
        lr = get_lr(optimizer)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr


    # Data loading code
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    train_dataset = KeypointsDataset(json_file = args.train_file, root_dir = args.root_dir,
                    transform=transforms.Compose([
                        IAA(insize,'val'),
                        ToNormalizedTensor()
                    ]), 
                    draw=False,
                    insize = insize,
                    outsize = outsize, 
                    keypoint_names = KEYPOINT_NAMES ,
                    local_grid_size = local_grid_size,
                    edges = EDGES
                    )


    val_dataset = KeypointsDataset(json_file = args.val_file, root_dir = args.root_dir,
                transform=transforms.Compose([
                    IAA(insize,'val'),
                    ToNormalizedTensor()
                ]), 
                draw=False,
                insize = insize,
                outsize = outsize, 
                keypoint_names = KEYPOINT_NAMES ,
                local_grid_size = local_grid_size,
                edges = EDGES
                )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

#    collate_fn = partial(custom_collate_fn, 
#                            insize = insize,
#                            outsize = outsize, 
#                            keypoint_names = KEYPOINT_NAMES ,
#                            local_grid_size = local_grid_size,
#                            edges = EDGES)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, 
        collate_fn = custom_collate_fn,
        pin_memory = True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, 
        collate_fn = custom_collate_fn,
        pin_memory = True,
        sampler=val_sampler)

    if args.evaluate:
#        val_dataset = KeypointsDataset(json_file = args.val_file, root_dir = args.root_dir,
#                    transform=transforms.Compose([
#                        IAA(insize,'test'),
#                        ToTensor()
#                    ]), draw=False,
#                    insize = insize,
#                    outsize = outsize, 
#                    keypoint_names = KEYPOINT_NAMES ,
#                    local_grid_size = local_grid_size,
#                    edges = EDGES
#                    )
#
#        val_loader = torch.utils.data.DataLoader(
#            val_dataset,
#            batch_size=args.batch_size, shuffle=False,
#            num_workers=args.workers, 
#            collate_fn = custom_collate_fn, 
#            pin_memory = True,
#            sampler = None)


        #test_output(val_loader, model, criterion, 1, outsize, local_grid_size, args)
        test_output(train_loader, model, criterion, 1, outsize, local_grid_size, args)
        return

    # Start trainin iterations
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        #adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_epoch_loss = train(train_loader, model, criterion, optimizer, epoch, args)


        # evaluate on validation set
        # epoch_loss = validate(val_loader, model, criterion, epoch, args)


        # remember best acc@1 and save checkpoint
        if args.local_rank == 0:

            plotter.plot('loss', 'train(epoch)', 'PPN Loss', epoch+1, train_epoch_loss) 
            #plotter.plot('loss', 'val', 'PPN Loss', epoch+1, epoch_loss) 

            #is_best = epoch_loss < best_loss
            is_best = False
            #best_loss = min(epoch_loss, best_loss)

            print("checkpoints checking")
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best, epoch+1, args.save)

class test_data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 , 0.456 , 0.406 ]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 , 0.224 , 0.225 ]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_delta, self.next_weight, self.next_weight_ij, self.next_tx_half, self.next_ty_half, self.next_tx, self.next_ty, self.next_tw, self.next_th, self.next_te = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_delta = None
            self.next_weight = None
            self.next_weight_ij = None
            self.next_tx_half = None
            self.next_ty_half = None
            self.next_tx = None
            self.next_ty = None
            self.next_tw = None
            self.next_th = None
            self.next_te = None
            return
        import copy 
        self.raw_img = copy.deepcopy(self.next_input)

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_delta = self.next_delta.cuda(non_blocking=True)
            self.next_weight = self.next_weight.cuda(non_blocking=True)
            self.next_weight_ij = self.next_weight_ij.cuda(non_blocking=True)
            self.next_tx_half = self.next_tx_half.cuda(non_blocking=True)
            self.next_ty_half = self.next_ty_half.cuda(non_blocking=True)
            self.next_tx = self.next_tx.cuda(non_blocking=True)
            self.next_ty = self.next_ty.cuda(non_blocking=True)
            self.next_tw = self.next_tw.cuda(non_blocking=True)
            self.next_th = self.next_th.cuda(non_blocking=True)
            self.next_te  = self.next_te.cuda(non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input

        delta = self.next_delta
        weight = self.next_weight
        weight_ij = self.next_weight_ij
        tx_half = self.next_tx_half
        ty_half = self.next_ty_half
        tx = self.next_tx
        ty = self.next_ty
        tw = self.next_tw
        th = self.next_th
        te = self.next_te 

        raw = self.raw_img

        self.preload()
        return input, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te, raw



class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 , 0.456 , 0.406 ]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 , 0.224 , 0.225 ]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            #self.next_input, self.next_delta, self.next_weight, self.next_weight_ij, self.next_tx_half, self.next_ty_half, self.next_tx, self.next_ty, self.next_tw, self.next_th, self.next_te = next(self.loader)
            self.next_input = next(self.loader)

        except StopIteration:
            self.next_input = None
            self.next_img = None
            self.next_delta =  None
            self.next_weight = None
            self.next_weight_ij = None
            self.next_tx_half = None
            self.next_ty_half = None
            self.next_tx = None
            self.next_ty = None
            self.next_tw = None
            self.next_th = None
            self.next_te  = None
            return

        with torch.cuda.stream(self.stream):
            self.next_img = self.next_input.image.cuda(non_blocking=True)
            self.next_delta = self.next_input.delta.cuda(non_blocking=True)
            self.next_weight = self.next_input.weight.cuda(non_blocking=True)
            self.next_weight_ij = self.next_input.weight_ij.cuda(non_blocking=True)
            self.next_tx_half = self.next_input.tx_half.cuda(non_blocking=True)
            self.next_ty_half = self.next_input.ty_half.cuda(non_blocking=True)
            self.next_tx = self.next_input.tx.cuda(non_blocking=True)
            self.next_ty = self.next_input.ty.cuda(non_blocking=True)
            self.next_tw = self.next_input.tw.cuda(non_blocking=True)
            self.next_th = self.next_input.th.cuda(non_blocking=True)
            self.next_te  = self.next_input.te.cuda(non_blocking=True)

            #self.next_img = self.next_img.float()
            self.next_img = self.next_img.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)

        input = self.next_img
        delta = self.next_delta
        weight = self.next_weight
        weight_ij = self.next_weight_ij
        tx_half = self.next_tx_half
        ty_half = self.next_ty_half
        tx = self.next_tx
        ty = self.next_ty
        tw = self.next_tw
        th = self.next_th
        te = self.next_te 

        self.preload()
        return input, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te

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
    #prefetcher = data_prefetcher(train_loader)

#    img, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te = prefetcher.next()
#    i = 0
#    while img is not None:
#        i += 1 
#
    #for i, (img, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te) in enumerate(train_loader):
    for i, samples in enumerate(train_loader):
        # measure data loading time
        img = samples.image.cuda()
        delta = samples.delta.cuda()

        weight = samples.weight.cuda()
        weight_ij = samples.weight_ij.cuda()
        tx_half = samples.tx_half.cuda()
        ty_half = samples.ty_half.cuda()

        tx = samples.tx.cuda()
        ty = samples.ty.cuda()
        tw = samples.tw.cuda()
        th = samples.th.cuda()
        te = samples.te.cuda()

        # compute output
        output = model(img)
        loss, loss_resp, loss_iou, loss_coor, loss_size, loss_limb = criterion(
                img, output, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.distributed:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:

            #TODO  Measure accuracy parts
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                reduced_loss_resp = reduce_tensor(loss_resp.data)
                reduced_loss_iou = reduce_tensor(loss_iou.data)
                reduced_loss_coor = reduce_tensor(loss_coor.data)
                reduced_loss_size = reduce_tensor(loss_size.data)
                reduced_loss_limb = reduce_tensor(loss_limb.data)
                reduced_resp_data = reduce_tensor(output.data)
            else:
                reduced_loss = loss.data
                reduced_loss_resp = loss_resp.data
                reduced_loss_iou = loss_iou.data
                reduced_loss_coor = loss_coor.data
                reduced_loss_size = loss_size.data
                reduced_loss_limb = loss_limb.data

            # measure accuracy and record loss
            if args.distributed:
                losses.update(to_python_float(reduced_loss), img.size(0))
                losses_resp.update(to_python_float(reduced_loss_resp), img.size(0))
                losses_iou.update(to_python_float(reduced_loss_iou), img.size(0))
                losses_coor.update(to_python_float(reduced_loss_coor), img.size(0))
                losses_size.update(to_python_float(reduced_loss_size), img.size(0))
                losses_limb.update(to_python_float(reduced_loss_limb), img.size(0))
            else:
                losses.update(loss.item(), img.size(0))
                losses_resp.update(loss_resp.item(), img.size(0))
                losses_iou.update(loss_iou.item(), img.size(0))
                losses_coor.update(loss_coor.item(), img.size(0))
                losses_size.update(loss_size.item(), img.size(0))
                losses_limb.update(loss_limb.item(), img.size(0))


            torch.cuda.synchronize()

            # measure elapsed time
            batch_time.update((time.time() - end)/args.print_freq)
            end = time.time()

            if args.local_rank == 0:

                K = len(KEYPOINT_NAMES)
                # Predicted Value
                if args.distributed:
                    resp = reduced_resp_data[:, 0 * K:1 * K, :, :].cpu().numpy() # delta
                    w = reduced_resp_data[:, 4 * K:5 * K, :, :].cpu().numpy() # delta
                    h = reduced_resp_data[:, 5 * K:6 * K, :, :].cpu().numpy() # delta
                else:
                    resp = output.data[:, 0 * K:1 * K, :, :].cpu().numpy() # delta
                    w = output.data[:, 4 * K:5 * K, :, :].cpu().numpy() # delta
                    h = output.data[:, 5 * K:6 * K, :, :].cpu().numpy() # delta
                # Ground Truth
                temp_delta = delta.cpu().numpy()
                temp_tw = tw.cpu().numpy()
                temp_th = th.cpu().numpy()

                if i % args.print_freq == 0 :
                    print("Trn max delta 0 value:"+str(temp_delta[0,0,:,:].reshape(-1)[np.argsort(temp_delta[0,0,:,:].reshape(-1))[-7:]]))
                    print("Trn max resp 0 value:"+str(resp[0,0,:,:].reshape(-1)[np.argsort(resp[0,0,:,:].reshape(-1))[-7:]]))

                    print("Trn max w 0 value:"+str(temp_tw[0,0,:,:].reshape(-1)[np.argsort(temp_tw[0,0,:,:].reshape(-1))[-7:]]))
                    print("Trn max tw 0 value:"+str(w[0,0,:,:].reshape(-1)[np.argsort(w[0,0,:,:].reshape(-1))[-7:]]))
                    print("Trn max h 0 value:"+str(temp_th[0,0,:,:].reshape(-1)[np.argsort(temp_th[0,0,:,:].reshape(-1))[-7:]]))
                    print("Trn max th 0 value:"+str(h[0,0,:,:].reshape(-1)[np.argsort(h[0,0,:,:].reshape(-1))[-7:]]))

#                    print("Trn max delta 1 value:"+str(temp_delta[0,1,:,:].reshape(-1)[np.argsort(temp_delta[0,1,:,:].reshape(-1))[-7:]]))
#                    print("Trn max resp 1 value:"+str(resp[0,1,:,:].reshape(-1)[np.argsort(resp[0,1,:,:].reshape(-1))[-7:]]))
#                    print("Trn max delta 2 value:"+str(temp_delta[0,2,:,:].reshape(-1)[np.argsort(temp_delta[0,2,:,:].reshape(-1))[-7:]]))
#                    print("Trn max resp 2 value:"+str(resp[0,2,:,:].reshape(-1)[np.argsort(resp[0,2,:,:].reshape(-1))[-7:]]))
#                    print("Trn max delta 3 value:"+str(temp_delta[0,3,:,:].reshape(-1)[np.argsort(temp_delta[0,3,:,:].reshape(-1))[-7:]]))
#                    print("Trn max resp 3 value:"+str(resp[0,3,:,:].reshape(-1)[np.argsort(resp[0,3,:,:].reshape(-1))[-7:]]))
#                    print("Trn max delta 4 value:"+str(temp_delta[0,4,:,:].reshape(-1)[np.argsort(temp_delta[0,4,:,:].reshape(-1))[-7:]]))
#                    print("Trn max resp 4 value:"+str(resp[0,4,:,:].reshape(-1)[np.argsort(resp[0,4,:,:].reshape(-1))[-7:]]))
#                    print("Trn max delta 5 value:"+str(temp_delta[0,5,:,:].reshape(-1)[np.argsort(temp_delta[0,5,:,:].reshape(-1))[-7:]]))
#                    print("Trn max resp 5 value:"+str(resp[0,5,:,:].reshape(-1)[np.argsort(resp[0,5,:,:].reshape(-1))[-7:]]))
#                    print("Trn max delta 6 value:"+str(temp_delta[0,6,:,:].reshape(-1)[np.argsort(temp_delta[0,6,:,:].reshape(-1))[-7:]]))
#                    print("Trn max resp 6 value:"+str(resp[0,6,:,:].reshape(-1)[np.argsort(resp[0,6,:,:].reshape(-1))[-7:]]))
#                    print("Trn max delta 7 value:"+str(temp_delta[0,7,:,:].reshape(-1)[np.argsort(temp_delta[0,7,:,:].reshape(-1))[-7:]]))
#                    print("Trn max resp 7 value:"+str(resp[0,7,:,:].reshape(-1)[np.argsort(resp[0,7,:,:].reshape(-1))[-7:]]))
#                    print("Trn max delta 8 value:"+str(temp_delta[0,8,:,:].reshape(-1)[np.argsort(temp_delta[0,8,:,:].reshape(-1))[-7:]]))
#                    print("Trn max resp 8 value:"+str(resp[0,8,:,:].reshape(-1)[np.argsort(resp[0,8,:,:].reshape(-1))[-7:]]))
#                    print("Trn max delta 9 value:"+str(temp_delta[0,9,:,:].reshape(-1)[np.argsort(temp_delta[0,9,:,:].reshape(-1))[-7:]]))
#                    print("Trn max resp 9 value:"+str(resp[0,9,:,:].reshape(-1)[np.argsort(resp[0,9,:,:].reshape(-1))[-7:]]))
#                    print("Trn max delta 10 value:"+str(temp_delta[0,10,:,:].reshape(-1)[np.argsort(temp_delta[0,10,:,:].reshape(-1))[-7:]]))
#                    print("Trn max resp 10 value:"+str(resp[0,10,:,:].reshape(-1)[np.argsort(resp[0,10,:,:].reshape(-1))[-7:]]))
#                    print("Trn max delta 11 value:"+str(temp_delta[0,11,:,:].reshape(-1)[np.argsort(temp_delta[0,11,:,:].reshape(-1))[-7:]]))
#                    print("Trn max resp 11 value:"+str(resp[0,11,:,:].reshape(-1)[np.argsort(resp[0,11,:,:].reshape(-1))[-7:]]))
#                    print("Trn max delta 12 value:"+str(temp_delta[0,12,:,:].reshape(-1)[np.argsort(temp_delta[0,12,:,:].reshape(-1))[-7:]]))
#                    print("Trn max resp 12 value:"+str(resp[0,12,:,:].reshape(-1)[np.argsort(resp[0,12,:,:].reshape(-1))[-7:]]))
#                    print("Trn max delta 13 value:"+str(temp_delta[0,13,:,:].reshape(-1)[np.argsort(temp_delta[0,13,:,:].reshape(-1))[-7:]]))
#                    print("Trn max resp 13 value:"+str(resp[0,13,:,:].reshape(-1)[np.argsort(resp[0,13,:,:].reshape(-1))[-7:]]))
#                    print("Trn max delta 14 value:"+str(temp_delta[0,14,:,:].reshape(-1)[np.argsort(temp_delta[0,14,:,:].reshape(-1))[-7:]]))
#                    print("Trn max resp 14 value:"+str(resp[0,14,:,:].reshape(-1)[np.argsort(resp[0,14,:,:].reshape(-1))[-7:]]))
#                    print("Trn max delta 15 value:"+str(temp_delta[0,15,:,:].reshape(-1)[np.argsort(temp_delta[0,15,:,:].reshape(-1))[-7:]]))
#                    print("Trn max resp 15 value:"+str(resp[0,15,:,:].reshape(-1)[np.argsort(resp[0,15,:,:].reshape(-1))[-7:]]))
#                    print("Trn max delta 16 value:"+str(temp_delta[0,16,:,:].reshape(-1)[np.argsort(temp_delta[0,16,:,:].reshape(-1))[-7:]]))
#                    print("Trn max resp 16 value:"+str(resp[0,16,:,:].reshape(-1)[np.argsort(resp[0,16,:,:].reshape(-1))[-7:]]))
#

                #print("max resp value:"+str(np.amax(resp)))
                print('Epoch: [{0}][{1}/{2}] {learning_rate:.7f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f}: {loss_resp.avg:.4f} + '
                    '{loss_iou.avg:.4f} + {loss_coor.avg:.4f} + '
                    '{loss_size.avg:.4f} + {loss_limb.avg:.4f})'.format(
                    epoch+1, i, len(train_loader), learning_rate=get_lr(optimizer), batch_time=batch_time,
                    loss=losses, loss_resp=losses_resp, loss_iou=losses_iou, loss_coor=losses_coor, loss_size=losses_size, loss_limb=losses_limb))
                sys.stdout.flush()
                plotter.plot('loss', 'train', 'PPN Loss', epoch + (i/len(train_loader)), losses.avg) 

        del output
        del loss
        del loss_resp
        del loss_iou
        del loss_limb
        del loss_coor
        del loss_size

        #img, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te = prefetcher.next()

    return losses.avg

#TODO function for evaluation like OKS

def test_output(val_loader, model, criterion, epoch, outsize, local_grid_size, args):
    from datatest import get_humans_by_feature
    from datatest import draw_humans
    data_time = AverageMeter()

    end = time.time()
    # revert normalized image to original image
    mean = torch.tensor([0.485 , 0.456 , 0.406 ]).cuda().view(1,3,1,1)
    std = torch.tensor([0.229 , 0.224 , 0.225 ]).cuda().view(1,3,1,1)

    # switch to evaluate mode
    model.eval()
    outW, outH = outsize

#    prefetcher = data_prefetcher(val_loader)
#    img, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te = prefetcher.next()
#    i = 0
#    while img is not None:
#        i += 1
#        # compute output
#        with torch.no_grad():

    with torch.no_grad():
        end=time.time()

        for i, samples in enumerate(val_loader):
            # measure data loading time
            img = samples.image.cuda()
            delta = samples.delta.cuda()

            weight = samples.weight.cuda()
            weight_ij = samples.weight_ij.cuda()
            tx_half = samples.tx_half.cuda()
            ty_half = samples.ty_half.cuda()

            tx = samples.tx.cuda()
            ty = samples.ty.cuda()
            tw = samples.tw.cuda()
            th = samples.th.cuda()
            te = samples.te.cuda()

            # compute output
            output = model(img)

            loss, loss_resp, loss_iou, loss_coor, loss_size, loss_limb = criterion(
                    img, output, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te)

            reduced_loss = loss.data
            reduced_loss_resp = loss_resp.data
            reduced_loss_iou = loss_iou.data
            reduced_loss_coor = loss_coor.data
            reduced_loss_size = loss_size.data
            reduced_loss_limb = loss_limb.data

            logger.info("resp loss:"+str(reduced_loss_resp))


            K = len(KEYPOINT_NAMES)
            B, _, _, _ = img.shape

            #loss function with torch
            resp = output[:, 0 * K:1 * K, :, :].cpu().numpy() # delta
            conf = output[:, 1 * K:2 * K, :, :].cpu().numpy()
            x = output[:, 2 * K:3 * K, :, :].cpu().numpy()
            y = output[:, 3 * K:4 * K, :, :].cpu().numpy()
            w = output[:, 4 * K:5 * K, :, :].cpu().numpy()
            h = output[:, 5 * K:6 * K, :, :].cpu().numpy()
            e = output[:, 6 * K:, :, :].reshape(
                B,
                len(EDGES),
                local_grid_size[1], local_grid_size[0],
                outH, outW
            ).cpu().numpy()

            logger.info("Non-Zero GT: "+str(np.count_nonzero(delta.cpu().numpy()[0])))
            #logger.info("Non-Zero GT1: "+str(np.count_nonzero(delta.cpu().numpy()[0][1])))
            logger.info("resp dim:"+str(resp.shape))
            #logger.info("resp value:"+str(resp[0]))

            resp = np.squeeze(resp, axis=0)
            conf = np.squeeze(conf, axis=0)
            x = np.squeeze(x, axis=0)
            y = np.squeeze(y, axis=0)
            w = np.squeeze(w, axis=0)
            h = np.squeeze(h, axis=0)
            e = np.squeeze(e, axis=0)

            logger.info("max human resp value:"+str(np.amax(resp[0])))
            logger.info("max 1 resp value:"+str(np.amax(resp[1])))
            logger.info("max 2 resp value:"+str(np.amax(resp[2])))
            logger.info("max 3 resp value:"+str(np.amax(resp[3])))
            logger.info("max 4 resp value:"+str(np.amax(resp[4])))
            logger.info("max 5 resp value:"+str(np.amax(resp[5])))
            logger.info("max 6 resp value:"+str(np.amax(resp[6])))
            logger.info("max 7 resp value:"+str(np.amax(resp[7])))
            logger.info("max 8 resp value:"+str(np.amax(resp[8])))
            logger.info("max 9 resp value:"+str(np.amax(resp[9])))
            logger.info("max 10 resp value:"+str(np.amax(resp[10])))
            logger.info("max 11 resp value:"+str(np.amax(resp[11])))
            logger.info("max 12 resp value:"+str(np.amax(resp[12])))
            logger.info("max 13 resp value:"+str(np.amax(resp[13])))
            logger.info("max 14 resp value:"+str(np.amax(resp[14])))
            logger.info("max 15 resp value:"+str(np.amax(resp[15])))
            logger.info("max 16 resp value:"+str(np.amax(resp[16])))
            logger.info("max conf value:"+str(np.amax(conf[0])))
            #delta = resp*conf
            #logger.info("max delta value:"+str(np.amax(delta[0])))
            humans = get_humans_by_feature(resp, x, y, w, h, e, detection_thresh=0.01)

            #self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            raw_img = img.mul_(std).add_(mean)
            pil_image = Image.fromarray(np.squeeze(raw_img.cpu().numpy(), axis=0).astype(np.uint8).transpose(1, 2, 0)) 

            pil_image = draw_humans(
                keypoint_names=KEYPOINT_NAMES,
                edges=EDGES,
                pil_image=pil_image,
                humans=humans,
                visbbox= False,
                gridOn = False
            )   

            pil_image.save('output/training_test/predict_test_result_'+str(i)+'.png', 'PNG')

            if i>100:
                break

            #img, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te = prefetcher.next()

def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()

    losses_resp = AverageMeter()
    losses_iou = AverageMeter()
    losses_coor = AverageMeter()
    losses_size = AverageMeter()
    losses_limb = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    prefetcher = data_prefetcher(val_loader)
    img, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te = prefetcher.next()
    i = 0
    while img is not None:
        i += 1


        # compute output
        with torch.no_grad():

#    with torch.no_grad():
#        for i, (target_img, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te) in enumerate(val_loader):
#            
#            img = target_img.cuda()
#
#            delta = delta.cuda()
#
#            weight = weight.cuda()
#            weight_ij = weight_ij.cuda()
#            tx_half = tx_half.cuda()
#            ty_half = ty_half.cuda()
#
#            tx = tx.cuda()
#            ty = ty.cuda()
#            tw = tw.cuda()
#            th = th.cuda()
#            te = te.cuda()

            output = model(img)

            loss, loss_resp, loss_iou, loss_coor, loss_size, loss_limb = criterion(
                    img, output, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                reduced_loss_resp = reduce_tensor(loss_resp.data)
                reduced_loss_iou = reduce_tensor(loss_iou.data)
                reduced_loss_coor = reduce_tensor(loss_coor.data)
                reduced_loss_size = reduce_tensor(loss_size.data)
                reduced_loss_limb = reduce_tensor(loss_limb.data)
                reduced_resp_data = reduce_tensor(output.data)
            else:
                reduced_loss = loss.data
                reduced_loss_resp = loss_resp.data
                reduced_loss_iou = loss_iou.data
                reduced_loss_coor = loss_coor.data
                reduced_loss_size = loss_size.data
                reduced_loss_limb = loss_limb.data

            # measure and record loss

            if args.distributed:
                losses.update(to_python_float(reduced_loss), img.size(0))
                losses_resp.update(to_python_float(reduced_loss_resp), img.size(0))
                losses_iou.update(to_python_float(reduced_loss_iou), img.size(0))
                losses_coor.update(to_python_float(reduced_loss_coor), img.size(0))
                losses_size.update(to_python_float(reduced_loss_size), img.size(0))
                losses_limb.update(to_python_float(reduced_loss_limb), img.size(0))
            else:
                losses.update(loss.item(), img.size(0))
                losses_resp.update(loss_resp.item(), img.size(0))
                losses_iou.update(loss_iou.item(), img.size(0))
                losses_coor.update(loss_coor.item(), img.size(0))
                losses_size.update(loss_size.item(), img.size(0))
                losses_limb.update(loss_limb.item(), img.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

#            K = len(KEYPOINT_NAMES)
#
#            if args.distributed:
#                resp = reduced_resp_data[:, 0 * K:1 * K, :, :].cpu().numpy() # delta
#            else:
#                resp = output.data[:, 0 * K:1 * K, :, :].cpu().numpy() # delta
#
#            temp_delta = delta.cpu().numpy()

            if i % args.print_freq == 0 and  args.local_rank == 0:
#                print("Val max delta 0 value:"+str(temp_delta[0,0,:,:].reshape(-1)[np.argsort(temp_delta[0,0,:,:].reshape(-1))[-10:]]))
#                print("Val max resp 0 value:"+str(resp[0,0,:,:].reshape(-1)[np.argsort(resp[0,0,:,:].reshape(-1))[-10:]]))
#                print("Val max delta 1 value:"+str(temp_delta[0,1,:,:].reshape(-1)[np.argsort(temp_delta[0,1,:,:].reshape(-1))[-10:]]))
#                print("Val max resp 1 value:"+str(resp[0,1,:,:].reshape(-1)[np.argsort(resp[0,1,:,:].reshape(-1))[-10:]]))
##                print("Val max delta 2 value:"+str(temp_delta[:,2,:,:].reshape(-1)[np.argsort(temp_delta[:,2,:,:].reshape(-1))[-110:]]))
##                print("Val max resp 2 value:"+str(resp[:,2,:,:].reshape(-1)[np.argsort(resp[:,2,:,:].reshape(-1))[-110:]]))
##                print("Val max delta 3 value:"+str(temp_delta[:,3,:,:].reshape(-1)[np.argsort(temp_delta[:,3,:,:].reshape(-1))[-110:]]))
##                print("Val max resp 3 value:"+str(resp[:,3,:,:].reshape(-1)[np.argsort(resp[:,3,:,:].reshape(-1))[-110:]]))
#                print("Val max delta 4 value:"+str(temp_delta[0,4,:,:].reshape(-1)[np.argsort(temp_delta[0,4,:,:].reshape(-1))[-10:]]))
#                print("Val max resp 4 value:"+str(resp[0,4,:,:].reshape(-1)[np.argsort(resp[0,4,:,:].reshape(-1))[-10:]]))
##                print("Val max delta 5 value:"+str(temp_delta[:,5,:,:].reshape(-1)[np.argsort(temp_delta[:,5,:,:].reshape(-1))[-110:]]))
##                print("Val max resp 5 value:"+str(resp[:,5,:,:].reshape(-1)[np.argsort(resp[:,5,:,:].reshape(-1))[-110:]]))
##                print("Val max delta 6 value:"+str(temp_delta[:,6,:,:].reshape(-1)[np.argsort(temp_delta[:,6,:,:].reshape(-1))[-110:]]))
##                print("Val max resp 6 value:"+str(resp[:,6,:,:].reshape(-1)[np.argsort(resp[:,6,:,:].reshape(-1))[-110:]]))
##                print("Val max delta 7 value:"+str(temp_delta[:,7,:,:].reshape(-1)[np.argsort(temp_delta[:,7,:,:].reshape(-1))[-110:]]))
##                print("Val max resp 7 value:"+str(resp[:,7,:,:].reshape(-1)[np.argsort(resp[:,7,:,:].reshape(-1))[-110:]]))
##                print("Val max delta 8 value:"+str(temp_delta[:,8,:,:].reshape(-1)[np.argsort(temp_delta[:,8,:,:].reshape(-1))[-110:]]))
##                print("Val max resp 8 value:"+str(resp[:,8,:,:].reshape(-1)[np.argsort(resp[:,8,:,:].reshape(-1))[-110:]]))
##                print("Val max delta 9 value:"+str(temp_delta[:,9,:,:].reshape(-1)[np.argsort(temp_delta[:,9,:,:].reshape(-1))[-110:]]))
##                print("Val max resp 9 value:"+str(resp[:,9,:,:].reshape(-1)[np.argsort(resp[:,9,:,:].reshape(-1))[-110:]]))
##                print("Val max delta 10 value:"+str(temp_delta[:,10,:,:].reshape(-1)[np.argsort(temp_delta[:,10,:,:].reshape(-1))[-110:]]))
##                print("Val max resp 10 value:"+str(resp[:,10,:,:].reshape(-1)[np.argsort(resp[:,10,:,:].reshape(-1))[-110:]]))
##                print("Val max delta 11 value:"+str(temp_delta[:,11,:,:].reshape(-1)[np.argsort(temp_delta[:,11,:,:].reshape(-1))[-110:]]))
##                print("Val max resp 11 value:"+str(resp[:,11,:,:].reshape(-1)[np.argsort(resp[:,11,:,:].reshape(-1))[-110:]]))
##                print("Val max delta 12 value:"+str(temp_delta[:,12,:,:].reshape(-1)[np.argsort(temp_delta[:,12,:,:].reshape(-1))[-110:]]))
##                print("Val max resp 12 value:"+str(resp[:,12,:,:].reshape(-1)[np.argsort(resp[:,12,:,:].reshape(-1))[-110:]]))
##                print("Val max delta 13 value:"+str(temp_delta[:,13,:,:].reshape(-1)[np.argsort(temp_delta[:,13,:,:].reshape(-1))[-110:]]))
##                print("Val max resp 13 value:"+str(resp[:,13,:,:].reshape(-1)[np.argsort(resp[:,13,:,:].reshape(-1))[-110:]]))
##                print("Val max delta 14 value:"+str(temp_delta[:,14,:,:].reshape(-1)[np.argsort(temp_delta[:,14,:,:].reshape(-1))[-110:]]))
##                print("Val max resp 14 value:"+str(resp[:,14,:,:].reshape(-1)[np.argsort(resp[:,14,:,:].reshape(-1))[-110:]]))
##                print("Val max delta 15 value:"+str(temp_delta[:,15,:,:].reshape(-1)[np.argsort(temp_delta[:,15,:,:].reshape(-1))[-110:]]))
##                print("Val max resp 15 value:"+str(resp[:,15,:,:].reshape(-1)[np.argsort(resp[:,15,:,:].reshape(-1))[-110:]]))
##                print("Val max delta 16 value:"+str(temp_delta[:,16,:,:].reshape(-1)[np.argsort(temp_delta[:,16,:,:].reshape(-1))[-110:]]))
##                print("Val max resp 16 value:"+str(resp[:,16,:,:].reshape(-1)[np.argsort(resp[:,16,:,:].reshape(-1))[-110:]]))

                print('Val Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f}: {loss_resp.avg:.4f} + '
                    '{loss_iou.avg:.4f} + {loss_coor.avg:.4f} + '
                    '{loss_size.avg:.4f} + {loss_limb.avg:.4f})'.format(
                    epoch+1, i, len(val_loader),batch_time=batch_time,
                    loss=losses, loss_resp=losses_resp, loss_iou=losses_iou, loss_coor=losses_coor, loss_size=losses_size, loss_limb=losses_limb))
                sys.stdout.flush()


            img, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te = prefetcher.next()

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
                xlabel='Epoch',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed linearly each iteration"""
    #lr = 0.007 * (1  - iters/260000)
    lr = get_lr(optimizer)

    if epoch % 200 == 0 and epoch > 400 :
        lr = 0.7* lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def reduce_tensor(tensor):
    global args
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    #TODO hardcoding world_size
    rt /= args.world_size
    return rt


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

