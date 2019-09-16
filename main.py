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
from functools import reduce

from tqdm import tqdm

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

parser.add_argument('--savefig', action='store_true')
parser.add_argument('--alp', '--alpha', default=0.12, type=float,
                    help='alpha (default: 0.12)',
                    dest='alpha')

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
                keypoint_names=KEYPOINT_NAMES, 
                local_grid_size=(21,21), 
                edges=EDGES
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

        # remake Binary cross entrophy function
#        raw_resp_weight = (outH*outW*B-torch.sum(delta).item())/torch.sum(delta).item()
#        resp_weight = torch.autograd.Variable(torch.FloatTensor([1,raw_resp_weight]))
#        resp_weight = resp_weight[delta.long()].cuda()
#	
#        loss_bce = nn.BCELoss(weight = resp_weight)
#
#        raw_limb_weight = (self.local_grid_size[1]*self.local_grid_size[0]*outH*outW*B-torch.sum(te).item())/torch.sum(te).item()
#        limb_weight = torch.autograd.Variable(torch.FloatTensor([1,raw_limb_weight]))
#        limb_weight = limb_weight[te.long()].cuda()
#
#        loss_bce_weight_ij = nn.BCELoss(weight=limb_weight)
#
#        loss_resp = loss_bce(resp, delta)
#        loss_limb = loss_bce_weight_ij(e, te)

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

        return loss_resp, loss_iou, loss_coor, loss_size, loss_limb


best_AP = -1.0
args = parser.parse_args()

def main():
    global best_AP, args, weightloss, weightloss0, weightloss1, weightloss2, weightloss3, weightloss4

    print("opt_level = {}".format(args.opt_level))
    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    #print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

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

    print("world_size:", args.world_size)
    sys.stdout.flush()
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
   

    #TODO Hardcoding local grid size
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


#    if args.local_rank ==0:
#        for ((name, value), value2) in zip(model.named_parameters(),model.parameters()):
#            print(name)
#            print(value.shape, value2.shape)
#    sys.exit(0)

    # Weight model
    weight_model = nn.Linear(5,1, bias=False).cuda()
    weight_model.weight.data.fill_(1.0)


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


    optimizerM = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizerR = torch.optim.Adam(weight_model.parameters(), lr=args.lr)

    if args.distributed:
        model, optimizerM = amp.initialize(model, optimizerM,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32= None,
                                      loss_scale=args.loss_scale
                                      )
        weight_model, optimizerR = amp.initialize(weight_model, optimizerR, opt_level = args.opt_level)
        
        model = DDP(model, delay_allreduce=True)


    # define loss function (criterion) - custom loss
    criterion = PPNLoss(
                insize=insize,
                outsize=outsize,
                keypoint_names = KEYPOINT_NAMES,
                local_grid_size = local_grid_size,
                edges = EDGES,
            ).cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                best_AP = checkpoint['best_AP']

                if args.parallel_ckpt:
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint['state_dict'].items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v

                    model.load_state_dict(new_state_dict)
                    weight_model.load_state_dict(checkpoint['weight_state_dict'])

                else:
                    model.load_state_dict(checkpoint['state_dict'])
                    weight_model.load_state_dict(checkpoint['weight_state_dict'])

                optimizerM.load_state_dict(checkpoint['optimizerM'])
                optimizerR.load_state_dict(checkpoint['optimizerR'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    print("resumed weight", weight_model.weight)
    sys.stdout.flush()

    if args.reset_lr:
        lr = get_lr(optimizerM)
        for param_group in optimizerM.param_groups:
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

    train_sampler = None
    val_sampler = None

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    print("Sampler:", train_sampler, val_sampler)
    sys.stdout.flush()

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        #train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, 
        collate_fn = custom_collate_fn,
        pin_memory = True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        #batch_size=args.batch_size, shuffle=True,
        batch_size=1, shuffle=False,
        num_workers=args.workers, 
        collate_fn = custom_collate_fn,
        pin_memory = True,
        sampler=val_sampler)

#    for name,param in model.named_parameters():
#        if param.requires_grad:
#            print(name)
#    sys.exit(0)

    if args.evaluate:

        #test_output(val_loader, val_dataset, model, criterion, 1, outsize, local_grid_size, args)
        test_output(val_loader, val_dataset, model, weight_model, criterion, outsize, local_grid_size, args)
        #test_output(train_loader, train_dataset, model, weight_model, criterion, outsize, local_grid_size, args)
        return


    # Start trainin iterations
    print("Training start")

    #base = get_baseloss(train_loader, model, criterion)

    base = [torch.tensor([3911.7922]).cuda(), torch.tensor([18.1335]).cuda(),torch.tensor([24.9189]).cuda(),torch.tensor([31.3605]).cuda(),torch.tensor([13727.9912]).cuda()]
            
    print("base loss:", base)
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        #TODO uncommect lr and evaluation code
        adjust_learning_rate(optimizerM, epoch, args)

        # train for one epoch
        train_epoch_loss = train(train_loader, model, weight_model, criterion, optimizerM, optimizerR, epoch, args, base)

        # evaluate on validation set
        epoch_loss ap_vals = validate(val_loader, model, weight_model, criterion, epoch, args)

        # remember best acc@1 and save checkpoint
        if args.local_rank == 0:

            plotter.plot('loss', 'train(epoch)', 'PPN Loss', epoch+1, train_epoch_loss) 
            plotter.plot('loss', 'val', 'PPN Loss', epoch+1, epoch_loss) 
            
            plotter.plot('AP', 'Head', 'Average Precision (AP)', epoch+1, ap_vals[0]) 
            plotter.plot('AP', 'Shoulder', 'Average Precision (AP)', epoch+1, ap_vals[1]) 
            plotter.plot('AP', 'Elbow', 'Average Precision (AP)', epoch+1, ap_vals[2]) 
            plotter.plot('AP', 'Wrist', 'Average Precision (AP)', epoch+1, ap_vals[3]) 
            plotter.plot('AP', 'Hip', 'Average Precision (AP)', epoch+1, ap_vals[4]) 
            plotter.plot('AP', 'Knee', 'Average Precision (AP)', epoch+1, ap_vals[5]) 
            plotter.plot('AP', 'Ankle', 'Average Precision (AP)', epoch+1, ap_vals[6]) 
            plotter.plot('AP', 'Total', 'Average Precision (AP)', epoch+1, ap_vals[7]) 

            #is_best = epoch_loss < best_AP

            is_best = ap_vals[7] < best_AP
            best_AP = max(ap_vals[7] best_AP)

            print("checkpoints checking")
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'weight_state_dict': weight_model.state_dict(),
                'best_AP': best_AP,
                'optimizerM' : optimizerM.state_dict(),
                'optimizerR' : optimizerR.state_dict(),
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

def get_baseloss(train_loader, model, criterion):
    base_l0 = 0
    base_l1 = 0
    base_l2 = 0
    base_l3 = 0
    base_l4 = 0

    model.eval()
    print("Get baseloss!")
    for samples in tqdm(train_loader):
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

        output = model(img)
        loss_resp, loss_iou, loss_coor, loss_size, loss_limb = criterion(
                img, output, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te)
        
        # Sum all theloss 
        base_l0 += loss_resp.detach()
        base_l1 += loss_iou.detach()
        base_l2 += loss_coor.detach()
        base_l3 += loss_size.detach()
        base_l4 += loss_limb.detach()

    base_l0 /= len(train_loader)
    base_l1 /= len(train_loader)
    base_l2 /= len(train_loader)
    base_l3 /= len(train_loader)
    base_l4 /= len(train_loader)

    return [base_l0.detach(), base_l1.detach(), base_l2.detach(), base_l3.detach(), base_l4.detach()]

def train(train_loader, model, weight_model, criterion, optimizerM, optimizerR, epoch, args, base):
    Gradloss = nn.L1Loss()

    #train function

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_resp = AverageMeter()
    losses_iou = AverageMeter()
    losses_coor = AverageMeter()
    losses_size = AverageMeter()
    losses_limb = AverageMeter()

    weightlosses_resp = AverageMeter()
    weightlosses_iou = AverageMeter()
    weightlosses_coor = AverageMeter()
    weightlosses_size = AverageMeter()
    weightlosses_limb = AverageMeter()

    alph = args.alpha
    # switch to train mode
    model.train()

    end = time.time()

    #print("model ready")
    #sys.stdout.flush()

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
        loss_resp, loss_iou, loss_coor, loss_size, loss_limb = criterion(
                img, output, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te)

        l0 = torch.mul(weight_model.weight[0][0], loss_resp)
        l1 = torch.mul(weight_model.weight[0][1], loss_iou)
        l2 = torch.mul(weight_model.weight[0][2], loss_coor)
        l3 = torch.mul(weight_model.weight[0][3], loss_size)
        l4 = torch.mul(weight_model.weight[0][4], loss_limb)

        loss = torch.div(l0+l1+l2+l3+l4, 5)

        # compute gradient and do step
        optimizerM.zero_grad()

        if args.distributed:
            with amp.scale_loss(loss, optimizerM) as scaled_loss:
                scaled_loss.backward(retain_graph=True)
        else:
            loss.backward(retain_graph=True)

        # Getting gradients of the first layers of each tower and calculate their l2-norm 
        param = list(model.parameters())

        # 0 -> -8 : conv2 weight
        G0R = torch.autograd.grad(l0, param[-8], retain_graph=True, create_graph=True)
        G1R = torch.autograd.grad(l1, param[-8], retain_graph=True, create_graph=True)
        G2R = torch.autograd.grad(l2, param[-8], retain_graph=True, create_graph=True)
        G3R = torch.autograd.grad(l3, param[-8], retain_graph=True, create_graph=True)
        G4R = torch.autograd.grad(l4, param[-8], retain_graph=True, create_graph=True)

#        G0R = torch.autograd.grad(l0, param[-9], retain_graph=True, create_graph=True)
#        G1R = torch.autograd.grad(l1, param[-9], retain_graph=True, create_graph=True)
#        G2R = torch.autograd.grad(l2, param[-9], retain_graph=True, create_graph=True)
#        G3R = torch.autograd.grad(l3, param[-9], retain_graph=True, create_graph=True)
#        G4R = torch.autograd.grad(l4, param[-9], retain_graph=True, create_graph=True)

        G0 = torch.norm(G0R[0], 2)
        G1 = torch.norm(G1R[0], 2)
        G2 = torch.norm(G2R[0], 2)
        G3 = torch.norm(G3R[0], 2)
        G4 = torch.norm(G4R[0], 2)

        G_avg = torch.div(G0+G1+G2+G3+G4, 5)

        # Calculating relative losses 
        lhat0 = torch.div(l0, base[0])
        lhat1 = torch.div(l1, base[1])
        lhat2 = torch.div(l2, base[2])
        lhat3 = torch.div(l3, base[3])
        lhat4 = torch.div(l4, base[4])

        lhat_avg = torch.div(lhat0+lhat1+lhat2+lhat3+lhat4, 5)

        # Calculating relative inverse training rates for tasks 
        inv_rate0 = torch.div(lhat0,lhat_avg)
        inv_rate1 = torch.div(lhat1,lhat_avg)
        inv_rate2 = torch.div(lhat2,lhat_avg)
        inv_rate3 = torch.div(lhat3,lhat_avg)
        inv_rate4 = torch.div(lhat4,lhat_avg)
        
        # Calculating the constant target for Eq. 2 in the GradNorm paper
        C0 = G_avg*(inv_rate0)**alph
        C1 = G_avg*(inv_rate1)**alph
        C2 = G_avg*(inv_rate2)**alph
        C3 = G_avg*(inv_rate3)**alph
        C4 = G_avg*(inv_rate4)**alph

        # detach: delete grad, squeeze: fit dimension
        C0 = C0.detach().squeeze()
        C1 = C1.detach().squeeze()
        C2 = C2.detach().squeeze()
        C3 = C3.detach().squeeze()
        C4 = C4.detach().squeeze()

        optimizerR.zero_grad()
        
        # Calculating the gradient loss according to Eq. 2 in the GradNorm paper
        Lgrad = Gradloss(G0,C0) + Gradloss(G1,C1) + Gradloss(G2,C2) + Gradloss(G3,C3) + Gradloss(G4,C4)

        Lgrad.backward()
        #print(args.local_rank, "backward Lgrad")

        # Updating loss weights 
        optimizerR.step()

        # Updating the model weights
        optimizerM.step()

        with torch.no_grad():

            if args.world_size > 1:
                dist.all_reduce(weight_model.weight)
                weight_model.weight.div_(args.world_size)

            # Clamp negative value
            weight_model.weight.clamp_(min=0.0)

            # Normalized weight value
            weight_model.weight.div_(torch.mean(weight_model.weight))

        del G0R, G1R, G2R, G3R, G4R

        if args.local_rank == 0:
            if i % args.print_freq == 0 or i == len(train_loader)-1:

                if args.distributed:
                    reduced_loss = reduce_tensor(loss.detach())
                    reduced_loss_resp = reduce_tensor(loss_resp.detach())
                    reduced_loss_iou = reduce_tensor(loss_iou.detach())
                    reduced_loss_coor = reduce_tensor(loss_coor.detach())
                    reduced_loss_size = reduce_tensor(loss_size.detach())
                    reduced_loss_limb = reduce_tensor(loss_limb.detach())
                    reduced_resp_data = reduce_tensor(output.detach())

                    # Weightlosses
                    reduced_weightloss_resp = reduce_tensor(weight_model.weight[0][0].detach())
                    reduced_weightloss_iou = reduce_tensor(weight_model.weight[0][1].detach())
                    reduced_weightloss_coor = reduce_tensor(weight_model.weight[0][2].detach())
                    reduced_weightloss_size = reduce_tensor(weight_model.weight[0][3].detach())
                    reduced_weightloss_limb = reduce_tensor(weight_model.weight[0][4].detach())

                else:
                    reduced_loss = loss.detach()
                    reduced_loss_resp = loss_resp.detach()
                    reduced_loss_iou = loss_iou.detach()
                    reduced_loss_coor = loss_coor.detach()
                    reduced_loss_size = loss_size.detach()
                    reduced_loss_limb = loss_limb.detach()

                # measure accuracy and record loss
                if args.distributed:
                    losses.update(to_python_float(reduced_loss), img.size(0))
                    losses_resp.update(to_python_float(reduced_loss_resp), img.size(0))
                    losses_iou.update(to_python_float(reduced_loss_iou), img.size(0))
                    losses_coor.update(to_python_float(reduced_loss_coor), img.size(0))
                    losses_size.update(to_python_float(reduced_loss_size), img.size(0))
                    losses_limb.update(to_python_float(reduced_loss_limb), img.size(0))
                    
                    # Weightlosses
                    weightlosses_resp.update(to_python_float(reduced_weightloss_resp), img.size(0))
                    weightlosses_iou.update(to_python_float(reduced_weightloss_iou), img.size(0))
                    weightlosses_coor.update(to_python_float(reduced_weightloss_coor), img.size(0))
                    weightlosses_size.update(to_python_float(reduced_weightloss_size), img.size(0))
                    weightlosses_limb.update(to_python_float(reduced_weightloss_limb), img.size(0))

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

                K = len(KEYPOINT_NAMES)

                # Predicted Value
                if args.distributed:
                    resp = reduced_resp_data[:, 0 * K:1 * K, :, :].cpu().numpy() # delta
                else:
                    resp = output.data[:, 0 * K:1 * K, :, :].cpu().numpy() # delta
                    
                print_resp(delta, resp)

                print('Epoch: [{0}][{1}/{2}] {learning_rate:.7f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f}: {loss_resp.avg:.4f} + '
                    '{loss_iou.avg:.4f} + {loss_coor.avg:.4f} + '
                    '{loss_size.avg:.4f} + {loss_limb.avg:.4f})'.format(
                    epoch+1, i, len(train_loader), learning_rate=get_lr(optimizerM), batch_time=batch_time,
                    loss=losses, loss_resp=losses_resp, loss_iou=losses_iou, loss_coor=losses_coor, loss_size=losses_size, loss_limb=losses_limb))

                print('Epoch: [{0}][{1}/{2}] \t'
                    'Loss Weight : {weightloss_resp.val:.4f} + '
                    '{weightloss_iou.val:.4f} + {weightloss_coor.val:.4f} + '
                    '{weightloss_size.val:.4f} + {weightloss_limb.val:.4f} = {total_weightloss:.4f})'.format(
                    epoch+1, i, len(train_loader),
                    weightloss_resp=weightlosses_resp, weightloss_iou=weightlosses_iou, weightloss_coor=weightlosses_coor, weightloss_size=weightlosses_size, weightloss_limb=weightlosses_limb, total_weightloss=weightlosses_resp.val +weightlosses_coor.val +weightlosses_iou.val +weightlosses_size.val +weightlosses_limb.val))
                sys.stdout.flush()

                plotter.plot('loss', 'train', 'PPN Loss', epoch + (i/len(train_loader)), losses.avg) 
                plotter.plot('weight', 'weight_resp', 'Weight Loss', epoch + (i/len(train_loader)), weightlosses_resp.val)
                plotter.plot('weight', 'weight_iou', 'Weight Loss', epoch + (i/len(train_loader)), weightlosses_iou.val)
                plotter.plot('weight', 'weight_coor', 'Weight Loss', epoch + (i/len(train_loader)), weightlosses_coor.val)
                plotter.plot('weight', 'weight_size', 'Weight Loss', epoch + (i/len(train_loader)), weightlosses_size.val)
                plotter.plot('weight', 'weight_limb', 'Weight Loss', epoch + (i/len(train_loader)), weightlosses_limb.val)

    return losses.avg

def test_output(val_loader, val_dataset, model, weight_model,  criterion, outsize, local_grid_size, args):
    from datatest import get_humans_by_feature, draw_humans, show_sample, evaluation
    # Process with one batch
    data_time = AverageMeter()

    # revert normalized image to original image
    mean = torch.tensor([0.485 , 0.456 , 0.406 ]).cuda().view(1,3,1,1)
    std = torch.tensor([0.229 , 0.224 , 0.225 ]).cuda().view(1,3,1,1)

    # switch to evaluate mode
    model.eval()
    outW, outH = outsize

    ## pck_object: fname, gt_KPs、 gt_bboxs, humans(pred_KPs, pred_bboxs), scores, is_visible, size
    pck_object = [[], [], [], [], [], [], []] 

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

            loss_resp, loss_iou, loss_coor, loss_size, loss_limb = criterion(
                    img, output, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te)

            l0 = torch.mul(weight_model.weight[0][0], loss_resp)
            l1 = torch.mul(weight_model.weight[0][1], loss_iou)
            l2 = torch.mul(weight_model.weight[0][2], loss_coor)
            l3 = torch.mul(weight_model.weight[0][3], loss_size)
            l4 = torch.mul(weight_model.weight[0][4], loss_limb)

            loss = torch.div(l0+l1+l2+l3+l4, 5)

            reduced_loss = loss.data
            reduced_loss_resp = loss_resp.data
            reduced_loss_iou = loss_iou.data
            reduced_loss_coor = loss_coor.data
            reduced_loss_size = loss_size.data
            reduced_loss_limb = loss_limb.data

            #logger.info("resp loss:"+str(reduced_loss_resp))

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

            resp = np.squeeze(resp, axis=0)
            conf = np.squeeze(conf, axis=0)
            x = np.squeeze(x, axis=0)
            y = np.squeeze(y, axis=0)
            w = np.squeeze(w, axis=0)
            h = np.squeeze(h, axis=0)
            e = np.squeeze(e, axis=0)

            resp = resp * conf

            delta = delta.cpu().numpy()

#            head_resp = resp[0,:,:]
#            print("Val head exist resp:\t\t Max(", head_resp[np.where(delta[0,0,:,:] == 1.0)].max(), "),\tMin(", head_resp[np.where(delta[0,0,:,:] == 1.0)].min(),"),\tMean(",  head_resp[np.where(delta[0,0,:,:] == 1.0)].mean(), ")")
#            print("Val head non-exist resp:\t Max(", head_resp[np.where(delta[0,0,:,:] == 0.0)].max(), "),\tMin(", head_resp[np.where(delta[0,0,:,:] == 0.0)].min(),"),\tMean(",  head_resp[np.where(delta[0,0,:,:] == 0.0)].mean(), ")")
#
#            ls_resp = resp[1,:,:]
#            print("Val left_shoulder exist resp:\t Max(", ls_resp[np.where(delta[0,1,:,:] == 1.0)].max(), "),\tMin(", ls_resp[np.where(delta[0,1,:,:] == 1.0)].min(),"),\tMean(",  ls_resp[np.where(delta[0,1,:,:] == 1.0)].mean(), ")")
#            print("Val left_shldr non-exist resp:\t Max(", ls_resp[np.where(delta[0,1,:,:] == 0.0)].max(), "),\tMin(", ls_resp[np.where(delta[0,1,:,:] == 0.0)].min(),"),\tMean(",  ls_resp[np.where(delta[0,1,:,:] == 0.0)].mean(), ")")
#
#
#            rs_resp = resp[2,:,:]
#            print("Val right_shoulder exist resp:\t Max(", rs_resp[np.where(delta[0,2,:,:] == 1.0)].max(), "),\tMin(", rs_resp[np.where(delta[0,2,:,:] == 1.0)].min(),"),\tMean(",  rs_resp[np.where(delta[0,2,:,:] == 1.0)].mean(), ")")
#            print("Val right_shldr non-exist resp:\t Max(", rs_resp[np.where(delta[0,2,:,:] == 0.0)].max(), "),\tMin(", rs_resp[np.where(delta[0,2,:,:] == 0.0)].min(),"),\tMean(",  rs_resp[np.where(delta[0,2,:,:] == 0.0)].mean(), ")")
#
#            le_resp = resp[3,:,:]
#            print("Val left_elbow exist resp:\t Max(", le_resp[np.where(delta[0,3,:,:] == 1.0)].max(), "),\tMin(", le_resp[np.where(delta[0,3,:,:] == 1.0)].min(),"),\tMean(",  le_resp[np.where(delta[0,3,:,:] == 1.0)].mean(), ")")
#            print("Val left_elbow non-exist resp:\t Max(", le_resp[np.where(delta[0,3,:,:] == 0.0)].max(), "),\tMin(", le_resp[np.where(delta[0,3,:,:] == 0.0)].min(),"),\tMean(",  le_resp[np.where(delta[0,3,:,:] == 0.0)].mean(), ")")
#
#            re_resp = resp[4,:,:]
#            print("Val right_elbow exist resp:\t Max(", re_resp[np.where(delta[0,4,:,:] == 1.0)].max(), "),\tMin(", re_resp[np.where(delta[0,4,:,:] == 1.0)].min(),"),\tMean(",  re_resp[np.where(delta[0,4,:,:] == 1.0)].mean(), ")")
#            print("Val right_elbow non-exist resp:\t Max(", re_resp[np.where(delta[0,4,:,:] == 0.0)].max(), "),\tMin(", re_resp[np.where(delta[0,4,:,:] == 0.0)].min(),"),\tMean(",  re_resp[np.where(delta[0,4,:,:] == 0.0)].mean(), ")")
#
#            lw_resp = resp[5,:,:]
#            print("Val left_wrist exist resp:\t Max(", lw_resp[np.where(delta[0,5,:,:] == 1.0)].max(), "),\tMin(", lw_resp[np.where(delta[0,5,:,:] == 1.0)].min(),"),\tMean(",  lw_resp[np.where(delta[0,5,:,:] == 1.0)].mean(), ")")
#            print("Val left_wrist non-exist resp:\t Max(", lw_resp[np.where(delta[0,5,:,:] == 0.0)].max(), "),\tMin(", lw_resp[np.where(delta[0,5,:,:] == 0.0)].min(),"),\tMean(",  lw_resp[np.where(delta[0,5,:,:] == 0.0)].mean(), ")")
#
#            rw_resp = resp[6,:,:]
#            print("Val right_wrist exist resp:\t Max(", rw_resp[np.where(delta[0,6,:,:] == 1.0)].max(), "),\tMin(", rw_resp[np.where(delta[0,6,:,:] == 1.0)].min(),"),\tMean(",  rw_resp[np.where(delta[0,6,:,:] == 1.0)].mean(), ")")
#            print("Val right_wrist non-exist resp:\t Max(", rw_resp[np.where(delta[0,6,:,:] == 0.0)].max(), "),\tMin(", rw_resp[np.where(delta[0,6,:,:] == 0.0)].min(),"),\tMean(",  rw_resp[np.where(delta[0,6,:,:] == 0.0)].mean(), ")")
#
#            lh_resp = resp[7,:,:]
#            print("Val left_hip exist resp:\t Max(", lh_resp[np.where(delta[0,7,:,:] == 1.0)].max(), "),\tMin(", lh_resp[np.where(delta[0,7,:,:] == 1.0)].min(),"),\tMean(",  lh_resp[np.where(delta[0,7,:,:] == 1.0)].mean(), ")")
#            print("Val left_hip non-exist resp:\t Max(", lh_resp[np.where(delta[0,7,:,:] == 0.0)].max(), "),\tMin(", lh_resp[np.where(delta[0,7,:,:] == 0.0)].min(),"),\tMean(",  lh_resp[np.where(delta[0,7,:,:] == 0.0)].mean(), ")")
#
#            rh_resp = resp[8,:,:]
#            print("Val right_hip exist resp:\t Max(", rh_resp[np.where(delta[0,8,:,:] == 1.0)].max(), "),\tMin(", rh_resp[np.where(delta[0,8,:,:] == 1.0)].min(),"),\tMean(",  rh_resp[np.where(delta[0,8,:,:] == 1.0)].mean(), ")")
#            print("Val right_hip non-exist resp:\t Max(", rh_resp[np.where(delta[0,8,:,:] == 0.0)].max(), "),\tMin(", rh_resp[np.where(delta[0,8,:,:] == 0.0)].min(),"),\tMean(",  rh_resp[np.where(delta[0,8,:,:] == 0.0)].mean(), ")")
#
#            lk_resp = resp[9,:,:]
#            print("Val left_knee exist resp:\t Max(", lk_resp[np.where(delta[0,9,:,:] == 1.0)].max(), "),\tMin(", lk_resp[np.where(delta[0,9,:,:] == 1.0)].min(),"),\tMean(",  lk_resp[np.where(delta[0,9,:,:] == 1.0)].mean(), ")")
#            print("Val left_knee non-exist resp:\t Max(", lk_resp[np.where(delta[0,9,:,:] == 0.0)].max(), "),\tMin(", lk_resp[np.where(delta[0,9,:,:] == 0.0)].min(),"),\tMean(",  lk_resp[np.where(delta[0,9,:,:] == 0.0)].mean(), ")")
#
#            rk_resp = resp[10,:,:]
#            print("Val right_knee exist resp:\t Max(", rk_resp[np.where(delta[0,10,:,:] == 1.0)].max(), "),\tMin(", rk_resp[np.where(delta[0,10,:,:] == 1.0)].min(),"),\tMean(",  rk_resp[np.where(delta[0,10,:,:] == 1.0)].mean(), ")")
#            print("Val right_knee non-exist resp:\t Max(", rk_resp[np.where(delta[0,10,:,:] == 0.0)].max(), "),\tMin(", rk_resp[np.where(delta[0,10,:,:] == 0.0)].min(),"),\tMean(",  rk_resp[np.where(delta[0,10,:,:] == 0.0)].mean(), ")")
#
#            la_resp = resp[11,:,:]
#            print("Val left_ankle exist resp:\t Max(", la_resp[np.where(delta[0,11,:,:] == 1.0)].max(), "),\tMin(", la_resp[np.where(delta[0,11,:,:] == 1.0)].min(),"),\tMean(",  la_resp[np.where(delta[0,11,:,:] == 1.0)].mean(), ")")
#            print("Val left_ankle non-exist resp:\t Max(", la_resp[np.where(delta[0,11,:,:] == 0.0)].max(), "),\tMin(", la_resp[np.where(delta[0,11,:,:] == 0.0)].min(),"),\tMean(",  la_resp[np.where(delta[0,11,:,:] == 0.0)].mean(), ")")
#
#            ra_resp = resp[12,:,:]
#            print("Val right_ankle exist resp:\t Max(", ra_resp[np.where(delta[0,12,:,:] == 1.0)].max(), "),\tMin(", ra_resp[np.where(delta[0,12,:,:] == 1.0)].min(),"),\tMean(",  ra_resp[np.where(delta[0,12,:,:] == 1.0)].mean(), ")")
#            print("Val right_ankle non-exist resp:\t Max(", ra_resp[np.where(delta[0,12,:,:] == 0.0)].max(), "),\tMin(", ra_resp[np.where(delta[0,12,:,:] == 0.0)].min(),"),\tMean(",  ra_resp[np.where(delta[0,12,:,:] == 0.0)].mean(), ")")
#
#            thorax_resp = resp[13,:,:]
#            print("Val thorax exist resp:\t\t Max(", thorax_resp[np.where(delta[0,13,:,:] == 1.0)].max(), "),\tMin(", thorax_resp[np.where(delta[0,13,:,:] == 1.0)].min(),"),\tMean(",  thorax_resp[np.where(delta[0,13,:,:] == 1.0)].mean(), ")")
#            print("Val thorax non-exist resp:\t Max(", thorax_resp[np.where(delta[0,13,:,:] == 0.0)].max(), "),\tMin(", thorax_resp[np.where(delta[0,13,:,:] == 0.0)].min(),"),\tMean(",  thorax_resp[np.where(delta[0,13,:,:] == 0.0)].mean(), ")")
#
#            pelvis_resp = resp[14,:,:]
#            print("Val pelvis exist resp:\t\t Max(", pelvis_resp[np.where(delta[0,14,:,:] == 1.0)].max(), "),\tMin(", pelvis_resp[np.where(delta[0,14,:,:] == 1.0)].min(),"),\tMean(",  pelvis_resp[np.where(delta[0,14,:,:] == 1.0)].mean(), ")")
#            print("Val pelvis non-exist resp:\t Max(", pelvis_resp[np.where(delta[0,14,:,:] == 0.0)].max(), "),\tMin(", pelvis_resp[np.where(delta[0,14,:,:] == 0.0)].min(),"),\tMean(",  pelvis_resp[np.where(delta[0,14,:,:] == 0.0)].mean(), ")")
#
#            neck_resp = resp[15,:,:]
#            print("Val neck exist resp:\t\t Max(", neck_resp[np.where(delta[0,15,:,:] == 1.0)].max(), "),\tMin(", neck_resp[np.where(delta[0,15,:,:] == 1.0)].min(),"),\tMean(",  neck_resp[np.where(delta[0,15,:,:] == 1.0)].mean(), ")")
#            print("Val neck non-exist resp:\t Max(", neck_resp[np.where(delta[0,15,:,:] == 0.0)].max(), "),\tMin(", neck_resp[np.where(delta[0,15,:,:] == 0.0)].min(),"),\tMean(",  neck_resp[np.where(delta[0,15,:,:] == 0.0)].mean(), ")")
#
#            top_resp = resp[16,:,:]
#            print("Val top exist resp:\t\t Max(", top_resp[np.where(delta[0,16,:,:] == 1.0)].max(), "),\tMin(", top_resp[np.where(delta[0,16,:,:] == 1.0)].min(),"),\tMean(",  top_resp[np.where(delta[0,16,:,:] == 1.0)].mean(), ")")
#            print("Val top non-exist resp:\t\t Max(", top_resp[np.where(delta[0,16,:,:] == 0.0)].max(), "),\tMin(", top_resp[np.where(delta[0,16,:,:] == 0.0)].min(),"),\tMean(",  top_resp[np.where(delta[0,16,:,:] == 0.0)].mean(), ")")
#
#            stomach_resp = resp[17,:,:]
#            print("Val stomach exist resp:\t\t Max(", stomach_resp[np.where(delta[0,17,:,:] == 1.0)].max(), "),\tMin(", stomach_resp[np.where(delta[0,17,:,:] == 1.0)].min(),"),\tMean(",  stomach_resp[np.where(delta[0,17,:,:] == 1.0)].mean(), ")")
#            print("Val stomach non-exist resp:\t Max(", stomach_resp[np.where(delta[0,17,:,:] == 0.0)].max(), "),\tMin(", stomach_resp[np.where(delta[0,17,:,:] == 0.0)].min(),"),\tMean(",  stomach_resp[np.where(delta[0,17,:,:] == 0.0)].mean(), ")")
#
#            sys.stdout.flush()

#            logger.info("max human delta value:"+str(delta[0,:,:].reshape(-1)[np.argsort(delta[0,:,:].reshape(-1))[-7:]]))
#            logger.info("max 1 delta value:"+str(delta[1,:,:].reshape(-1)[np.argsort(delta[1,:,:].reshape(-1))[-7:]]))
#            logger.info("max 2 delta value:"+str(delta[2,:,:].reshape(-1)[np.argsort(delta[2,:,:].reshape(-1))[-7:]]))
#            logger.info("max 3 delta value:"+str(delta[3,:,:].reshape(-1)[np.argsort(delta[3,:,:].reshape(-1))[-7:]]))
#            logger.info("max 4 delta value:"+str(delta[4,:,:].reshape(-1)[np.argsort(delta[4,:,:].reshape(-1))[-7:]]))
#            logger.info("max 5 delta value:"+str(delta[5,:,:].reshape(-1)[np.argsort(delta[5,:,:].reshape(-1))[-7:]]))
#            logger.info("max 6 delta value:"+str(delta[6,:,:].reshape(-1)[np.argsort(delta[6,:,:].reshape(-1))[-7:]]))
#            logger.info("max 7 delta value:"+str(delta[7,:,:].reshape(-1)[np.argsort(delta[7,:,:].reshape(-1))[-7:]]))
#            logger.info("max 8 delta value:"+str(delta[8,:,:].reshape(-1)[np.argsort(delta[8,:,:].reshape(-1))[-7:]]))
#            logger.info("max 9 delta value:"+str(delta[9,:,:].reshape(-1)[np.argsort(delta[9,:,:].reshape(-1))[-7:]]))
#            logger.info("max 10 delta value:"+str(delta[10,:,:].reshape(-1)[np.argsort(delta[10,:,:].reshape(-1))[-7:]]))
#            logger.info("max 11 delta value:"+str(delta[11,:,:].reshape(-1)[np.argsort(delta[11,:,:].reshape(-1))[-7:]]))
#            logger.info("max 12 delta value:"+str(delta[12,:,:].reshape(-1)[np.argsort(delta[12,:,:].reshape(-1))[-7:]]))
#            logger.info("max 13 delta value:"+str(delta[13,:,:].reshape(-1)[np.argsort(delta[13,:,:].reshape(-1))[-7:]]))
#            logger.info("max 14 delta value:"+str(delta[14,:,:].reshape(-1)[np.argsort(delta[14,:,:].reshape(-1))[-7:]]))
#            logger.info("max 15 delta value:"+str(delta[15,:,:].reshape(-1)[np.argsort(delta[15,:,:].reshape(-1))[-7:]]))
#            logger.info("max 16 delta value:"+str(delta[16,:,:].reshape(-1)[np.argsort(delta[16,:,:].reshape(-1))[-7:]]))
#            logger.info("max 17 delta value:"+str(delta[17,:,:].reshape(-1)[np.argsort(delta[17,:,:].reshape(-1))[-7:]]))

            # basic detection_thresh = 0.15
            humans, scores = get_humans_by_feature(resp, x, y, w, h, e, detection_thresh=0.15)

            # PCKh metric
            fname = val_dataset.filename_list[i]
            #gt_kps = np.array(val_dataset.data[fname][0], dtype='float32').reshape(-1,17,2)
            gt_kps = np.array(val_dataset.data[fname][0], dtype='float32').reshape(-1,17,2)

            gt_bboxes = val_dataset.data[fname][1]    # [center_x, center_y , width, height] for head
            is_visible = val_dataset.data[fname][2]
            size = val_dataset.data[fname][3]

            # include pred_KPs, pred_bbox
            pck_object[0].append(fname)
            pck_object[1].append(gt_kps)
            pck_object[2].append(humans)
            pck_object[3].append(scores)
            pck_object[4].append(gt_bboxes)
            pck_object[5].append(is_visible)
            pck_object[6].append(size)

            if args.savefig and i < 100:
            #if args.savefig :
                raw_img = img.mul_(std).add_(mean)
                raw_pil_image = Image.fromarray(np.squeeze(raw_img.cpu().numpy(), axis=0).astype(np.uint8).transpose(1, 2, 0))

                pil_image = draw_humans(
                    keypoint_names=KEYPOINT_NAMES,
                    edges=EDGES,
                    pil_image=raw_pil_image.copy(),
                    humans=humans,
                    visbbox= False,
                    gridOn = True
                )   

                #sample_fig = show_sample(raw_pil_image, delta[0], pil_image)
                sample_fig = show_sample(raw_pil_image, resp, pil_image)

                pil_image.save('output/training_test/predict_test_result_'+str(i)+'.png', 'PNG')
                sample_fig.savefig('output/training_test/predict_test_resp_result_'+str(i)+'.png')

            if i>= 10:
#                print("gt_kps", pck_object[1])
#                print("gt_bbox", pck_object[4])
#                print("human", pck_object[2])
#                print("size", pck_object[6])
                break

        _ = evaluation(pck_object)


def validate(val_loader, val_dataset,  model, weight_model, criterion, epoch, args):
    from datatest import get_humans_by_feature, draw_humans, show_sample, evaluation

    batch_time = AverageMeter()
    losses = AverageMeter()

    losses_resp = AverageMeter()
    losses_iou = AverageMeter()
    losses_coor = AverageMeter()
    losses_size = AverageMeter()
    losses_limb = AverageMeter()

    # switch to evaluate mode
    model.eval()
    outW, outH = outsize

    end = time.time()

    # revert normalized image to original image
    mean = torch.tensor([0.485 , 0.456 , 0.406 ]).cuda().view(1,3,1,1)
    std = torch.tensor([0.229 , 0.224 , 0.225 ]).cuda().view(1,3,1,1)

    ## pck_object: fname, gt_KPs、 gt_bboxs, humans(pred_KPs, pred_bboxs), scores, is_visible, size
    pck_object = [[], [], [], [], [], [], []] 

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

            loss_resp, loss_iou, loss_coor, loss_size, loss_limb = criterion(
                    img, output, delta, weight, weight_ij, tx_half, ty_half, tx, ty, tw, th, te)

            l0 = torch.mul(weight_model.weight[0][0], loss_resp)
            l1 = torch.mul(weight_model.weight[0][1], loss_iou)
            l2 = torch.mul(weight_model.weight[0][2], loss_coor)
            l3 = torch.mul(weight_model.weight[0][3], loss_size)
            l4 = torch.mul(weight_model.weight[0][4], loss_limb)

            loss = torch.div(l0+l1+l2+l3+l4, 5)

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

            resp = np.squeeze(resp, axis=0)
            conf = np.squeeze(conf, axis=0)
            x = np.squeeze(x, axis=0)
            y = np.squeeze(y, axis=0)
            w = np.squeeze(w, axis=0)
            h = np.squeeze(h, axis=0)
            e = np.squeeze(e, axis=0)

            resp = resp * conf

            delta = delta.cpu().numpy()

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

            # basic detection_thresh = 0.15
            humans, scores = get_humans_by_feature(resp, x, y, w, h, e, detection_thresh=0.15)

            # PCKh metric
            fname = val_dataset.filename_list[i]
            gt_kps = np.array(val_dataset.data[fname][0], dtype='float32').reshape(-1,17,2)

            gt_bboxes = val_dataset.data[fname][1]    # [center_x, center_y , width, height] for head
            is_visible = val_dataset.data[fname][2]
            size = val_dataset.data[fname][3]

            # include pred_KPs, pred_bbox
            pck_object[0].append(fname)
            pck_object[1].append(gt_kps)
            pck_object[2].append(humans)
            pck_object[3].append(scores)
            pck_object[4].append(gt_bboxes)
            pck_object[5].append(is_visible)
            pck_object[6].append(size)

            if i % args.print_freq == 0 and  args.local_rank == 0:

#                K = len(KEYPOINT_NAMES)
#
#                if args.distributed:
#                    resp = reduced_resp_data[:, 0 * K:1 * K, :, :].cpu().numpy() # delta
#                else:
#                    resp = output.data[:, 0 * K:1 * K, :, :].cpu().numpy() # delta
#
#                temp_delta = delta.cpu().numpy()
#
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

        ap_vals = evaluation(pck_object)

    return losses.avg, ap_vals


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

    if epoch % 300 == 0 and epoch > 1 :
        lr = 0.5* lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def reduce_tensor(tensor):
    global args
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def print_resp(delta, resp):
    # Ground Truth
    temp_delta = delta.cpu().numpy()

    head_resp = resp[:,0,:,:]
    print("Trn head exist resp:\t\t Max(", head_resp[np.where(temp_delta[:,0,:,:] == 1.0)].max(), "),\tMin(", head_resp[np.where(temp_delta[:,0,:,:] == 1.0)].min(),"),\tMean(",  head_resp[np.where(temp_delta[:,0,:,:] == 1.0)].mean(), ")")
    print("Trn head non-exist resp:\t Max(", head_resp[np.where(temp_delta[:,0,:,:] == 0.0)].max(), "),\tMin(", head_resp[np.where(temp_delta[:,0,:,:] == 0.0)].min(),"),\tMean(",  head_resp[np.where(temp_delta[:,0,:,:] == 0.0)].mean(), ")")

    ls_resp = resp[:,1,:,:]
    print("Trn left_shoulder exist resp:\t Max(", ls_resp[np.where(temp_delta[:,1,:,:] == 1.0)].max(), "),\tMin(", ls_resp[np.where(temp_delta[:,1,:,:] == 1.0)].min(),"),\tMean(",  ls_resp[np.where(temp_delta[:,1,:,:] == 1.0)].mean(), ")")
    print("Trn left_shldr non-exist resp:\t Max(", ls_resp[np.where(temp_delta[:,1,:,:] == 0.0)].max(), "),\tMin(", ls_resp[np.where(temp_delta[:,1,:,:] == 0.0)].min(),"),\tMean(",  ls_resp[np.where(temp_delta[:,1,:,:] == 0.0)].mean(), ")")

    rs_resp = resp[:,2,:,:]
    print("Trn right_shoulder exist resp:\t Max(", rs_resp[np.where(temp_delta[:,2,:,:] == 1.0)].max(), "),\tMin(", rs_resp[np.where(temp_delta[:,2,:,:] == 1.0)].min(),"),\tMean(",  rs_resp[np.where(temp_delta[:,2,:,:] == 1.0)].mean(), ")")
    print("Trn right_shldr non-exist resp:\t Max(", rs_resp[np.where(temp_delta[:,2,:,:] == 0.0)].max(), "),\tMin(", rs_resp[np.where(temp_delta[:,2,:,:] == 0.0)].min(),"),\tMean(",  rs_resp[np.where(temp_delta[:,2,:,:] == 0.0)].mean(), ")")

    le_resp = resp[:,3,:,:]
    print("Trn left_elbow exist resp:\t Max(", le_resp[np.where(temp_delta[:,3,:,:] == 1.0)].max(), "),\tMin(", le_resp[np.where(temp_delta[:,3,:,:] == 1.0)].min(),"),\tMean(",  le_resp[np.where(temp_delta[:,3,:,:] == 1.0)].mean(), ")")
    print("Trn left_elbow non-exist resp:\t Max(", le_resp[np.where(temp_delta[:,3,:,:] == 0.0)].max(), "),\tMin(", le_resp[np.where(temp_delta[:,3,:,:] == 0.0)].min(),"),\tMean(",  le_resp[np.where(temp_delta[:,3,:,:] == 0.0)].mean(), ")")

    re_resp = resp[:,4,:,:]
    print("Trn right_elbow exist resp:\t Max(", re_resp[np.where(temp_delta[:,4,:,:] == 1.0)].max(), "),\tMin(", re_resp[np.where(temp_delta[:,4,:,:] == 1.0)].min(),"),\tMean(",  re_resp[np.where(temp_delta[:,4,:,:] == 1.0)].mean(), ")")
    print("Trn right_elbow non-exist resp:\t Max(", re_resp[np.where(temp_delta[:,4,:,:] == 0.0)].max(), "),\tMin(", re_resp[np.where(temp_delta[:,4,:,:] == 0.0)].min(),"),\tMean(",  re_resp[np.where(temp_delta[:,4,:,:] == 0.0)].mean(), ")")

    lw_resp = resp[:,5,:,:]
    print("Trn left_wrist exist resp:\t Max(", lw_resp[np.where(temp_delta[:,5,:,:] == 1.0)].max(), "),\tMin(", lw_resp[np.where(temp_delta[:,5,:,:] == 1.0)].min(),"),\tMean(",  lw_resp[np.where(temp_delta[:,5,:,:] == 1.0)].mean(), ")")
    print("Trn left_wrist non-exist resp:\t Max(", lw_resp[np.where(temp_delta[:,5,:,:] == 0.0)].max(), "),\tMin(", lw_resp[np.where(temp_delta[:,5,:,:] == 0.0)].min(),"),\tMean(",  lw_resp[np.where(temp_delta[:,5,:,:] == 0.0)].mean(), ")")

    rw_resp = resp[:,6,:,:]
    print("Trn right_wrist exist resp:\t Max(", rw_resp[np.where(temp_delta[:,6,:,:] == 1.0)].max(), "),\tMin(", rw_resp[np.where(temp_delta[:,6,:,:] == 1.0)].min(),"),\tMean(",  rw_resp[np.where(temp_delta[:,6,:,:] == 1.0)].mean(), ")")
    print("Trn right_wrist non-exist resp:\t Max(", rw_resp[np.where(temp_delta[:,6,:,:] == 0.0)].max(), "),\tMin(", rw_resp[np.where(temp_delta[:,6,:,:] == 0.0)].min(),"),\tMean(",  rw_resp[np.where(temp_delta[:,6,:,:] == 0.0)].mean(), ")")

    lh_resp = resp[:,7,:,:]
    print("Trn left_hip exist resp:\t Max(", lh_resp[np.where(temp_delta[:,7,:,:] == 1.0)].max(), "),\tMin(", lh_resp[np.where(temp_delta[:,7,:,:] == 1.0)].min(),"),\tMean(",  lh_resp[np.where(temp_delta[:,7,:,:] == 1.0)].mean(), ")")
    print("Trn left_hip non-exist resp:\t Max(", lh_resp[np.where(temp_delta[:,7,:,:] == 0.0)].max(), "),\tMin(", lh_resp[np.where(temp_delta[:,7,:,:] == 0.0)].min(),"),\tMean(",  lh_resp[np.where(temp_delta[:,7,:,:] == 0.0)].mean(), ")")

    rh_resp = resp[:,8,:,:]
    print("Trn right_hip exist resp:\t Max(", rh_resp[np.where(temp_delta[:,8,:,:] == 1.0)].max(), "),\tMin(", rh_resp[np.where(temp_delta[:,8,:,:] == 1.0)].min(),"),\tMean(",  rh_resp[np.where(temp_delta[:,8,:,:] == 1.0)].mean(), ")")
    print("Trn right_hip non-exist resp:\t Max(", rh_resp[np.where(temp_delta[:,8,:,:] == 0.0)].max(), "),\tMin(", rh_resp[np.where(temp_delta[:,8,:,:] == 0.0)].min(),"),\tMean(",  rh_resp[np.where(temp_delta[:,8,:,:] == 0.0)].mean(), ")")

    lk_resp = resp[:,9,:,:]
    print("Trn left_knee exist resp:\t Max(", lk_resp[np.where(temp_delta[:,9,:,:] == 1.0)].max(), "),\tMin(", lk_resp[np.where(temp_delta[:,9,:,:] == 1.0)].min(),"),\tMean(",  lk_resp[np.where(temp_delta[:,9,:,:] == 1.0)].mean(), ")")
    print("Trn left_knee non-exist resp:\t Max(", lk_resp[np.where(temp_delta[:,9,:,:] == 0.0)].max(), "),\tMin(", lk_resp[np.where(temp_delta[:,9,:,:] == 0.0)].min(),"),\tMean(",  lk_resp[np.where(temp_delta[:,9,:,:] == 0.0)].mean(), ")")

    rk_resp = resp[:,10,:,:]
    print("Trn right_knee exist resp:\t Max(", rk_resp[np.where(temp_delta[:,10,:,:] == 1.0)].max(), "),\tMin(", rk_resp[np.where(temp_delta[:,10,:,:] == 1.0)].min(),"),\tMean(",  rk_resp[np.where(temp_delta[:,10,:,:] == 1.0)].mean(), ")")
    print("Trn right_knee non-exist resp:\t Max(", rk_resp[np.where(temp_delta[:,10,:,:] == 0.0)].max(), "),\tMin(", rk_resp[np.where(temp_delta[:,10,:,:] == 0.0)].min(),"),\tMean(",  rk_resp[np.where(temp_delta[:,10,:,:] == 0.0)].mean(), ")")

    la_resp = resp[:,11,:,:]
    print("Trn left_ankle exist resp:\t Max(", la_resp[np.where(temp_delta[:,11,:,:] == 1.0)].max(), "),\tMin(", la_resp[np.where(temp_delta[:,11,:,:] == 1.0)].min(),"),\tMean(",  la_resp[np.where(temp_delta[:,11,:,:] == 1.0)].mean(), ")")
    print("Trn left_ankle non-exist resp:\t Max(", la_resp[np.where(temp_delta[:,11,:,:] == 0.0)].max(), "),\tMin(", la_resp[np.where(temp_delta[:,11,:,:] == 0.0)].min(),"),\tMean(",  la_resp[np.where(temp_delta[:,11,:,:] == 0.0)].mean(), ")")

    ra_resp = resp[:,12,:,:]
    print("Trn right_ankle exist resp:\t Max(", ra_resp[np.where(temp_delta[:,12,:,:] == 1.0)].max(), "),\tMin(", ra_resp[np.where(temp_delta[:,12,:,:] == 1.0)].min(),"),\tMean(",  ra_resp[np.where(temp_delta[:,12,:,:] == 1.0)].mean(), ")")
    print("Trn right_ankle non-exist resp:\t Max(", ra_resp[np.where(temp_delta[:,12,:,:] == 0.0)].max(), "),\tMin(", ra_resp[np.where(temp_delta[:,12,:,:] == 0.0)].min(),"),\tMean(",  ra_resp[np.where(temp_delta[:,12,:,:] == 0.0)].mean(), ")")

    thorax_resp = resp[:,13,:,:]
    print("Trn thorax exist resp:\t\t Max(", thorax_resp[np.where(temp_delta[:,13,:,:] == 1.0)].max(), "),\tMin(", thorax_resp[np.where(temp_delta[:,13,:,:] == 1.0)].min(),"),\tMean(",  thorax_resp[np.where(temp_delta[:,13,:,:] == 1.0)].mean(), ")")
    print("Trn thorax non-exist resp:\t Max(", thorax_resp[np.where(temp_delta[:,13,:,:] == 0.0)].max(), "),\tMin(", thorax_resp[np.where(temp_delta[:,13,:,:] == 0.0)].min(),"),\tMean(",  thorax_resp[np.where(temp_delta[:,13,:,:] == 0.0)].mean(), ")")

    pelvis_resp = resp[:,14,:,:]
    print("Trn pelvis exist resp:\t\t Max(", pelvis_resp[np.where(temp_delta[:,14,:,:] == 1.0)].max(), "),\tMin(", pelvis_resp[np.where(temp_delta[:,14,:,:] == 1.0)].min(),"),\tMean(",  pelvis_resp[np.where(temp_delta[:,14,:,:] == 1.0)].mean(), ")")
    print("Trn pelvis non-exist resp:\t Max(", pelvis_resp[np.where(temp_delta[:,14,:,:] == 0.0)].max(), "),\tMin(", pelvis_resp[np.where(temp_delta[:,14,:,:] == 0.0)].min(),"),\tMean(",  pelvis_resp[np.where(temp_delta[:,14,:,:] == 0.0)].mean(), ")")

    neck_resp = resp[:,15,:,:]
    print("Trn neck exist resp:\t\t Max(", neck_resp[np.where(temp_delta[:,15,:,:] == 1.0)].max(), "),\tMin(", neck_resp[np.where(temp_delta[:,15,:,:] == 1.0)].min(),"),\tMean(",  neck_resp[np.where(temp_delta[:,15,:,:] == 1.0)].mean(), ")")
    print("Trn neck non-exist resp:\t Max(", neck_resp[np.where(temp_delta[:,15,:,:] == 0.0)].max(), "),\tMin(", neck_resp[np.where(temp_delta[:,15,:,:] == 0.0)].min(),"),\tMean(",  neck_resp[np.where(temp_delta[:,15,:,:] == 0.0)].mean(), ")")

    top_resp = resp[:,16,:,:]
    print("Trn top exist resp:\t\t Max(", top_resp[np.where(temp_delta[:,16,:,:] == 1.0)].max(), "),\tMin(", top_resp[np.where(temp_delta[:,16,:,:] == 1.0)].min(),"),\tMean(",  top_resp[np.where(temp_delta[:,16,:,:] == 1.0)].mean(), ")")
    print("Trn top non-exist resp:\t\t Max(", top_resp[np.where(temp_delta[:,16,:,:] == 0.0)].max(), "),\tMin(", top_resp[np.where(temp_delta[:,16,:,:] == 0.0)].min(),"),\tMean(",  top_resp[np.where(temp_delta[:,16,:,:] == 0.0)].mean(), ")")

    stomach_resp = resp[:,17,:,:]
    print("Trn stomach exist resp:\t\t Max(", stomach_resp[np.where(temp_delta[:,17,:,:] == 1.0)].max(), "),\tMin(", stomach_resp[np.where(temp_delta[:,17,:,:] == 1.0)].min(),"),\tMean(",  stomach_resp[np.where(temp_delta[:,17,:,:] == 1.0)].mean(), ")")
    print("Trn stomach non-exist resp:\t Max(", stomach_resp[np.where(temp_delta[:,17,:,:] == 0.0)].max(), "),\tMin(", stomach_resp[np.where(temp_delta[:,17,:,:] == 0.0)].min(),"),\tMean(",  stomach_resp[np.where(temp_delta[:,17,:,:] == 0.0)].mean(), ")")

if __name__ == '__main__':

    import logging
    global plotter
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    plotter = VisdomLinePlotter(env_name="PoseProposalNet")
    main()
