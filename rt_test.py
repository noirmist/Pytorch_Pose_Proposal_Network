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

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Custom models
import drn #dilated residual network
from model import *
from config import *
from dataset import *
from datatest import get_humans_by_feature, draw_humans
from aug import *

parser = argparse.ArgumentParser(description='PyTorch PoseProposalNet Training')
parser.add_argument("-imsize", "--image_size", default=384, type=int, help="set input image size")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()



def network():
    global args

    # Prepare model
    model = drn.drn_d_22()

    local_grid_size=(21, 21)

    modules = list(model.children())[:-2]
    model = nn.Sequential(*modules)
    model = PoseProposalNet(model, local_grid_size=local_grid_size)
    model = model.cuda()

    inputs = torch.randn(1,3, args.image_size, args.image_size).cuda()
    y =  model(inputs)
    _, _, outH, outW =y.shape
    outsize = (outW, outH)

    # Load weights
    def resume():
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(0))
            model.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    resume()

    sys.stdout.flush()

    return model, outsize, local_grid_size

def inference(image, model, outsize, local_grid_size):
 
    # revert normalized image to original image
    mean = torch.tensor([0.485 , 0.456 , 0.406 ]).cuda().view(1,3,1,1)
    std = torch.tensor([0.229 , 0.224 , 0.225 ]).cuda().view(1,3,1,1)

    # switch to evaluate mode
    model.eval()
    outW, outH = outsize

    image = np.array(image).transpose((2, 0, 1))
    image = torch.from_numpy(image).cuda().view(1,3,args.image_size,args.image_size)
    # Normalize 
    image = image.float()
    image = image.sub_(mean).div_(std)

    output = model(image).detach()


    K = len(KEYPOINT_NAMES)

    #loss function with torch
    resp = output[:, 0 * K:1 * K, :, :].cpu().numpy() # delta
    conf = output[:, 1 * K:2 * K, :, :].cpu().numpy()
    x = output[:, 2 * K:3 * K, :, :].cpu().numpy()
    y = output[:, 3 * K:4 * K, :, :].cpu().numpy()
    w = output[:, 4 * K:5 * K, :, :].cpu().numpy()
    h = output[:, 5 * K:6 * K, :, :].cpu().numpy()
    e = output[:, 6 * K:, :, :].reshape(
        1,
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

    # basic detection_thresh = 0.15
    humans, scores = get_humans_by_feature(resp, x, y, w, h, e, detection_thresh=0.15)

    raw_img = image.mul_(std).add_(mean)
    raw_pil_image = Image.fromarray(np.squeeze(raw_img.cpu().numpy(), axis=0).astype(np.uint8).transpose(1, 2, 0))

    pil_image = draw_humans(
        keypoint_names=KEYPOINT_NAMES,
        edges=EDGES,
        pil_image=raw_pil_image.copy(),
        humans=humans,
        visbbox= False,
        gridOn = False
    )

    return pil_image


def grab_frame(cap):
    ret,frame = cap.read()
    frame = cv2.resize(frame, dsize=(384,384))
    
    frame = cv2.flip(frame, 0)
    frame = cv2.flip(frame, 1)

    return ret, cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


if __name__ == '__main__':

    model, outsize, local_grid_size = network()

    cap = cv2.VideoCapture(0)
    cap.set(14, 0.5)

    plt.figure(figsize=(14,7))

    ax1 = plt.subplot(1,2,1)
    ax1.axis("off")

    ax2 = plt.subplot(1,2,2)
    ax2.axis("off")


    ret, input_img = grab_frame(cap)

    im1 = ax1.imshow(input_img)
    im2 = ax2.imshow(input_img)

    def update(i):
        start = time.time()
        ret, input_img = grab_frame(cap)
        end = time.time()
        #print("grab_frame:", end-start)

        #DRN function
        start = time.time()
        output_img = inference(input_img, model, outsize, local_grid_size)
        end = time.time()
        #print("Processing time:", end-start)
        #sys.stdout.flush()
        im1.set_data(input_img)

        im2.set_data(output_img)

    def close(event):
        if event.key == 'q':
            plt.close(event.canvas.figure)


    ani = FuncAnimation(plt.gcf(), update, interval=10)
    cid = plt.gcf().canvas.mpl_connect("key_press_event", close)

    plt.show()
