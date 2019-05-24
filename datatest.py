from __future__ import print_function, division
import time
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

from aug import *
from config import *
from dataset import *
import functools
#from augment import *
from PIL import ImageDraw, Image

import imgaug as ia
from imgaug import augmenters as iaa

from sys import maxsize
from numpy import set_printoptions

set_printoptions(threshold=maxsize)

#parser = argparse.ArgumentParser()
#parser.add_argument("-d", "--file", help="json file path")
#parser.add_argument("-img", "--root_dir", help="path to train2017 or val2018")
#
#args = parser.parse_args()
#root_dir = args.root_dir
#
insize = (384, 384)
#outsize = (12, 12)
#local_grid_size = (9, 9)

#outsize = (32, 32)
#local_grid_size = (29, 29)

outsize = (24, 24)
local_grid_size = (21, 21)

inW, inH = insize
outW, outH = outsize
sW, sH = local_grid_size
gridsize = (int(inW / outW), int(inH / outH))


# Recover size
def restore_xy(x, y): 
    gridW, gridH = gridsize
    outW, outH = outsize
    X, Y = np.meshgrid(np.arange(outW, dtype=np.float32), np.arange(outH, dtype=np.float32))
    return (x + X) * gridW, (y + Y) * gridH

def restore_size(w, h): 
    inW, inH = insize
    return inW * w, inH * h 
 

# Parse result
def get_humans_by_feature(delta, x, y, w, h, e, detection_thresh=0.015, min_num_keypoints=1):
    start = time.time()

    #delta = resp * conf
    #K = len(KEYPOINT_NAMES)
    outW, outH = outsize
    ROOT_NODE = 0  # instance

    rx, ry = restore_xy(x, y)
    rw, rh = restore_size(w, h)
    ymin, ymax = ry - rh / 2, ry + rh / 2
    xmin, xmax = rx - rw / 2, rx + rw / 2 
    bbox = np.array([ymin, xmin, ymax, xmax])
    #logger.info('bbox: %s', bbox.shape) #bbox: (4, 17, 24, 24)
    bbox = bbox.transpose(1, 2, 3, 0)
    #logger.info('tr bbox: %s', bbox.shape)  #tr bbox: (17, 24, 24, 4)
    root_bbox = bbox[ROOT_NODE]
    score = delta[ROOT_NODE]
    #logger.info('score: %s', score.shape)
    # Find person boxes which better confidence than the threshold
    candidate = np.where(score > detection_thresh)
    #logger.info('candidate: %s', candidate)
    #logger.info('score: %s', score)
    #logger.info('outsize: %s, %s', outW, outH)
        
    score = score[candidate]

    root_bbox = root_bbox[candidate]
    selected = non_maximum_suppression(
        bbox=root_bbox, thresh=0.3, score=score)
    logger.info('selected: %s', selected)
    root_bbox = root_bbox[selected]
    logger.info('detect instance {:.5f}'.format(time.time() - start))

    start = time.time()

    humans = []

    #logger.info('delta shape: %s', delta.shape)
#    logger.info('x shape: %s', x.shape)
#    logger.info('y shape: %s', y.shape)
#    logger.info('w shape: %s', w.shape)
#    logger.info('h shape: %s', h.shape)
#    logger.info('e shape: %s', e.shape)
    e = e.transpose(0, 3, 4, 1, 2)
    #logger.info('e shape: %s', e.shape)
    ei = 0  # index of edges which contains ROOT_NODE as begin
    # alchemy_on_humans
    for hxw in zip(candidate[0][selected], candidate[1][selected]):
        human = {ROOT_NODE: bbox[(ROOT_NODE, hxw[0], hxw[1])]}  # initial
        for graph in DIRECTED_GRAPHS:
            
            eis, ts = graph
            i_h, i_w = hxw
            for ei, t in zip(eis, ts):
                index = (ei, i_h, i_w)  # must be tuple
                u_ind = np.unravel_index(np.argmax(e[index]), e[index].shape)
                logger.info('u_ind: %s', u_ind)
                j_h = i_h + u_ind[0] - (local_grid_size[1] // 2)
                j_w = i_w + u_ind[1] - (local_grid_size[0] // 2)

                if j_h < 0 or j_w < 0 or j_h >= outH or j_w >= outW:
                    break

                logger.info('t: %s, j_h: %s, j_w: %s',t, j_h, j_w)
                logger.info('delta: %s', delta[t, j_h, j_w])
                logger.info('bbox: %s ', bbox[(t, j_h, j_w)])
                if delta[t, j_h, j_w] < detection_thresh:
                    break
                human[t] = bbox[(t, j_h, j_w)]
                logger.info('human[t]: %s', human[t])
                
                i_h, i_w = j_h, j_w
        if min_num_keypoints <= len(human) - 1:
            humans.append(human)
    logger.info('alchemy time {:.5f}'.format(time.time() - start))
    logger.info('num humans = {}'.format(len(humans)))
    if len(humans) >0:
        logger.info("human detected!")
    return humans
# NMS
def non_maximum_suppression(bbox, thresh, score=None, limit=None):
    #logger.info('nms bbox= {}'.format(bbox))

    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selec = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        tl = np.maximum(b[:2], bbox[selec, :2])
        br = np.minimum(b[2:], bbox[selec, 2:])
        area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)

        iou = area / (bbox_area[i] + bbox_area[selec] - area)
        if (iou >= thresh).any():
            continue

        selec[i] = True
        if limit is not None and np.count_nonzero(selec) >= limit:
            break

    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)

def draw_humans(keypoint_names, edges, pil_image, humans, mask=None, visbbox=False, gridOn = False):
    """
    This is what happens when you use alchemy on humans...
    note that image should be PIL object
    """
    start = time.time()
    drawer = ImageDraw.Draw(pil_image)
    for human in humans:
        logger.info("human : %s", human.items())
        
        for k, b in human.items():
            if mask:
                fill = (255, 255, 255) if k == 0 else None
            else:
                fill = None
            ymin, xmin, ymax, xmax = b
            if k == 0:  # human instance
                # adjust size
                t = 1
                xmin = int(xmin * t + xmax * (1 - t))
                xmax = int(xmin * (1 - t) + xmax * t)
                ymin = int(ymin * t + ymax * (1 - t))
                ymax = int(ymin * (1 - t) + ymax * t)
                if mask:
                    resized = mask.resize(((xmax - xmin), (ymax - ymin)))
                    pil_image.paste(resized, (xmin, ymin), mask=resized)
                else:
                    drawer.rectangle(xy=[xmin, ymin, xmax, ymax],
                                     fill=fill,
                                     outline=COLOR_MAP[keypoint_names[k]])
                    drawer.rectangle(xy=[xmin+1, ymin+1, xmax-1, ymax-1],
                                     fill=fill,
                                     outline=COLOR_MAP[keypoint_names[k]])
            else:
                if visbbox:
                    drawer.rectangle(xy=[xmin, ymin, xmax, ymax],
                                     fill=fill,
                                     outline=COLOR_MAP[keypoint_names[k]])
                else:
                    r = 2
                    x = (xmin + xmax) / 2
                    y = (ymin + ymax) / 2
                    #logger.info("%3d : %s, %.2f, %.2f",k, keypoint_names[k], x, y)

                    drawer.ellipse((x - r, y - r, x + r, y + r),
                                   fill=COLOR_MAP[keypoint_names[k]])

        for s, t in edges:
            if s in human and t in human:
                by = (human[s][0] + human[s][2]) / 2
                bx = (human[s][1] + human[s][3]) / 2
                ey = (human[t][0] + human[t][2]) / 2
                ex = (human[t][1] + human[t][3]) / 2

                drawer.line([bx, by, ex, ey],
                            fill=COLOR_MAP[keypoint_names[s]], width=2)

    # Add the grid
    if gridOn:
        y_start = 0
        y_end = pil_image.height
        step_size = int(pil_image.width / outW)

        for x in range(0, pil_image.width, step_size):
            line = ((x, y_start), (x, y_end))
            drawer.line(line, fill=(110,110,110), width=1)

        x_start = 0
        x_end = pil_image.width

        for y in range(0, pil_image.height, step_size):
            line = ((x_start, y), (x_end, y))
            drawer.line(line, fill=(110, 110, 110), width=1)


    logger.info('draw humans {: .5f}'.format(time.time() - start))
    return pil_image

if __name__== '__main__':

    train_set = KeypointsDataset(json_file = args.file, root_dir = args.root_dir,
            transform=transforms.Compose([
                IAA((384,384),'train'),
                ToTensor()
            ]) , draw = True)

      
    #print("insize:", insize)

    collate_fn = functools.partial(custom_collate_fn,
                                insize=insize,
                                outsize = outsize,
                                keypoint_names = KEYPOINT_NAMES ,
                                local_grid_size = local_grid_size,
                                edges = EDGES)

    dataloader = DataLoader(
        train_set, batch_size=1, shuffle=False,
        num_workers=1, collate_fn = collate_fn, pin_memory=True)

    # Drawing Ground Truth 
    for i, (img, delta, max_delta_ij, x, y, w, h, e) in enumerate(dataloader):
        if i>100:
            break

        delta = np.squeeze(delta.numpy(), axis=0)

        x = np.squeeze(x.numpy(), axis=0)
        y = np.squeeze(y.numpy(), axis=0)
        w = np.squeeze(w.numpy(), axis=0)
        h = np.squeeze(h.numpy(), axis=0)
        e = np.squeeze(e.numpy(), axis=0)
        
        humans = get_humans_by_feature(delta, x, y, w, h, e)

        pil_image = Image.fromarray(np.squeeze(img.numpy(), axis=0).astype(np.uint8).transpose(1, 2, 0))

        pil_image = draw_humans(
            keypoint_names=KEYPOINT_NAMES,
            edges=EDGES,
            pil_image=pil_image,
            humans=humans
        )

        pil_image.save('output/gt_test/predict_test_result_'+str(i)+'.png', 'PNG')

        logger.info('file_number {: d}'.format(i))

    #for i in range(len(train_set)):
    #    if i>10:
    #        break
    #    train_set.show_landmarks(train_set[i]['image'], train_set[i]['keypoints'], train_set[i]['bbox'], i) 
    #    test_img = train_set[i]['image'].numpy().astype(np.uint8).transpose(1,2,0)
    #    plt.imshow(test_img)
    #    plt.show()
     
