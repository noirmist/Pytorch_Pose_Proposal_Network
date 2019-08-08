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

import scipy

#Evaluation library
from evaluateAP import evaluateAP

import eval_helpers
from eval_helpers import Joint

set_printoptions(threshold=maxsize)

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
def get_humans_by_feature(delta, x, y, w, h, e, detection_thresh=0.15, min_num_keypoints=1):
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
    #logger.info('bbox: %s', bbox.shape) #bbox: (4, 18, 24, 24)
    bbox = bbox.transpose(1, 2, 3, 0)
    #logger.info('tr bbox: %s', bbox.shape)  #tr bbox: (18, 24, 24, 4)
    root_bbox = bbox[ROOT_NODE]
    score = delta[ROOT_NODE]
    logger.info('score: %s', score.shape)
    # Find person boxes which better confidence than the threshold
    candidate = np.where(score > detection_thresh)
    logger.info('candidate: %s', candidate)
    #logger.info('score: %s', score)
    logger.info('outsize: %s, %s', outW, outH)
        
    score = score[candidate]

    root_bbox = root_bbox[candidate]
    selected = non_maximum_suppression(
        bbox=root_bbox, thresh=0.3, score=score)
    logger.info('selected: %s', selected)
    root_bbox = root_bbox[selected]
    logger.info('root_bbox: %s', root_bbox)
    
    logger.info('detect instance {:.5f}'.format(time.time() - start))
    start = time.time()

    humans = []
    scores = []
    
#    logger.info('delta shape: %s', delta.shape)
#    logger.info('x shape: %s', x.shape)
#    logger.info('y shape: %s', y.shape)
#    logger.info('w shape: %s', w.shape)
#    logger.info('h shape: %s', h.shape)
#    logger.info('e shape: %s', e.shape)

    e = e.transpose(0, 3, 4, 1, 2)
#    logger.info('e shape: %s', e.shape)
    ei = 0  # index of edges which contains ROOT_NODE as begin
    # alchemy_on_humans
    for hxw in zip(candidate[0][selected], candidate[1][selected]):
        human = {ROOT_NODE: bbox[(ROOT_NODE, hxw[0], hxw[1])]}  # initial
        score = {ROOT_NODE : delta[(ROOT_NODE, hxw[0], hxw[1])]}
        for graph in DIRECTED_GRAPHS:
            
            eis, ts = graph
            i_h, i_w = hxw
            for ei, t in zip(eis, ts):
                index = (ei, i_h, i_w)  # must be tuple
                #logger.info('e[idx] shape: %s', e[index].shape)
                #logger.info('e[idx]: %s', e[index])

                u_ind = np.unravel_index(np.argmax(e[index]), e[index].shape) # max value in e[index]

                logger.info('index: %s',index)
                logger.info('u_ind: %s', u_ind)
                logger.info('local_grid_size: %s', local_grid_size)
                j_h = i_h + u_ind[0] - (local_grid_size[1] // 2)
                j_w = i_w + u_ind[1] - (local_grid_size[0] // 2)
                logger.info('t: %s, j_h: %s, j_w: %s',t, j_h, j_w)

                if j_h < 0 or j_w < 0 or j_h >= outH or j_w >= outW:
                    logger.info('Out of bound: t: %s, j_h: %s, j_w: %s',t, j_h, j_w)
                    break

                #logger.info('delta[%s]: %s',t, delta[t])
                #logger.info('bbox: %s ', bbox[(t, j_h, j_w)])
                if delta[t, j_h, j_w] < detection_thresh:
                    logger.info('cut by threhold: %s', delta[t, j_h, j_w])
                    break
                human[t] = bbox[(t, j_h, j_w)]
                score[t] = delta[(t, j_h, j_w)]

                logger.info('human[t]: %s', human[t])
                
                i_h, i_w = j_h, j_w

        if min_num_keypoints <= len(human) - 1:
            humans.append(human)
            scores.append(score)
    logger.info('alchemy time {:.5f}'.format(time.time() - start))
    logger.info('num humans = {}'.format(len(humans)))

    if len(humans) >0:
        logger.info("human detected!")
    return humans, scores
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



def gauss(x, a, b, c, d=0):
        return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

def color_heatmap(x):

    color = np.zeros((x.shape[0],x.shape[1],3))
    color[:,:,0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    color[:,:,1] = gauss(x, 1, .5, .3)
    color[:,:,2] = gauss(x, 1, .2, .3)
    color[color > 1] = 1
    color = (color*255 ).astype(np.uint8)
    return color

def show_joints(img, pts):
    imshow(img)

    for i in range(pts.size(0)):
        if pts[i, 2] > 0:
            plt.plot(pts[i, 0], pts[i, 1], 'yo')
    plt.axis('off')

def show_sample(inputs, target, result):

    num_joints = target.shape[0]
    height = target.shape[1]
    width = target.shape[2]
  
    inp = inputs.resize((width, height))

    fig, axes = plt.subplots(nrows=4, ncols=5)
    plt.tight_layout()

    ax = axes.ravel()

    #ax[0].imshow(inputs)
    ax[0].imshow(result)

    for p in range(num_joints):
        tgt = np.asarray(inp)*0.5 + color_heatmap(target[p,:,:])*0.5
        ax[p+1].imshow(tgt.astype(np.uint8))
        ax[p+1].set_title(str(KEYPOINT_NAMES[p]))

    fig.set_size_inches(18, 9)

    return fig


def evaluation(list_):
    # file_name
    fname_list = list_[0]

    # gt_key points
    gt_kps_list = list_[1]

    # humans
    humans_list = list_[2]

    # scores
    scores_list = list_[3]

    # gt bboxes
    gt_bboxes_list = list_[4]

    # is_visible
    is_visible_list = list_[5]

    # size
    size_list = list_[6]

    # PCKh metric
    if False:
        # prediction bboxes list
        pred_bboxs_list = []

        # person[0] = bboxes head size ? 
        # predicted keypoint to gt kps shape
        pred_kps_list = []
        for humans in humans_list:
            pred_bboxs = []
            pred_kps = []
            # humans maybe have several person
            for person in humans:
                item_pred = []
                pred_bboxs.append(person[0]) # head 
                for num in range(1,len(KEYPOINT_NAMES)): # others
                    if num in person:
                        y = (person[num][0] + person[num][2]) / 2
                        x = (person[num][1] + person[num][3]) / 2
                        item_pred.append([y, x])
                    else:
                        item_pred.append([0,0])
                pred_kps.append(item_pred) 
            pred_kps_list.append(pred_kps)
            pred_bboxs_list.append(pred_bboxs)


        #assert dataset_type == 'mpii', 'only work on mpii'
        factor = 0.5 * 0.6

        # pred_boxs_list and gt_boxs_list's center point
        pred_cp_list = []
        gt_cp_list = []
        for item in pred_bboxs_list:
            pred_cp = []
            for per_people in item:
                ymin, xmin, ymax, xmax = per_people
                cp_x = xmin + (xmax-xmin)/2
                cp_y = ymin + (ymax-ymin)/2
                pred_cp.append((cp_x,cp_y))
            pred_cp_list.append(pred_cp)

        for item in gt_bboxes_list:
            gt_cp = []
            for per_people in item:
                cp_x, cp_y, _, _ = per_people
                gt_cp.append((cp_x,cp_y))
            gt_cp_list.append(gt_cp)

        results = {}

        for k in range(len(gt_cp_list)):
            pred_cp, gt_cp, pred_kps, gt_kps, gt_boxs,is_visible = pred_cp_list[k], gt_cp_list[k], pred_kps_list[k], gt_kps_list[k], gt_bboxes_list[k], is_visible_list[k]

            # find out a matched person if image have several person
            for p_index in range(len(pred_cp)):
                p_cp = pred_cp[p_index]

                # distance check
                dists = [np.linalg.norm(np.array(p_cp) - np.array(g_cp)) for g_cp in gt_cp]
                if dists == []:
                    continue

                # find closest one
                g_index = dists.index(min(dists))

                # select matched points
                pred_kps_item = pred_kps[p_index]
                gt_kps_item = gt_kps[g_index]
                is_visible_item = is_visible[g_index]

                # measure diagonal from head bounding boxes
                gt_box = gt_boxs[g_index]
                h,w = gt_box[2:]
                length = np.sqrt((pow(h,2) + pow(w, 2))) * factor

                # check pred kpt is in target(gt)
                for i, name in enumerate(name_list):

                    if not results.get(name):
                        results[name] = []

                    is_v = is_visible_item[i]

                    # ignore thorax, pelvis, neck, top ,stomach and invisible keypoints 
                    if is_v == 0 and i>=12 :
                        continue

                    pred_point = pred_kps_item[i]
                    gt_point = gt_kps_item[i]
                    # compute pred gt point distance
                    distance_pt = np.linalg.norm(np.array(pred_point)-np.array(gt_point))


                    if distance_pt < length:
                        results[name].append(1)
                    else:
                        results[name].append(0)

                # detect single person in a image
                break

        # accuracy
        total_results = []
        head = []
        shoulder = []
        ankle = []
        elbow = []
        wrist = []
        hip = []
        knee = []
        for item in name_list+['instance']:
            total_results += results[item]
            if 'knee' in item:
                knee += results[item]
            elif 'shoulder' in item:
                shoulder += results[name]
            elif 'ankle' in item:
                ankle += results[item]
            elif 'elbow' in item:
                elbow += results[item]
            elif 'wrist' in item:
                wrist += results[item]
            elif 'hip' in item:
                hip += results[item]
            else:
                head += results[item]


        print(len(head), len(hip))
        pck = np.sum(total_results)/len(total_results)
        pck_head = np.sum(head)/len(head)
        pck_shoulder = np.sum(shoulder)/len(shoulder)
        pck_ankle = np.sum(ankle)/len(ankle)
        pck_elbow = np.sum(elbow)/len(elbow)
        pck_wrist = np.sum(wrist)/len(wrist)
        pck_hip = np.sum(hip)/len(hip)
        pck_knee = np.sum(knee)/len(knee)

        pck, pck_head, pck_shoulder, pck_ankle, pck_elbow, pck_wrist, pck_hip, pck_knee = round(pck,2), round(pck_head,2), round(pck_shoulder, 2), round(pck_ankle, 2), round(pck_elbow, 2), round(pck_wrist, 2), round(pck_hip, 2), round(pck_knee, 2)

        logger.info('the total pck is: {}'.format(pck))
        logger.info('head: {}\tshoulder: {}\tankle: {}\telbow: {}\twrist: {}\thip: {}\n\n'.format(pck_head, pck_shoulder, pck_ankle, pck_elbow, pck_wrist, pck_hip))

    # Mean Average Precision
    
    # Make dictionary
    gtFramesAll = []
    prFramesAll = []
    
    for fname, humans, scores, gt_kps, gt_bboxes in zip(fname_list, humans_list, scores_list, gt_kps_list, gt_bboxes_list) :

        pred_list = {"annolist": []}
        gt_list = {"annolist": []}

        pred_data = {"image":[], "annorect":[]}
        gt_data= {"image":[], "annorect":[]}

        pred_data["image"].append(fname)
        gt_data["image"].append(fname)

        print("empty person check:", humans)
        sys.stdout.flush() 

        for person , score in zip(humans, scores):
            # head keypoints
            y1, x1, y2, x2 = person[0]

            pp = {'x1':[x1], 'y1':[y1], 'x2':[x2], 'y2':[y2], 'score':[score[0]],'annopoints':[{"point":[]}]}
            for num in range(1,len(KEYPOINT_NAMES)): # others
                if num in person:
                    y = (person[num][0] + person[num][2]) / 2
                    x = (person[num][1] + person[num][3]) / 2
                    s = score[num]

                else:
                    y, x, s = 0, 0, 0

                p = {'id':[num-1], 'x':[x], 'y': [y], 'score':[s]}
                pp['annopoints'][0]['point'].append(p)

        pred_data['annorect'].append(pp)


        print("# gt bboxes:", len(gt_bboxes)) 
        print("# gt keypoints:", len(gt_kps)) 
        sys.stdout.flush() 
        for person , keypoints in zip(gt_bboxes, gt_kps):

            print("person:", person)
            sys.stdout.flush() 

            cx, cy, w, h = person
            x1 = cx - w//2
            x2 = cx + w//2
            y1 = cy - h//2
            y2 = cy + h//2

            gtperson = {'x1':[x1], 'y1':[y1], 'x2':[x2], 'y2':[y2], 'annopoints':[{"point":[]}]}

            for idx, pt in enumerate(keypoints):
                gtpt = {'id':[idx], 'x':[pt[0]], 'y': [pt[1]]}

            gtperson['annopoints'][0]['point'].append(gtpt)

        gt_data['annorect'].append(gtperson)

        pred_list['annolist'].append(pred_data)
        gt_list['annolist'].append(gt_data)

        gtFramesAll += gt_list['annolist']
        prFramesAll += pred_list['annolist']

    # Poseeval code
    gtFramesAll, prFramesAll = eval_helpers.load_data(gtFramesAll, prFramesAll)

    print("# gt frames  :", len(gtFramesAll))
    print("# pred frames:", len(prFramesAll))

    #####################################################
    # evaluate per-frame multi-person pose estimation (AP)

    # compute AP
    print("Evaluation of per-frame multi-person pose estimation")
    apAll,preAll,recAll = evaluateAP(gtFramesAll,prFramesAll, True)

    # print AP
    print("Average Precision (AP) metric:")
    eval_helpers.printTable(apAll)




if __name__== '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--file", help="json file path")
    parser.add_argument("-img", "--root_dir", help="path to train2017 or val2018")

    args = parser.parse_args()
    root_dir = args.root_dir


    train_set = KeypointsDataset(json_file = args.file, root_dir = args.root_dir,
            transform=transforms.Compose([
                IAA((384,384),'val'),
                ToNormalizedTensor()
            ]),
            draw = False,
            insize = insize,
            outsize = outsize, 
            keypoint_names = KEYPOINT_NAMES ,
            local_grid_size = local_grid_size,
            edges = EDGES
            )

    dataloader = DataLoader(
        train_set, batch_size=1, shuffle=False,
        num_workers=1, collate_fn = custom_collate_fn, sampler=None)

    # Drtawing Ground Truth 
    #for i, (img, delta, weight, weight_ij, tx_half, ty_half, x, y, w, h, e) in enumerate(dataloader):
    mean = torch.tensor([0.485 , 0.456 , 0.406 ]).view(1,3,1,1)
    std = torch.tensor([0.229 , 0.224 , 0.225 ]).view(1,3,1,1)

    for i, samples in enumerate(dataloader):

        delta = np.squeeze(samples.delta.numpy(), axis=0)
        x = np.squeeze(samples.tx.numpy(), axis=0)
        y = np.squeeze(samples.ty.numpy(), axis=0)
        w = np.squeeze(samples.tw.numpy(), axis=0)
        h = np.squeeze(samples.th.numpy(), axis=0)
        e = np.squeeze(samples.te.numpy(), axis=0)


        humans, scores = get_humans_by_feature(delta, x, y, w, h, e)

        raw_img = samples.image.cpu().mul_(std).add_(mean)


        raw_pil_image = Image.fromarray(np.squeeze(raw_img.numpy(), axis=0).astype(np.uint8).transpose(1, 2, 0))

        pil_image = draw_humans(
            keypoint_names=KEYPOINT_NAMES,
            edges=EDGES,
            pil_image=raw_pil_image,
            humans=humans,
            visbbox = False,
            gridOn = True
        )

        sample_fig = show_sample(raw_pil_image, delta, pil_image)
        
        pil_image.save('output/refine_data_test_0708/predict_test_result_'+str(i)+'.png', 'PNG')

        sample_fig.savefig('output/refine_data_test_0708/predict_test_resp_result_'+str(i)+'.png')
        plt.close()
        logger.info('file_number {: d}'.format(i))

    #for i in range(len(train_set)):
    #    if i>10:
    #        break
    #    train_set.show_landmarks(train_set[i]['image'], train_set[i]['keypoints'], train_set[i]['bbox'], i) 
    #    test_img = train_set[i]['image'].numpy().astype(np.uint8).transpose(1,2,0)
    #    plt.imshow(test_img)
    #    plt.show()
     
