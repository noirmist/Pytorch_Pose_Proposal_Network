import json, os

from skimage import io, transform
from skimage.color import gray2rgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from config import *

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
        # filename => keypoints, bbox, is_visible, is_labeled
        self.data = {}

        self.filename_list = np.unique([anno['file_name'] for anno in self.annotations]).tolist()

        for filename in np.unique([anno['file_name'] for anno in self.annotations]):
            self.data[filename] = [], [], [], []

        min_num_keypoints = 1
        for anno in self.annotations:
            is_visible = anno['is_visible']
            if sum(is_visible) < min_num_keypoints:
                continue
            #keypoints = [anno['joint_pos'][k][::-1] for k in KEYPOINT_NAMES[1:]]
            #keypoints = [anno['keypoints']]

            entry = self.data[anno['file_name']]
            entry[0].append(np.array(anno['keypoints']))  # array of x,y
            entry[1].append(np.array(anno['bbox']))  # cx, cy, w, h
            entry[2].append(np.array(is_visible, dtype=np.bool))
            entry[3].append(anno['size'])
            #is_labeled = np.ones(len(is_visible), dtype=np.bool)
            #entry[3].append(is_labeled)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname = self.filename_list[idx]
        img_name = os.path.join(self.root_dir, fname)
        #image = gray2rgb(io.imread(img_name).astype(np.uint8))
        image = gray2rgb(io.imread(img_name))
        
        # center_x, center_y, visible_coco, width, height
        keypoints = np.array(self.data[fname][0], dtype='float32').reshape(-1,2)
        bboxes = self.data[fname][1]    # [center_x, center_y , width, height]
        is_visible = self.data[fname][2]
        size = self.data[fname][3]

        sample = {'image': image, 'keypoints': keypoints, 'bbox': bboxes, 'is_visible':is_visible, 'size': size}

        #print("original_keypoints:", sample['keypoints'])
        if self.transform:
            sample = self.transform(sample)
        #To Tensor
        #print("test file name:", self.filename_list[idx])
        #print("image_shape", sample['image'].shape)
        #print("bbox:", sample['bbox'].shape)
        #, len(bboxes), len(sample['bbox']), sample['bbox'])
        #print("size:",len(size), "is_vis:", len(is_visible))
        #print("old keypoints:", sample['keypoints'].shape)
        sample['keypoints'] = sample['keypoints'].reshape(-1,21,2) 
        #print("new keypoints:", sample['keypoints'].shape)
        #print(sample['keypoints'])

        #show_landmarks(sample['image'], sample['keypoints'], sample['bbox'], fname)
        return sample


def custom_collate_fn(datas,
                    insize=(384,384),
                    outsize=(12,12),
                    keypoint_names = KEYPOINT_NAMES ,
                    local_grid_size= (9,9),
                    edges = EDGES
                    ):

    inW, inH = insize
    outW, outH = outsize
    gridsize = (int(inW / outW), int(inH / outH))
    gridW, gridH = gridsize

    images = []
    deltas = []
    max_deltas_ij = []
    txs = []
    tys = []
    tws = []
    ths = []
    tes = []
    
    for data in datas:
        image = data['image']

        keypoints = data['keypoints']
        bbox = data['bbox']
        is_visible = data['is_visible']
        size = data['size']

        K = len(keypoint_names)

        delta = np.zeros((K, outH, outW), dtype=np.float32)
        tx = np.zeros((K, outH, outW), dtype=np.float32)
        ty = np.zeros((K, outH, outW), dtype=np.float32)
        tw = np.zeros((K, outH, outW), dtype=np.float32)
        th = np.zeros((K, outH, outW), dtype=np.float32)
        te = np.zeros((
            len(edges),
            local_grid_size[1], local_grid_size[0],
            outH, outW), dtype=np.float32)

        # Set delta^i_k
        # points(x,y)

        for (cx, cy, w, h), points, labeled, parts in zip(bbox, keypoints, is_visible, size):
            partsW, partsH = parts, parts
            instanceW, instanceH = w, h

            points = [[cx, cy]] + list(points)

            if w > 0 and h > 0:
                labeled = [True] + list(labeled)
            else:
                labeled = [False] + list(labeled)

            for k, (xy, l) in enumerate(zip(points, labeled)):
                if not l:
                    continue
                cx = xy[0] / gridW
                cy = xy[1] / gridH

                ix, iy = int(cx), int(cy)
                sizeW = instanceW if k == 0 else partsW
                sizeH = instanceH if k == 0 else partsH
                if 0 <= iy < outH and 0 <= ix < outW:
                    delta[k, iy, ix] = 1 
                    tx[k, iy, ix] = cx - ix
                    ty[k, iy, ix] = cy - iy
                    tw[k, iy, ix] = sizeW / inW 
                    th[k, iy, ix] = sizeH / inH 

            for ei, (s, t) in enumerate(edges):
                if not labeled[s]:
                    continue
                if not labeled[t]:
                    continue
                src_xy = points[s]
                tar_xy = points[t]
                iyx = (int(src_xy[1] / gridH), int(src_xy[0] / gridW))
                jyx = (int(tar_xy[1] / gridH) - iyx[0] + local_grid_size[1] // 2,
                       int(tar_xy[0] / gridW) - iyx[1] + local_grid_size[0] // 2)

                if iyx[0] < 0 or iyx[1] < 0 or iyx[0] >= outH or iyx[1] >= outW:
                    continue
                if jyx[0] < 0 or jyx[1] < 0 or jyx[0] >= local_grid_size[1] or jyx[1] >= local_grid_size[0]:
                    continue

                te[ei, jyx[0], jyx[1], iyx[0], iyx[1]] = 1

        # define max(delta^i_k1, delta^j_k2) which is used for loss_limb
        max_delta_ij = np.zeros((len(edges),
                                outH, outW,
                                local_grid_size[1], local_grid_size[0]), dtype=np.float32)
        for ei,(s,t) in enumerate(edges):
            max_delta_ij[ei][delta[s]!=0]=1
            pad_delta_t=np.pad(delta[t],(local_grid_size[1]//2,local_grid_size[0]//2),'constant')
            # Convolve filter
            for r,c in zip(*np.where(delta[s]==0)):
                rr=r+local_grid_size[1]//2
                cc=c+local_grid_size[0]//2
                max_delta_ij[ei][r,c]=pad_delta_t[
                    rr-local_grid_size[1]//2:rr+local_grid_size[1]//2+1,
                    cc-local_grid_size[0]//2:cc+local_grid_size[1]//2+1,
                ]

        max_delta_ij = max_delta_ij.transpose(0,3,4,1,2)

        # Make Sequence of data 
        images.append(image)
        deltas.append(torch.from_numpy(delta))
        max_deltas_ij.append(torch.from_numpy(max_delta_ij))
        txs.append(torch.from_numpy(tx))
        tys.append(torch.from_numpy(ty))
        tws.append(torch.from_numpy(tw))
        ths.append(torch.from_numpy(th))
        tes.append(torch.from_numpy(te))

    # Stack data 
    image = torch.stack(images,0)
    delta = torch.stack(deltas,0)
    max_delta_ij = torch.stack(max_deltas_ij,0)
    tx = torch.stack(txs,0)
    ty = torch.stack(tys,0)
    tw = torch.stack(tws,0)
    th = torch.stack(ths,0)
    te = torch.stack(tes,0)

    return image, delta, max_delta_ij, tx, ty, tw, th, te



