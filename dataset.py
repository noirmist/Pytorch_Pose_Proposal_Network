import json, os, sys

from skimage import io, transform
from skimage.color import gray2rgb
import numpy as np
import pandas as pd
from PIL import ImageDraw, Image
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from config import *

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class KeypointsDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None, draw=False,
                insize=(384,384),
                outsize=(12,12),
                keypoint_names = KEYPOINT_NAMES ,
                local_grid_size= (9,9),
                edges = EDGES
            ):
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
        self.draw = draw
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

        # for encoding
        self.insize = insize
        self.outsize = outsize
        self.keypoint_names = keypoint_names
        self.local_grid_size = local_grid_size
        self.edges = edges

        self.inW, self.inH = insize
        self.outW, self.outH = outsize
        self.gridW = int(self.inW/ self.outW) 
        self.gridH = int(self.inH/ self.outH) 
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname = self.filename_list[idx]
        img_name = os.path.join(self.root_dir, fname)
        #image = gray2rgb(io.imread(img_name).astype(np.uint8))
        #image = gray2rgb(io.imread(img_name))
        image = io.imread(img_name)
        
        # center_x, center_y, visible_coco, width, height
        keypoints = np.array(self.data[fname][0], dtype='float32').reshape(-1,2)
        bboxes = self.data[fname][1]    # [center_x, center_y , width, height]
        is_visible = self.data[fname][2]
        size = self.data[fname][3]

        sample = {'image': image, 'keypoints': keypoints, 'bbox': bboxes, 'is_visible':is_visible, 'size': size, 'name': img_name }

#        print("test file name:", self.filename_list[idx])
#        sys.stdout.flush()
#        print("old keypoints:", sample['keypoints'].shape)
#        sys.stdout.flush()
          
        #print("original_keypoints:", sample['keypoints'])
        if self.transform:
            sample = self.transform(sample)

        # Draw image
        if self.draw:
            self.show_landmarks(sample['image'], sample['keypoints'], sample['bbox'], fname, idx)

        # Encode samples 
        image = sample['image']
        keypoints = sample['keypoints']
        bbox = sample['bbox']
        is_visible = sample['is_visible']
        size = sample['size']

        K = len(self.keypoint_names)

        delta = np.zeros((K, self.outH, self.outW), dtype=np.float32)
        tx = np.zeros((K, self.outH, self.outW), dtype=np.float32)
        ty = np.zeros((K, self.outH, self.outW), dtype=np.float32)
        tw = np.zeros((K, self.outH, self.outW), dtype=np.float32)
        th = np.zeros((K, self.outH, self.outW), dtype=np.float32)
        te = np.zeros((
            len(self.edges),
            self.local_grid_size[1], self.local_grid_size[0],
            self.outH, self.outW), dtype=np.float32)


        for (cx, cy, w, h), points, labeled, parts in zip(bbox, keypoints, is_visible, size):
            partsW, partsH = parts, parts
            instanceW, instanceH = w, h

            points = [torch.tensor([cx.item(), cy.item()])] + list(points)

            if w > 0 and h > 0:
                labeled = [True] + list(labeled)
            else:
                labeled = [False] + list(labeled)

            for k, (xy, l) in enumerate(zip(points, labeled)):
                if not l:
                    continue
                cx = xy[0] / self.gridW
                cy = xy[1] / self.gridH

                ix, iy = int(cx), int(cy)
                sizeW = instanceW if k == 0 else partsW
                sizeH = instanceH if k == 0 else partsH

                if 0 <= iy < self.outH and 0 <= ix < self.outW:
                    delta[k, iy, ix] = 1
                    tx[k, iy, ix] = cx - ix
                    ty[k, iy, ix] = cy - iy
                    tw[k, iy, ix] = sizeW / self.inW
                    th[k, iy, ix] = sizeH / self.inH

            for ei, (s, t) in enumerate(self.edges):
                if not labeled[s]:
                    continue
                if not labeled[t]:
                    continue
                src_xy = points[s]
                tar_xy = points[t]
                iyx = (int(src_xy[1] / self.gridH), int(src_xy[0] / self.gridW))
                jyx = (int(tar_xy[1] / self.gridH) - iyx[0] + self.local_grid_size[1] // 2,
                       int(tar_xy[0] / self.gridW) - iyx[1] + self.local_grid_size[0] // 2)

                if iyx[0] <= 0 or iyx[1] <= 0 or iyx[0] >= self.outH or iyx[1] >= self.outW:
                    continue
                if jyx[0] <= 0 or jyx[1] <= 0 or jyx[0] >= self.local_grid_size[1] or jyx[1] >= self.local_grid_size[0]:
                    continue

                te[ei, jyx[0], jyx[1], iyx[0], iyx[1]] = 1

        # define max(delta^i_k1, delta^j_k2) which is used for loss_limb
        max_delta_ij = np.zeros((len(self.edges),
                                self.outH, self.outW,
                                self.local_grid_size[1], self.local_grid_size[0]), dtype=np.float32)
        for ei,(s,t) in enumerate(self.edges):
            max_delta_ij[ei][delta[s]!=0]=1.0
            pad_delta_t=np.pad(delta[t],(self.local_grid_size[1]//2,self.local_grid_size[0]//2),'constant')
            # Convolve filter
            for r,c in zip(*np.where(delta[s]==0)):
                rr=r+self.local_grid_size[1]//2
                cc=c+self.local_grid_size[0]//2
                max_delta_ij[ei][r,c]=pad_delta_t[
                    rr-self.local_grid_size[1]//2:rr+self.local_grid_size[1]//2+1,
                    cc-self.local_grid_size[0]//2:cc+self.local_grid_size[1]//2+1,
                ]

        max_delta_ij = max_delta_ij.transpose(0,3,4,1,2)

        # Post Processing
        zero_place = np.zeros(max_delta_ij.shape, dtype=np.float32)
        zero_place[max_delta_ij < 0.5] = 0.0005
        weight_ij = np.minimum(max_delta_ij + zero_place, 1.0)

        # add weight where can't find keypoint
        zero_place = np.zeros(delta.shape).astype(np.float32)
        zero_place[delta < 0.5] = 0.0005
        weight = np.minimum(delta + zero_place, 1.0)

        half = np.zeros(delta.shape).astype(np.float32)
        half[delta < 0.5] = 0.5
        tx_half = tx + half
        ty_half = ty + half

        # Convert numpy to tensor
        delta = torch.from_numpy(delta)        
        weight = torch.from_numpy(weight)        
        weight_ij = torch.from_numpy(weight_ij)
        tx = torch.from_numpy(tx)
        ty = torch.from_numpy(ty)
        tw = torch.from_numpy(tw)
        th = torch.from_numpy(th)
        te = torch.from_numpy(te)
        tx_half = torch.from_numpy(tx_half)
        ty_half = torch.from_numpy(ty_half)

#        deltas.append(torch.from_numpy(delta))
#        weights.append(torch.from_numpy(weight))
#        weights_ij.append(torch.from_numpy(weight_ij))
#        tx_halfs.append(torch.from_numpy(tx_half))
#        ty_halfs.append(torch.from_numpy(ty_half))
#
#        txs.append(torch.from_numpy(tx))
#        tys.append(torch.from_numpy(ty))
#        tws.append(torch.from_numpy(tw))
#        ths.append(torch.from_numpy(th))
#        tes.append(torch.from_numpy(te))

        #sample = {'image':image, 'delta':delta, 'weight':weight, 'weight_ij':weight_ij, 'tx':tx, 'ty':ty, 'tx_half':tx_half, 'ty_half':ty_half, 'tw':tw, 'th':th, 'te':te  }
        #sample = [image, delta, weight, weight_ij, tx, ty, tx_half, ty_half, tw, th, te]
        del sample
        return [image, delta, weight, weight_ij, tx, ty, tx_half, ty_half, tw, th, te]

    # Visutalization
    def show_landmarks(self, img, keypoints, bboxes, fname, idx):
        """Show image with keypoints"""

        pil_image = Image.fromarray(img.numpy().astype(np.uint8).transpose(1, 2, 0))
        drawer = ImageDraw.Draw(pil_image)

        keypoints = list(keypoints.reshape(-1,2).numpy())
        for i, key in enumerate(keypoints):
            r = 2
            x = key[0]
            y = key[1]
            drawer.ellipse((x - r, y - r, x + r, y + r), fill=COLOR_MAP[name_list[i%len(name_list)]])

        for (cx1,cy1,w,h) in bboxes:
            k = 0
            xmin = cx1 - w//2
            ymin = cy1 - h//2
            xmax = cx1 + w//2
            ymax = cy1 + h//2

            drawer.rectangle(xy=[xmin, ymin, xmax, ymax],
                             fill=None,
                             outline=COLOR_MAP[KEYPOINT_NAMES[k]])
            drawer.rectangle(xy=[xmin+1, ymin+1, xmax-1, ymax-1],
                             fill=None,
                             outline=COLOR_MAP[KEYPOINT_NAMES[k]])
        pil_image.save("/media/hci-gpu/hdd/PPN/input_check_0527/"+fname) 

class CustomBatch:
    def __init__(self, datas):
        # [image, delta, weight, weight_ij, tx, ty, tx_half, ty_half, tw, th, te]
        data = list(zip(*datas))

        # Stack data 
        self.image = torch.stack(data[0], 0)
        self.delta = torch.stack(data[1], 0)
        self.weight = torch.stack(data[2], 0)
        self.weight_ij = torch.stack(data[3], 0)
        self.tx = torch.stack(data[4], 0)
        self.ty = torch.stack(data[5], 0)
        self.tx_half = torch.stack(data[6], 0)
        self.ty_half = torch.stack(data[7], 0)
        self.tw = torch.stack(data[8], 0)
        self.th = torch.stack(data[9], 0)
        self.te = torch.stack(data[10], 0)

    #    pront("collate_fn_shape: image", image.shape)
    #    sys.stdout.flush()
    #    print("delta", delta.shape)
    #    print("max_delta_ij", max_delta_ij.shape)
    #    print("max_delta_ij", np.unique(max_delta_ij.numpy()))
    #    print("tx", tx.shape)
    #    print("ty", ty.shape)
    #    print("tw", tw.shape)
    #    print("th", th.shape)
    #    print("te", te.shape)

    def pin_memory(self):
        self.image = self.image.pin_memory()
        self.delta = self.delta.pin_memory()
        self.weight = self.weight.pin_memory()
        self.weight_ij = self.weight_ij.pin_memory()
        self.tx_half = self.tx_half.pin_memory()
        self.ty_half = self.ty_half.pin_memory()
        self.tx = self.tx.pin_memory()
        self.ty = self.ty.pin_memory()
        self.tw = self.tw.pin_memory()
        self.th = self.th.pin_memory()
        self.te = self.te .pin_memory()

        return self

def custom_collate_fn(batch):
    return CustomBatch(batch)
 
