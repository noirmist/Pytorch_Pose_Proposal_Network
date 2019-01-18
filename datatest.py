from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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
        "lEye",
        "rEye",
        "lEar",
        "rEar",
        "lShoulder",
        "rShoulder",
        "lElbow",
        "rElbow",
        "lWrist",
        "rWrist",
        "lHip",
        "rHip",
        "lKnee",
        "rKnee",
        "lAnkle",
        "rAnkle",
        "thorax",
        "pelvis",
        "neck",
        "top"
]

def show_landmarks(image, keypoints, bbox):
    """Show image with keypoints"""
    print("show_landmarks:", type(image), image.dtype)
    print(image)
    plt.imshow(image)

    # change 0 to nan
    x = keypoints[:,0]
    x[x==0] = np.nan

    y = keypoints[:,1]
    y[y==0] = np.nan
    x1,y1,w,h = bbox
    rect = patches.Rectangle((x1,y1),w,h,linewidth=2,edgecolor='b',facecolor='none')
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
        keypoints = np.array(self.annotations[idx]["keypoints"], dtype='float32').reshape(-1, 5)
        bbox = self.annotations[idx]['bbox']
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
        bbox = [ bbox[0] * diff_w , bbox[1] *diff_h, bbox[2] *diff_w, bbox[3]*diff_h]
        #print('rescale image', landmarks)
        
        return {'image': img, 'keypoints': landmarks, 'bbox': bbox}

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
        #print("ToTensor image dtype:", image.dtype)
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(keypoints),
		'bbox': torch.from_numpy(np.asarray(bbox))}



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
            Rescale((224,224)),
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

# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['keypoints']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1,2,0)))

    #print(landmarks_batch.shape, grid.shape)
    for i in range(batch_size):
        #print(i, ", x:", landmarks_batch[i, :, 0].numpy(), ", y:", landmarks_batch[i, :, 1].numpy())
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['keypoints'].size())

    # observe 4th batch and stop.
    if i_batch == 1:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break

#Load model
#encode data

#apply transformation

#show examples
