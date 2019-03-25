import sys
import random
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import torch
from config import *

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
                    #fit_output=True
                ), # random rotate by -40-40deg and scale to 35-250%, affects keypoints
                iaa.Multiply((0.5, 1.5)), # change brightness, doesn't affect keypoints
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Crop(
                    px = (int(0.1*random.random()*h), int(0.1*random.random()*w), int(0.1*random.random()*h), int(0.1*random.random()*w)), 
                    #percent=(0, 0.2),
                ),
                iaa.Resize({"height": self.output_size[0], "width": self.output_size[1]})
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
        #bbs_aug = bbs_aug[0].remove_out_of_image().clip_out_of_image() 

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
        # Delete empty keypoints
        keypoints = list(keypoints.reshape(-1,len(name_list),2))
        blacklist = []
        for idx, (temp, tempbb, tempvis) in enumerate(zip(keypoints, bbs_aug.bounding_boxes, is_visible)):

            if tempbb.x1 < 0.0:
                tempbb.x1 = 0.0
            elif tempbb.x1 > image_aug.shape[1]:
                tempbb.x1 = image_aug.shape[1]-1

            if tempbb.y1 < 0.0:
                tempbb.y1 = 0.0
            elif tempbb.y1 > image_aug.shape[0]:
                tempbb.y1 = image_aug.shape[0]-1

            if tempbb.x2 < 0.0:
                tempbb.x2 = 0.0
            elif tempbb.x2 > image_aug.shape[1]:
                tempbb.x2 = image_aug.shape[1]-1

            if tempbb.y2 < 0.0:
                tempbb.y2 = 0.0
            elif tempbb.y2 > image_aug.shape[0]:
                tempbb.y2 = image_aug.shape[0]-1

            if (np.unique(temp) == 0).all():
                blacklist.append(idx)
            
            # Update keypoint visibility
            for jdx, (yx, vis) in enumerate(zip(list(temp), tempvis)):
                if (np.unique(yx) == 0).all():
                    tempvis[jdx] = False
#                print("check vis:", yx, vis)
#                sys.stdout.flush()

#            print(idx, "-th keypoints:", temp, "\n",idx, "-th aug bbox:", tempbb, "\n",idx, "-th fixed vis:", tempvis )
#            sys.stdout.flush()
            
#        print("blacklist:", blacklist)
#        sys.stdout.flush()
        for i in sorted(blacklist, reverse=True):
            del keypoints[i]
            del bbs_aug.bounding_boxes[i]
            del is_visible[i]

        keypoints = np.asarray(keypoints, dtype= np.float32)

#        for idx2, (temp, tempbb, tempvis) in enumerate(zip(keypoints, bbs_aug.bounding_boxes, is_visible)):
#            print("refined", idx2, "-th keypoints:", temp, "\n",idx2, "-th aug bbox:", tempbb, "\n" , idx2,'-th visible', tempvis )
#            sys.stdout.flush()
#
#        print("total boxes:", bbs_aug.bounding_boxes)
#        sys.stdout.flush()

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

#        print("total new boxes:", new_bboxes)
#        sys.stdout.flush()
        #sample['keypoints'][:,[0,1]] = keypoints 
        #logger.info('keypoints : %s', sample['keypoints'])
        #logger.info('bbox : %s', new_bboxes)

        return {'image': image_aug, 'keypoints': keypoints, 'bbox': new_bboxes, 'is_visible':is_visible, 'size': size}


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


