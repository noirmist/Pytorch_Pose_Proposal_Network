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
        #image, keypoints, bboxes, is_visible, size ,name, people = sample['image'], sample['keypoints'], sample['bbox'], sample['is_visible'], sample['size'], sample['name'], sample['ppl']
        image, keypoints, bboxes, is_visible, size ,name = sample['image'], sample['keypoints'], sample['bbox'], sample['is_visible'], sample['size'], sample['name']

        h, w = image.shape[:2]

        #filter existed keypoints , aka exclude zero value
        kps_coords = []
        kps = []
        #keypoints = keypoints.reshape(-1,2).tolist()
        for temp in keypoints:
            if temp[0] !=0 or temp[1] !=0:
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

#        ppl_box_list = []
#        for bbox in people:
#            ppl_box_list.append(ia.BoundingBox(x1 = bbox[0]-bbox[2]//2,
#                                y1=bbox[1]-bbox[3]//2,
#                                x2=bbox[0]+bbox[2]//2,
#                                y2=bbox[1]+bbox[3]//2))
#
#        pbbs = ia.BoundingBoxesOnImage(ppl_box_list, shape=image.shape)
#
#

        kps_oi = ia.KeypointsOnImage(kps, shape=image.shape)
        if self.mode =='train':
            seq = iaa.Sequential([
                iaa.Affine(
                    rotate=(-40, 40),
                    scale=(0.35, 2.5),
                    #fit_output=True
                ), # random rotate by -40-40deg and scale to 35-250%, affects keypoints
                #iaa.Multiply((0.5, 1.5)), # change brightness, doesn't affect keypoints
                #iaa.Fliplr(0.5),
                #iaa.Flipud(0.5),
                iaa.Crop(
                    px = (int(0.1*random.random()*h), int(0.1*random.random()*w), int(0.1*random.random()*h), int(0.1*random.random()*w)), 
                    #percent=(0, 0.2),
                ),
                iaa.Resize({"height": self.output_size[0], "width": self.output_size[1]})
            ])
        else:
            seq = iaa.Sequential([
                #iaa.Fliplr(0.5),
                iaa.Scale({"height": self.output_size[0], "width": self.output_size[1]})
            ])

        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_images([image])[0]
        keypoints_aug = seq_det.augment_keypoints([kps_oi])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
        #bbs_aug = bbs_aug[0].remove_out_of_image().clip_out_of_image() 
        #pbbs_aug = seq_det.augment_bounding_boxes([pbbs])[0]

        # update keypoints and bbox
        cnt = 0
        for ck, temp in enumerate(keypoints):
            #print(ck, "-th", temp)
            if temp[0] != 0 or temp[1] != 0:
                # ignore outside keypoints
                #print("list index error:", temp, cnt, len(keypoints_aug.keypoints))
                #sys.stdout.flush()
                if keypoints_aug.keypoints[cnt].x >=0 and keypoints_aug.keypoints[cnt].x < image_aug.shape[1] and \
                    keypoints_aug.keypoints[cnt].y >=0 and keypoints_aug.keypoints[cnt].y < image_aug.shape[0]:
                    temp[0] = keypoints_aug.keypoints[cnt].x
                    temp[1] = keypoints_aug.keypoints[cnt].y
                else:
                    temp[0] = 0.0 
                    temp[1] = 0.0 
                #print(temp, cnt, len(keypoints_aug.keypoints))
                #sys.stdout.flush()
                cnt +=1 
#        print(name, "updated keypoint:", keypoints)
#        sys.stdout.flush()

        keypoints = np.asarray(keypoints, dtype= np.float32)
        # Delete empty keypoints
        keypoints = list(keypoints.reshape(-1,len(name_list),2))

        blacklist = []
        #for idx, (temp, tempbb, temp_pbb, tempvis) in enumerate(zip(keypoints, bbs_aug.bounding_boxes, pbbs_aug.bounding_boxes, is_visible)):
        for idx, (temp, tempbb, tempvis) in enumerate(zip(keypoints, bbs_aug.bounding_boxes, is_visible)):

            if tempbb.x1 < 0.0:
                tempbb.x1 = 0.0
            elif tempbb.x1 > image_aug.shape[1]:
                tempbb.x1 = image_aug.shape[1]

            if tempbb.y1 < 0.0:
                tempbb.y1 = 0.0
            elif tempbb.y1 > image_aug.shape[0]:
                tempbb.y1 = image_aug.shape[0]

            if tempbb.x2 < 0.0:
                tempbb.x2 = 0.0
            elif tempbb.x2 > image_aug.shape[1]:
                tempbb.x2 = image_aug.shape[1]

            if tempbb.y2 < 0.0:
                tempbb.y2 = 0.0
            elif tempbb.y2 > image_aug.shape[0]:
                tempbb.y2 = image_aug.shape[0]

#            if temp_pbb.x1 < 0.0:
#                temp_pbb.x1 = 0.0
#            elif temp_pbb.x1 > image_aug.shape[1]:
#                temp_pbb.x1 = image_aug.shape[1]
#
#            if temp_pbb.y1 < 0.0:
#                temp_pbb.y1 = 0.0
#            elif temp_pbb.y1 > image_aug.shape[0]:
#                temp_pbb.y1 = image_aug.shape[0]
#
#            if temp_pbb.x2 < 0.0:
#                temp_pbb.x2 = 0.0
#            elif temp_pbb.x2 > image_aug.shape[1]:
#                temp_pbb.x2 = image_aug.shape[1]
#
#            if temp_pbb.y2 < 0.0:
#                temp_pbb.y2 = 0.0
#            elif temp_pbb.y2 > image_aug.shape[0]:
#                temp_pbb.y2 = image_aug.shape[0]

            if (np.unique(temp) == 0).all():
                blacklist.append(idx)
            
            # Update keypoint visibility
            for jdx, (yx, vis) in enumerate(zip(list(temp), tempvis)):
                if (np.unique(yx) == 0).all():
                    tempvis[jdx] = False

#                print("check vis:", yx, vis)
#                sys.stdout.flush()

#            print(name, idx, "-th keypoints:", temp, "\n",idx, "-th aug bbox:", tempbb, "\n",idx, "-th fixed vis:", tempvis )
#            sys.stdout.flush()
#           
#        print("blacklist:", blacklist)
#        sys.stdout.flush()

        for i in sorted(blacklist, reverse=True):
            del keypoints[i]
            del bbs_aug.bounding_boxes[i]
            #del pbbs_aug.bounding_boxes[i]
            del is_visible[i]

        if len(keypoints) == 0:
            keypoints.append(np.zeros((len(name_list),2) ))
            
        keypoints = np.asarray(keypoints, dtype= np.float32)
        
#        for idx2, (temp, tempbb, tempvis) in enumerate(zip(keypoints, bbs_aug.bounding_boxes, is_visible)):
#            print(name, "refined", idx2, "-th keypoints:", temp, "\n",idx2, "-th aug bbox:", tempbb, "\n" , idx2,'-th visible', tempvis )
#            sys.stdout.flush()
#
#        print(name, "total boxes:", bbs_aug.bounding_boxes)
#        sys.stdout.flush()

        # bbox
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
#        else:
#            new_bbox = torch.zeros(1,4)
#            is_visible.append(np.zeros((20), dtype=bool))

        # People
#        new_people_bboxes = []
#        if len(pbbs_aug.bounding_boxes) > 0:
#            for i in range(len(pbbs_aug.bounding_boxes)):
#                new_people_bbox = []
#                temp = pbbs_aug.bounding_boxes[i]
#                new_people_bbox.append((temp.x2+temp.x1)/2)    #center x
#                new_people_bbox.append((temp.y2+temp.y1)/2)    #center y
#                new_people_bbox.append((temp.x2-temp.x1))      #width
#                new_people_bbox.append((temp.y2-temp.y1))      #height
#                new_people_bboxes.append(new_people_bbox)

#        else:
#            new_people_bbox = torch.zeros(1,4)
#            is_visible.append(np.zeros((20), dtype=bool))

       
        #print("total new boxes:", new_bboxes)
        #sys.stdout.flush()
        #logger.info('keypoints : %s', sample['keypoints'])
        #logger.info('bbox : %s', new_bboxes)

        #return {'image': image_aug, 'keypoints': keypoints, 'bbox': new_bboxes, 'ppl': new_people_bboxes, 'is_visible':is_visible, 'size': size}
        return {'image': image_aug, 'keypoints': keypoints, 'bbox': new_bboxes, 'is_visible':is_visible, 'size': size}


class ToNormalizedTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __call__(self, sample):
        #image, keypoints, bbox, ppl= sample['image'], sample['keypoints'], sample['bbox'], sample['ppl']
        image, keypoints, bbox = sample['image'], sample['keypoints'], sample['bbox']

        # swap color axis because
        # PIL  image 
        # numpy image: H x W x C
        # torch image: C X H X W
        #print("totensor img shape:", image.shape)
        #sys.stdout.flush()

        image = np.array(image).transpose((2, 0, 1))
        image = torch.from_numpy(image)
        # Normalize 
        image = image.float()
        image = image.sub_(self.mean).div_(self.std)
        #'ppl': torch.from_numpy(np.asarray(ppl)),
        return {'image': image,
                'keypoints': torch.from_numpy(keypoints),
                'bbox': torch.from_numpy(np.asarray(bbox)),
                'size': sample['size'],
                'is_visible': sample['is_visible']
                }


