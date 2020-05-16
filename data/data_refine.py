import json
import os, sys
import argparse
import math
import cv2 
import numpy as np
import random
import imgaug as ia
from imgaug import augmenters as iaa 

from skimage import io, transform
from skimage.color import gray2rgb
from skimage.util import img_as_ubyte

import pandas as pd
from PIL import ImageDraw, Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import *

parser = argparse.ArgumentParser()
parser.add_argument("-mpi", "--mpifile", help="json file path")

parser.add_argument("-mpiimg", "--mpi_image_folder", help="path to train2017 or val2018")
parser.add_argument("-o", "--output_file", help="output file name")
parser.add_argument("-val_o", "--val_file", help="output val file name")


args = parser.parse_args()
json_file = args.mpifile
root_dir = args.mpi_image_folder
output_file = args.output_file
val_file = args.val_file


# Visutalization
def show_landmarks(img, keypoints, bboxes, people, fname):
    """Show image with keypoints"""

    pil_image = Image.fromarray(img_as_ubyte(img))
    print(pil_image, img.dtype)
    
    drawer = ImageDraw.Draw(pil_image)

    keypoints = list(keypoints)
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

    for (cx1,cy1,w,h) in people:
        k = len(KEYPOINT_NAMES) - 1
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

    pil_image.save("./GT/"+fname)

d = {"annotations":[]}

# Aggregate annoatation
json_data = open(json_file)
annotations = json.load(json_data)["annotations"]

# filename => keypoints, bbox, is_visible,
data = {}

filename_list = np.unique([anno['file_name'] for anno in annotations]).tolist()

for filename in np.unique([anno['file_name'] for anno in annotations]):
    data[filename] = [], [], [], [], []

min_num_keypoints = 1 
for anno in annotations:
    is_visible = anno['is_visible']
    if sum(is_visible) < min_num_keypoints:
        continue
    entry = data[anno['file_name']]
    entry[0].append(np.array(anno['keypoints']))  # array of x,y
    entry[1].append(np.array(anno['bbox']))  # cx, cy, w, h for head
    entry[2].append(np.array(is_visible, dtype=np.bool))
    entry[3].append(anno['size'])
    entry[4].append(np.array(anno['person']))  # cx, cy, w, h for person

for data_idx in range(len(data)):
    # Load image
    fname = filename_list[data_idx]
    img_name = os.path.join(root_dir, fname)
    image = io.imread(img_name)

    # center_x, center_y, visible, width, height
    keypoints = np.array(data[fname][0], dtype='float32').reshape(-1,16,2)
    bboxes = data[fname][1]    # [center_x, center_y , width, height] for head
    is_visible = data[fname][2]
    size = data[fname][3]
    people = data[fname][4]    # [center_x, center_y , width, height] for person instance

    keypoints = keypoints.tolist()

    # Sorting instance values (from left to right)
    people, bboxes, keypoints, is_visible, size = zip(*sorted(zip(people, bboxes, keypoints, is_visible, size), key=lambda x:x[0][0]))

    keypoints = np.array(keypoints).reshape(-1,16,2)

    # for each people images
    for pidx, (pcx, pcy, pw, ph) in enumerate(people):
        save_name = str(data_idx)
        # height :ph 
        if ph> pw:
            s = 200/ph
        else:
            s = 200/pw

        # Scale image
        image_resized = transform.resize(image, (int(image.shape[0]*s), int(image.shape[1]*s)), anti_aliasing=True)

        scaled_bboxes = []
        scaled_people = []
        scaled_keypoints = []
        scaled_size = []
        # Scale keypoints
        for idx, tmp in enumerate(zip(bboxes, people, keypoints, size)):

            bbox, person, points, parts = map(lambda i: i*s, tmp)
            scaled_bboxes.append(bbox)
            scaled_people.append(person)
            scaled_keypoints.append(points)
            scaled_size.append(parts)

        scaled_keypoints = np.array(scaled_keypoints, dtype='float32').reshape(-1,2)

        # Padding image
        pad_image = np.pad(image_resized.copy(),((192, 192), (192, 192), (0, 0)) , mode='constant', constant_values=0.5)

        # Set annotation center
        crop_center_x , crop_center_y, _, _ = scaled_people[pidx]

        # Crop image 384, 384 from annotation center
        min_x = int(crop_center_x)
        min_y = int(crop_center_y)
        max_x = int(crop_center_x) + 384
        max_y = int(crop_center_y) + 384
      
        img = pad_image[min_y:max_y, min_x:max_x]

        # Pad keypoints, bboxes_xy, people_xy
        scaled_keypoints += 192
        scaled_keypoints[:,0] -= crop_center_x
        scaled_keypoints[:,1] -= crop_center_y

        scaled_people = np.array(scaled_people, dtype='float32')
        scaled_people[:,0:2] += 192
        scaled_people[:,0] -= crop_center_x
        scaled_people[:,1] -= crop_center_y

        scaled_bboxes = np.array(scaled_bboxes, dtype='float32')
        scaled_bboxes[:,0:2] += 192
        scaled_bboxes[:,0] -= crop_center_x
        scaled_bboxes[:,1] -= crop_center_y

        # Cut bounding_boxes outside of image by using imgaug
        box_list = []
        for bbox in scaled_bboxes:
            box_list.append(ia.BoundingBox(x1 = bbox[0]-bbox[2]//2,
                                y1=bbox[1]-bbox[3]//2,
                                x2=bbox[0]+bbox[2]//2,
                                y2=bbox[1]+bbox[3]//2))

        ppl_box_list = []
        for bbox in scaled_people:
            ppl_box_list.append(ia.BoundingBox(x1 = bbox[0]-bbox[2]//2,
                                y1=bbox[1]-bbox[3]//2,
                                x2=bbox[0]+bbox[2]//2,
                                y2=bbox[1]+bbox[3]//2))

        delete_list = []
        # Checking bbox which place outside of image 
        for idx, bb in enumerate(box_list):
            if bb.is_out_of_image(img.shape):
                delete_list.append(idx)

        # Delete bounding_boxes 
        for i in list(reversed(delete_list)):
            box_list.pop(i)
            ppl_box_list.pop(i)

        # Cut bounding boxes
        bbs = ia.BoundingBoxesOnImage(box_list, shape=pad_image[min_y:max_y, min_x:max_x].shape).remove_out_of_image().clip_out_of_image()
        pbbs = ia.BoundingBoxesOnImage(ppl_box_list, shape=pad_image[min_y:max_y, min_x:max_x].shape).remove_out_of_image().clip_out_of_image()

        # bbox
        new_bboxes = []
        if len(bbs.bounding_boxes) > 0:
            for i in range(len(bbs.bounding_boxes)):
                new_bbox = []
                temp = bbs.bounding_boxes[i]
                new_bbox.append((temp.x2+temp.x1)/2)    #center x
                new_bbox.append((temp.y2+temp.y1)/2)    #center y
                new_bbox.append((temp.x2-temp.x1))      #width
                new_bbox.append((temp.y2-temp.y1))      #height
                new_bboxes.append(new_bbox)

        # People
        new_people_bboxes = []
        if len(pbbs.bounding_boxes) > 0:
            for i in range(len(pbbs.bounding_boxes)):
                new_people_bbox = []
                temp = pbbs.bounding_boxes[i]
                new_people_bbox.append((temp.x2+temp.x1)/2)    #center x
                new_people_bbox.append((temp.y2+temp.y1)/2)    #center y
                new_people_bbox.append((temp.x2-temp.x1))      #width
                new_people_bbox.append((temp.y2-temp.y1))      #height
                new_people_bboxes.append(new_people_bbox)

        # Number of keypoints
        skpt_counting_list = []
        # Cut out keypoints outside of image
        skpt = scaled_keypoints.reshape(-1,16,2)
        # Add stomach keypoint 
        skpt = np.append(skpt, ((skpt[:,12,:] + skpt[:,13,:])/2).reshape(-1,1,2), axis=1)
        new_visible = list(is_visible)
        for i, vis in enumerate(new_visible):
            new_visible[i] = np.append(vis, [1], axis=0)

        for skpt_idx, (s, vis) in enumerate(zip(skpt, new_visible)):
            skpt_counting_list.append(17)
            for i in range(17):
                x, y  = s[i,:]
                if x < 0 or x > 384 or y <0 or y>384:
                    s[i,:] = 0.0
                    vis[i] = 0
                    skpt_counting_list[skpt_idx] -= 1

        skpt = np.delete(skpt, delete_list, 0)
        new_visible = np.delete(np.array(new_visible), delete_list, 0)

        for i in list(reversed(delete_list)):
            scaled_size.pop(i)
            skpt_counting_list.pop(i)

        # Save image
        if len(str(pidx)) == 1:
            save_name += "00"+str(pidx)
        elif len(str(pidx)) == 2:
            save_name += "0"+str(pidx)
        else:
            save_name += str(pidx)

        for i in range(9-len(save_name)):
            save_name = '0' + save_name

        save_name += ".png"
        pil_image = Image.fromarray(img_as_ubyte(img))
        pil_image.save("images/"+save_name)

        # save Ground truth with keypoints
        #show_landmarks(pad_image[min_y:max_y, min_x:max_x], skpt.reshape(-1,2), new_bboxes, new_people_bboxes, save_name)

        assert len(new_people_bboxes) == len(new_bboxes) == len(list(skpt)) == len(list(new_visible)) == len(scaled_size)
        # Append keypoints
        for i in range(len(new_bboxes)):
            new_anno= {}
            new_anno["file_name"] = save_name
            new_anno['person'] = new_people_bboxes[i]
            new_anno['bbox'] = new_bboxes[i]
            new_anno['keypoints'] = skpt[i].reshape(-1).tolist()
            new_anno['num_keypoints'] = skpt_counting_list[i]
            new_anno['is_visible'] = new_visible[i].astype(int).tolist()
            new_anno["size"] = scaled_size[i]

            d["annotations"].append(new_anno)

file_name_list = [x["file_name"] for x in d["annotations"]]
file_name_list = list(set(file_name_list))

core_name_list = []
for tmp in file_name_list:
    core_name_list.append(tmp[:-7])

core_name_list = list(set(core_name_list))
print("number of total image:", len(file_name_list))

val_file_list = []
import random
for i in range(int(0.2*len(core_name_list))):
    random.shuffle(core_name_list)
    seed = core_name_list.pop()
    for tmp in file_name_list:
        if tmp[:-7] == seed:
            val_file_list.append(tmp)

print("number of train file:", len(file_name_list)- len(val_file_list))
print("number of val file:", len(val_file_list))

val_d = {"annotations":[]}
train_d = {"annotations":[]}

for temp in d["annotations"]:
    if temp["file_name"] in val_file_list:
        val_d["annotations"].append(temp)
    else:
        train_d["annotations"].append(temp)

with open(output_file, 'w') as f:
    json.dump(train_d, f)

with open(val_file, 'w') as f:
    json.dump(val_d, f)
        
