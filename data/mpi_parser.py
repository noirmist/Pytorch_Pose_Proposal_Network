import json
import argparse
import math
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-mpi", "--mpifile", help="json file path")

parser.add_argument("-mpiimg", "--mpi_image_folder", help="path to train2017 or val2018")
parser.add_argument("-o", "--output_file", help="output file name")
parser.add_argument("-val_o", "--val_file", help="output val file name")

args = parser.parse_args()
mpi_image_folder = args.mpi_image_folder
output_file = args.output_file
val_file = args.val_file

name_list = [ 
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

d = {"annotations":[]}

'''
anno_format

anno = 
{
"area": float
"bbox": [float, float, float, float]
"category_id": int
"id": int
"image_id": int
"file_name": str
"iscrowd": binary
"keypoints": [x,y,v, ..., x,y,v]
"num_keypoints": int
"segmentations":[[x,y]]
}
'''

'''
new_anno_format

new_anno = 
{
"bbox": [float, float, float, float] # center_x, center_y, width, height
"file_name": str
"keypoints": [x,y, ..., x,y]
"is_visible": [v,v,v,v,v, ..., v,v,v]
"size": w # w = h
"num_keypoints": int
}
'''
with open(args.mpifile) as mpi_json_data:
    m = json.load(mpi_json_data)


for idx, anno in enumerate(m):
    print(anno["filename"])
    f_name = anno["filename"]

    new_anno = {}
    new_anno["file_name"] = f_name
    
    head_width = anno["head_rect"][2] - anno["head_rect"][0]
    head_height = anno["head_rect"][3] - anno["head_rect"][1]
    
    head_area = head_width * head_height
    w = h = head_square_edge = math.sqrt(head_area)//2
    new_anno['size']= math.sqrt(head_area)//2
    
    joints = anno["joint_pos"]
    n = len(joints)

    # parse mpi joint_pose to key
    #r_hip, r_elbow, l_wrist, r_knee, thorax, pelvis, head_top, upper_neck, l_ankle, l_hip, r_wrist, r_shoulder, l_shoulder, l_knee, l_elbow

    #lShoulder",
    l_shoulder = joints["l_shoulder"]
    labeled = anno['is_visible']['l_shoulder'] + 1 
    key = np.array([*l_shoulder, labeled, w, h], dtype='float32')
    
    #rShoulder",
    r_shoulder = joints["r_shoulder"]
    labeled = anno['is_visible']['r_shoulder'] + 1 
    key = np.vstack([key, np.array([*r_shoulder, labeled, w, h])])

    #lElbow",
    l_elbow = joints["l_elbow"]
    labeled = anno['is_visible']['l_elbow'] + 1 
    key = np.vstack([key, np.array([*l_elbow, labeled, w, h])])

    #rElbow",
    r_elbow = joints["r_elbow"]
    labeled = anno['is_visible']['r_elbow'] + 1 
    key = np.vstack([key, np.array([*r_elbow, labeled, w, h])])

    #lWrist",
    l_wrist = joints["l_wrist"]
    labeled = anno['is_visible']['l_wrist'] + 1 
    key = np.vstack([key, np.array([*l_wrist, labeled, w, h])])

    #rWrist",
    r_wrist = joints["r_wrist"]
    labeled = anno['is_visible']['r_wrist'] + 1 
    key = np.vstack([key, np.array([*r_wrist, labeled, w, h])])

    #lHip",
    l_hip = joints["l_hip"]
    labeled = anno['is_visible']['l_hip'] + 1 
    key = np.vstack([key, np.array([*l_hip, labeled, w, h])])

    #rHip",
    r_hip = joints["r_hip"]
    labeled = anno['is_visible']['r_hip'] + 1 
    key = np.vstack([key, np.array([*r_hip, labeled, w, h])])

    #lKnee",
    l_knee = joints["l_knee"]
    labeled = anno['is_visible']['l_knee'] + 1 
    key = np.vstack([key, np.array([*l_knee, labeled, w, h])])

    #rKnee",
    r_knee = joints["r_knee"]
    labeled = anno['is_visible']['r_knee'] + 1 
    key = np.vstack([key, np.array([*r_knee, labeled, w, h])])

    #lAnkle",
    l_ankle = joints["l_ankle"]
    labeled = anno['is_visible']['l_ankle'] + 1 
    key = np.vstack([key, np.array([*l_ankle, labeled, w, h])])

    #rAnkle",
    r_ankle = joints["r_ankle"]
    labeled = anno['is_visible']['r_ankle'] + 1 
    key = np.vstack([key, np.array([*r_ankle, labeled, w, h])])

    #thorax",
    thorax = joints["thorax"]
    labeled = anno['is_visible']['thorax'] + 1 
    key = np.vstack([key, np.array([*thorax, labeled, w, h])])
    
    #pelvis",
    pelvis = joints["pelvis"]
    labeled = anno['is_visible']['pelvis'] + 1 
    key = np.vstack([key, np.array([*pelvis, labeled, w, h])])

    #neck",
    neck = joints["upper_neck"]
    labeled = anno['is_visible']['upper_neck'] + 1 
    key = np.vstack([key, np.array([*neck, labeled, w, h])])

    #top
    top = joints["head_top"]
    labeled = anno['is_visible']['head_top'] + 1 
    key = np.vstack([key, np.array([*top, labeled, w, h])])
    
    # clean width height for invisible data
    for k in key:
        if k[2] < 1:
            k[3] = 0.0
            k[4] = 0.0

    new_anno['is_visible'] = key[:,2].reshape(-1).tolist()
    new_anno['keypoints'] = key[:,0:2].reshape(-1).tolist()
    new_anno['size'] = w
    #new_anno['keypoints'] = key.reshape(-1).tolist()

    new_anno['num_keypoints'] = n
    
    x_axis = key.reshape(-1, 5).T[0][5:]
    #print("x:", x_axis)
    y_axis = key.reshape(-1, 5).T[1][5:]
    #print("y:", y_axis)
    center_x = (x_axis.max() + x_axis.min())/2
    center_y = (y_axis.max() + y_axis.min())/2
    width = x_axis.max() - x_axis.min() + 2*w
    height = y_axis.max() - y_axis.min() + 2*h
    new_anno['bbox'] = [center_x, center_y, width, height]
    #new_anno['bbox'] = [x_axis.min()-w/2, y_axis.min()-h/2, x_axis.max()+w/2, y_axis.max()+h/2]
    

#    #drawing test
#    if f_name == '061805081.jpg': #'00000000536.jpg':
#        img = cv2.imread(mpi_image_folder+'/'+f_name)
#        #cv2.rectangle(img, (int(x_axis.min()-int(w/2)), int(y_axis.min())-int(w/2)), (int(x_axis.max()+int(w/2)), int(y_axis.max()+int(w/2)) ), (0,128,255), 2)
#        cv2.rectangle(img, (int(center_x - width/2), int(center_y - height/2)), (int(center_x + width/2), int(center_y + height/2)), (0,128,255), 2)
#        cv2.rectangle(img, (int(anno["head_rect"][0]),int(anno["head_rect"][1])), (int(anno["head_rect"][2]),int(anno["head_rect"][3])), (0,128,255),2)
#
#        for i, k in enumerate(key):
#            #x,y,v,w,h 
#            w = k[3]
#            h = k[4]
#            x1 = k[0] - w//2
#            y1 = k[1] - h//2
#            v = k[2]
#
#            if v == 1:
#                cv2.rectangle(img, (int(x1),int(y1)), (int(x1+w), int(y1+h)), (0,0,255), 2)
#            elif v == 2:
#                cv2.rectangle(img, (int(x1),int(y1)), (int(x1+w), int(y1+h)), (255,0,0), 2)
#        cv2.imshow("img",img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()


    d["annotations"].append(new_anno)

import random
random.shuffle(d["annotations"])
print(len(d["annotations"]))
val_d = {"annotations":d["annotations"][:int(0.2*len(d["annotations"]))+1]}
train_d = {"annotations":d["annotations"][int(0.2*len(d["annotations"]))+1:]}

with open(output_file, 'w') as f:
    json.dump(train_d, f)

with open(val_file, 'w') as f:
    json.dump(val_d, f)
