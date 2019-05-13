import json
import argparse
import math
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-coco", "--cocofile", help="json file path")
#parser.add_argument("-mpi", "--mpifile", help="json file path")

parser.add_argument("-cocoimg", "--coco_image_folder", help="path to train2017 or val2018")
#parser.add_argument("-mpiimg", "--mpi_image_folder", help="path to train2017 or val2018")
parser.add_argument("-o", "--output_file", help="output file name")

args = parser.parse_args()
coco_image_folder = args.coco_image_folder
#mpi_image_folder = args.mpi_image_folder
output_file = args.output_file

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
with open(args.cocofile) as coco_json_data:
    c = json.load(coco_json_data)

#with open(args.mpifile) as mpi_json_data:
#    m = json.load(mpi_json_data)

for idx, anno in enumerate(c["annotations"]):

    if anno['iscrowd'] != 0:
        continue
    if anno['num_keypoints'] == 0:
        continue

    new_anno = {}
    #new_anno['bbox'] = anno['bbox'].copy()
    bbox_x = anno['bbox'][0]
    bbox_y = anno['bbox'][1]
    bbox_w = anno['bbox'][2]
    bbox_h = anno['bbox'][3]

    new_anno['bbox'] = [bbox_x+bbox_w/2, bbox_y+bbox_h/2, bbox_w, bbox_h].copy()
    
    image_id = anno["image_id"]
    for j in range(12 - len(str(image_id))):
        if j == 0:
            f_space = str(0)
        else:
            f_space = f_space + str(0)
    f_name = f_space + str(image_id) + ".jpg"
    print(f_name)
    new_anno['file_name'] = f_name
    n = anno['num_keypoints']

    key = np.array(anno['keypoints'], dtype='float32').reshape(-1, 3)

    left_shoulder_idx = name_list.index('lShoulder')
    right_shoulder_idx = name_list.index('rShoulder')

    left_hip_idx = name_list.index('lHip')
    right_hip_idx = name_list.index('rHip')

    neck_idx = name_list.index('neck')
    pelvis_idx = name_list.index('pelvis')
    #thorax_idx = name_list.index('thorax')
    top_idx =  name_list.index('top')
    nose_idx = name_list.index('nose')

    lear_idx = name_list.index('lEar')
    rear_idx = name_list.index('rEar')


    left_shoulder, left_shoulder_v = key[left_shoulder_idx][:2], key[left_shoulder_idx][2]
    right_shoulder, right_shoulder_v = key[right_shoulder_idx][:2], key[right_shoulder_idx][2]

    left_hip, left_hip_v = key[left_hip_idx][:2], key[left_hip_idx][2]
    right_hip, right_hip_v = key[right_hip_idx][:2], key[right_hip_idx][2]

    # pelvis 
    if left_hip_v >= 1 and right_hip_v >=1:
        pelvis = (left_hip + right_hip) / 2.
        labeled = 1
        key = np.vstack([key, np.array([*pelvis, labeled])])
        n += 1
    else:
        labeled = 0
        #dummy data
        key = np.vstack([key, np.array([0.0, 0.0, labeled])])
   
    nose, nose_v = key[nose_idx][:2], key[nose_idx][2]
    
    # neck
    if left_shoulder_v >= 1 and right_shoulder_v >=1:
        neck = (left_shoulder + right_shoulder) / 2.
        labeled = 1
        key = np.vstack([key, np.array([*neck, labeled])])
        n += 1
    else:
        labeled = 0
        #dummy data
        key = np.vstack([key, np.array([0.0, 0.0, labeled])])
    
    lear_v = key[lear_idx][2]
    rear_v = key[rear_idx][2]
    neck, neck_v = key[neck_idx][:2], key[neck_idx][2]

    # top
#    if nose_v >=1 and neck_v>=1 and lear_v >=1 and rear_v >=1:
#        top = nose+(nose-neck)
#        labeled = 1
#        key = np.vstack([key, np.array([*top, labeled])])
#        n += 1
#    else:

    labeled = 0
    #dummy data
    key = np.vstack([key, np.array([0.0, 0.0, labeled])])
    
    w = h = math.sqrt(anno['area']/n)
    new_anno['size'] = math.sqrt(anno['area']/n)

    key = np.hstack([key, (np.ones(len(name_list))*w).reshape(-1,1)])
    key = np.hstack([key, (np.ones(len(name_list))*h).reshape(-1,1)])

    # clear width and height for not labeled keypoints
    for k in key:
        if k[2] < 1:
            k[3] = 0.0
            k[4] = 0.0

#    #drawing test
#    img = cv2.imread(coco_image_folder+'/'+f_name)
#    for i, k in enumerate(key):
#        #x,y,v,w,h 
#        w = k[3]
#        h = k[4]
#        x1 = k[0] - w//2
#        y1 = k[1] - h//2
#        v = k[2]
#
#        if v == 1:
#            cv2.rectangle(img, (int(x1),int(y1)), (int(x1+w), int(y1+h)), (0,0,255), 2)
#        elif v == 2:
#            cv2.rectangle(img, (int(x1),int(y1)), (int(x1+w), int(y1+h)), (255,0,0), 2)
#        elif v==4:
#            cv2.rectangle(img, (int(x1),int(y1)), (int(x1+w), int(y1+h)), (255,0,128), 2)
#
#    cv2.imshow("img",img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
   
    #key = key.reshape(-1).tolist()
    new_anno['is_visible'] = key[:,2].reshape(-1).tolist()
    new_anno['keypoints'] = key[:,0:2].reshape(-1).tolist()

#    new_anno['keypoints'] = key
    new_anno['num_keypoints'] = n

    d['annotations'].append(new_anno)


'''
for idx, anno in enumerate(m):
    print(anno["filename"])
    f_name = anno["filename"]

    new_anno = {}
    new_anno["file_name"] = f_name
    
    head_width = anno["head_rect"][2] - anno["head_rect"][0]
    head_height = anno["head_rect"][3] - anno["head_rect"][1]
    
    head_area = head_width * head_height
    w = h = head_square_edge = math.sqrt(head_area)//2
    
    joints = anno["joint_pos"]
    n = len(joints)

    # parse mpi joint_pose to key
    #r_hip, r_elbow, l_wrist, r_knee, thorax, pelvis, head_top, upper_neck, l_ankle, l_hip, r_wrist, r_shoulder, l_shoulder, l_knee, l_elbow
    
    # nose
    key = np.array([0.0, 0.0, 0, w, h] , dtype='float32')
    #LeYE
    key = np.vstack([key, np.array([0.0, 0.0, 0, w, h])])
    #rEye
    key = np.vstack([key, np.array([0.0, 0.0, 0, w, h])])
    #lEar
    key = np.vstack([key, np.array([0.0, 0.0, 0, w, h])])
    #rEar
    key = np.vstack([key, np.array([0.0, 0.0, 0, w, h])])
    #lShoulder",
    l_shoulder = joints["l_shoulder"]
    labeled = anno['is_visible']['l_shoulder'] + 1 
    key = np.vstack([key, np.array([*l_shoulder, labeled, w, h])])
    
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

    #neck",
    neck = joints["upper_neck"]
    labeled = anno['is_visible']['upper_neck'] + 1 
    key = np.vstack([key, np.array([*neck, labeled, w, h])])

    #pelvis",
    pelvis = joints["pelvis"]
    labeled = anno['is_visible']['pelvis'] + 1 
    key = np.vstack([key, np.array([*pelvis, labeled, w, h])])

    #thorax",
    thorax = joints["thorax"]
    labeled = anno['is_visible']['thorax'] + 1 
    key = np.vstack([key, np.array([*thorax, labeled, w, h])])
    
    #top
    top = joints["head_top"]
    labeled = anno['is_visible']['head_top'] + 1 
    key = np.vstack([key, np.array([*top, labeled, w, h])])
    
    # clean width height for invisible data
    for k in key:
        if k[2] < 1:
            k[3] = 0.0
            k[4] = 0.0

    new_anno['keypoints'] = key.reshape(-1).tolist()

    new_anno['num_keypoints'] = n
    
    x_axis = key.reshape(-1, 5).T[0][5:]
    y_axis = key.reshape(-1, 5).T[1][5:]

    new_anno['bbox'] = [x_axis.min()-w/2, y_axis.min()-w/2, x_axis.max()-w/2, y_axis.max()-w/2]
    

#    #drawing test
#    img = cv2.imread(mpi_image_folder+'/'+f_name)
#    cv2.rectangle(img, (int(x_axis.min()-int(w/2)), int(y_axis.min())-int(w/2)), (int(x_axis.max()+int(w/2)), int(y_axis.max()+int(w/2)) ), (0,128,255), 2)
#    cv2.rectangle(img, (int(anno["head_rect"][0]),int(anno["head_rect"][1])), (int(anno["head_rect"][2]),int(anno["head_rect"][3])), (0,128,255),2)
#
#    for i, k in enumerate(key):
#        #x,y,v,w,h 
#        w = k[3]
#        h = k[4]
#        x1 = k[0] - w//2
#        y1 = k[1] - h//2
#        v = k[2]
#
#        if v == 1:
#            cv2.rectangle(img, (int(x1),int(y1)), (int(x1+w), int(y1+h)), (0,0,255), 2)
#        elif v == 2:
#            cv2.rectangle(img, (int(x1),int(y1)), (int(x1+w), int(y1+h)), (255,0,0), 2)
#    cv2.imshow("img",img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


    d["annotations"].append(new_anno)
'''
with open(output_file, 'w') as f:
    json.dump(d, f)

