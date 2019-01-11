import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import json
import argparse
from PIL import Image
import math
import cv2

#TODO list
#reduce region for image

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="json file path")
parser.add_argument("-img", "--image_folder", help="path to train2017 or val2017")
parser.add_argument("-o", "--output_folder", help="path to output_folder labels")

args = parser.parse_args()
#print args.file
image_folder = args.image_folder
output_folder = args.output_folder

def convert(size, x, y, w, h):
    dw = 1./(size[0])
    dh = 1./(size[1])
#    x = (box[0] + box[1])/2.0 - 1
#    y = (box[2] + box[3])/2.0 - 1
#    w = box[1] - box[0]
#    h = box[3] - box[2]
    x = (x+(w/2.0))*dw
    w = w*dw
    y = (y+(h/2.0))*dh
    h = h*dh
    return (x,y,w,h)

keycounter = {
        "person":0,
        "nose_v":0,
        "nose_nv":0,
        "lEye_v":0,
        "lEye_nv":0,
        "rEye_v":0,
        "rEye_nv":0,
        "lEar_v":0,
        "lEar_nv":0,
        "rEar_v":0,
        "rEar_nv":0,
        "lShoulder_v":0,
        "lShoulder_nv":0,
        "rShoulder_v":0,
        "rShoulder_nv":0,
        "lElbow_v":0,
        "lElbow_nv":0,
        "rElbow_v":0,
        "rElbow_nv":0,
        "lWrist_v":0,
        "lWrist_nv":0,
        "rWrist_v":0,
        "rWrist_nv":0,
        "lHip_v":0,
        "lHip_nv":0,
        "rHip_v":0,
        "rHip_nv":0,
        "lKnee_v":0,
        "lKnee_nv":0,
        "rKnee_v":0,
        "rKnee_nv":0,
        "lAnkle_v":0,
        "lAnkle_nv":0,
        "rAnkle_v":0,
        "rAnkle_nv":0
        }
name_list = [
        "nose_v",
        "lEye_v",
        "rEye_v",
        "lEar_v",
        "rEar_v",
        "lShoulder_v",
        "rShoulder_v",
        "lElbow_v",
        "rElbow_v",
        "lWrist_v",
        "rWrist_v",
        "lHip_v",
        "rHip_v",
        "lKnee_v",
        "rKnee_v",
        "lAnkle_v",
        "rAnkle_v"
]

nv_name_list = [
        "nose_nv",
        "lEye_nv",
        "rEye_nv",
        "lEar_nv",
        "rEar_nv",
        "lShoulder_nv",
        "rShoulder_nv",
        "lElbow_nv",
        "rElbow_nv",
        "lWrist_nv",
        "rWrist_nv",
        "lHip_nv",
        "rHip_nv",
        "lKnee_nv",
        "rKnee_nv",
        "lAnkle_nv",
        "rAnkle_nv"
]

image_list = []
people_image_list = []

with open(args.file) as json_data:
    d = json.load(json_data)

    #file read by index
    #for i in range(len(d["annotations"])):
    for i in range(11,20):
	#file name
	image_id = d["annotations"][i]["image_id"]
	#make real filename = insert 0 at front of image_id
	for j in range(12-len(str(image_id))):
	    if j == 0:
		f_space = str(0)
	    else :
		f_space = f_space + str(0)

	f_name = f_space+str(image_id)

	#open label file "append"
	#outpath = output_folder + f_name + '.txt'
	#print outpath
	#outfile = open(outpath, "a+")

       
	if d["annotations"][i]["category_id"] == 1 and d["annotations"][i]["iscrowd"] == 0:
            img = cv2.imread(image_folder+'/'+f_name+".jpg")
            height, width = img.shape[:2]
            print f_name, height, width

            image_list.append(image_id)
            keycounter["person"] +=1
            keypoints_list = d["annotations"][i]["keypoints"]

            x = keypoints_list[0::3]
            y = keypoints_list[1::3]
            visible = keypoints_list[2::3]
            print x
            print y
            print d["annotations"][i]["area"]
            

            if d["annotations"][i]["num_keypoints"] != 0:
                key_length = int(round(math.sqrt(d["annotations"][i]["area"]/d["annotations"][i]["num_keypoints"])))
            else:
                key_length = 0

            for i, v in enumerate(visible):
                x1 = x[i]-(key_length//2)
                y1 = y[i]-(key_length//2)
                if v == 1:
                    keycounter[nv_name_list[i]] +=1
                    cv2.rectangle(img, (x1,y1), (x1+key_length, y1+key_length), (0,0,255), 2)
                elif v == 2:
                    keycounter[name_list[i]] +=1
                    cv2.rectangle(img, (x1,y1), (x1+key_length, y1+key_length), (255,0,0), 2)
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            people_image_list.append(image_id)

            

#	x =  d["annotations"][i]["bbox"][0]
#	y =  d["annotations"][i]["bbox"][1]
#	w =  d["annotations"][i]["bbox"][2]
#	h =  d["annotations"][i]["bbox"][3]
#
#	converted_box = convert((img_w,img_h), x, y, w, h)
       
	#outfile.write(str(clas-1) + " " + " ".join([str(a) for a in converted_box]) + '\n')

	#outfile.close()

print "total annotations: ", len(d["annotations"])
print "number of person images: ", len(set(image_list))
print "number of people images: ", len(set(people_image_list))
print "total images: ", len(set(image_list+people_image_list))

print "person :\t" , keycounter["person"]
print "nose_v :\t" , keycounter["nose_v"]
print "nose_nv :\t" , keycounter["nose_nv"]
print "lEye_v :\t" , keycounter["lEye_v"]
print "lEye_nv :\t" , keycounter["lEye_nv"]
print "rEye_v :\t" , keycounter["rEye_v"]
print "rEye_nv :\t" , keycounter["rEye_nv"]
print "lEar_v :\t" , keycounter["lEar_v"]
print "lEar_nv :\t" , keycounter["lEar_nv"]
print "rEar_v :\t" , keycounter["rEar_v"]
print "rEar_nv :\t" , keycounter["rEar_nv"]
print "lShoulder_v :\t" , keycounter["lShoulder_v"]
print "lShoulder_nv :\t" , keycounter["lShoulder_nv"]
print "rShoulder_v :\t" , keycounter["rShoulder_v"]
print "rShoulder_nv :\t" , keycounter["rShoulder_nv"]
print "lElbow_v :\t" , keycounter["lElbow_v"]
print "lElbow_nv :\t" , keycounter["lElbow_nv"]
print "rElbow_v :\t" , keycounter["rElbow_v"]
print "rElbow_nv :\t" , keycounter["rElbow_nv"]
print "lWrist_v :\t" , keycounter["lWrist_v"]
print "lWrist_nv :\t" , keycounter["lWrist_nv"]
print "rWrist_v :\t" , keycounter["rWrist_v"]
print "rWrist_nv :\t" , keycounter["rWrist_nv"]
print "lHip_v :\t" , keycounter["lHip_v"]
print "lHip_nv :\t" , keycounter["lHip_nv"]
print "rHip_v :\t" , keycounter["rHip_v"]
print "rHip_nv :\t" , keycounter["rHip_nv"]
print "lKnee_v :\t" , keycounter["lKnee_v"]
print "lKnee_nv :\t" , keycounter["lKnee_nv"]
print "rKnee_v :\t" , keycounter["rKnee_v"]
print "rKnee_nv :\t" , keycounter["rKnee_nv"]
print "lAnkle_v :\t" , keycounter["lAnkle_v"]
print "lAnkle_nv :\t" , keycounter["lAnkle_nv"]
print "rAnkle_v :\t" , keycounter["rAnkle_v"]
print "rAnkle_nv :\t" , keycounter["rAnkle_nv"]
