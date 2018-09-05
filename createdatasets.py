# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import  numpy as np
from scipy.misc import imread
import os,pickle,copy
import random, csv

def overlap_area(top_xmin, top_ymin, top_xmax, top_ymax, btop_xmin, btop_ymin, btop_xmax, btop_ymax):
    ov_xmin = max(top_xmin, btop_xmin)
    ov_ymin = max(top_ymin, btop_ymin)
    ov_xmax = min(top_xmax, btop_xmax)
    ov_ymax = min(top_ymax, btop_ymax)
    aw = top_xmax - top_xmin
    bw = btop_xmax - btop_xmin
    ow = ov_xmax - ov_xmin
    ah = top_ymax - top_ymin
    bh = btop_ymax - btop_ymin
    oh = ov_ymax - ov_ymin

    quad1 = aw * ah
    quad2 = bw * bh
    oquad = ow * oh
    #iou = oquad / (quad1 + quad2 - oquad)
    return (oquad / quad1) * 100

def overlap(top_xmin, top_ymin, top_xmax, top_ymax, btop_xmin, btop_ymin, btop_xmax, btop_ymax): #2つの四角形が重なっていたら,return1
    xs = btop_xmin - top_xmin
    y = btop_ymin - top_ymin
    aw = top_xmax - top_xmin
    bw = btop_xmax - btop_xmin
    ah = top_ymax - top_ymin
    bh = btop_ymax - btop_ymin
    if -bw < xs and xs < aw and -bh < y and y < ah: #重なり判定
        area = overlap_area(top_xmin, top_ymin, top_xmax, top_ymax, btop_xmin, btop_ymin, btop_xmax, btop_ymax)
        if area >= 0.2:
            return area
    return 0

def include_flower(hpoint,wpoint,bboxs,wlength,hlength):
    xmin = bboxs[0] * wlength
    ymin = bboxs[1] * hlength
    xmax = bboxs[2] * wlength
    ymax = bboxs[3] * hlength
    if wpoint < xmin and xmax < wpoint+window_size and hpoint < ymin and ymax < hpoint+window_size:
        xmin2 = (xmin - wpoint) / window_size
        ymin2 = (ymin - hpoint) / window_size
        xmax2 = (xmax - wpoint) / window_size
        ymax2 = (ymax - hpoint) / window_size
        box = np.array([xmin2, ymin2, xmax2, ymax2, 1])
        box = box.reshape(1, 5)
        return box
    area = overlap(wpoint,hpoint,wpoint+window_size,hpoint+window_size,xmin,ymin,xmax,ymax)
    if area != 0:
        xmin2 = (max(xmin,wpoint) - wpoint) / window_size
        ymin2 = (max(ymin,hpoint) - hpoint) / window_size
        xmax2 = (min(xmax,wpoint+window_size) - wpoint) / window_size
        ymax2 = (min(ymax,hpoint+window_size) - hpoint) / window_size
        box = np.array([xmin2, ymin2, xmax2, ymax2, 1])
        box = box.reshape(1, 5)
        #return box
        return 1
    return 0

move = 450
window_size = 900

path = "/home/minelab/dataset/rename/"
f = open('pkl/flip_flower.pkl', 'rb')
data = pickle.load(f)
data2 = copy.deepcopy(data)
f.close()
a = 0
for imgName in data.keys():
    #if a > 10:
    #    break
    img = cv2.imread(path + imgName)
    count = 0
    hpoint = 0 #高さ
    bunkatuh = range((img.shape[0] / move) - 1)
    bunkatuw = range((img.shape[1] / move) - 1)
    for h in range((img.shape[0] / move)-1):
        wpoint = 0 #横
        for w in range((img.shape[1] / move)-1): #per画像
            new = np.empty([0, 5])
            save_name = imgName + "_" + str(count) + ".jpg"
            print save_name
            for bboxs in data[imgName]:  #perBBox
                box = include_flower(hpoint,wpoint,bboxs,img.shape[1],img.shape[0])
                if type(box) != int:
                    new = np.r_[new, box]
                elif box == 1:
                    break
            if len(new) != 0:
                data2.update({save_name: new})
                tmpimg = img[hpoint:hpoint+window_size,wpoint:wpoint+window_size]
                cv2.imwrite("/home/minelab/dataset/crop/" + save_name, tmpimg)
            count += 1
            wpoint += move
        hpoint += move
    #a += 1

pickle.dump(data2,open('pkl/crop_flower.pkl','wb'))
