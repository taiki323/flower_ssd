# -*- coding: utf-8 -*-
import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import  numpy as np
from scipy.misc import imread
import tensorflow as tf
import os,pickle,copy
import random, csv
from  Exclusion import *
from sklearn.metrics import label_ranking_average_precision_score
from ssd import SSD300
from ssd_utils import BBoxUtility


def boxsize(data):
    for valuer in data.values():
        for v in valuer:
            boxsizelist.append((abs(v[2]-v[0]) * abs(v[3]-v[1])) * 1000)

def caulcsizeratio():
    sizeacc.append((len(filter(lambda size: size < 10, detectsizelist)) / float(len(filter(lambda size: size < 10, boxsizelist)))) * 100)
    #sizeacc.append((len(filter(lambda size: size < 80 and size >= 10, detectsizelist)) / float(len(filter(lambda size: size < 80 and size >= 10, boxsizelist)))) * 100)
    sizeacc.append((len(filter(lambda size: size >= 10, detectsizelist)) / float(len(filter(lambda size: size >= 10, boxsizelist)))) * 100)

def calcIOU(top_xmin, top_ymin, top_xmax, top_ymax, btop_xmin, btop_ymin, btop_xmax, btop_ymax):
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
    iou = oquad / (quad1 + quad2 - oquad)
    return iou

def overlap(top_xmin, top_ymin, top_xmax, top_ymax, btop_xmin, btop_ymin, btop_xmax, btop_ymax, iou_rate): #2つの四角形が重なっていたら,return1
    xs = btop_xmin - top_xmin
    y = btop_ymin - top_ymin
    aw = top_xmax - top_xmin
    bw = btop_xmax - btop_xmin
    ah = top_ymax - top_ymin
    bh = btop_ymax - btop_ymin
    if -bw < xs and xs < aw and -bh < y and y < ah: #重なり判定
        iou = calcIOU(top_xmin, top_ymin, top_xmax, top_ymax, btop_xmin, btop_ymin, btop_xmax, btop_ymax)
        if iou >= iou_rate:
            return iou
    return 0


def makelist(top_xmin, top_ymin, top_xmax, top_ymax, score, label, data):
    global aplist
    for i in range(len(top_conf)):
        flag = 0
        for j in range(data.shape[0]): #正解BB
            if (label[i] == int(data[j][4]) and not(j in detectedboxlist)): #同じラベルか
                iou = overlap(top_xmin[i], top_ymin[i], top_xmax[i], top_ymax[i],data[j][0],data[j][1],data[j][2],data[j][3], 0.5)
                if iou != 0: #正解したときリストに追加
                    detectedboxlist.append(j)
                    aplist.append((1,score[i]))
                    detectsizelist.append((abs(data[j][2]-data[j][0]) * abs(data[j][3]-data[j][1])) * 1000 )
                    flag = 1
                    break
        if flag == 0:
            aplist.append((0,score[i]))
    return 1

def precisionlist(aplist):
    global presicion
    tp = 0
    fp = 0
    count = 0

    for ap in aplist:
        if ap[0] == 1:
            count += 1
    try:
        presicion = count / float(len(aplist))
    except:
        presicion = 0

    for p in aplist:
        if p[0] == 1:
            tp += 1
        else:
            fp += 1
        ap = tp / float(tp + fp)
        tmp = list(p)
        tmp.append(ap)
        tmp.append(tp/float(label_num))
        maplist.append(tmp)

def truecalcmap(aplist):
    for a in aplist:
        ans.append(a[0])
        sc.append(a[1])
    map = label_ranking_average_precision_score([ans],[sc])
    return map

def calcmap(maplist):
    map_sum = 0
    ap_sum = 0
    counts = 0
    acc_num = 0
    for i, data in enumerate(maplist): #AP計算
        if(data[0] == 1): #正解
            acc_num += 1
            ap_sum = 0
            counts = 0
            for x in range(i+1): #今までのP
                if(maplist[x][0] == 1): #正解
                    ap_sum += maplist[x][2]
                    counts += 1
            ap = ap_sum / counts
            map_sum += ap
    map = map_sum / acc_num
    return map

def non_maximum_supression(top_xmin,top_ymin,top_xmax,top_ymax,top_conf,top_label_indices, iou_range=0.2):
    reducelist = []
    for i in range(len(top_conf)):
        if i in reducelist:
            continue
        for j in range(i+1,len(top_conf)):
            if j in reducelist:
                continue
            iou = overlap(top_xmin[i], top_ymin[i], top_xmax[i], top_ymax[i], top_xmin[j], top_ymin[j], top_xmax[j], top_ymax[j], iou_range)
            if iou != 0: #もし2つのボックスが重なっていたら
                ioulist.append(iou)
                if top_conf[i] > top_conf[j]: #compare score
                    reducelist.append(j)
                else:
                    reducelist.append(i)
    return reducelist

###############################
hdf5 = "normal/flower_weights_batch=4_lr=3e-05.hdf5"
hdf5 = "flip/flip_adam_flower_weights_batch=4_lr=3e-05,0.001004.hdf5"
hdf5 = "randomErasing_flip/randomErasing0.3_flip_adam_flower_weights_batch=4_lr=3e-05.hdf5"
#hdf5 = "mask/flip/mask_flip_adam_flower_weights_batch=4_lr=3e-05.hdf5"
#hdf5 = "roll_flip/roll_flip_adam_flower_weights_batch=4_lr=3e-05,0.001004.hdf5"
testflag = 1 #1 = nougi, 0=nougakubu
path = "/home/minelab/dataset/rename/"
#path = "/home/minelab/dataset/mask2/"
conf_rate = 0.2
io_range = 0.1
##############################

if testflag == 1:
    testdir = "nougi_test/"
    xmlpath = "/home/minelab/dataset/xml/test_nougi/"
    pk = 'nougi_flower3.pkl'
else:
    xmlpath = "/home/minelab/dataset/xml/test_nougakubu/"
    pk = 'test0.001004.pkl'
    testdir = ""

###
#xmlpath = "/home/minelab/dataset/xml/test/"
#pk = 'test_map.pkl'
###
pkl = 'pkl/' + pk
f = open(pkl, 'rb')
data = pickle.load(f)
f.close()

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

voc_classes = ['flower']
NUM_CLASSES = len(voc_classes) + 1

input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('./checkpoints/' + hdf5, by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)

imgNames = os.listdir(xmlpath)
random.shuffle(imgNames)
for i, img in enumerate(imgNames):
    imgNames[i] = img.replace("xml","jpg")


areas = []
inputs = []
images = []
names = []
count = 0
for img_path in imgNames:
   # if count >= 100:
   #     break
    count += 1
    img = image.load_img(path + img_path, target_size=(300, 300))
    img = image.img_to_array(img)
    images.append(imread(path + img_path))
    names.append(img_path)
    inputs.append(img.copy())
inputs = preprocess_input(np.array(inputs))

preds = model.predict(inputs, batch_size=1, verbose=1)

results = bbox_util.detection_out(preds)

a = model.predict(inputs, batch_size=1)
b = bbox_util.detection_out(preds)
print ("\n")
dir = "result/" + testdir + hdf5
print dir
count = 0
for a in range(1):
#for a in range(1):
    dir = "result/" + testdir + hdf5 + "_conf=" + str(conf_rate)
    types = [('a',int),('b', float)]
    global aplist #予測が正解かどうか(0,1)とスコア
    detectsizelist = []
    boxsizelist = []
    sizeacc = []
    ioulist = []
    aplist = []
    ans = []
    sc = []
    maplist = []
    label_num = 0
    count = 0
    presicion = 0
    recall = 0

#    if os.path.isdir(dir) == False:
#        os.mkdir(dir)

    for i, img in enumerate(images): #出力200 per 1 image
        bb_num = 0  #出力BB数
        plt.figure()
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_rate]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        reducelist = non_maximum_supression(top_xmin,top_ymin,top_xmax,top_ymax,top_conf,top_label_indices, io_range)
        top_reducedindices = list(set(top_indices) - set(reducelist))

        top_conf = det_conf[top_reducedindices]
        top_label_indices = det_label[top_reducedindices].tolist()
        top_xmin = det_xmin[top_reducedindices]
        top_ymin = det_ymin[top_reducedindices]
        top_xmax = det_xmax[top_reducedindices]
        top_ymax = det_ymax[top_reducedindices]

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

        #oriimg = imread("/home/minelab/dataset/rename/" + names[count])
        #count += 1
        #plt.imshow(oriimg / 255.) #mask
        plt.imshow(img / 255.)
        currentAxis = plt.gca()
        pred = []
        detectedboxlist = []

        makelist(top_xmin, top_ymin, top_xmax, top_ymax, top_conf, top_label_indices, data[imgNames[count]])
        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = voc_classes[label - 1]
            display_txt = '{:0.2f}'.format(score)
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor="red", linewidth=1))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': "red", 'alpha': 0.5})
            bb_num += 1
            #makelist(top_xmin[i], top_ymin[i], top_xmax[i], top_ymax[i], score, label, data[imgNames[count]])

            loc = [top_xmin[i],top_ymin[i],top_xmax[i],top_ymax[i],top_conf[i]]
            areas.append(calcRectangle(loc))

        # true BB
        for bb in data[imgNames[count]]:
            xmin = int(round(bb[0] * img.shape[1]))
            ymin = int(round(bb[1] * img.shape[0]))
            xmax = int(round(bb[2] * img.shape[1]))
            ymax = int(round(bb[3] * img.shape[0]))
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor="blue", linewidth=1))
            label_num += 1

        #plt.show()
        #print str(count) + imgNames[count]
        #plt.savefig(dir + "/result_" + imgNames[count])
        count += 1
        plt.close()
        plt.close('all') #メモリ解放

    boxsize(data)
    caulcsizeratio()
