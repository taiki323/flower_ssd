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
import os,pickle
import random
from  Exclusion import *

from ssd import SSD300
from ssd_utils import BBoxUtility

hdf5 = "flower_weights_batch=4_lr=3e-05,0.001004.hdf5"
xmlpath = "/home/minelab/dataset/xml/test/"
path = "/home/minelab/dataset/rename/"
conf_rate = 0.6
dir = "result/" + hdf5 + "_conf=" + str(conf_rate)

f = open('pkl/test0.001004.pkl', 'rb')
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
count = 0
for img_path in imgNames:
   # if count >= 100:
   #     break
    count += 1
    img = image.load_img(path + img_path, target_size=(300, 300))
    img = image.img_to_array(img)
    images.append(imread(path + img_path))
    inputs.append(img.copy())
inputs = preprocess_input(np.array(inputs))

preds = model.predict(inputs, batch_size=1, verbose=1)

results = bbox_util.detection_out(preds)

a = model.predict(inputs, batch_size=1)
b = bbox_util.detection_out(preds)

count = 0
if os.path.isdir(dir) == False:
    os.mkdir(dir)

for i, img in enumerate(images):
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

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    plt.imshow(img / 255.)
    currentAxis = plt.gca()

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
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

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

    #plt.show()
    #print str(count) + imgNames[count]
    plt.savefig(dir + "/result_" + imgNames[count])
    count += 1
    plt.close()
    plt.close('all') #メモリ解放

print "end"
print min(areas)