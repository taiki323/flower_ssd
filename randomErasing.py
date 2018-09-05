import numpy as np
import pickle
import copy
import cv2
from fractions import Fraction
import random
import math

###########
p = 0.5
sl = 0.02
sh = 0.3
r1 = Fraction(1,3)
r2 = 3
mean = [255 * 0.4914, 255 * 0.4822, 255 * 0.4465]
###########

imgpath = "/home/minelab/dataset/rename/"
pklfile = "flip_flower.pkl"
f = open('../pkl/' + pklfile, 'rb')
data = pickle.load(f)
f.close()

aug = "randomErasing_"
data2 = copy.deepcopy(data)
count = 0
for key, value in  data.iteritems():
    if count % 2 == 0:
        count += 1
        continue
    count += 1
    keyname = aug + key
    img = cv2.imread(imgpath + key)
    for v in value:
        wei = int((v[2] - v[0]) * img.shape[1])
        hei = int((v[3] - v[1]) * img.shape[0])
        area = wei * hei
        target_area = random.uniform(sl,sh) * area
        aspect_ratio = random.uniform(r1,r2)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w <= wei and h <= hei:
            x1 = int((v[1] * img.shape[0]) + random.randint(0,hei - h))
            y1 = int((v[0] * img.shape[1]) + random.randint(0,wei - w))
            if img.shape[2] == 3:
                img[x1:x1 + h, y1:y1 + w, 0] = mean[0]
                img[x1:x1 + h, y1:y1 + w, 1] = mean[1]
                img[x1:x1 + h, y1:y1 + w, 2] = mean[2]
            else:
                img[x1:x1 + h, y1:y1 + w, 0] = mean[0]
    cv2.imwrite(imgpath + keyname,img)
    data2.update({keyname: value})

pickle.dump(data2,open('../pkl/randomErasing_' + pklfile,'wb'))