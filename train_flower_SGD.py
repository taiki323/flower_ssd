# -*- coding: utf-8 -*-
import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pickle, csv, random
from random import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import pandas as pd
import tensorflow as tf

from ssd import SSD300
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility
from keras.callbacks import LambdaCallback
from keras.callbacks import EarlyStopping
from keras_func import *



plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True) #指数表示の禁止

# 21
NUM_CLASSES = 2 #4
input_shape = (300, 300, 3)

priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

pkl = 'pkl/flower.pkl'
# gt = pickle.load(open('gt_pascal.pkl', 'rb'))
gt = pickle.load(open(pkl, 'rb')) #教師データロード
keys = sorted(gt.keys())     #ファイル名でソート
#num_train = int(round(0.8 * len(keys)))  #データの8割を訓練データに
num_train = 900
train_keys = keys[:num_train]
val_keys = keys[num_train:]              #データの2割を検証データに
num_val = len(val_keys)                  #検証データ数

def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

class Generator(object):
    def __init__(self, gt, bbox_util,
                 batch_size, path_prefix,
                 train_keys, val_keys, image_size,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3./4., 4./3.]):
        self.gt = gt
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_batches = len(train_keys)
        self.val_batches = len(val_keys)
        self.image_size = image_size
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range

    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)

    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y

    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y

    def random_sized_crop(self, img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w_rel = w / img_w
        w = int(w)
        h = min(h, img_h)
        h_rel = h / img_h
        h = int(h)
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y+h, x:x+w]
        new_targets = []
        for box in targets:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
        return img, new_targets

    def generate(self, train=True):
        while True:
            if train:
                shuffle(self.train_keys)
                keys = self.train_keys
            else:
                shuffle(self.val_keys)
                keys = self.val_keys
            inputs = []
            targets = []
            for key in keys:
                img_path = self.path_prefix + key
                img = imread(img_path).astype('float32')
                y = self.gt[key].copy()
                if train and self.do_crop:
                    img, y = self.random_sized_crop(img, y)
                img = imresize(img, self.image_size).astype('float32')
                # boxの位置は正規化されているから画像をリサイズしても
                # 教師信号としては問題ない
                if train:
                    shuffle(self.color_jitter)
                    for jitter in self.color_jitter:
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)
                # 訓練データ生成時にbbox_utilを使っているのはここだけらしい
                #print(y)
                y = self.bbox_util.assign_boxes(y)
                #print(y)
                inputs.append(img)
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield preprocess_input(tmp_inp), tmp_targets

column = ["pickle", "optimizer","batch_size", "base_lr", "train_loss", "val_loss"]
path_prefix = '/home/minelab/dataset/flower_300/' #原画像パス
batch_sizes = [4,8,16,32]
SGD_flag = 1

for base_lr in range(1):
    #batch_size = random.choice(batch_sizes)
    batch_size = 32
    base_lr = random.uniform(0.0002, 0.0004)
    decay_lr = random.uniform(0.0001, 0.00001)
    base_lr = 0.000276594597079214
    decay_lr = 0.0000666831799375426


    print "batch_size = " + str(batch_size)
    print "base_lr = " + str(base_lr)
    print "decay = "  + str(decay_lr)

    gen = Generator(gt, bbox_util, batch_size, path_prefix,
                    train_keys, val_keys,
                    (input_shape[0], input_shape[1]), do_crop=False)

    model = SSD300(input_shape, num_classes=NUM_CLASSES)
    model.load_weights('weights_SSD300.hdf5', by_name=True)

    freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',  #VGGをフリーズ
              'conv2_1', 'conv2_2', 'pool2',
              'conv3_1', 'conv3_2', 'conv3_3', 'pool3']#,
    #           'conv4_1', 'conv4_2', 'conv4_3', 'pool4']

    for L in model.layers: #freezeで指定した層は訓練しない
        if L.name in freeze:
            L.trainable = False

    early_stop = EarlyStopping(patience=30)

    #plot_loss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch), logs['loss']))
    #callbacks = [keras.callbacks.ModelCheckpoint('./checkpoints/weights_flower.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5',
    callbacks = [keras.callbacks.ModelCheckpoint('./checkpoints/SGD/flower_weights_batch=' + str(batch_size) + '_lr=' + str(base_lr) + ',0.001004.hdf5',
                                                     verbose=1,
                                                     save_best_only = True,
                                                     mode = "auto",
                                                     save_weights_only=True), #各エポック終了後にモデルを保存
                     early_stop,
    #                 keras.callbacks.LearningRateScheduler(schedule)
    ] #学習係数を動的に変更

    optim = keras.optimizers.SGD(lr=base_lr,momentum=0.9,decay=decay_lr)
    #optim = keras.optimizers.Adam(lr=base_lr)
    model.compile(optimizer=optim,
                  loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)  #?


    nb_epoch = 1000
    history = model.fit_generator(gen.generate(True), gen.train_batches, #学習
                                  nb_epoch, verbose=1,
                                  callbacks=callbacks,
                                  validation_data=gen.generate(False), #?
                                  nb_val_samples=gen.val_batches,
                                  nb_worker=1)


    drowpltloss(history, "gragh/SGD/flower_batch=" + str(batch_size) + "_lr=" + str(base_lr) + "_decay=" + str(decay_lr) + ",0.001004.png", 0, 4.5)
    cs = [pkl.split("/")[1],"SGD",batch_size,base_lr,min(history.history['loss']),min(history.history['val_loss']), decay_lr]
    with open('result_' + pkl.split("/")[1] + '.csv', 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(cs)


""""
inputs = []     #inputsには画像配列 (画像数,300,300,3)
images = []     #imagesには画像データ
img_path = path_prefix + sorted(val_keys)[0]  #検証画像から1枚取得
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
inputs = preprocess_input(np.array(inputs)) #VGG16用の平均を引く前処理

preds = model.predict(inputs, batch_size=1, verbose=1) #(画像数,7308,33)
results = bbox_util.detection_out(preds)  #?

for i, img in enumerate(images): #iは0,1,2と増えていく
    # Parse the outputs.
    det_label = results[i][:, 0] #len:200
    det_conf = results[i][:, 1] #信頼度  降順
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6] #BB数 しきい値

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist() #各ラベル番号
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, NUM_CLASSES)).tolist() #BBグラデーション

    plt.imshow(img / 255.)
    currentAxis = plt.gca() #ax取得

    for i in range(top_conf.shape[0]): #BB数
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i]) #数字
        # label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label) #'0.29, 15'
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1 #(x,y), width, height
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2)) #長方形プロット
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5}) #label付け

  #  plt.show()
"""""