import numpy as np
import os
import cv2
from xml.etree import ElementTree

class XML_preprocessor(object):

    def __init__(self, data_path):
        self.path_prefix = data_path
        self.num_classes = 1
        self.data = dict()
        self._preprocess_XML()

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            imgname = filename.split(".")[0] + ".jpg"
            img = cv2.imread("/media/ubtaiki/disk/dataset/VOCdevkit/VOC2012/JPEGImages/" + imgname)
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text)
                    ymin = float(bounding_box.find('ymin').text)
                    xmax = float(bounding_box.find('xmax').text)
                    ymax = float(bounding_box.find('ymax').text)
                    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 3)
            cv2.imwrite("/media/ubtaiki/disk/dataset/VOCdevkit/VOC2012/BBinJPEGImages/" + imgname, img)


## example on how to use it
import pickle
data = XML_preprocessor('/media/ubtaiki/disk/dataset/VOCdevkit/VOC2012/Annotations/').data

