import keras
import pickle
from videotest import VideoTest
import cv2

import sys
sys.path.append("..")
from ssd import SSD300 as SSD
video = cv2.VideoCapture('/home/minelab/dataset/video/GOPR9632_2.mkv')
input_shape = (300,300,3)

# Change this if you run with other classes than VOC
class_names = ["flower"];
NUM_CLASSES = len(class_names) + 1

model = SSD(input_shape, num_classes=NUM_CLASSES)

# Change this path if you want to use your own trained weights
model.load_weights('../checkpoints/flip/flip_adam_flower_weights_batch=4_lr=3e-05,0.001004.hdf5') 
        
vid_test = VideoTest(class_names, model, input_shape)

# To test on webcam 0, remove the parameter (or change it to another number
# to test on that webcam)
vid_test.run('/home/minelab/dataset/video/GOPR9632_2.avi')
