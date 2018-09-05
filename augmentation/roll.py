# -*- coding: utf-8 -*-

from PIL import Image
import os

rolling = 90
ori_path = "/home/minelab/dataset/flower_300/"
save_path = "/home/minelab/dataset/flower_300/"
def readImg(imgName):
    try:
        img_src = Image.open(ori_path + imgName) #d
        #print("read img!")
    except:
        print("{} is not image file!".format(imgName))
        img_src = 1
    return img_src


def spinImg(imgNames):
    for imgName in imgNames:
        img_src = readImg(imgName)
        if img_src == 1:continue
        else:
            #左右反転
            tmp = img_src.rotate(rolling)
            tmp.save(save_path + "roll" + str(rolling) + "_" + imgName)
         #   print("{} is done!".format(imgName))

if __name__ == '__main__':
    #read imgs names
    imgNames = os.listdir(ori_path)#画像が保存されてるディレクトリへのpathを書きます
    spinImg(imgNames)
    print "fnish"