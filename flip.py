# -*- coding: utf-8 -*-

from PIL import Image
import os

def readImg(imgName):
    try:
        img_src = Image.open("/home/minelab/dataset/rename/" + imgName)
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
            tmp = img_src.transpose(Image.FLIP_LEFT_RIGHT)
            tmp.save("/home/minelab/dataset/rename/flipLR_" + imgName)
         #   print("{} is done!".format(imgName))

if __name__ == '__main__':
    #read imgs names
    imgNames = os.listdir("/home/minelab/dataset/rename")#画像が保存されてるディレクトリへのpathを書きます
    spinImg(imgNames)
    print "fnish"