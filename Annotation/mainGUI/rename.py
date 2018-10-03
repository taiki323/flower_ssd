#coding: UTF-8

from PIL import Image
import os

####指定######
fromdirname = "/media/ubtaiki/disk/dataset/flower/test2" #画像ファイル保存されているフォルダ名
save_path = "/media/ubtaiki/disk/dataset/flower/test2"
filename = "flower"
##############

path = fromdirname
imgNames = os.listdir(path)
count = 1

if not os.path.isdir(save_path):  #保存するフォルダ存在しないとき、フォルダ作成
    os.mkdir(save_path)

def readImg(imgName):
    try: #tryを使ってエラーでプログラムが止まっちゃうのを回避します。
        img_src = Image.open(path + "/" + imgName)
    except: #ゴミを読み込んだらこれちゃうで！って言います。
        print("{} is not image file!".format(imgName))
        img_src = 1
    return img_src

for imgName in imgNames: #1つの画像ずつ
    img_src = readImg(imgName)
    if img_src == 1:continue
    else:        	
        img_src.save(save_path + "/" + filename + '{0:05d}'.format(count) + ".jpg")
        #resizedImg.save(save_path + "/" + "256_256_" + str(count) + ".jpg")
	count += 1

print "finish"
