import cv2
import pickle

height = 300
width = 300
f = open('/home/ubtaiki/git/flower_ssd/flower89.pkl', 'rb')
datas = pickle.load(f)
f.close()
img = cv2.imread("/media/ubtaiki/disk/dataset/flower/test2/flower00009.jpg")

for data in datas["flower00009.jpg"]:
    xmin = int(width * data[0])
    ymin = int(height * data[1])
    xmax = int(width * data[2])
    ymax = int(height * data[3])
    cv2.rectangle(img,(xmin,ymin), (xmax,ymax), (0,0,255),1)
cv2.imwrite("result.jpg",img)

