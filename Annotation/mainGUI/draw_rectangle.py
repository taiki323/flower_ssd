import cv2

img = cv2.imread("/media/ubtaiki/disk/dataset/flower/test2/flower00009.jpg")

cv2.rectangle(img,(162,191), (189,223), (0,0,255),3)
cv2.rectangle(img,(1230,850), (1372,960), (0,0,255),3)
cv2.imwrite("result.jpg",img)  
