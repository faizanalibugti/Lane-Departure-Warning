import cv2
import numpy as np
img=cv2.imread('out3.png')
# print(img.shape[1],img.shape[0])
# imgcrop=img[300:310,200:210]
def isGreen(img):
    thr=(img.shape[0]*img.shape[1])*0.15
    lowerGreen = np.array([0,150,0])
    upperGreen = np.array([0,255,0])
    newimage=cv2.inRange(img,lowerGreen,upperGreen)
    count=cv2.countNonZero(newimage)
    if count>thr:
        return True,count
    else:
        return False,count
a,c=isGreen(img)
print(a,c)
# print(img.shape)
# print(c)
# while True:
#     # cv2.imshow('img',newimage)
#     if cv2.waitKey(1) & 0xFF==ord('q'):
#         break
# cv2.destroyAllWindows()