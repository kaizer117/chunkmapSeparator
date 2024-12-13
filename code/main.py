
# import packages
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('./resources/example.png', cv.IMREAD_UNCHANGED)

opac=cv.split(img)[3]
hist = cv.calcHist([opac],[0],None,[256],[0,256])

# plotting opacity histogram

# plt.plot(hist)
# plt.title("Opacity Histogram")
# plt.xlim([250,256])
# plt.show()

ret,thresh1 = cv.threshold(opac,253,255,cv.THRESH_BINARY)



contours, hierarchy = cv.findContours(thresh1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# code block to replace transparent pixels with not transparent stuff.
trans_mask = img[:,:,3] == 0
img[trans_mask] = [255, 255, 255, 255]



cv.drawContours(img,contours,0,(0,255,0),1)
cv.imshow("contours",img)
k = cv.waitKey(0)
# code block to show images
# cv.imshow("Display window", img)

