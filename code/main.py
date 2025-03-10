
# import packages
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cvutils as cutils
import programutils as putils
import graphutils as gutils
import fileutils as futils

# load input image

inputimg='big-example-wo-border.png'
img = cv.imread('./resources/'+inputimg, cv.IMREAD_UNCHANGED)

# creating a histogram and splitting opacity
opac,hist=cutils.histimg(img)


ret,thresh1 = cv.threshold(opac,252,255,cv.THRESH_BINARY)



contours, hierarchy = cv.findContours(thresh1, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)


cons1=cutils.cons(contours,hierarchy)



#plt=gutils.plotconsHierarchy(cons1,cm.spring,'Feature visualizaton')
#plt.show()


# contour to svg

img_size=np.shape(img)
futils.saveCons(img_size,contours)




print('end')

# code block to replace transparent pixels with not transparent stuff.
# trans_mask = img[:,:,3] == 0
# img[trans_mask] = [255, 255, 255, 255]



# cv.drawContours(img,contours,0,(0,255,0),1)
# cv.imshow("contours",img)
# k = cv.waitKey(0)
# code block to show images
# cv.imshow("Display window", img)

