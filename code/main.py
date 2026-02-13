
# import packages
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cvutils as cutils
import programutils as putils
import graphutils as gutils
import fileutils as futils
import colorspace as colutils
import vectorization as vect
import segmentation as seg

# load input image

inputimg='example-cuba.png'
img = cv.imread('./resources/'+inputimg, cv.IMREAD_UNCHANGED)

# creating a histogram and splitting opacity
opac,hist=cutils.histimg(img)

# why deos this function return two values and what is ret
ret,thresh1 = cv.threshold(opac,252,255,cv.THRESH_BINARY)


# doing the computer vision contours
contours, hierarchy = cv.findContours(thresh1, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

#reshaping the contours array for ease
contours = cutils.contoursRehsaper(contours)

# compute and display the curvature of contours and a histogram of the curvature
i = 40
# curv, stats = vect.computCurvature(contours[i])
# # fig,ax,stats = gutils.plotcurvaturewithhistogram(contours[i],curv,((max(curv)-min(curv))/100))
# print(f"n={stats['n']} mean={stats['mean']:.3f} median={stats['median']:.3f} std={stats['std']:.3f} range=[{stats['min']:.3f},{stats['max']:.3f}]")

# shifting and reshaping the shape for visualization
shapeTuple = vect.get_xy_extent(contours[i],output="minmaxex")

data = contours[i]

#segmentation 
# segments = vect.polygonSegmentation(contours[i])
vect.visualizeSegmentation(data)

# bezier curve fitting and visualization
# controlPoints = vect.vectorizeContour(contours[i])
# ctrlCon = list(map(vect.fitBezierCurve,segments))
# reshapedCtrl = vect.reshapeCtrlSVG(controlPoints)
# futils.saveIndCon(reshapedCtrl,contours[i])

#plt=gutils.plotconsHierarchy(cons1,cm.spring,'Feature visualizaton')
#plt.show()

# contour to svg
#cmap=colutils.cmapHex(12,len(contours),'random','shuffle')

#img_size=np.shape(img)
# futils.saveCons(img_size,contours,cmap,'outputs','cuba_ex')

print('end')

# code block to replace transparent pixels with not transparent stuff.
# trans_mask = img[:,:,3] == 0
# img[trans_mask] = [255, 255, 255, 255]



# cv.drawContours(img,contours,0,(0,255,0),1)
# cv.imshow("contours",img)
# k = cv.waitKey(0)
# code block to show images
# cv.imshow("Display window", img)

