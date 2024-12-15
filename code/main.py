
# import packages
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import binarization as binar

inputimg='big-example-wo-border.png'
img = cv.imread('./resources/'+inputimg, cv.IMREAD_UNCHANGED)

opac,hist=binar.histimg(img)

# plotting opacity histogram

# plt.plot(hist)
# plt.title("Opacity Histogram")
# plt.xlim([250,256])
# plt.show()

ret,thresh1 = cv.threshold(opac,252,255,cv.THRESH_BINARY)



contours, hierarchy = cv.findContours(thresh1, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

hierarchy_n=hierarchy[0,:,2]

def norm01(unnormalized):
    return (unnormalized-np.min(unnormalized))/(np.max(unnormalized)-np.min(unnormalized))




# Get the moments
mu = [None]*len(contours)
for i in range(len(contours)):
    mu[i] = cv.moments(contours[i])
# Get the mass centers
mc = [None]*len(contours)
ma = [None]*len(contours)
for i in range(len(contours)):
    # add 1e-5 to avoid division by zero
    mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
    ma[i] = mu[i]['m00']

cmap=cm.rainbow(norm01(hierarchy_n))
areanorm=norm01(ma)
plt.imshow(img)
i=0
for con in contours:
    plt.plot(con[:,0,0],con[:,0,1],color=cmap[i],marker='o',markersize=0.5)
    plt.text(mc[i][0],mc[i][1],str(i),fontdict={'fontsize':areanorm[i]*10+7})
    i+=1
plt.show()
print('end')

# code block to replace transparent pixels with not transparent stuff.
# trans_mask = img[:,:,3] == 0
# img[trans_mask] = [255, 255, 255, 255]



# cv.drawContours(img,contours,0,(0,255,0),1)
# cv.imshow("contours",img)
# k = cv.waitKey(0)
# code block to show images
# cv.imshow("Display window", img)

