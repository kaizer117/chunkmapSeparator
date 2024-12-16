# import packages
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import programutils as putil

class cons():
    
    '''
    object: con
        a container class for contour and its' associated data. 
    '''
    
    def __init__(self,contours,hierarchy):
        self.contours=contours
        self.hierarchy=hierarchy # hierarchy values
        
        # hiearchy values normalized.
        self.hierarchy_norm=[] 
        for i in range(4):
            self.hierarchy_norm.append(putil.norm01(hierarchy[0,:,i]))
            
        # add moments
        self.mu=list(map(lambda con: cv.moments(con),contours))
        # add centeroid
        self.centeroid=list(map(lambda d: (d['m10'] / (d['m00'] + 1e-5), d['m01'] / (d['m00'] + 1e-5)),self.mu))
        # add contor area
        self.area=list(map(lambda d: d['m00'],self.mu))
            
    def setcmap(self,cmap,i):
        
        '''
        params: cmap
            cmap: matplotlib.cm object
        output: ret
            ret: lists of colors at lenght(contour) for the respective colormap

        '''
        return cmap(self.hierarchy_norm[i])
        
        


def histimg(img):
    '''
    params: img
        img: for now, img is a 4 channel matrix, the fourth channel being the opacity channel
    output: opac,hist
    '''
    opac=cv.split(img)[3]
    hist = cv.calcHist([opac],[0],None,[256],[0,256]) # for manually selecting the value
    return opac,hist

def binarize(img):
    return None