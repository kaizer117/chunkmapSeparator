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
        params: cmap,i
            cmap: matplotlib.cm object
            i: int 0 to 3. picks the
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

def con2ll(con):
    '''
    Is this really needed?
    function to convert list of points to a linked list with front and back traversal properties
    '''
    return None

def linearShift(sX,sY):
    """
    THis function will shift the raster contour data so the final image can have have good cropping
    """
    pass

def conReshaper(con):
    """
    This function changes single contours
    The cv function returns the countours in a peculiar format.

    It will wrap the points in 1 more layer of list as exampled below
    [ [ [x1 , y1] ] , [ [x2 , y2] ] ] => [ [x1 , y1] , [x2 , y2] ]

    """
    return np.reshape(con,(len(con),2))

def contoursRehsaper(cons):
    """
    This function changes multiple contours (list of lists)
    Wrapper for individual conReshaper
    """
    l = []
    for con in cons:
        l.append(conReshaper(con))
    return l