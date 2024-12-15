# import packages
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def histimg(img):
    '''
    params: img
        img: for now, img is a 4 channel matrix, the fourth channel being the opacity channel
    outout:opac,hist
    '''
    opac=cv.split(img)[3]
    hist = cv.calcHist([opac],[0],None,[256],[0,256]) # for manually selecting the value
    return opac,hist

def binarize(img):
    return None