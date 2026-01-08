'''
This submodule will handle creating folders, checking existence of 
folders and output version management
'''
import os
import datetime
import cv2 as cv

import numpy as np
import matplotlib.pyplot as plt
import re
import colorspace as colutils

cwd = os.getcwd()

def importImg(name,mode=cv.IMREAD_UNCHANGED):

    return cv.imread(cwd+'/resources/'+name, mode)
    
def getDay():
    current_time=datetime.datetime.now()
    return str(current_time.year)+str(current_time.month)+str(current_time.day)

def newSession():
    '''
    Description: When the codebase is cloned to a new machine, initialize the output folder
    '''
    if (not (os.path.exists(cwd+'\\outputs') )):
        os.mkdir(cwd+'\\outputs')
    return None

def createFolder(sub):
    prefix=getDay()
    i=0
    while(os.path.exists(cwd+'\\'+sub+'\\'+prefix+'_'+str(i))):
        i+=1
    
    # create folder
    new_path=cwd+'\\'+sub+'\\'+prefix+'_'+str(i)
    os.mkdir(new_path)
    # return the folder path
    return new_path
    
    pass

def savePlot(plt,sub=cwd,name='plot.png',create=False):
    '''
    params: plt,sub,name,create
        plt: plot object
        sub: if create == True, the subdirectory where new folder will be made and plot saved
             if create == False, the direct destination where plot will be saved
        name: file name of the plot
        create: create a subdirectory in the location specified in sub
    
    outputs:
        None, will save the plot according to the specified params
    '''
    if (create==True):
        plt.savefig(createFolder(sub)+'\\'+name)
    else:
        plt.savefig(sub+'\\'+name)
    
    return None

def saveCons(img_size,cons,cmap,loc='outputs',filename='mymap'):
    '''
    save contours in one svg file
    '''
    blr=''
    blr+='<?xml version="1.0" standalone="no"?>\n'


    blr+='<svg width="'+str(img_size[1])+'px"'
    blr+=' height="'+str(img_size[0])+'px"'
    blr+=' viewbox="'+'0'+' '+'0'+' '+str(img_size[1])+' '+str(img_size[0])+'"'
    blr+=' xmlns="http://www.w3.org/2000/svg" version="1.1">\n'
    for j,ex in enumerate(cons):
        s='\n'
        for i,p in enumerate(ex):
            # p=p[0] # bypassing the extra dimension # obsolete due to cvutils contourReshaper function
            if (i==1):
                s+='L '+str(p[0])+' '+str(p[1])+' '
                continue
            s+=str(p[0])+' '+str(p[1])+' '

        s='M '+s+'z'
        
        blr+='<path\nstyle="fill:'+cmap[j]+';fill-opacity:0.75;stroke:none;stroke-width:0.52916667;stroke-dasharray:none"\nd="'
        blr+=s+'"\n/>'
    blr+='\n</svg>'
    
    save_path=createFolder(loc)
    f = open(save_path+'\\'+filename+'.svg', "w")
    f.write(blr)
    f.close()
    return True

class SVGHandler():
    """
    OOP implementation of the svg editing functions for better utility and reusability
    
    :var returns: Description
    """
    def __init__(self):
        return None

def svgfileinit(img_size):
    '''
    This function will generate the boiler plate stuff of the svg

    :param img_size: (X,Y) size a tuple or a list of 2 pairs of values usually in px values

    returns:
    :param blr: string containing the contents of svg file
    '''
    blr=''
    blr+='<?xml version="1.0" standalone="no"?>\n'


    blr+='<svg width="'+str(img_size[1])+'px"'
    blr+=' height="'+str(img_size[0])+'px"'
    blr+=' viewbox="'+'0'+' '+'0'+' '+str(img_size[1])+' '+str(img_size[0])+'"'

    blr+=' xmlns="http://www.w3.org/2000/svg" version="1.1">\n'
    return blr

def svgfileclose(blr):
    '''
    This function simply closes the svg file by applying the closing tag
    '''
    blr+='\n</svg>'
    return blr

def svgaddstyletag(blr,d):
    '''
    blr: svg text to save
    d: dictionary of style guides
    structure of d:
    d => { styleName :{ property : value,... } }
    '''
    blr+='<style type="text/css"><![CDATA[\n'
    for stl in d.keys():
        blr+='.'+stl+' {'
        for prop in d[stl].keys():
            blr+=prop+':'+d[stl][prop]+'; '
        blr=blr[:-2]
        blr+='}\n'
    blr+=']]></style>\n'
    return blr

def svgdrawcircle(blr,c,r,stl):
    blr+='<circle class="'+stl+'" cx="'+str(c[0])+'" cy="'+str(c[1])+'"'+' r="'+str(r)+'" />\n'
    return blr

def svgdrawline(blr,p,stl):
    """
    Appends a path tag to the blr string, for straight lines
    
    :param blr: string containing the contents of svg file
    :param p: list of cartesian coordinates
    :param stl: name of the object in the style tag

    #returns
    :param blr: string containing the contents of svg file
    """
    
    blr+='<path class="'+stl+'" d="M '+str(p[0][0])+','+str(p[0][1])+' L '
    
    for point in p[1:]:
        blr+=str(point[0])+','+str(point[1])+' '
        
    blr=blr[:-1]+'" />\n'
    return blr



def svgdrawcubicbezier(blr,c,stl,closed=True):
    """
    Appends a path tag to the blr string, for closed cubic bezier curves
    
    :param blr: string containing the contents of svg file
    :param c: list of controlpoints for cubic bezier curve
    :param stl: name of the object in the style tag

    returns:
    :param blr: string containing the contents of svg file
    """
    blr+='<path class="'+stl+'" d="M '+str(c[0][0])+','+str(c[0][1])+' C '
    
    for point in c[1:]:
        blr+=str(point[0])+','+str(point[1])+' '
    
    if (closed == True):
        blr=blr[:-1]+'z" />\n'
    else:
        blr=blr[:-1]+'" />\n'
    return blr

def svgsave(blr,loc='outputs',filename='drawing'):
    save_path=createFolder(loc)
    f = open(save_path+'\\'+filename+'.svg', "w")
    f.write(blr)
    f.close()
    return save_path

def saveIndCon(controlPoints, con, styleguide = None, padding = 10):
    """
    This function will save the raster contours as well as the 
    
    :param controlPoints: Description
    """

    # defining a stylegyide if nothing is given
    if (styleguide is None):
        styleguide = {
            "thick-line": {
                "stroke": "black",
                "stroke-width": "5",
                "fill": "none"
                },
            "thin-line": {
                "stroke": "red",
                "stroke-width": "1",
                "fill": "none"
                }
        }
    # check styleguide conformity
    if ( not("thick-line" in styleguide.keys() or "thick-line" in styleguide.keys() )):
        raise ValueError("Styleguide does not have appropriate style classes")

    # setting the size of the SVG
    exX=max(con[:,0]) - min(con[:,0])
    exY=max(con[:,1]) - min(con[:,1])
    img_size = (exX + 2*padding , exY + 2*padding)

    # linear shift the contours and control points
    minX = min(con[:,0])
    minY = min(con[:,1])

    con[:,0] = con[:,0] - minX + padding
    con[:,1] = con[:,1] - minY + padding

    controlPoints[:,0] = controlPoints[:,0] - minX + padding
    controlPoints[:,1] = controlPoints[:,1] - minY + padding

    # write svg file
    blr = ''
    blr = svgfileinit(img_size)
    blr = svgaddstyletag(blr,styleguide)
    
    blr = svgdrawline(blr,con,"thick-line")
    blr = svgdrawcubicbezier(blr,controlPoints,"thin-line")
    blr = svgfileclose(blr)
    return blr

def saveConsVectorized(controlPoints):
    """
    This function should use the following
    svgfileinit
    svgaddstyletag
    svgdrawcubicbezier
    svgfileclose

    To output an SVG file with the proper vectorized shapes as a cubic bezier curve
    """
    pass

if(__name__=='__main__'):
    x = np.random.random(9)
    y = np.random.random(9)
    p=np.column_stack([x,y])

    stl = {"pa":{},"po":{}}
    svgsave(saveIndCon(p,p))


    print("Hi`")