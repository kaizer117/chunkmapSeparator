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
            p=p[0] # bypassing the extra dimension
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

def saveConsDepricated(img_size,cons):
    save_path=createFolder('outputs')
    
    for i,con in enumerate(cons):
        saveConDepricated(con,img_size,save_path+'\\'+str(i)+'.svg')
    return None

def saveConDepricated(ex,img_size,file_name,fill_color='#000000'):
    '''
    save each contour as one svg
    '''
    s=''
    for i,p in enumerate(ex):
        p=p[0] # bypassing the extra dimension
        if (i==1):
            s+='L '+str(p[0])+' '+str(p[1])+' '
            continue
        s+=str(p[0])+' '+str(p[1])+' '

    s='M '+s+'z'

    f = open(file_name, "w")


    blr=''
    blr+='<?xml version="1.0" standalone="no"?>\n'


    blr+='<svg width="'+str(img_size[1])+'px"'
    blr+=' height="'+str(img_size[0])+'px"'
    blr+=' viewbox="'+'0'+' '+'0'+' '+str(img_size[1])+' '+str(img_size[0])+'"'

    blr+=' xmlns="http://www.w3.org/2000/svg" version="1.1">\n'

    blr+='<path\nstyle="fill:'+fill_color+';fill-opacity:0.75;stroke:none;stroke-width:0.52916667;stroke-dasharray:none"\nd="'
    
    
    blr+=s+'"\n/>'
    blr+='\n</svg>'

    f.write(blr)

    f.close()
    return None

def svgfileinit(img_size):
    '''
    This function will generate the boiler plate stuff of the svg
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
    blr+='<path class="'+stl+'" d="M '
    
    for point in p:
        blr+=str(point[0])+','+str(point[1])+' '
        
    blr=blr[:-1]+'" />\n'
    return blr

def svgdrawcubicbezier(blr,c,stl):
    blr+='<path class="'+stl+'" d="M '+str(c[0][0])+','+str(c[0][1])+' C '
    
    for point in c[1:]:
        blr+=str(point[0])+','+str(point[1])+' '
    blr=blr[:-1]+'" />\n'
    return blr

def svgdrawclosedcubicbezier(blr,c,stl):
    blr+='<path class="'+stl+'" d="M '+str(c[0][0])+','+str(c[0][1])+' C '
    
    for point in c[1:]:
        blr+=str(point[0])+','+str(point[1])+' '
    blr=blr[:-1]+'" Z/>\n'
    return blr

def svgsave(blr,loc='outputs',filename='drawing'):
    save_path=createFolder(loc)
    f = open(save_path+'\\'+filename+'.svg', "w")
    f.write(blr)
    f.close()
    return save_path

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
def newSession():
    '''
    I have forgotten what this is supposed to be.
    '''
    pass

if(__name__=='__main__'):
    # d={'Border':{ 'fill':'none', 'stroke':'blue', 'stroke-width':'1' },
    #    'Connect': { 'fill':'none', 'stroke':'#888888', 'stroke-width':'2' }}
    # print(svgaddstyletag('',d))
    
    print(svgdrawcircle('',[2,3],0.5,'datapoint'))
    print(svgdrawline('',[[1,2],[3,4]],'line'))
    print(svgdrawcubicbezier('',[[1,2],[3,4],[4,5],[5,6]],'the'))
    print('end')