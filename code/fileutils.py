'''
This submodule will handle creating folders, checking existence of 
folders and output version management
'''
import os
import datetime
import cv2 as cv

import numpy as np
import matplotlib.pyplot as plt

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

def saveCons(img_size,cons):
    save_path=createFolder('outputs')
    
    for i,con in enumerate(cons):
        saveCon(con,img_size,save_path+'\\'+str(i)+'.svg')
    return None

def saveCon(ex,img_size,file_name,fill_color='#000000'):
    '''
    save contour as a svg
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

def newSession():
    pass

if(__name__=='__main__'):
    print('hallo')
    x=np.linspace(-4,4,100)
    y=np.sin(x)
    plt.plot(x,y)
    savePlot(plt,'outputs','good.png',True)
    plt.show()
    
    
    print('end')