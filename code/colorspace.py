import fileutils as futils
import numpy as np



def rgb2hex(rgb):
    '''
    rgb values: [0,1]
    '''
    r,g,b=round(rgb[0]*255),round(rgb[1]*255),round(rgb[2]*255) # force converting to integer
    s=str(hex((r<<(4*4))+(g<<(4*2))+b))[2:]
    return '#'+'0'*(6-len(s))+s

def hsl2rgb(hsl):
    '''
    h: degrees [0, 360]
    s: saturation [0,1]
    l: lightness [0,1]
    '''
    h,s,l=hsl[0],hsl[1],hsl[2]
    
    C=(1-np.abs(2*l-1))*s
    Hdash=h/60
    X=C*(1-np.abs((Hdash%2)-1))
    
    if (Hdash<1):
        r1=C
        g1=X
        b1=0
    elif(Hdash<2):
        r1=X
        g1=C
        b1=0
    elif(Hdash<3):
        r1=0
        g1=C
        b1=X
    elif(Hdash<4):
        r1=0
        g1=X
        b1=C
    elif(Hdash<5):
        r1=X
        g1=0
        b1=C
    else:
        r1=C
        g1=0
        b1=X
    
    m=l-(C/2)
    
    return (r1+m,g1+m,b1+m)

def lightnessIndex(d):
    pass

def populatePalette(n,sTolerance,lTolerance):
    pass



if (__name__=="__main__"):
    
    # save_path=futils.createFolder('outputs')

    # f=open(save_path+'\\colorspaces.svg','w')

    print('Hi')