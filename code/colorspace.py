import fileutils as futils
import numpy as np
'''
Generating color palettes
'''

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
    
    # what used to be in the code block below was a bunch of if statements
    # refer to \resources\hsl_kmaps.xlsx to understand the working behind the code below
    # refer from row 58 onwards
    
    lTmp=[0,C,X]
    lTmpindex=[1,2,0,0,2,1]
    dt= int(Hdash)
    
    r1,g1,b1=lTmp[lTmpindex[(dt%6)]],lTmp[lTmpindex[((dt+2)%6)]],lTmp[lTmpindex[((dt+4)%6)]]
    
    # end of what used to be if statements, what continues is normal calculation
    m=l-(C/2)
    
    return (r1+m,g1+m,b1+m)

def hsl2hex(hsl):
    return rgb2hex(hsl2rgb(hsl))


def populatePalette(h,n,sampling='linear',picking='linear'):
    '''
    This function takes a hue value and populated a color palatte with adjacent values
    '''
    m=12         #for each 7 colours, new hue value
    hN=n//m
    hVar=4 #degrees
    
    
    sClampParams=[0.3,0.8,0.1] # start end var
    lClampParams=[0.3,0.8,0.1] # start end var
    
    
    
    
    #create s and l value min and max
    hClamp=[h-(hN/2)*hVar,h+(hN/2)*hVar]
    sClamp=[sClampParams[0]+sClampParams[2]*np.random.rand(),sClampParams[1]-sClampParams[2]*np.random.rand()]
    lClamp=[lClampParams[0]+lClampParams[2]*np.random.rand(),lClampParams[1]-lClampParams[2]*np.random.rand()]
    
    #generate m number of s and l values
    
    if (sampling=='linear'):
        # generate h s l values within the clamp values linearly
        hVals=np.linspace(hClamp[0],hClamp[1],hN+2) #    <----h---->
        sVals=np.linspace(sClamp[0],sClamp[1],m)
        lVals=np.linspace(lClamp[0],lClamp[1],m)
    elif (sampling=='random'):
        hVals=np.random.rand(hN+2)*(hClamp[1]-hClamp[0])+hClamp[0]
        sVals=np.random.rand(m)*(sClamp[1]-sClamp[0])+sClamp[0]
        lVals=np.random.rand(m)*(lClamp[1]-lClamp[0])+lClamp[0]
        
    elif (sampling=='random_justified'):
        pass
    
    #ensuring boundary conformance
    for i,v in enumerate(hVals):
        if (v>360):
            hVals[i]=hVals[i]-360
            continue
        if (v<0):
            hVals[i]=hVals[i]+360
            continue
    
    #if n is not divisible by m, dealing with the remainder
    cmap=[]
    r=n%m
    if (r):
        r0=np.random.randint(r)
        r1=r-r0
        if (picking=='linear'):
            cmap.extend(list(map(lambda i: [hVals[0],sVals[np.random.randint(m)],sVals[np.random.randint(m)]],range(r0))))
            cmap.extend(list(map(lambda i: [hVals[-1],sVals[np.random.randint(m)],sVals[np.random.randint(m)]],range(r1))))
            
        elif (picking=='shuffle'):
            np.random.shuffle(hVals)
            np.random.shuffle(sVals)
            np.random.shuffle(lVals)
            cmap.extend(list(map(lambda i: [hVals[0],sVals[np.random.randint(m)],sVals[np.random.randint(m)]],range(r0))))
            cmap.extend(list(map(lambda i: [hVals[-1],sVals[np.random.randint(m)],sVals[np.random.randint(m)]],range(r1))))
    
    # populating cmap
    if (picking=='linear'):
        for i in range(1,hN+1):
            cmap.extend(list(map(lambda j: [hVals[i],sVals[j],lVals[j]],range(m))))
    elif (picking=='shuffle'):
        for i in range(1,hN+1):
            np.random.shuffle(sVals)
            np.random.shuffle(lVals)
            cmap.extend(list(map(lambda j: [hVals[i],sVals[j],lVals[j]],range(m))))
    return cmap

def lightnessIndex(d):
    pass

if (__name__=="__main__"):
    
    save_path=futils.createFolder('outputs')
    n=143
    n+=1
    for hue in range(360):    
        
        cmap=populatePalette(hue,n,sampling='random',picking='shuffle')
        cmap.append([hue,0.5,0.5])
        
        b=np.round(np.sqrt(n))
        w=b-int(b*0.1)
        h=(n//w)+((n%w)!=0)*1
        
        f=open(save_path+'\\'+str(hue)+'.svg','w')
        
        s=''
        s+='<?xml version="1.0" standalone="no"?>\n'
        s+='<svg width="'+str(w*50)+'px"'
        s+=' height="'+str(h*50)+'px"'
        s+=' viewbox="'+'0'+' '+'0'+' '+str(w*50)+' '+str(h*50)+'"'
        s+=' xmlns="http://www.w3.org/2000/svg" version="1.1">\n'
        
        for i in range(n):
            s+='<rect'
            s+=' style="fill:'+hsl2hex(cmap[i])+';stroke-width:1;stroke-linejoin:round;fill-opacity:1;stroke:none"'
            s+=' width="50"'
            s+=' height="50"'
            s+=' x="'+str((i%w)*50)+'"'
            s+=' y="'+str((i//w)*50)+'" />'
            s+='\n'
        s+='\n</svg>'
        
        f.write(s)
        f.close()
    print('Hi')