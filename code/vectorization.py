import numpy as np
import fileutils as f

def generateDatapoints(method='true_random',n=4):
    if (method=='true_random'):
        return np.column_stack((np.round(np.random.rand(n)*200,7),np.round(np.random.rand(n)*200,7)))
    elif(method=='one_dimension_justified'):
        '''
        Generates one row (x or y) first.
        Algorithm makes sure that they are generated in one direction are not back stepping.
        
        '''
        
        p1=np.sort(np.random.rand(n)*200)
        p2=np.random.rand(n)*200
        
        t=np.random.rand()
        
        if(t>0.5):
            return np.column_stack((np.round(p1,7),np.round(p2,7)))
        else:
            return np.column_stack((np.round(p2,7),np.round(p1,7)))
    
    
def sampleT(datapoints,method='square_root_distance'):
    array=[]
    if (method=='square_root_distance'):
        for i in range(len(datapoints)-1):
            array.append(np.sqrt((datapoints[i][0]-datapoints[i+1][0])**2+(datapoints[i][1]-datapoints[i+1][1])**2))
    elif (method=='squared_distance'):
        for i in range(len(datapoints)-1):
            array.append((datapoints[i][0]-datapoints[i+1][0])**2+(datapoints[i][1]-datapoints[i+1][1])**2)
    elif (method=='unsquared_sum'):
        for i in range(len(datapoints)-1):
            array.append(np.abs(datapoints[i][0]-datapoints[i+1][0])+np.abs(datapoints[i][1]-datapoints[i+1][1]))
    
    return np.insert(np.cumsum(array),0,0)/np.sum(array)

def vandermonde(ts):
    n=len(ts)
    # calculate and store factorials
    # precomputes the factorials instead of recalculating every cycle of the loop
    factorials=[1]*(n+1)
    
    for i in range(1,n+1):
        factorials[i]=factorials[i-1]*i
    
    # ith element will correspond to the ith factorial
    
    # columnwise bernstien basis calculation
    vandermonde_holder=[]
    for i in range(n):
        binomial=factorials[n]/(factorials[i]*factorials[n-i])
        vandermonde_holder.append(list(map(lambda t: binomial*t**i*(1-t)**(n-i),ts)))
        
        
    return np.array(vandermonde_holder)

def controlpointcalc(v,d):
    pass

if (__name__=="__main__"):
    
    
    styleguide={'Datapoint':{'fill':'#b0303c','fill-opacity':'1','stroke':'none',
                            'stroke-width':'none','stroke-linejoin':'round','stroke-dasharray':'none','stroke-opacity':'1'},
                'Connectlinedash':{'opacity':'1','fill':'none','fill-opacity':'1','stroke':'#9e9e9e','stroke-width':'1',
                               'stroke-linejoin':'round','stroke-dasharray':'1.20000005,0.30000001','stroke-opacity':'1','stroke-dashoffset':'0'},
                'Controlpoint':{'fill':'#b0303c','fill-opacity':'1','stroke':'none','stroke-width':'none',
                                'stroke-linejoin':'round','stroke-dasharray':'none','stroke-opacity':'1'},
                'Handles':{'opacity':'1','fill':'none','fill-opacity':'1','stroke':'#9e9e9e','stroke-width':'1',
                           'stroke-linejoin':'round','stroke-dasharray':'none','stroke-opacity':'1'},
                'Curve':{'opacity':'1','fill':'none','fill-opacity':'1','stroke':'#000000','stroke-width':'1',
                           'stroke-linejoin':'round','stroke-dasharray':'none','stroke-opacity':'1'}
                }
    
    
    # datapoints=generateDatapoints('one_dimension_justified')
    # print(sampleT(datapoints))
    # print(sampleT(datapoints,method='squared_distance'))
    # print(sampleT(datapoints,method='unsquared_sum'))
    
    
    #generating datapoints

    datapoints=generateDatapoints('one_dimension_justified')
    
    #creating two dummy points
    
    def dummypoint(p0,k=10):
        return [p0[0]+(-1)**np.random.randint(1,3)*np.random.rand()*k,p0[1]+(-1)**np.random.randint(1,3)*np.random.rand()*k]
    
    datapoints=np.vstack((datapoints,dummypoint(datapoints[-1])))
    datapoints=np.vstack((dummypoint(datapoints[0]),datapoints))
    ts=sampleT(datapoints,method='square_root_distance')
    
    # a small correction factor so that the vandermonde matrix is invertible
    correction=1e-16
    ts[0]=ts[0]+correction
    ts[-1]=ts[-1]-correction
    
    # creating vandermonde matrix
    
    v=vandermonde(ts)
    
    A_1=np.linalg.inv(v)
    c=np.matmul(A_1,datapoints)
    c[0]=datapoints[0]
    c[-1]=datapoints[-1]
    print("stop")
    #
    
    blr=f.svgfileinit((200,200))
    blr=f.svgaddstyletag(blr,styleguide)
    blr=f.svgdrawline(blr,datapoints,'Connectlinedash')
    blr=f.svgdrawcubicbezier(blr,c,'Curve')
    
    blr=f.svgdrawline(blr,c[0:2],'Handles')
    blr=f.svgdrawline(blr,c[2:],'Handles')
    
    for cr in c[1:-1]:
        blr=f.svgdrawcircle(blr,cr,2,'Controlpoint')
    
    for cr in datapoints:
        blr=f.svgdrawcircle(blr,cr,2,'Datapoint')
    blr=f.svgfileclose(blr)
    print(f.svgsave(blr))
    
    print('Hi')