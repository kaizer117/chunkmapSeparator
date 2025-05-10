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
    factorials=[1]*(n+1)
    
    for i in range(1,n+1):
        factorials[i]=factorials[i-1]*i
    
    # ith element will correspond to the ith factorial
    
    # columnwise bernstien basis calculation
    vandermonde_holder=[]
    for i in range(n+1):
        pass
        
    return None

if (__name__=="__main__"):
    
    
    # styleguide={'Datapoint':{'fill':'#b0303c','fill-opacity':'1','stroke':'none',
    #                         'stroke-width':'none','stroke-linejoin':'round','stroke-dasharray':'none','stroke-opacity':'1'},
    #             'Connectlinedash':{'opacity':'1','fill':'none','fill-opacity':'1','stroke':'#9e9e9e','stroke-width':'1',
    #                            'stroke-linejoin':'round','stroke-dasharray':'1.20000005,0.30000001','stroke-opacity':'1','stroke-dashoffset':'0'}
    #             }
    
    
    # datapoints=generateDatapoints('one_dimension_justified')
    # print(sampleT(datapoints))
    # print(sampleT(datapoints,method='squared_distance'))
    # print(sampleT(datapoints,method='unsquared_sum'))
    # blr=f.svgfileinit((200,200))
    # blr=f.svgaddstyletag(blr,styleguide)
    # for i in range(len(datapoints)-1):
    #     blr=f.svgdrawline(blr,datapoints[i],datapoints[i+1],'Connectlinedash')
    # for c in datapoints:
    #     blr=f.svgdrawcircle(blr,c,2,'Datapoint')
    # blr=f.svgfileclose(blr)
    # print(blr)
    # print(f.svgsave(blr))
    
    
    vandermonde([0]*5)
    print('Hi')