'''
This submodule contains graphing utilities for visualizing contours
'''


import matplotlib.pyplot as plt
import programutils as putils



def plotconsHierarchy(consObj,mplcmap,suptitle):
    
    '''
    plots the 4 features of a hierarchy dataset onto the 
    '''
    
    fig, axs = plt.subplots(2, 2)
    textSizer=putils.norm01(consObj.area)
    
    
    for i in range(4):
        plt.subplot(2,2,i+1)

        cmap=mplcmap(consObj.hierarchy_norm[i])

        
        j=0
        for con in consObj.contours:
            plt.plot(con[:,0,0],con[:,0,1],color=cmap[j],marker='o',markersize=0.1)
            #plt.text(consObj.centeroid[i][0],consObj.centeroid[i][1],str(i),fontdict={'fontsize':textSizer*13+4})
            j+=1
        plt.title('hierarchy '+str(i)+' hierarchy feature')
    
    plt.suptitle(suptitle)
    
    return plt

def plotCon(img_size,con):
    # plots one contour
    plt.plot(con[:,0,0],-1*con[:,0,1],marker='o',markersize=0.1)
    plt.xlim(0,img_size[1])
    plt.ylim(-1*img_size[0],0)
    return plt
def plotCons():
    pass

def plotconsIndexes():
    pass

def plothist():
    pass