'''
This submodule contains graphing utilities for visualizing contours
'''


import matplotlib.pyplot as plt
import programutils as putils



def plotconsHierarchy(consObj,mplcmap,suptitle):
    
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
    plt.show()
    return None

def plotconsIndexes():
    pass

def plothist():
    pass