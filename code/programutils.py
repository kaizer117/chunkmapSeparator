import numpy as np

def norm01(unnormalized):
    return (unnormalized-np.min(unnormalized))/(np.max(unnormalized)-np.min(unnormalized))