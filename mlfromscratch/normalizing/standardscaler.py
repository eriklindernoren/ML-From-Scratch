from __future__ import print_function, division
import numpy as np

def StandardScaler(x,standard=True):
    """
    Parameters :
    x : Dataset
    standard : True or False (True = setting the mean of 0 and var of 1, False = computing mean and var of data)

    """
    if standard == True:
        normalized = (x-0)/np.sqrt(1)
        
    else:
        normalized = (x-np.mean(x))/np.std(x)
    return normalized
