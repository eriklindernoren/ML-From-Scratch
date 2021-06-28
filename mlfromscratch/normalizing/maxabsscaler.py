from __future__ import print_function, division
import numpy as np

def MaxAbsScaler(x):
    """
    Parameters :
    x : Dataset
    
    """
    normalized = x/np.max(abs(x))
    return normalized
