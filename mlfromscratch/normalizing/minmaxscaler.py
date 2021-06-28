from __future__ import print_function, division
import numpy as np

def MinMaxScaler(x,minmax):
    """
    Parameters :
    x : Dataset
    minmax : the minimum and maximum range of your features

    """
    normalized = (x-np.min(x))/(np.max(x)-np.min(x))
    scale = (normalized * (np.max(minmax)-np.min(minmax))) + np.min(minmax)
    return scale
