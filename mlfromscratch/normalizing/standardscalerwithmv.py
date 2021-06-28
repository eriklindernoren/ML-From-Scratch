from __future__ import print_function, division
import numpy as np

def StandardScalerAdv(x,mean,var):
    """
    Parameters :
    x : Dataset
    mean : your favorite mean for scaling features
    var : your favorite variance for scaling features

    """
    normalized = (x-mean)/np.sqrt(var)
