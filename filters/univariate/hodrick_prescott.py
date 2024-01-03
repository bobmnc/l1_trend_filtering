import numpy as np
import scipy
from scipy.linalg import toeplitz


def Hodrick_Prescott(signal :np.ndarray,penalty : np.ndarray):
    '''
    compute a Hodrick prescott filtering 
    compute min_X |X-signal|^2 + penalty x |DX|^2
    
    Args : 
        - signal (array): ndarray representing the signal of shape n
        with n the length of the signal
        - penalty (float) : penalty used for the filtering in HP filters

    Returns :
        - filtered (array) : ndarray representing the filtered signal with HP
    '''
    n = len(signal)
    row = np.zeros(n)
    row[:3] = np.array([1,-2,1])
    col= np.zeros(n-2)
    col[0] = 1
    D = toeplitz(c=col,r=row)
    filtering_matrix = np.eye(n)+penalty*D.transpose()@D
    filtering_matrix = np.linalg.inv(filtering_matrix)
    filtered = filtering_matrix@signal
    return filtered