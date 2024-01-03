
import numpy as np
from scipy.linalg import toeplitz
import cvxpy as cp
import scipy


### only grey level is supported for the moment
def l1_filtering_spatial(img : np.ndarray, penalty : float):
    '''
    Implementation of the algorithm described in 7.6 of 
        https://web.stanford.edu/~gorin/papers/l1_trend_filter.pdf
    Function that implement spatial trend filtering on 2d array
    of shape n,m, only 2d array with one channel are supported 
    for the moment
    

    Args : 
        - img (array) : 2d array of shape (n,m) to filter spatially
        - penalty for the l1 spatial trend filtering
    
    Returns :
        - filtered (array) : A 2d filtered array of the same shape
        as img, which is a filtered version of img


    '''
    n,m = img.shape # to add C channels 
    X = cp.Variable((n,m))
    cost = cp.sum_squares(X-img)
    for i in range(1,n-1):
        for j in range(1,m-1):
            cost += cp.pnorm(cp.hstack([X[i,j-1]-2*X[i,j]+X[i,j+1],X[i-1,j] -2*X[i,j]+X[i+1,j]]),2)
            
    objective =  cp.Minimize(cost)

    prob =  cp.Problem(objective=objective)
    prob.solve(solver = cp.CLARABEL)
    return X.value