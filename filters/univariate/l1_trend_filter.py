import numpy as np
from scipy.linalg import toeplitz
import cvxpy as cp
import scipy


def l1_trend_filter(signal,penalty):
    '''
    Compute univariate L1 trend filtering as done in
      https://web.stanford.edu/~gorin/papers/l1_trend_filter.pdf
    Compute min_X |signal - X|^2 + penalty x |DX|_1

   Args : 
        - signal (array): ndarray representing the signal of shape n
        with n the length of the signal
        - penalty (float) : penalty used for the l1 trend filtering

    Returns :
        - filtered (array) : ndarray representing the filtered signal
    '''
    n = len(signal)
    x = cp.Variable(n)
    row = np.zeros(n)
    row[:3] = np.array([1,-2,1])
    col= np.zeros(n-2)
    col[0] = 1
    D = toeplitz(c=col,r=row)
    objective=  cp.Minimize(cp.sum_squares(x-signal)+penalty*cp.norm(D@x,1))
    prob =  cp.Problem(objective=objective)
    prob.solve(solver = cp.CLARABEL)
    return x.value