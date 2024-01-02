import numpy as np
from scipy.linalg import toeplitz
import cvxpy as cp
import scipy


## TO DO : compare with first implementation for speed
def l1_trend_filter2(signal,penalty):
    '''
    Compute univariate L1 trend filtering as done in
      https://web.stanford.edu/~gorin/papers/l1_trend_filter.pdf
    but with the A \theta formulation
    Compute min_X |signal - A \theta|^2 + penalty x |\theta|_1

   Args : 
        - signal (array): ndarray representing the signal of shape n
        with n the length of the signal
        - penalty (float) : penalty used for the l1 trend filtering

    Returns :
        - filtered (array) : ndarray representing the filtered signal
    '''
    n = len(signal)
    theta = cp.Variable(n)
    mat_A = compute_matrix_A(n)
    print(mat_A.shape)
    objective=  cp.Minimize(cp.sum_squares(mat_A@theta-signal)+penalty*cp.norm(theta[3:],1))

    prob =  cp.Problem(objective=objective)
    prob.solve(solver = cp.CLARABEL)
    return mat_A@theta.value


def compute_matrix_A(n):
    '''
    compute the matrix A described in paper 
        https://web.stanford.edu/~gorin/papers/l1_trend_filter.pdf
    
    Args :
        - n (int) : shape of the matrix A
    
    Returns:
        - A (array) : the matrix used for reconstructing the signal
        from theta
    '''
    A_ = np.diag(np.ones(n),0)
    for k in range(2,n):
        A_ += np.diag(k*np.ones(n-k+1),-k+1)
    A_[:,0] = np.ones(n)
    #A_ = scipy.sparse.dia_array(A_)

    return A_