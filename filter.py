import numpy as np
from scipy.linalg import toeplitz
import cvxpy as cp
import scipy


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

if __name__=='__main__':
    A_ = compute_matrix_A(4)
    X = np.random.randn(10,10)

    test = l1_filtering_spatial(X)
    print('test',test)

    #l1_trend_filter2(np.load('./data/signals_test.npy')[0],penalty=10) 





