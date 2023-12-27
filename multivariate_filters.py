import numpy as np
from scipy.linalg import toeplitz
from filter import compute_matrix_A
from create_tests import create_multivariate_signals
from tqdm import tqdm
from numba import jit
## formulation 1
## to do : numba compatible toeplitz
#@jit(nopython=False)
def m_focuss(signal : np.ndarray,penalty :float,n_iter :int = 20):
    n = signal.shape[0]
    dim = signal.shape[1]
    row = np.zeros(n)
    row[:3] = np.array([1,-2,1])
    col= np.zeros(n-2)
    col[0] = 1
    D = toeplitz(c=col,r=row)
    X = signal # we copy the signal at the beginning
    for k in tqdm(range(n_iter)):
        W = np.diag(np.sqrt(np.linalg.norm(D@X,
                                   axis=1)))
        W_inv_2 = np.clip(np.diag(1/np.linalg.norm(D@X,
                                   axis=1)),
                                   -10**6,
                                   10**6)
        filtering_matrix = (np.eye(n)+penalty*D.T@W_inv_2@D)
        filtering_matrix = np.linalg.pinv(filtering_matrix)
        filtered = filtering_matrix@signal

        X = filtered
    
    return X

def m_focuss_A(signal : np.ndarray,penalty :float,n_iter :int = 20):
    n = signal.shape[0]
    dim = signal.shape[1]
    A  = compute_matrix_A(n)
    theta  = np.clip(np.linalg.pinv(A)@signal,
                     -0.1,0.1) # we do some hard thresholding (test)
    P = np.eye(n-2,n,k=2) # projector in the space spanned by n-2 last vectors
    AtA = A.T@A
    Aty = A.T@signal

    for k in tqdm(range(n_iter)):
        W = np.diag(np.sqrt(np.linalg.norm(P@theta,
                                   axis=1)))
        W_inv_2 = np.clip(np.diag(1/np.linalg.norm(P@theta,
                                   axis=1)),
                                   -10**6,
                                   10**6)
        filtering_matrix = (AtA+penalty*P.T@W_inv_2@P)
        filtering_matrix = np.clip(np.linalg.pinv(filtering_matrix),
                                   -10**6,
                                   10**6)
        filtered = filtering_matrix@Aty
        theta = filtered
    
    return A@theta

if __name__=='__main__':
    signal_test = create_multivariate_signals(N =1,signal_length=1000,

                                    D =5, 
                                    max_slope=0.01,
                                    p_trend_change =0.2,
                                    noise_level=0.1,shift =100)

    filtered = m_focuss(signal_test[0],penalty=1)
    np.save('./data/signals_test_multivariate.npy',signal_test)
    np.save('./data/filtered_signals_test_multivariate.npy',filtered)