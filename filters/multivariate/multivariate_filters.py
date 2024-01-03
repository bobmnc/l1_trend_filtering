import numpy as np
from scipy.linalg import toeplitz
from filters.univariate.filter import compute_matrix_A
from tqdm import tqdm

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
