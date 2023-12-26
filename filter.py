import numpy as np
from scipy.linalg import toeplitz
import cvxpy as cp
import scipy


def Hodrick_Prescott(signal,penalty):
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

def l1_trend_filter2(signal,penalty):
    n = len(signal)
    theta = cp.Variable(n)
    mat_A = compute_matrix_A(n)
    print(mat_A.shape)
    objective=  cp.Minimize(cp.sum_squares(mat_A@theta-signal)+penalty*cp.norm(theta[3:],1))

    prob =  cp.Problem(objective=objective)
    prob.solve(solver = cp.CLARABEL)
    return mat_A@theta.value

def compute_matrix_A(n):
    A_ = np.diag(np.ones(n),0)
    for k in range(2,n):
        A_ += np.diag(k*np.ones(n-k+1),-k+1)
    A_[:,0] = np.ones(n)
    #A_ = scipy.sparse.dia_array(A_)

    return A_


A_ = compute_matrix_A(4)

l1_trend_filter2(np.load('./data/signals_test.npy')[0],penalty=10) 





