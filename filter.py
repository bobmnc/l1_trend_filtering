import numpy as np
from scipy.linalg import toeplitz
import cvxpy as cp


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



