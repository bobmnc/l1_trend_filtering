import numpy as np
import cvxpy as cp


def l1_trend_filter_multivariate(signal, penalty):
    """
    l1_trend_filter_multivariate algorithm described in section 7.5
    """
    n, d = signal.shape
    x = cp.Variable((n, d))
    # convex minimizer
    left = cp.sum_squares(x - signal)
    right = 0
    for t in range(1, n - 1):
        right += cp.norm(x[t - 1] - 2 * x[t] + x[t + 1], 2)
    objective = cp.Minimize(left + 2*penalty * right)
    prob = cp.Problem(objective=objective)
    prob.solve(solver=cp.CLARABEL)
    return x.value