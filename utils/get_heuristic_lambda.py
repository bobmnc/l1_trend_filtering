import numpy as np


def get_moving_average(signal, window_size):
    moving_average = np.convolve(signal, np.ones(window_size), 'same') / window_size
    return moving_average

def get_Dx(trend):
    Dx = np.convolve(trend, np.array([1,-2,1]), mode = 'valid')
    Dx = np.abs(Dx).sum()
    return Dx

def get_reconstruction_error(trend, signal):
    reconstruction_error = np.sum((trend - signal)**2)
    return reconstruction_error


def get_heuristic_lambda(signal):
    """
    Gives a heuristic to choose a default value of lambda (in the l1 trend filtering)
    to start with
    """
    trend = get_moving_average(signal, window_size = len(signal) // 5)
    heuristic_lambda = 15 * get_reconstruction_error(trend, signal) / (2*get_Dx(trend))
    return heuristic_lambda