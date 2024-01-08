import numpy as np


def get_breakpoints(trend):
    """
    Takes in argument the output of l1_trend or l1_trend_multivariate of one signal
    and returns the breakpoints.
    """
    second_diff = np.convolve(trend, np.array([1,-2,1]), mode = 'valid')
    breakpoints = np.where(1 - np.isclose(second_diff, 0., atol=5e-04))[0]
    # get rid of consecutive breakpoints
    if len(breakpoints) == 0:
        breakpoints_list = []
    else:
        breakpoints_list = [breakpoints[0]]
        for i in range(1,len(breakpoints)):
            if breakpoints[i] != breakpoints[i-1] + 1:
                breakpoints_list.append(breakpoints[i])
                
    return breakpoints_list