import numpy as np

def create_signals(N : int,signal_length : int,max_slope : float,p_trend_change : float,noise_level:float):
    '''create N signals of size signal_length that have linear trend 
    and changes trend every 1/(1-p) in mean

    outputs the generated signal and the breakpoints
    '''
    signals =  np.zeros((N,signal_length))
    breakpoints_list = []
    ## moments where we change the slope
    change_slope = np.random.random((N,signal_length))
    change_slope = (change_slope>=p_trend_change)
    for k in range(N):
        breakpoints = np.where(change_slope[k])
        breakpoints_list.append(breakpoints)
        signals[k][0] = np.random.randn()
        slope = (np.random.random()-.5)*2*max_slope
        for j in range(1,signal_length):
            if not change_slope[k,j]:
                signals[k,j] = signals[k,j-1]+slope
            else:
                slope = (np.random.random()-.5)*2*max_slope
                signals[k,j] = signals[k,j-1]+slope
    return signals, breakpoints_list