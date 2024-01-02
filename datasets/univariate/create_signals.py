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
    change_slope = (change_slope<=p_trend_change)
    # we remove breakpoints that are too close to each other
    change_slope = remove_close_true(change_slope, min_distance= signal_length / 10)
    for k in range(N):
        breakpoints = np.where(change_slope[k])[0]
        breakpoints = np.append(breakpoints, signal_length)
        breakpoints_list.append(breakpoints)
        signals[k][0] = np.random.randn()
        slope = (np.random.random()-.5)*2*max_slope # initial slope
        for j in range(1,signal_length):
            if not change_slope[k,j]:
                signals[k,j] = signals[k,j-1]+slope
            else:
                slope = (np.random.random()-.5)*2*max_slope
                signals[k,j] = signals[k,j-1]+slope

    # adding noise
    noise = noise_level * np.random.randn(N,signal_length)
    signals += noise

    return signals, breakpoints_list



def remove_close_true(arr, min_distance=2):
    N = arr.shape[0]
    for row in range(N):
        indices_true = np.where(arr[row])[0]
        last_true_index = indices_true[0]
        
        for i in range(1, len(indices_true)):
            if (indices_true[i] - last_true_index < min_distance):
                arr[row, indices_true[i]] = False
            else:
                last_true_index = indices_true[i]
            
    return arr