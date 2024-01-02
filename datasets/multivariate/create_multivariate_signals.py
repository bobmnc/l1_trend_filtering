import numpy as np


def create_multivariate_signals(N : int,signal_length : int,
                                D :int, 
                                max_slope : float,
                                p_trend_change : float,
                                noise_level:float,shift :int =50):
    '''create N signals of size signal_length that have linear trend 
    and changes trend every 1/(1-p) in mean,
    the dim of the signal is D 
    and every signal is just a shift of the 1st signal
    '''
    signals =  np.zeros((N,D,signal_length))
    
    first_signal = create_signals(N,signal_length,max_slope,
                                  p_trend_change,noise_level)
    for k in range(D):
        signals[:,k,:] = np.roll(first_signal,axis=1,
                                 shift=shift*k)
    signals = np.transpose(signals,(0,2,1))
    return signals