import numpy as np
from datasets.univariate.create_signals import create_signals


def create_multivariate_signals(
    N: int,
    signal_length: int,
    D: int,
    max_slope: float,
    p_trend_change: float,
    noise_level: float):
    """create N signals (D dimensional) of size signal_length that have linear trend
    and changes trend every 1/(1-p) in mean,
    the dim every signal is D
    for every signal, we have one underlying "big" signal
    and on every dimension is just the big signal + a smaller signal

    output shape : (N, signal_length, D)
    """
    signals = np.zeros((N, signal_length, D))

    big_signal, breakpoints_list = create_signals(
        N, signal_length, max_slope, p_trend_change, noise_level
    )

    signals = np.tile(big_signal, (D, 1, 1))
    signals = np.transpose(signals, (1, 2, 0))

    for k in range(D):
        small_signal, small_bkps = create_signals(
            N, signal_length, max_slope, p_trend_change, noise_level
        )
        signals[:, :, k] += 0.3 * small_signal
    return signals, breakpoints_list