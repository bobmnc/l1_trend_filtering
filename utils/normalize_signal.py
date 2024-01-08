def normalize_signal(signal):
    """
    Normalizes a univariate or multivariate signal 
    (centered and unit variance signal)
    """
    # N, d = signal.shape

    normalized_signal = signal
    normalized_signal -= signal.mean(axis=0)
    normalized_signal /= signal.std(axis=0)

    return normalized_signal