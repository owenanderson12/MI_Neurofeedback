from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    """
    Apply a bandpass filter to the EEG signal.
    
    Parameters:
    -----------
    signal : array-like
        The EEG signal data
    fs : float
        Sampling rate in Hz
    lowcut : float
        Lower cutoff frequency in Hz
    highcut : float
        Upper cutoff frequency in Hz
    order : int
        Filter order
        
    Returns:
    --------
    filtered_signal : array-like
        Filtered EEG signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal