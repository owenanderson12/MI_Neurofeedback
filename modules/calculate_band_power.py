import numpy as np
from scipy.signal import welch

def calculate_band_power(signal, fs, freq_band):
    """
    Calculate power in a specific frequency band using Welch's method.
    
    Parameters:
    -----------
    signal : array-like
        The EEG signal data
    fs : float
        Sampling rate in Hz
    freq_band : tuple
        Frequency band (low, high) in Hz
        
    Returns:
    --------
    band_power : float
        Power in the specified frequency band
    f : array-like
        Frequency array
    Pxx : array-like
        Power spectral density
    """
    # Calculate power spectral density using Welch's method
    # Use nperseg that's appropriate for the signal length
    nperseg = min(len(signal), fs)  # Use 1s window or smaller if signal is shorter
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
    
    # Get the indices corresponding to the frequency band
    idx_band = np.logical_and(f >= freq_band[0], f <= freq_band[1])
    
    # Calculate mean power in the band
    band_power = np.mean(Pxx[idx_band]) if np.any(idx_band) else 0
    
    return band_power, f, Pxx