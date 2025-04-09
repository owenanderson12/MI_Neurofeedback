import numpy as np

def fft_power_update(prev_fft_data, new_sample, old_sample, fs, freq_band):
    """
    Efficient update of power spectrum when a new sample is added and old sample removed.
    This is more efficient than recomputing the entire FFT for each new sample.
    
    Parameters:
    -----------
    prev_fft_data : tuple
        (frequencies, power spectrum) from previous calculation
    new_sample : float
        The new sample being added to the window
    old_sample : float
        The old sample being removed from the window
    fs : float
        Sampling rate in Hz
    freq_band : tuple
        Frequency band (low, high) in Hz
    
    Returns:
    --------
    band_power : float
        Updated power in the specified frequency band
    updated_fft_data : tuple
        Updated (frequencies, power spectrum)
    """
    f, Pxx = prev_fft_data
    
    # Simple implementation for demo - in practice, this would use a more
    # sophisticated algorithm for incremental FFT updates
    # This is a placeholder for the concept
    
    # Get the indices corresponding to the frequency band
    idx_band = np.logical_and(f >= freq_band[0], f <= freq_band[1])
    
    # Calculate mean power in the band
    band_power = np.mean(Pxx[idx_band]) if np.any(idx_band) else 0
    
    return band_power, (f, Pxx)