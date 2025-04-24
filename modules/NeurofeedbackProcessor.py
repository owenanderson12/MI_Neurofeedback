import logging
import numpy as np
import time
from collections import deque
from scipy.signal import butter, filtfilt

from modules.config import *
from modules.calculate_band_power import calculate_band_power
from modules.fft_power_update import fft_power_update
from modules.bandpass_filter import bandpass_filter

class NeurofeedbackProcessor:
    """
    Processes EEG data for real-time neurofeedback.
    """
    def __init__(self, fs=SAMPLE_RATE):
        self.fs = fs
        self.window_size = int(WINDOW_LENGTH * fs)
        self.step_size = int(WINDOW_SHIFT * fs)
        
        # Initialize buffers as numpy arrays for better performance
        self.window_buffer = np.zeros(self.window_size)
        self.window_count = 0  # Track how many samples in the buffer
        
        # Pre-allocate baseline buffer
        self.baseline_buffer_size = int((BASELINE_END - BASELINE_START) * fs)
        self.baseline_buffer = np.zeros(self.baseline_buffer_size)
        self.baseline_count = 0
        
        # Initial baseline buffer (10 seconds)
        self.initial_baseline_buffer_size = int(INITIAL_BASELINE_DURATION * fs)
        self.initial_baseline_buffer = np.zeros(self.initial_baseline_buffer_size)
        self.initial_baseline_count = 0
        self.initial_baseline_power_mu = None
        self.initial_baseline_power_beta = None
        self.is_collecting_initial_baseline = False
        
        self.baseline_power_mu = None
        self.baseline_power_beta = None
        self.current_power_mu = None
        self.current_power_beta = None
        self.current_erd_mu = 0  # Default ERD (will be percentage change from baseline)
        self.current_erd_beta = 0  # Beta ERD
        self.max_observed_erd_mu = MAX_INITIAL_ERD  # Start with initial conservative value
        self.max_observed_erd_beta = MAX_INITIAL_ERD  # Same for beta
        self.is_collecting_baseline = False
        self.is_processing = False
        self.channel_index = 2  # Default to CH3 (index 2)
        self.active_hand = "right"  # Default hand
        
        # Smoothing variables for visual feedback
        self.smoothed_erd_mu = 0
        self.smoothed_erd_beta = 0
        
        # For FFT buffer optimization
        self.use_fft_buffer = USE_FFT_BUFFER
        self.fft_data = None
        
        # Filter coefficients (precompute for efficiency)
        nyq = 0.5 * fs
        low = FILTER_BAND[0] / nyq
        high = FILTER_BAND[1] / nyq
        self.filter_b, self.filter_a = butter(FILTER_ORDER, [low, high], btype='band')
        
        # Performance metrics
        self.process_times = deque(maxlen=100)  # Store processing times
        self.last_process_time = time.time()
        
    def set_active_hand(self, hand):
        """Set the active hand for the current trial."""
        self.active_hand = hand
        # Use contralateral hemisphere: left hand -> right hemisphere (CH6), right hand -> left hemisphere (CH3)
        if hand == "no_movement":
            # For no-movement trials, use both channels or a default channel
            self.channel_index = 2  # Default to CH3 (can be modified as needed)
            logging.info(f"No movement trial, using channel {EEG_CHANNELS[self.channel_index]} for baseline")
        else:
            self.channel_index = 5 if hand == "left" else 2  # CH6 (index 5) or CH3 (index 2)
            logging.info(f"Active hand set to {hand}, using channel {EEG_CHANNELS[self.channel_index]}")
        
    def start_initial_baseline_collection(self):
        """Start collecting data for initial baseline calculation."""
        self.initial_baseline_count = 0
        self.is_collecting_initial_baseline = True
        logging.info("Started initial baseline collection")
        
    def stop_initial_baseline_collection(self):
        """Stop initial baseline collection and calculate baseline power."""
        self.is_collecting_initial_baseline = False
        if self.initial_baseline_count > 0:
            # Only use the actual data collected
            baseline_signal = self.initial_baseline_buffer[:self.initial_baseline_count]
            # Use the bandpass_filter function for consistency with modular approach
            filtered_signal = bandpass_filter(baseline_signal, self.fs, FILTER_BAND[0], FILTER_BAND[1], FILTER_ORDER)
            self.initial_baseline_power_mu, _, _ = calculate_band_power(filtered_signal, self.fs, MU_BAND)
            self.initial_baseline_power_beta, _, _ = calculate_band_power(filtered_signal, self.fs, BETA_BAND)
            logging.info(f"Initial baseline calculated: mu={self.initial_baseline_power_mu}, beta={self.initial_baseline_power_beta}")
        else:
            logging.warning("No data collected for initial baseline")
            
    def start_baseline_collection(self):
        """Start collecting data for baseline calculation."""
        self.baseline_count = 0
        self.is_collecting_baseline = True
        logging.info("Started trial baseline collection")
        
    def stop_baseline_collection(self):
        """Stop baseline collection and calculate baseline power."""
        self.is_collecting_baseline = False
        if self.baseline_count > 0:
            # Only use the actual data collected
            baseline_signal = self.baseline_buffer[:self.baseline_count]
            # Use the bandpass_filter function for consistency with modular approach
            filtered_signal = bandpass_filter(baseline_signal, self.fs, FILTER_BAND[0], FILTER_BAND[1], FILTER_ORDER)
            
            # Calculate power for both frequency bands
            trial_baseline_power_mu, _, _ = calculate_band_power(filtered_signal, self.fs, MU_BAND)
            trial_baseline_power_beta, _, _ = calculate_band_power(filtered_signal, self.fs, BETA_BAND)
            
            logging.info(f"Trial baseline calculated: mu={trial_baseline_power_mu}, beta={trial_baseline_power_beta}")
            
            # Combine initial and trial baseline using weighted average for mu band
            if self.initial_baseline_power_mu is not None and trial_baseline_power_mu is not None:
                self.baseline_power_mu = (BASELINE_WEIGHT_INITIAL * self.initial_baseline_power_mu + 
                                      BASELINE_WEIGHT_TRIAL * trial_baseline_power_mu)
                logging.info(f"Combined mu baseline: {self.baseline_power_mu} "
                            f"(initial: {self.initial_baseline_power_mu}, trial: {trial_baseline_power_mu})")
            elif self.initial_baseline_power_mu is not None:
                self.baseline_power_mu = self.initial_baseline_power_mu
                logging.info(f"Using only initial mu baseline: {self.baseline_power_mu}")
            elif trial_baseline_power_mu is not None:
                self.baseline_power_mu = trial_baseline_power_mu
                logging.info(f"Using only trial mu baseline: {self.baseline_power_mu}")
            else:
                logging.warning("No mu baseline data available")
                
            # Combine initial and trial baseline for beta band
            if self.initial_baseline_power_beta is not None and trial_baseline_power_beta is not None:
                self.baseline_power_beta = (BASELINE_WEIGHT_INITIAL * self.initial_baseline_power_beta + 
                                        BASELINE_WEIGHT_TRIAL * trial_baseline_power_beta)
                logging.info(f"Combined beta baseline: {self.baseline_power_beta} "
                            f"(initial: {self.initial_baseline_power_beta}, trial: {trial_baseline_power_beta})")
            elif self.initial_baseline_power_beta is not None:
                self.baseline_power_beta = self.initial_baseline_power_beta
                logging.info(f"Using only initial beta baseline: {self.baseline_power_beta}")
            elif trial_baseline_power_beta is not None:
                self.baseline_power_beta = trial_baseline_power_beta
                logging.info(f"Using only trial beta baseline: {self.baseline_power_beta}")
            else:
                logging.warning("No beta baseline data available")
        else:
            logging.warning("No data collected for trial baseline")
            
    def start_processing(self):
        """Start real-time processing of EEG data."""
        self.window_count = 0
        self.is_processing = True
        self.fft_data = None
        self.max_observed_erd_mu = MAX_INITIAL_ERD  # Reset for new trial
        self.max_observed_erd_beta = MAX_INITIAL_ERD
        self.smoothed_erd_mu = 0
        self.smoothed_erd_beta = 0
        logging.info("Started real-time processing")
        
    def stop_processing(self):
        """Stop real-time processing."""
        self.is_processing = False
        # Calculate and log average processing time
        if self.process_times:
            avg_time = sum(self.process_times) / len(self.process_times)
            logging.info(f"Average sample processing time: {avg_time*1000:.2f} ms")
        logging.info("Stopped real-time processing")
        
    def process_sample(self, eeg_sample):
        """
        Process a new EEG sample.
        
        Parameters:
        -----------
        eeg_sample : list
            EEG data for all channels
            
        Returns:
        --------
        tuple : (mu_erd, beta_erd)
            Calculated ERD values (0 if not enough data or not processing)
        """
        start_time = time.time()
        channel_data = eeg_sample[self.channel_index]
        
        # Store sample in appropriate buffer
        if self.is_collecting_initial_baseline:
            if self.initial_baseline_count < len(self.initial_baseline_buffer):
                self.initial_baseline_buffer[self.initial_baseline_count] = channel_data
                self.initial_baseline_count += 1
            return 0, 0
            
        if self.is_collecting_baseline:
            if self.baseline_count < len(self.baseline_buffer):
                self.baseline_buffer[self.baseline_count] = channel_data
                self.baseline_count += 1
            return 0, 0
            
        if self.is_processing:
            # Update the circular buffer
            if self.window_count < self.window_size:
                # Still filling the buffer
                self.window_buffer[self.window_count] = channel_data
                self.window_count += 1
                
                # If buffer just filled, initialize FFT data
                if self.window_count == self.window_size and self.use_fft_buffer:
                    filtered_data = bandpass_filter(self.window_buffer, self.fs, FILTER_BAND[0], FILTER_BAND[1], FILTER_ORDER)
                    _, f, Pxx = calculate_band_power(filtered_data, self.fs, MU_BAND)
                    self.fft_data = (f, Pxx)
            else:
                # Shift buffer and add new sample (efficient numpy operation)
                old_sample = self.window_buffer[0]
                self.window_buffer = np.roll(self.window_buffer, -1)
                self.window_buffer[-1] = channel_data
                
                # Filter the data once for both frequency bands
                filtered_data = bandpass_filter(self.window_buffer, self.fs, FILTER_BAND[0], FILTER_BAND[1], FILTER_ORDER)
                
                # Calculate current power for both bands
                self.current_power_mu, _, _ = calculate_band_power(filtered_data, self.fs, MU_BAND)
                self.current_power_beta, _, _ = calculate_band_power(filtered_data, self.fs, BETA_BAND)
                
                # Calculate ERD for mu band
                if self.baseline_power_mu and self.baseline_power_mu > 0:
                    self.current_erd_mu = ((self.current_power_mu - self.baseline_power_mu) / self.baseline_power_mu) * 100
                    # Invert so that negative ERD (power decrease) corresponds to positive feedback
                    self.current_erd_mu = -self.current_erd_mu
                    
                    # Adapt maximum ERD if enabled
                    if ADAPTIVE_ERD_SCALING and self.current_erd_mu > 0:
                        if self.current_erd_mu > self.max_observed_erd_mu:
                            self.max_observed_erd_mu = self.current_erd_mu * ERD_SCALING_FACTOR
                            logging.debug(f"Adjusted max mu ERD to {self.max_observed_erd_mu}")
                
                # Calculate ERD for beta band
                if self.baseline_power_beta and self.baseline_power_beta > 0:
                    self.current_erd_beta = ((self.current_power_beta - self.baseline_power_beta) / self.baseline_power_beta) * 100
                    # Invert so that negative ERD (power decrease) corresponds to positive feedback
                    self.current_erd_beta = -self.current_erd_beta
                    
                    # Adapt maximum ERD if enabled
                    if ADAPTIVE_ERD_SCALING and self.current_erd_beta > 0:
                        if self.current_erd_beta > self.max_observed_erd_beta:
                            self.max_observed_erd_beta = self.current_erd_beta * ERD_SCALING_FACTOR
                            logging.debug(f"Adjusted max beta ERD to {self.max_observed_erd_beta}")
                
                # Apply smoothing for visual feedback
                normalized_mu = self.get_normalized_erd_mu()
                normalized_beta = self.get_normalized_erd_beta()
                
                # Apply emphasis to larger desynchronizations
                emphasized_mu = normalized_mu ** (1/ERD_EMPHASIS_FACTOR)  # Power < 1 emphasizes higher values
                emphasized_beta = normalized_beta ** (1/ERD_EMPHASIS_FACTOR)
                
                # Smooth the values for display
                self.smoothed_erd_mu = (SMOOTHING_FACTOR * self.smoothed_erd_mu + 
                                     (1-SMOOTHING_FACTOR) * emphasized_mu)
                self.smoothed_erd_beta = (SMOOTHING_FACTOR * self.smoothed_erd_beta + 
                                       (1-SMOOTHING_FACTOR) * emphasized_beta)
                
                # Record processing time
                process_duration = time.time() - start_time
                self.process_times.append(process_duration)
                
                return self.current_erd_mu, self.current_erd_beta
                
        return 0, 0
        
    def get_normalized_erd_mu(self):
        """Return the mu ERD value normalized to a 0-1 range based on adaptive maximum."""
        if self.current_erd_mu <= 0:
            return 0
        return min(self.current_erd_mu / self.max_observed_erd_mu, 1.0)
        
    def get_normalized_erd_beta(self):
        """Return the beta ERD value normalized to a 0-1 range based on adaptive maximum."""
        if self.current_erd_beta <= 0:
            return 0
        return min(self.current_erd_beta / self.max_observed_erd_beta, 1.0)
        
    def get_smoothed_erd_mu(self):
        """Return the smoothed mu ERD value for visual feedback."""
        return self.smoothed_erd_mu
        
    def get_smoothed_erd_beta(self):
        """Return the smoothed beta ERD value for visual feedback."""
        return self.smoothed_erd_beta