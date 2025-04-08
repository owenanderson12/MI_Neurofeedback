#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neurofeedback_BCI.py - Real-time EEG Motor Imagery Neurofeedback System

Extended from motor_imagery_eeg.py, this script implements a motor imagery neurofeedback
system using a sliding window approach for real-time bandpower calculation.

Features:
  - Real-time Mu-band (8-12 Hz) power calculation with sliding window
  - Baseline normalization for ERD calculation
  - Dynamic visual feedback bar representing ERD strength
  - Continuous updating of feedback every 50ms

Requirements:
  pip install pylsl psychopy pandas numpy scipy
"""

import threading
import time
import random
import csv
import os
import sys
import numpy as np
from collections import deque
from datetime import datetime
import logging
from scipy.signal import welch, butter, filtfilt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- Configuration Parameters ---
EEG_STREAM_NAME = "OpenBCI_EEG"
MARKER_STREAM_NAME = "MI_MarkerStream"

SAMPLE_RATE = 250  # Hz
EEG_CHANNELS = ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8"]

# Motor Imagery Experiment Parameters
NUM_TRIALS = 40  # Configurable number of trials
INSTRUCTION_DURATION = 2.0  # seconds to show left/right instruction
IMAGERY_DURATION = 5.0  # seconds for motor imagery
INTER_TRIAL_INTERVAL = 3.0  # seconds between trials

# Marker values
MARKER_RIGHT = "1"  # right hand imagery
MARKER_LEFT = "2"   # left hand imagery
MARKER_STOP = "3"   # end of imagery period

# File/directories
RAW_DATA_DIR = "data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Data collection parameters
MERGE_THRESHOLD = 0.002  # seconds threshold for aligning EEG and marker timestamps
POLL_SLEEP = 0.001      # sleep time between polls in collector loop

# Neurofeedback Parameters
MU_BAND = (8, 12)       # Mu frequency band (Hz)
BETA_BAND = (13, 30)    # Beta frequency band (Hz)
WINDOW_LENGTH = 1.0     # Window length for bandpower calculation (seconds)
WINDOW_OVERLAP = 0.95   # Window overlap (seconds)
WINDOW_SHIFT = 0.05     # Window shift (seconds)
BASELINE_START = -3.0   # Start time for baseline period (seconds relative to imagery start)
BASELINE_END = -1.0     # End time for baseline period (seconds relative to imagery start)
INITIAL_BASELINE_DURATION = 10.0  # Duration of initial baseline period in seconds
BASELINE_WEIGHT_INITIAL = 0.7     # Weight for initial baseline (0-1)
BASELINE_WEIGHT_TRIAL = 0.3       # Weight for per-trial baseline (0-1)
DISPLAY_REFRESH_RATE = 60  # Hz, target refresh rate for display updates
DISPLAY_UPDATE_INTERVAL = 1.0/DISPLAY_REFRESH_RATE  # seconds between display updates
FEEDBACK_BAR_MAX_WIDTH = 1.2  # Maximum width of the feedback bar (normalized units)

# Visual feedback parameters
SMOOTHING_FACTOR = 0.8  # Weight given to previous value (0-1) for smooth transitions
ERD_EMPHASIS_FACTOR = 1.5  # Emphasis factor for larger desynchronizations

# Real-time processing optimization
USE_FFT_BUFFER = True   # Use rolling FFT update instead of recomputing full FFT
ADAPTIVE_ERD_SCALING = True  # Adapt ERD scaling based on subject performance
MAX_INITIAL_ERD = 50    # Initial maximum expected ERD value (%)
ERD_SCALING_FACTOR = 1.2  # Factor for adjusting max ERD based on performance

# Bandpass filter parameters for preprocessing
FILTER_ORDER = 4
FILTER_BAND = (1, 45)   # Hz

###############################################################################
#                          SIGNAL PROCESSING FUNCTIONS
###############################################################################
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

###############################################################################
#                         NEUROFEEDBACK PROCESSOR
###############################################################################
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
            filtered_signal = filtfilt(self.filter_b, self.filter_a, baseline_signal)
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
            filtered_signal = filtfilt(self.filter_b, self.filter_a, baseline_signal)
            
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
                    filtered_data = filtfilt(self.filter_b, self.filter_a, self.window_buffer)
                    _, f, Pxx = calculate_band_power(filtered_data, self.fs, MU_BAND)
                    self.fft_data = (f, Pxx)
            else:
                # Shift buffer and add new sample (efficient numpy operation)
                old_sample = self.window_buffer[0]
                self.window_buffer = np.roll(self.window_buffer, -1)
                self.window_buffer[-1] = channel_data
                
                # Filter the data once for both frequency bands
                filtered_data = filtfilt(self.filter_b, self.filter_a, self.window_buffer)
                
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

###############################################################################
#                         DATA COLLECTOR
###############################################################################
class LSLDataCollector(threading.Thread):
    """
    Background thread that collects and synchronizes EEG and marker data.
    """
    def __init__(self, stop_event, neurofeedback_processor):
        super().__init__()
        self.daemon = True  # Make thread a daemon so it exits when main program does
        self.stop_event = stop_event
        self.neurofeedback_processor = neurofeedback_processor
        self.eeg_buffer = deque()
        self.marker_buffer = deque()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_csv = os.path.join(RAW_DATA_DIR, f"MI_EEG_{timestamp_str}.csv")
        self.eeg_inlet = None
        self.marker_inlet = None
        self.clock_offset = 0.0
        
        # Create a separate buffer for high-priority neurofeedback processing
        self.neurofeedback_queue = deque(maxlen=5)  # Small buffer to prevent overflow

    def resolve_streams(self):
        try:
            logging.info("Resolving EEG LSL stream...")
            from pylsl import resolve_byprop, StreamInlet
            eeg_streams = resolve_byprop("name", EEG_STREAM_NAME, timeout=10)
            if not eeg_streams:
                logging.error(f"No EEG stream found with name '{EEG_STREAM_NAME}'. Exiting.")
                sys.exit(1)
            self.eeg_inlet = StreamInlet(eeg_streams[0], max_buflen=360)
            self.clock_offset = self.eeg_inlet.time_correction()
            logging.info(f"Computed clock offset: {self.clock_offset:.6f} seconds")

            logging.info("Resolving Marker LSL stream...")
            marker_streams = resolve_byprop("name", MARKER_STREAM_NAME, timeout=10)
            if not marker_streams:
                logging.error(f"No Marker stream found with name '{MARKER_STREAM_NAME}'. Exiting.")
                sys.exit(1)
            self.marker_inlet = StreamInlet(marker_streams[0], max_buflen=360)

            logging.info("LSL streams resolved. Starting data collection...")
        except Exception as e:
            logging.exception("Exception during stream resolution:")
            sys.exit(1)

    def flush_remaining(self, writer):
        """Flush any remaining buffered data to the CSV."""
        while self.eeg_buffer or self.marker_buffer:
            if self.eeg_buffer and self.marker_buffer:
                ts_eeg, eeg_data = self.eeg_buffer[0]
                ts_marker, marker = self.marker_buffer[0]
                if abs(ts_marker - ts_eeg) < MERGE_THRESHOLD:
                    row = [ts_eeg] + eeg_data + [marker]
                    writer.writerow(row)
                    self.eeg_buffer.popleft()
                    self.marker_buffer.popleft()
                elif ts_marker < ts_eeg:
                    row = [ts_marker] + ([""] * len(eeg_data)) + [marker]
                    writer.writerow(row)
                    self.marker_buffer.popleft()
                else:
                    row = [ts_eeg] + eeg_data + [""]
                    writer.writerow(row)
                    self.eeg_buffer.popleft()
            elif self.eeg_buffer:
                ts_eeg, eeg_data = self.eeg_buffer.popleft()
                row = [ts_eeg] + eeg_data + [""]
                writer.writerow(row)
            elif self.marker_buffer:
                ts_marker, marker = self.marker_buffer.popleft()
                row = [ts_marker] + ([""] * len(EEG_CHANNELS)) + [marker]
                writer.writerow(row)

    def run(self):
        from pylsl import resolve_byprop, StreamInlet
        self.resolve_streams()
        try:
            with open(self.output_csv, mode="w", newline="") as f:
                writer = csv.writer(f)
                header = ["lsl_timestamp"] + EEG_CHANNELS + ["marker"]
                writer.writerow(header)
                while not self.stop_event.is_set():
                    try:
                        # Use chunk processing for efficiency when available
                        samples, timestamps = self.eeg_inlet.pull_chunk(timeout=0.0, max_samples=32)
                        if samples and timestamps:
                            for sample, ts in zip(samples, timestamps):
                                # Process for neurofeedback immediately
                                self.neurofeedback_processor.process_sample(sample)
                                
                                # Buffer for data saving
                                self.eeg_buffer.append((ts, sample))
                    except Exception as e:
                        # If chunk not available, fallback to single sample
                        try:
                            sample_eeg, ts_eeg = self.eeg_inlet.pull_sample(timeout=0.0)
                            if sample_eeg is not None and ts_eeg is not None:
                                self.eeg_buffer.append((ts_eeg, sample_eeg))
                                # Process sample for neurofeedback
                                self.neurofeedback_processor.process_sample(sample_eeg)
                        except Exception as inner_e:
                            logging.exception("Error pulling EEG sample:")
                    
                    try:
                        sample_marker, ts_marker = self.marker_inlet.pull_sample(timeout=0.0)
                        if sample_marker is not None and ts_marker is not None:
                            adjusted_ts_marker = ts_marker - self.clock_offset
                            marker_val = sample_marker[0]
                            self.marker_buffer.append((adjusted_ts_marker, marker_val))
                    except Exception as e:
                        pass

                    # Merge data from both buffers (file saving is lower priority than processing)
                    while self.eeg_buffer and self.marker_buffer:
                        ts_eeg, eeg_data = self.eeg_buffer[0]
                        ts_marker, marker = self.marker_buffer[0]
                        if abs(ts_marker - ts_eeg) < MERGE_THRESHOLD:
                            row = [ts_eeg] + eeg_data + [marker]
                            writer.writerow(row)
                            self.eeg_buffer.popleft()
                            self.marker_buffer.popleft()
                        elif ts_marker < ts_eeg:
                            row = [ts_marker] + ([""] * len(eeg_data)) + [marker]
                            writer.writerow(row)
                            self.marker_buffer.popleft()
                        else:
                            row = [ts_eeg] + eeg_data + [""]
                            writer.writerow(row)
                            self.eeg_buffer.popleft()
                    
                    # Flush buffer periodically to avoid excessive memory use
                    if len(self.eeg_buffer) > 1000:
                        while len(self.eeg_buffer) > 500:
                            ts_eeg, eeg_data = self.eeg_buffer.popleft()
                            row = [ts_eeg] + eeg_data + [""]
                            writer.writerow(row)
                            
                    time.sleep(POLL_SLEEP)
                logging.info("Stop event set. Flushing remaining data...")
                self.flush_remaining(writer)
        except Exception as e:
            logging.exception("Exception in data collector run loop:")
        finally:
            logging.info(f"Data collection stopped. Data saved to {self.output_csv}")

###############################################################################
#                        EXPERIMENT LOGIC
###############################################################################
def run_neurofeedback_experiment(neurofeedback_processor):
    from pylsl import StreamInfo, StreamOutlet
    # Create LSL Marker stream
    try:
        marker_info = StreamInfo(MARKER_STREAM_NAME, "Markers", 1, 0, "string", "marker_id")
        marker_outlet = StreamOutlet(marker_info)
        logging.info("Marker stream created.")
    except Exception as e:
        logging.exception("Failed to create LSL Marker stream:")
        sys.exit(1)

    # PsychoPy setup
    try:
        from psychopy import visual, core, event, clock
        
        # Set up monitor for optimal performance
        win = visual.Window(
            size=(1024, 768),
            color="black",
            units="norm",
            fullscr=False,
            allowGUI=False,  # Disable GUI for better performance
            waitBlanking=False  # Disable waiting for screen refresh for lower latency
        )
        win.recordFrameIntervals = True  # Track frame timing
        
        logging.info("PsychoPy window created.")
    except Exception as e:
        logging.exception("Error setting up PsychoPy window:")
        sys.exit(1)

    # Create visual stimuli
    instruction_text = visual.TextStim(win, text="", height=0.15, color="white")
    ready_text = visual.TextStim(win, text="Get Ready", height=0.15, color="white")
    countdown_text = visual.TextStim(win, text="", height=0.2, color="white", pos=(0, 0.2))
    
    # Create channel indicator
    channel_text = visual.TextStim(win, text="", height=0.05, color="white",
                               pos=(0, -0.7))
    
    # Create Mu band feedback meter (right side) - similar to train_response.py
    meter_width = 0.2
    meter_height = 0.6
    mu_meter_pos = (0.7, 0)  # Position on right side of screen
    mu_meter_bg = visual.Rect(
        win=win, 
        width=meter_width, 
        height=meter_height, 
        pos=mu_meter_pos, 
        fillColor='gray', 
        lineColor='white',
        autoLog=False
    )
    mu_meter_fill = visual.Rect(
        win=win, 
        width=meter_width, 
        height=0, 
        pos=mu_meter_pos, 
        fillColor='green',
        autoLog=False
    )
    mu_meter_text = visual.TextStim(
        win=win, 
        text="Mu Band Power", 
        height=0.05,
        pos=(mu_meter_pos[0], mu_meter_pos[1] + meter_height/2 + 0.1),
        autoLog=False
    )
    
    # Create Beta band feedback meter (left side)
    beta_meter_pos = (-0.7, 0)  # Position on left side of screen
    beta_meter_bg = visual.Rect(
        win=win, 
        width=meter_width, 
        height=meter_height, 
        pos=beta_meter_pos, 
        fillColor='gray', 
        lineColor='white',
        autoLog=False
    )
    beta_meter_fill = visual.Rect(
        win=win, 
        width=meter_width, 
        height=0, 
        pos=beta_meter_pos, 
        fillColor='blue',
        autoLog=False
    )
    beta_meter_text = visual.TextStim(
        win=win, 
        text="Beta Band Power", 
        height=0.05,
        pos=(beta_meter_pos[0], beta_meter_pos[1] + meter_height/2 + 0.1),
        autoLog=False
    )
    
    # Create central feedback bar with optimized settings (keep this from original)
    feedback_bar_bg = visual.Rect(
        win=win,
        width=FEEDBACK_BAR_MAX_WIDTH,
        height=0.2,
        fillColor="gray",
        lineColor="white",
        pos=(0, -0.5),
        autoLog=False
    )
    
    feedback_bar = visual.Rect(
        win=win,
        width=0,  # Will be updated based on ERD
        height=0.15,
        fillColor="green",
        lineColor=None,
        pos=(-FEEDBACK_BAR_MAX_WIDTH/2, -0.5),  # Centered position
        anchor="left",  # Anchor to left for width updates
        autoLog=False
    )

    # Generate randomized trial list
    trials = ["right", "left"] * (NUM_TRIALS // 2)
    random.shuffle(trials)
    
    # Show initial instructions
    instruction_text.text = "Motor Imagery Neurofeedback\n\nImagine moving your hand when instructed\n\nThe feedback bars will show your brain activity\n\nPress SPACE to begin"
    instruction_text.draw()
    win.flip()
    event.waitKeys(keyList=["space"])
    
    # Initial 10-second baseline collection
    instruction_text.text = "Collecting baseline\n\nPlease relax and remain still"
    neurofeedback_processor.start_initial_baseline_collection()
    
    baseline_clock = core.Clock()
    baseline_clock.reset()
    
    # Countdown for initial baseline
    while baseline_clock.getTime() < INITIAL_BASELINE_DURATION:
        remaining = int(INITIAL_BASELINE_DURATION - baseline_clock.getTime())
        instruction_text.draw()
        countdown_text.text = f"{remaining}s"
        countdown_text.draw()
        win.flip()
        
        # Check for quit
        if event.getKeys(keyList=["escape"]):
            win.close()
            neurofeedback_processor.stop_initial_baseline_collection()
            return
    
    neurofeedback_processor.stop_initial_baseline_collection()
    
    # Indicate baseline completion
    instruction_text.text = "Baseline Completed\n\nThe experiment will begin shortly"
    instruction_text.draw()
    win.flip()
    core.wait(2.0)

    # Main experiment loop
    for trial_num, hand in enumerate(trials, 1):
        # Set the active hand for this trial
        neurofeedback_processor.set_active_hand(hand)
        
        # Update the channel indicator
        channel_used = "CH3" if hand == "right" else "CH6"
        channel_text.text = f"Using {channel_used} (Contralateral Hemisphere)"
        
        # Display get ready message
        ready_text.draw()
        channel_text.draw()
        win.flip()
        core.wait(1.0)
        
        # Show instruction (right or left hand)
        instruction_text.text = f"{hand.upper()} HAND"
        instruction_text.draw()
        channel_text.draw()
        win.flip()
        core.wait(INSTRUCTION_DURATION)
        
        # Start collecting baseline data 3 seconds before the START cue
        # We use a separate clock to time the baseline collection
        baseline_clock = core.Clock()
        baseline_clock.reset()
        neurofeedback_processor.start_baseline_collection()
        
        # Wait for baseline collection (-3s to -1s before START)
        # Continue showing instruction during this time
        while baseline_clock.getTime() < abs(BASELINE_START - BASELINE_END):  # 2 seconds
            instruction_text.draw()
            channel_text.draw()
            win.flip()
            
            # Check for quit
            if event.getKeys(keyList=["escape"]):
                win.close()
                return
        
        # Stop baseline collection and prepare for processing
        neurofeedback_processor.stop_baseline_collection()
        
        # Show START cue and send marker
        instruction_text.text = "START"
        instruction_text.draw()
        channel_text.draw()
        # Schedule marker to be sent on next flip
        marker_val = MARKER_RIGHT if hand == "right" else MARKER_LEFT
        win.callOnFlip(lambda m=marker_val: marker_outlet.push_sample([m]))
        win.flip()
        
        # Start real-time processing after the START cue
        neurofeedback_processor.start_processing()
        
        # Initialize feedback timer and display timer
        feedback_timer = core.Clock()
        display_timer = core.Clock()
        feedback_timer.reset()
        display_timer.reset()
        
        # Continuous feedback during imagery period
        while feedback_timer.getTime() < IMAGERY_DURATION:
            # Only update display at the target refresh rate
            if display_timer.getTime() >= DISPLAY_UPDATE_INTERVAL:
                # Reset display timer
                display_timer.reset()
                
                # Get current ERD values with smoothing for visual appeal
                smoothed_mu = neurofeedback_processor.get_smoothed_erd_mu()
                smoothed_beta = neurofeedback_processor.get_smoothed_erd_beta()
                
                # Update central feedback bar width based on mu ERD value
                bar_width = min(smoothed_mu * FEEDBACK_BAR_MAX_WIDTH, FEEDBACK_BAR_MAX_WIDTH)
                feedback_bar.width = bar_width
                
                # Update mu meter height
                mu_meter_fill.height = smoothed_mu * meter_height
                mu_meter_fill.pos = (mu_meter_pos[0], mu_meter_pos[1] - (meter_height/2) + (smoothed_mu * meter_height/2))
                
                # Update beta meter height
                beta_meter_fill.height = smoothed_beta * meter_height
                beta_meter_fill.pos = (beta_meter_pos[0], beta_meter_pos[1] - (meter_height/2) + (smoothed_beta * meter_height/2))
                
                # Update central bar color based on mu ERD value
                if smoothed_mu < 0.2:
                    feedback_bar.fillColor = "red"
                elif smoothed_mu < 0.6:
                    feedback_bar.fillColor = "yellow"
                else:
                    feedback_bar.fillColor = "green"
                
                # Draw feedback
                instruction_text.text = "HOLD"
                instruction_text.draw()
                
                # Draw mu and beta meters
                mu_meter_bg.draw()
                mu_meter_fill.draw()
                mu_meter_text.draw()
                beta_meter_bg.draw()
                beta_meter_fill.draw()
                beta_meter_text.draw()
                
                # Draw central feedback bar
                feedback_bar_bg.draw()
                feedback_bar.draw()
                
                # Draw channel indicator
                channel_text.draw()
                
                win.flip()
            
            # Process any events to prevent blocking
            if event.getKeys(keyList=["escape"]):
                win.close()
                return
        
        # Stop real-time processing
        neurofeedback_processor.stop_processing()
        
        # Show STOP and send stop marker
        instruction_text.text = "STOP"
        instruction_text.draw()
        channel_text.draw()
        win.callOnFlip(lambda: marker_outlet.push_sample([MARKER_STOP]))
        win.flip()
        core.wait(1.0)
        
        # Inter-trial interval
        win.flip()  # clear screen
        core.wait(INTER_TRIAL_INTERVAL)
        
        # Report frame timing stats
        if win.recordFrameIntervals:
            frame_times = win.frameIntervals
            if frame_times:
                mean_frame_time = np.mean(frame_times)
                std_frame_time = np.std(frame_times)
                logging.info(f"Frame timing: Mean={mean_frame_time*1000:.1f}ms, Std={std_frame_time*1000:.1f}ms")
                win.frameIntervals = []  # Reset for next trial
        
        logging.info(f"Completed trial {trial_num}/{len(trials)}: {hand} hand")
        
        # Check for quit
        if event.getKeys(keyList=["escape"]):
            break

    # Cleanup
    win.close()
    logging.info("Neurofeedback experiment finished.")

###############################################################################
#                                   MAIN
###############################################################################
def main():
    # Set higher priority for this process (platform-specific)
    try:
        import psutil
        process = psutil.Process()
        if hasattr(process, 'nice'):
            process.nice(-10)  # Unix: higher priority (negative is higher)
        elif hasattr(process, 'prioritize'):
            process.prioritize(psutil.HIGH_PRIORITY_CLASS)  # Windows
        logging.info("Set process to high priority")
    except ImportError:
        logging.warning("psutil not available, process priority not adjusted")
    except Exception as e:
        logging.warning(f"Could not adjust process priority: {e}")
    
    logging.info("Starting Neurofeedback BCI experiment script")
    
    # Create neurofeedback processor
    neurofeedback_processor = NeurofeedbackProcessor()
    
    # Start data collector thread
    stop_event = threading.Event()
    collector = LSLDataCollector(stop_event, neurofeedback_processor)
    collector.start()

    # Run the experiment
    run_neurofeedback_experiment(neurofeedback_processor)

    # Stop the collector thread gracefully
    logging.info("Experiment complete. Stopping data collector thread...")
    stop_event.set()
    collector.join(timeout=2.0)  # Wait max 2 seconds for clean shutdown

    logging.info("All processes complete. Exiting script.")

if __name__ == "__main__":
    main() 