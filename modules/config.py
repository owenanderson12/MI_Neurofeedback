"""
config.py - Centralized configuration parameters for the MI Neurofeedback system

This file contains all configuration constants used throughout the MI Neurofeedback system.
All modules should import parameters from this file to ensure consistency.
"""

import os

# --- Configuration Parameters ---
EEG_STREAM_NAME = "OpenBCI_EEG"
MARKER_STREAM_NAME = "MI_MarkerStream"

SAMPLE_RATE = 250  # Hz
EEG_CHANNELS = ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8"]

# Motor Imagery Experiment Parameters
NUM_TRIALS = 35  # Configurable number of trials
INSTRUCTION_DURATION = 2.0  # seconds to show left/right instruction
IMAGERY_DURATION = 5.0  # seconds for motor imagery
INTER_TRIAL_INTERVAL = 3.0  # seconds between trials
NO_MOVEMENT_PROBABILITY = 0.1  # Probability of a no-movement trial occurring

# Marker values
MARKER_RIGHT = "1"  # right hand imagery
MARKER_LEFT = "2"   # left hand imagery
MARKER_STOP = "3"   # end of imagery period
MARKER_NO_MOVEMENT = "0"  # no movement/rest trials

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