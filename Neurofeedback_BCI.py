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
import logging
import os
import sys

# Import configuration and modules
from modules.config import *
from modules.NeurofeedbackProcessor import NeurofeedbackProcessor
from modules.LSLDataCollector import LSLDataCollector
from modules.run_neurofeedback_experiment import run_neurofeedback_experiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

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