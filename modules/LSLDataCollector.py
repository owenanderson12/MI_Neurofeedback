import threading
import csv
import logging
import os
import time
import sys
from datetime import datetime
from collections import deque

from modules.config import *

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
        
        # Ensure data directory exists before creating files
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
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