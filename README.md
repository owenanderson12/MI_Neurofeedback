# MI Neurofeedback

**Project**: Real-time Motor Imagery Neurofeedback System

This repository implements a closed-loop motor imagery (MI) neurofeedback experiment using EEG data streamed over LSL and visual feedback presented via PsychoPy. Participants perform imagined left/right hand movements, and the system computes real-time bandpower in the Mu (8–12 Hz) and Beta (13–30 Hz) bands to generate dynamic feedback. ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/tree/main))

---

## Repository Structure

```
MI_Neurofeedback/
├── Neurofeedback_BCI.py                # Main script: orchestrates data collection, processing, and feedback display
├── .gitignore                          # Files and directories to ignore in Git
├── modules/                            # Modular components
│   ├── config.py                       # Centralized parameters and directories ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/config.py))
│   ├── LSLDataCollector.py             # LSL stream resolution, data synchronization, CSV logging ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/LSLDataCollector.py))
│   ├── NeurofeedbackProcessor.py       # Real-time sliding-window bandpower & ERD computation ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/NeurofeedbackProcessor.py))
│   ├── run_neurofeedback_experiment.py # PsychoPy-based experiment loop and visual feedback ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/run_neurofeedback_experiment.py))
│   ├── calculate_band_power.py         # Welch’s method for bandpower estimation ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/calculate_band_power.py))
│   ├── fft_power_update.py             # Placeholder for incremental FFT updates ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/fft_power_update.py))
│   └── bandpass_filter.py              # Butterworth bandpass filter utility ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/bandpass_filter.py))
```

---

## Features

* **Real-time EEG acquisition** via LSL (OpenBCI or any LSL-compatible EEG) ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/LSLDataCollector.py))
* **Sliding-window Mu/Beta bandpower** estimation with baseline normalization and adaptive scaling ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/NeurofeedbackProcessor.py))
* **Visual feedback**: central bar for Mu ERD, side meters for Mu and Beta power, colored by strength ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/run_neurofeedback_experiment.py))
* **Configurable experiment**: number/duration of trials, inter-trial intervals, no-movement probability, etc. ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/config.py))
* **CSV logging**: synchronized EEG samples and markers for offline analysis ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/LSLDataCollector.py))

---

## Requirements

* Python 3.8+
* **pylsl**: Lab Streaming Layer interface
* **PsychoPy**: visual stimuli and timing
* **NumPy**, **SciPy**, **pandas**
* **logging** (standard library)
* **psutil** (optional, for process prioritization)

Install dependencies via pip:

```bash
pip install pylsl psychopy numpy scipy pandas psutil
```

---

## Usage

1. **Connect EEG & Marker Streams**: Ensure an EEG LSL stream named `OpenBCI_EEG` and a marker stream named `MI_MarkerStream` are active (as defined in `modules/config.py`).
2. **Run the experiment**:

   ```bash
   python Neurofeedback_BCI.py
   ```
3. **Data output**:

   * Real-time CSV logs saved to `data/raw/MI_EEG_<timestamp>.csv`.
   * Visual feedback window will present instructions, baseline collection, and trial-by-trial feedback.

Press **ESC** at any time to exit the experiment prematurely.

---

## Configuration

All key parameters are defined in `modules/config.py`, including:

* Stream names, sampling rate, EEG channels
* Trial counts, durations (`INSTRUCTION_DURATION`, `IMAGERY_DURATION`, `INTER_TRIAL_INTERVAL`)
* Baseline settings and weights
* Mu/Beta band definitions and windowing parameters
* Visual feedback properties (bar width, refresh rates)

Modify these values to suit your experimental protocol. ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/config.py))

---

## Module Overview

* **LSLDataCollector**: Resolves EEG and marker streams, buffers samples, synchronizes timestamps, writes CSV logs, and forwards samples to the processor thread. ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/LSLDataCollector.py))
* **NeurofeedbackProcessor**: Maintains circular buffers for sliding-window processing, computes bandpower via `calculate_band_power` and optional FFT updates, calculates event-related desynchronization (ERD), applies smoothing and adaptive scaling. ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/NeurofeedbackProcessor.py))
* **run\_neurofeedback\_experiment**: Implements the PsychoPy experiment loop: instructions, baseline collection, randomized trial sequence, marker sending, real-time feedback updates, and performance logging. ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/run_neurofeedback_experiment.py))
* **Utility modules** (`calculate_band_power`, `fft_power_update`, `bandpass_filter`): Provide signal-processing functions used by the processor. ([github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/calculate_band_power.py), [github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/fft_power_update.py), [github.com](https://github.com/owenanderson12/MI_Neurofeedback/raw/main/modules/bandpass_filter.py))

---

## License & Author

**Author**: Owen Anderson<br>
---

Enjoy exploring and extending this MI neurofeedback system! Feel free to open issues or contribute. 🎉
