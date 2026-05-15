# Elephant Rumble Denoiser

Multi-stage DSP pipeline for removing mechanical noise (airplanes, vehicles, generators) from elephant bioacoustic recordings.

## Features

- **Multi-stage denoising pipeline**:
  1. Butterworth band-pass filtering (20-1000 Hz)
  2. Generator harmonic notch filtering (60/120/180 Hz...)
  3. Spectral gating with noise profile
  4. Harmonic-Percussive Source Separation (HPSS)
  5. Wiener filtering
  
- **Noise-type specific processing**: Automatic detection and tailored parameters for airplane, vehicle, generator, and background noise
- **Adaptive noise profile extraction**: Automatically finds clean noise segments before/after calls
- **Batch processing**: Process hundreds of calls efficiently
- **Visualization**: B/W spectrograms and before/after comparisons
- **Parameter tuning**: Test multiple configurations to optimize results

## Installation

```bash
# Clone or download the project
cd elephant_denoiser

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

You need:
- **CSV file** with columns: `Selection`, `Sound_file`, `Start_time`, `End_time`
- **Audio folder** containing the referenced audio files

Example CSV:
```csv
Selection,Sound_file,Start_time,End_time
1,airplane_noise.wav,5.2,8.7
2,vehicle_noise.wav,12.1,15.3
```

### 2. Test on Single Call

```bash
python main.py --csv /path/to/calls.csv --audio /path/to/audio/folder --test
```

### 3. Process Full Batch

```bash
python main.py --csv /path/to/calls.csv --audio /path/to/audio/folder
```

This will:
- Process all calls
- Generate cleaned audio files → `outputs/audio/`
- Create spectrograms → `outputs/spectrograms/`
- Save processing logs → `outputs/logs/`
- Create a timestamped `.zip` archive

## Project Structure

```
elephant_denoiser/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── config/
│   └── config.py          # Pipeline configuration
├── src/
│   ├── algorithms.py      # DSP algorithms (filters, HPSS, etc.)
│   ├── noise_utils.py     # Noise profile extraction/validation
│   ├── visualization.py   # Spectrogram generation
│   ├── pipeline.py        # Main processing pipeline
│   ├── batch_process.py   # Batch processing logic
│   └── tuning.py          # Parameter tuning utilities
├── outputs/               # Generated outputs
│   ├── audio/            # Cleaned WAV files
│   ├── spectrograms/     # PNG visualizations
│   └── logs/             # Processing results CSV
├── data/                  # (Your input data goes here)
├── tests/                 # Unit tests (optional)
└── notebooks/             # Jupyter notebooks for exploration
```

## Configuration

Edit `config/config.py` to customize:

```python
# STFT parameters
n_fft = 4096
hop_length = 512

# Band-pass filter
bp_lowcut = 20.0    # Hz
bp_highcut = 1000.0 # Hz

# Noise type parameters
noise_params = {
    'airplane': {'prop_decrease': 0.85, 'stationary': False},
    'vehicle':  {'prop_decrease': 0.80, 'stationary': False},
    'generator': {'prop_decrease': 0.90, 'stationary': True}
}

# HPSS
hpss_kernel_size = 31
hpss_margin = 3.0
```

## Parameter Tuning

Use the tuning module to test different parameters:

```python
from src.batch_process import load_and_validate_data
from src.tuning import test_parameters

# Load data
df = load_and_validate_data('calls.csv', 'audio_folder')

# Test different configurations on Selection 1
results = test_parameters(
    df,
    selection_id=1,
    prop_decrease_values=[0.75, 0.85, 0.90],
    hpss_margins=[2.0, 3.0, 4.0]
)
```

## Understanding the Pipeline

### Stage 1: Band-Pass Filter
Removes frequencies outside elephant vocalization range (20-1000 Hz).

### Stage 2: Generator Notch (Conditional)
For recordings with electrical interference, removes 60Hz harmonics.

### Stage 3: Spectral Gating
Uses noise profile to subtract noise in frequency domain. Adaptive based on noise type.

### Stage 4: HPSS
Separates harmonic (elephant rumbles) from percussive (machinery clicks/pops) components.

### Stage 5: Wiener Filter
MSE-optimal noise reduction using local signal statistics.

## Output Analysis

After batch processing, check `outputs/logs/processing_results.csv` for:
- Processing status (success/failed)
- Noise type classification
- Duration, sample rate
- Noise profile source and validation metrics

## Research Context

This pipeline was developed for the elephant bioacoustics to denoise mechanical interference from African elephant rumble recordings. The frequency range (20-300 Hz for fundamentals) requires specialized low-frequency processing.

## Contributing

Contributions welcome! Areas for improvement:
- [ ] Deep learning denoising (U-Net, Conv-TasNet)
- [ ] Real-time processing
- [ ] GUI interface
- [ ] More noise type classifiers
- [ ] Automated quality metrics


## License

MIT License - feel free to use and modify for your research.

## Acknowledgments

Built for elephant bioacoustics research. Uses librosa, noisereduce, and scipy for signal processing.
