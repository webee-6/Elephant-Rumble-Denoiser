<div align="center">

# Kumki Radar - Elephant Rumble Denoiser

### Multi-stage DSP Pipeline for Bioacoustic Denoising

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Remove mechanical noise from elephant bioacoustic recordings**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-understanding-the-pipeline)

</div>

---

## About

Multi-stage DSP pipeline for removing mechanical noise (airplanes, vehicles, generators) from elephant bioacoustic recordings. Built for the elephant bioacoustics to denoise mechanical interference from elephant rumble recordings in the 20-300 Hz frequency range.

## Built With

<table>
<tr>
<td width="33%">

### Core DSP Libraries
- **[librosa](https://librosa.org/)** `0.10+` - Audio analysis & STFT
- **[scipy](https://scipy.org/)** `1.9+` - IIR filters (Butterworth, Notch)
- **[noisereduce](https://github.com/timsainb/noisereduce)** `3.0+` - Spectral gating

</td>
<td width="33%">

### Scientific Computing
- **[numpy](https://numpy.org/)** `1.21+` - Numerical operations
- **[pandas](https://pandas.pydata.org/)** `1.5+` - Data processing
- **[soundfile](https://github.com/bastibe/python-soundfile)** `0.12+` - Audio I/O

</td>
<td width="33%">

### Visualization
- **[matplotlib](https://matplotlib.org/)** `3.5+` - Spectrogram plots
- **[tqdm](https://github.com/tqdm/tqdm)** `4.64+` - Progress bars

</td>
</tr>
</table>

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

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment tool

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/elephant-rumble-denoiser.git
   cd elephant-rumble-denoiser
   ```

2. **Create virtual environment (recommended)**
   ```bash
   # Using venv
   python -m venv venv
   
   # Activate on Linux/Mac
   source venv/bin/activate
   
   # Activate on Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Requirements

The project requires the following packages:

```txt
numpy>=1.21.0          # Numerical computing
matplotlib>=3.5.0      # Visualization
librosa>=0.10.0        # Audio signal processing
soundfile>=0.12.0      # Audio file I/O
noisereduce>=3.0.0     # Spectral noise reduction
scipy>=1.9.0           # Scientific computing & filters
pandas>=1.5.0          # Data manipulation
tqdm>=4.64.0           # Progress bars
```

### Verify Installation

```bash
python -c "import librosa, noisereduce, scipy; print('✅ All dependencies installed!')"
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

---

## Windowing & Segmentation

### Built-in Windowing

The pipeline uses windowing in several stages:

1. **HPSS**: Hann window (4096 samples, 88% overlap) for STFT
2. **Spectral Gating**: Internal STFT windowing (2048 samples)
3. **Wiener Filter**: Local 29-sample sliding window

### Optional Segmentation for Long Calls

For recordings > 30 seconds, use the enhanced segmented pipeline:

```python
from src.pipeline_segmented import process_single_call_segmented

result = process_single_call_segmented(
    audio_path='long_call.wav',
    start_time=0,
    end_time=300,  # 5 minutes
    selection_id=1,
    use_segmentation=True,    # Auto-segment long calls
    segment_duration=30.0      # 30-second segments
)
```

**Benefits:**
- ✅ Memory-efficient for very long recordings
- ✅ Prevents edge artifacts with 25% overlap
- ✅ Auto-detects when segmentation is needed
- ✅ No quality loss vs. processing entire signal

**See [WINDOWING.md](docs/WINDOWING.md) for technical details.**

---

## Unsupervised Learning (No Labels Required!)

**Discover natural patterns in your rumbles without manual labeling!**

### Quick Start - Clustering & Pattern Discovery

```bash
# One command - extracts features, finds clusters, detects anomalies
python analyze_rumbles_unsupervised.py --audio outputs/audio
```

**What happens:**
1. ✅ Extracts **88 bioacoustic features** per rumble (openSMILE)
2. ✅ Automatically finds **optimal number of clusters** (2-10)
3. ✅ Groups acoustically similar rumbles together
4. ✅ Detects **rare/unusual calls** (anomaly detection)
5. ✅ Creates **2D visualizations** (UMAP/t-SNE/PCA)
6. ✅ Generates **CSV reports** with assignments

**Example output:**
```
🔬 Extracting acoustic features (opensmile)...
✅ Extracted features: (100, 88)

🔍 Finding optimal number of clusters...
   → Recommended: 4 clusters

🎯 Clustering with KMEANS (k=4)...
   Cluster 0: 28 rumbles (28.0%) - Contact calls
   Cluster 1: 35 rumbles (35.0%) - Greeting calls  
   Cluster 2: 22 rumbles (22.0%) - Alarm calls
   Cluster 3: 15 rumbles (15.0%) - Mixed patterns

🔍 Detecting anomalies...
   Found 10 anomalous rumbles (10.0%)

📊 Results saved to: unsupervised_results/
   - cluster_assignments.csv
   - anomalies.csv
   - visualization.png
```

### Feature Extraction Options

**openSMILE** (88 features - BEST for vocalizations):
```bash
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --features opensmile
```
- Prosodic features (pitch, energy contours)
- Voice quality (jitter, shimmer)
- Spectral features (formants, MFCC)

### Advanced Options

**Specify cluster count:**
```bash
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --clusters 5 \
    --method gmm  # or kmeans, hierarchical, dbscan
```

**Focus on anomaly detection:**
```bash
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --anomaly-only \
    --contamination 0.05  # Expect 5% anomalies
```

**Change visualization:**
```bash
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --dim-reduction umap  # or tsne, pca
```

### Understanding Results

**cluster_assignments.csv**: Which rumble belongs to which cluster
```csv
filename,cluster
selection_001_cleaned.wav,0
selection_002_cleaned.wav,1
...
```

**anomalies.csv**: Unusual/rare rumbles
```csv
filename,is_anomaly,anomaly_score
selection_047_cleaned.wav,True,-0.128
...
```

**visualization.png**: 2D scatter plots showing:
- Left: Clusters (different colors)
- Right: Anomalies (red = unusual)

**See [UNSUPERVISED_GUIDE.md](docs/UNSUPERVISED_GUIDE.md) for more details.**

---


### Neural Network with Windowing (Recommended)

**Complete pipeline with segmentation:**

```bash
# 1. Create labels (see docs/LABEL_CREATION.md)
python scripts/create_labels.py --csv data/annotations.csv

# 2. Train neural network with windowing
python train_neural_classifier.py \
    --audio outputs/audio \
    --labels data/labels.json \
    --window 1.0 \
    --overlap 0.5 \
    --epochs 100
```

**What it does:**
- ✅ Segments long rumbles into 1s windows (50% overlap)
- ✅ Extracts 39 acoustic features per window
- ✅ LSTM processes temporal sequence
- ✅ Attention focuses on important moments
- ✅ Achieves **85-90% accuracy** on 200+ samples

**Example output:**
```
Epoch 67/100
  Train: Loss=0.214, Acc=0.925
  Val:   Loss=0.342, Acc=0.886

✅ Best val accuracy: 0.886

Prediction on new rumble:
  Class: 1 (Greeting)
  Confidence: 0.82
  Windows analyzed: 8
  Key moments: Windows 3-5 (1.0-3.0s)
```

### Supported Tasks

| Task | Method |  Use Case |
|------|--------|----------|
| **Call Classification** | Random Forest  | Rumble type (contact/greeting/alarm) |
| **Individual ID** | SVM  | Which elephant made the call |
| **Call Detection** | CNN  | Rumble vs background noise |
| **Unsupervised Clustering** | Autoencoder + K-Means | Discover call patterns |

### Features Extracted

- **Temporal**: Duration, zero-crossing rate, energy
- **Spectral**: Centroid, rolloff, bandwidth, contrast
- **Low-Frequency**: Fundamental F0 (10-300 Hz), harmonic ratio
- **MFCCs**: 20 coefficients + deltas for pattern recognition
- **Mel Spectrogram**: 128 bands for deep learning

**Documentation:**
- **[AI_LEARNING_GUIDE.md](docs/AI_LEARNING_GUIDE.md)** - Classical ML & deep learning
- **[NEURAL_NETWORK_GUIDE.md](docs/NEURAL_NETWORK_GUIDE.md)** - Windowing & segmentation strategy ⭐
- **[LABEL_CREATION.md](docs/LABEL_CREATION.md)** - How to create training labels

## Output Analysis

After batch processing, check `outputs/logs/processing_results.csv` for:
- Processing status (success/failed)
- Noise type classification
- Duration, sample rate
- Noise profile source and validation metrics

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Signal Processing** | Librosa, SciPy | STFT, filtering, decomposition |
| **Noise Reduction** | Noisereduce | Spectral gating algorithm |
| **Harmonic Separation** | Librosa HPSS | Median filtering decomposition |
| **Optimal Filtering** | SciPy Wiener | MSE-optimal noise reduction |
| **Data Pipeline** | Pandas, NumPy | Batch processing & analysis |
| **Visualization** | Matplotlib | B/W spectrograms |

## Contributing

Contributions welcome! Areas for improvement:

- [ ] **Deep Learning**: U-Net, Conv-TasNet denoising models
- [ ] **Real-time Processing**: Streaming audio support
- [ ] **GUI Interface**: Electron/PyQt application
- [ ] **Enhanced Classification**: ML-based noise type detection
- [ ] **Quality Metrics**: SNR, PESQ, STOI evaluation
- [ ] **Documentation**: More usage examples
- [ ] **Testing**: Expanded test coverage

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python -m pytest tests/

# Format code
black src/ config/ tests/
```

## License

MIT License - feel free to use and modify for your research.

## Acknowledgments

- Built for **Elephant Voices** and SMU hackathon - **HackSMU VII**
- **Librosa** team for audio processing tools
- **Noisereduce** by Tim Sainburg for spectral gating implementation
- **SciPy** community for robust signal processing algorithms
- Elephant conservation researchers worldwide 

---

<div align="center">

**[⬆ back to top](#-elephant-rumble-denoiser)**

Made with ❤️ for elephant conservation research

</div>
