# 📂 Outputs Directory

This directory contains all processed results from the elephant rumble denoiser.

## Directory Structure

```
outputs/
├── audio/              # Cleaned audio files
├── spectrograms/       # Visualization images
└── logs/              # Processing metadata
```

---

## 📁 audio/

Contains denoised WAV files.

**Naming convention:**
```
selection_XXX_<original_filename>_cleaned.wav
```

Example:
```
selection_001_airplane_noise_cleaned.wav
selection_002_vehicle_noise_cleaned.wav
```

**Format:**
- Sample rate: 44100 Hz (or original)
- Channels: Mono
- Bit depth: 16-bit PCM
- Normalization: Peak normalized to 0.7

---

## 📊 spectrograms/

Contains before/after comparison plots and individual spectrograms.

**File types:**

1. **Comparison plots:**
   ```
   selection_XXX_<filename>_comparison.png
   ```
   Side-by-side before/after spectrograms

2. **Cleaned spectrograms:**
   ```
   selection_XXX_<filename>_cleaned.png
   ```
   B/W spectrogram of denoised signal

**Properties:**
- Format: PNG
- DPI: 150
- Colormap: Binary (black & white)
- Frequency range: 10-1000 Hz (log scale)
- Reference lines: 20 Hz (red), 300 Hz (blue) for elephant range

---

## 📋 logs/

Processing metadata and results.

**Files:**

1. **`processing_results.csv`**
   
   Complete processing log with columns:
   ```
   selection_id, filename, start_time, end_time, status, error,
   noise_type, noise_source, duration, sample_rate, 
   output_audio, spectrogram, comparison_plot
   ```

2. **`summary_statistics.txt`**
   
   High-level summary:
   - Total/successful/failed counts
   - Processing by noise type
   - Duration statistics
   - Noise profile sources

---

## 🔍 Example Output Files

After processing selection #5 from `airplane_noise.wav`:

```
outputs/
├── audio/
│   └── selection_005_airplane_noise_cleaned.wav
├── spectrograms/
│   ├── selection_005_airplane_noise_cleaned.png
│   └── selection_005_airplane_noise_comparison.png
└── logs/
    └── processing_results.csv  (contains row for selection 5)
```

---

## 📊 Reading Processing Results

### Python

```python
import pandas as pd

# Load results
results = pd.read_csv('outputs/logs/processing_results.csv')

# View successful calls
successful = results[results['status'] == 'success']
print(f"Successfully processed: {len(successful)}")

# Group by noise type
by_noise = successful.groupby('noise_type').size()
print(by_noise)

# Find failed selections
failed = results[results['status'] == 'failed']
print(failed[['selection_id', 'filename', 'error']])
```

### Command Line

```bash
# Count successful
grep "success" outputs/logs/processing_results.csv | wc -l

# List failed selections
grep "failed" outputs/logs/processing_results.csv

# View summary
cat outputs/logs/summary_statistics.txt
```

---

## 🎵 Playing Cleaned Audio

### Python
```python
from IPython.display import Audio
Audio('outputs/audio/selection_001_airplane_noise_cleaned.wav')
```

### Command Line
```bash
# Linux
aplay outputs/audio/selection_001_airplane_noise_cleaned.wav

# Mac
afplay outputs/audio/selection_001_airplane_noise_cleaned.wav

# Windows
start outputs/audio/selection_001_airplane_noise_cleaned.wav
```

---

## 🗜️ Archive Creation

After batch processing, results are automatically archived:

```
elephant_denoiser_results_YYYYMMDD_HHMMSS.zip
```

Contains complete `outputs/` directory structure.

**Extract archive:**
```bash
unzip elephant_denoiser_results_20260515_143022.zip
```

---

## 🧹 Cleanup

To clear outputs and start fresh:

```bash
# Linux/Mac
rm -rf outputs/audio/* outputs/spectrograms/* outputs/logs/*

# Windows
del /s /q outputs\audio\* outputs\spectrograms\* outputs\logs\*
```

Or keep outputs but create new batch:
```bash
# Rename existing
mv outputs outputs_backup_$(date +%Y%m%d)

# Create fresh
mkdir -p outputs/{audio,spectrograms,logs}
```

---

## 📏 Disk Space Estimates

Typical file sizes:

| File Type | Size per File | 10 Calls | 100 Calls |
|-----------|--------------|----------|-----------|
| Cleaned WAV | ~400-800 KB | ~6 MB | ~60 MB |
| Spectrogram PNG | ~150-300 KB | ~2 MB | ~20 MB |
| Comparison PNG | ~300-500 KB | ~4 MB | ~40 MB |
| **Total** | ~1 MB | **~12 MB** | **~120 MB** |

**Note:** Actual sizes vary based on audio duration and sample rate.
