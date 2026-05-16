# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### 1. Filename Errors: "System error" when saving files

**Error:**
```
Error opening 'outputs/audio/selection_001_filename with spaces_cleaned.wav': System error
```

**Cause:** Filenames with spaces or special characters

**Solution:** The pipeline now automatically sanitizes filenames. Update your code:
```python
# Updated in v1.0.0 - filenames are auto-sanitized
result = process_single_call(...)  # Just works now!
```

**Manual fix** (if using old version):
```python
import re
from pathlib import Path

def sanitize_filename(filename):
    name = Path(filename).stem
    name = name.replace(' ', '_')
    name = re.sub(r'[^\w\-.]', '_', name)
    return name
```

---

### 2. Output Directory Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'outputs/audio/...'
```

**Solution:** 
```python
import os

# Create directories before processing
os.makedirs('outputs/audio', exist_ok=True)
os.makedirs('outputs/spectrograms', exist_ok=True)
os.makedirs('outputs/logs', exist_ok=True)
```

Or use the notebook helpers:
```python
from notebook_utils import setup_notebook_environment
output_dir = setup_notebook_environment()
```

---

### 3. Module Import Errors in Notebooks

**Error:**
```
ModuleNotFoundError: No module named 'config'
```

**Solution:** Add parent directory to path at notebook start:
```python
import sys
from pathlib import Path

# If running from notebooks/ folder
notebook_dir = Path.cwd()
project_root = notebook_dir.parent
sys.path.insert(0, str(project_root))

# Now imports work
from config.config import CONFIG
```

Or use the helper:
```python
from notebook_utils import setup_notebook_environment
setup_notebook_environment()
```

---

### 4. Audio File Not Found

**Error:**
```
FileNotFoundError: could not find audio file
```

**Solution:** Use full paths in CSV or verify audio folder:
```python
import os

# Check if file exists
audio_path = 'path/to/audio.wav'
if not os.path.exists(audio_path):
    print(f"File not found: {audio_path}")

# Use load_and_validate_data to check all files
from src.batch_process import load_and_validate_data
df = load_and_validate_data('calls.csv', 'audio_folder/')
# This will report missing files
```

---

### 5. Memory Error with Large Files

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:** Process files in smaller batches:
```python
# Process in chunks of 10
chunk_size = 10
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    results = batch_process(chunk)
```

Or increase system memory / use lower sample rate:
```python
# Load with lower sample rate (in algorithms.py)
y, sr = librosa.load(audio_path, sr=22050)  # Instead of sr=None
```

---

### 6. Filter Instability Warnings

**Warning:**
```
⚠️ Butterworth filter unstable, returning original
```

**Cause:** Extreme frequency cutoffs or high filter orders

**Solution:** Adjust filter parameters in `config/config.py`:
```python
# More conservative settings
bp_lowcut = 20.0   # Don't go below 15 Hz
bp_highcut = 1000.0  # Don't exceed 0.99 * Nyquist
bp_order = 2        # Keep at 2 or 3 max
```

---

### 7. Poor Denoising Results

**Symptoms:** Still too noisy or elephant call removed

**Solutions:**

**A. Adjust spectral gating strength:**
```python
# In config/config.py
noise_params = {
    'vehicle': {
        'prop_decrease': 0.70,  # Lower = less aggressive (keep more signal)
        # Or
        'prop_decrease': 0.95,  # Higher = more aggressive (remove more)
    }
}
```

**B. Change HPSS parameters:**
```python
CONFIG.hpss_kernel_size = 41  # Larger = smoother separation
CONFIG.hpss_margin = 4.0      # Higher = stricter separation
```

**C. Use parameter tuning:**
```python
from src.tuning import test_parameters

results = test_parameters(
    df, selection_id=1,
    prop_decrease_values=[0.70, 0.80, 0.90],
    hpss_margins=[2.0, 3.0, 4.0, 5.0]
)
# Listen to outputs to find best params
```

---

### 8. Jupyter Notebook Kernel Crashes

**Cause:** Running out of memory with large batch processing

**Solution:**
```python
# Process one at a time instead of batch
for idx, row in df.iterrows():
    result = process_single_call(...)
    # Force garbage collection
    import gc
    gc.collect()
```

---

### 9. Matplotlib Backend Errors in Notebook

**Error:**
```
RuntimeError: main thread is not in main loop
```

**Solution:** Set backend at notebook start:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
```

---

### 10. Slow Processing Speed

**Solutions:**

**A. Reduce STFT resolution:**
```python
# In config/config.py
CONFIG.n_fft = 2048      # Instead of 4096
CONFIG.hop_length = 1024  # Instead of 512
```

**B. Skip visualization:**
```python
# Edit pipeline.py to comment out spectrogram generation
# Or process audio-only
```

**C. Use multiprocessing:**
```python
from multiprocessing import Pool

def process_wrapper(row):
    return process_single_call(
        row['file_path'], 
        row['Start_time'],
        row['End_time'],
        row['Selection']
    )

with Pool(4) as p:  # 4 parallel workers
    results = p.map(process_wrapper, [row for _, row in df.iterrows()])
```

---

## Getting Help

If none of these solutions work:

1. **Check your versions:**
   ```bash
   python --version
   pip list | grep -E "librosa|numpy|scipy"
   ```

2. **Enable debug output:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Create minimal reproduction:**
   ```python
   # Test on single simple sine wave
   import numpy as np
   sr = 44100
   t = np.linspace(0, 1, sr)
   signal = np.sin(2*np.pi*200*t)
   
   from src.algorithms import algorithm_bandpass_butterworth
   filtered = algorithm_bandpass_butterworth(signal, sr)
   print("Success!" if len(filtered) == len(signal) else "Failed")
   ```

4. **Open GitHub issue:** Include:
   - Error message (full traceback)
   - Python version
   - OS (Windows/Mac/Linux)
   - Package versions
   - Minimal code to reproduce

🔗 [Report Issue](https://github.com/Krish-008/elephant-rumble-denoiser/issues/new)
