# 🚨 Quick Fix Reference Card

## The Error You Just Had: Filename with Spaces

**Error Message:**
```
Error opening 'outputs/audio/selection_001_99-37_airplane_02 copy_cleaned.wav': System error.
```

**What Happened:**
The filename `99-37_airplane_02 copy.wav` had a space in it, which caused `soundfile.write()` to fail.

**✅ Fixed in Latest Version:**
The pipeline now automatically sanitizes filenames:
- Replaces spaces with underscores: `copy.wav` → `copy.wav`
- Removes special characters
- Ensures filesystem compatibility

**Your file will now be saved as:**
```
selection_001_99-37_airplane_02_copy_cleaned.wav
```

---

## Quick Fixes for Common Errors

### 1. "No such file or directory: 'outputs/audio'"
```python
import os
os.makedirs('outputs/audio', exist_ok=True)
os.makedirs('outputs/spectrograms', exist_ok=True)
os.makedirs('outputs/logs', exist_ok=True)
```

### 2. "ModuleNotFoundError: No module named 'config'" (in notebooks)
```python
from notebook_utils import setup_notebook_environment
setup_notebook_environment()
```

### 3. Results too noisy / elephant removed
```python
# Edit config/config.py
CONFIG.noise_params['vehicle']['prop_decrease'] = 0.75  # Less aggressive
# Or
CONFIG.noise_params['vehicle']['prop_decrease'] = 0.90  # More aggressive
```

### 4. Slow processing
```python
# Edit config/config.py
CONFIG.n_fft = 2048  # Faster (less frequency resolution)
CONFIG.hop_length = 1024
```

---

## Notebook Quick Start

```python
# Cell 1: Setup
from notebook_utils import setup_notebook_environment, quick_test
output_dir = setup_notebook_environment()

from config.config import CONFIG
from src.batch_process import load_and_validate_data

# Cell 2: Load data
df = load_and_validate_data('path/to/calls.csv', 'path/to/audio/')

# Cell 3: Test first call
result = quick_test(df, selection_index=0)
```

---

## See Full Solutions

- **Detailed troubleshooting:** See `TROUBLESHOOTING.md`
- **Usage examples:** See `EXAMPLES.md`
- **Quick start:** See `QUICKSTART.md`
- **Full docs:** See `README.md`
