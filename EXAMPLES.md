# Usage Examples

## Basic Usage

### 1. Command Line - Test Single Call

```bash
python main.py --csv data/calls.csv --audio data/audio --test
```

### 2. Command Line - Batch Process All Calls

```bash
python main.py --csv data/calls.csv --audio data/audio
```

### 3. Command Line - Skip Archive Creation

```bash
python main.py --csv data/calls.csv --audio data/audio --no-archive
```

## Python API Usage

### Process Single Call

```python
from src.pipeline import process_single_call

result = process_single_call(
    audio_path='data/audio/recording.wav',
    start_time=5.2,
    end_time=8.7,
    selection_id=1,
    output_dir='outputs'
)

print(f"Status: {result['status']}")
print(f"Output: {result['output_audio']}")
```

### Batch Processing

```python
from src.batch_process import load_and_validate_data, batch_process

# Load data
df = load_and_validate_data(
    csv_path='data/calls.csv',
    audio_folder='data/audio'
)

# Process all calls
results_df = batch_process(df, output_dir='outputs')

# View summary
print(results_df[results_df['status']=='success'].groupby('noise_type').size())
```

### Custom Configuration

```python
from config.config import CONFIG

# Modify global config
CONFIG.bp_lowcut = 15.0  # Lower cutoff to 15 Hz
CONFIG.bp_highcut = 1200.0  # Raise upper cutoff to 1200 Hz
CONFIG.hpss_margin = 4.0  # Stricter HPSS separation

CONFIG.print_summary()

# Now process with new settings
from src.pipeline import process_single_call
result = process_single_call(...)
```

### Parameter Tuning

```python
from src.batch_process import load_and_validate_data
from src.tuning import test_parameters

df = load_and_validate_data('data/calls.csv', 'data/audio')

# Test different configurations on Selection 1
results = test_parameters(
    df,
    selection_id=1,
    prop_decrease_values=[0.70, 0.80, 0.90],  # Spectral gating strength
    hpss_margins=[2.0, 3.0, 4.0, 5.0]         # HPSS separation
)

# View results
print(results[['prop_decrease', 'hpss_margin', 'status', 'output_audio']])
```

### Noise Profile Validation

```python
import librosa
from src.noise_utils import extract_noise_profile, validate_noise_profile

# Load audio
y, sr = librosa.load('data/audio/recording.wav')

# Extract noise profile before call (samples 0-44100)
noise_profile, source = extract_noise_profile(
    audio=y,
    sr=sr,
    call_start=44100,  # Call starts at 1 second
    call_end=88200,    # Call ends at 2 seconds
    mode='before'
)

# Validate noise profile
metrics = validate_noise_profile(noise_profile, sr)

print(f"Noise source: {source}")
print(f"Fundamental ratio: {metrics['fundamental_ratio']:.4f}")
print(f"Elephant band ratio: {metrics['elephant_ratio']:.4f}")
```

### Direct Algorithm Usage

```python
import numpy as np
import librosa
from src.algorithms import (
    algorithm_bandpass_butterworth,
    algorithm_spectral_gating,
    algorithm_hpss,
    algorithm_wiener_filter
)

# Load signal
signal, sr = librosa.load('audio.wav')

# Apply individual algorithms
bp_filtered = algorithm_bandpass_butterworth(signal, sr, lowcut=20, highcut=1000)
harmonic, percussive = algorithm_hpss(bp_filtered, sr, kernel_size=31, margin=3.0)
wiener_filtered = algorithm_wiener_filter(harmonic, window_size=29)
```

### Visualization Only

```python
import librosa
from src.visualization import create_bw_spectrogram, create_comparison_plot

# Load signals
original, sr = librosa.load('original.wav')
cleaned, _ = librosa.load('cleaned.wav')

# Create spectrogram
spec_path = create_bw_spectrogram(
    signal=cleaned,
    sr=sr,
    title='Cleaned Elephant Rumble',
    save_path='outputs/my_spec.png',
    show_colorbar=True
)

# Create comparison
comp_path = create_comparison_plot(
    original=original,
    cleaned=cleaned,
    sr=sr,
    title='Before/After Denoising',
    save_path='outputs/comparison.png'
)
```

## Advanced: Custom Pipeline

```python
import numpy as np
import librosa
import soundfile as sf
from config.config import CONFIG
from src.algorithms import *
from src.noise_utils import *

# Load your data
audio, sr = librosa.load('my_recording.wav')
call_segment = audio[10000:50000]  # Extract segment

# Extract noise profile
noise_profile, _ = extract_noise_profile(
    audio, sr, 
    call_start=10000, 
    call_end=50000,
    mode='adaptive'
)

# Custom pipeline
step1 = algorithm_bandpass_butterworth(call_segment, sr, lowcut=25, highcut=800)
step2 = algorithm_spectral_gating(step1, noise_profile, sr, noise_type='vehicle')
step3_harm, step3_perc = algorithm_hpss(step2, sr, kernel_size=41, margin=4.0)
final = algorithm_wiener_filter(step3_harm, window_size=21)

# Normalize and save
final = final / np.max(np.abs(final)) * 0.7
sf.write('outputs/my_cleaned.wav', final, sr)
```

## Jupyter Notebook

See `notebooks/interactive_demo.ipynb` for an interactive demonstration with:
- Parameter exploration
- Live audio playback
- Visualization
- Iterative tuning
