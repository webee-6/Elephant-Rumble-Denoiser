# 🪟 Windowing & Segmentation in the Pipeline

## Current Implementation

### ✅ **Algorithms with Built-in Windowing**

#### 1. **HPSS (Harmonic-Percussive Source Separation)**
```python
# src/algorithms.py line 165
D = librosa.stft(signal, n_fft=4096, hop_length=512)
```

**Windowing Details:**
- **Window Type**: Hann window (default in librosa)
- **Window Size**: 4096 samples (~93ms at 44.1kHz)
- **Hop Size**: 512 samples (~11.6ms)
- **Overlap**: 88% ((4096-512)/4096)
- **Purpose**: Convert to time-frequency representation for median filtering

**Why Hann Window?**
- Smooth edges → no spectral leakage
- Good frequency resolution
- Standard for STFT analysis

#### 2. **Spectral Gating (noisereduce)**
```python
# src/algorithms.py line 124
denoised = nr.reduce_noise(y=signal, y_noise=noise_profile, sr=sr, ...)
```

**Windowing Details:**
- Uses STFT internally (noisereduce library)
- Default: 2048 samples window
- Overlapping frames for spectral subtraction
- Smoothing in time/frequency via parameters:
  - `freq_mask_smooth_hz`: 100 Hz
  - `time_mask_smooth_ms`: 50 ms

#### 3. **Wiener Filter**
```python
# src/algorithms.py line 192
filtered = wiener(signal, mysize=29)
```

**Windowing Details:**
- **Window Size**: 29 samples (local neighborhood)
- **Type**: Sliding rectangular window
- **Purpose**: Local mean/variance estimation for MSE-optimal filtering

### ❌ **Algorithms WITHOUT Explicit Windowing**

#### 1. **Butterworth Band-Pass Filter**
```python
# src/algorithms.py line 41
filtered = filtfilt(b, a, signal)
```

**Current Approach:**
- Applied to **entire signal** at once
- No segmentation
- Uses scipy.signal.filtfilt (zero-phase)

**Potential Issues:**
- Long signals (>5 min) → memory intensive
- Edge effects at very beginning/end
- Not parallelizable

#### 2. **Notch Filter (Generator Harmonics)**
```python
# src/algorithms.py line 62
filtered = filtfilt(b, a, filtered)  # Applied 10 times
```

**Current Approach:**
- Applied to entire signal
- 10 sequential applications (one per harmonic)

**Potential Issues:**
- Same as Butterworth
- Cascaded filters can accumulate phase distortion

---

## 🚀 NEW: Enhanced Segmentation Module

### When to Use Segmentation

**Enable for:**
- ✅ Calls > 30 seconds
- ✅ Very long recordings (5+ minutes)
- ✅ Memory-constrained environments
- ✅ Real-time processing requirements

**Skip for:**
- ❌ Short calls (< 30 seconds) - overhead not worth it
- ❌ When you need absolute phase coherence across entire signal

### New `segmentation.py` Module

#### 1. **Overlap-Add Segmentation**
```python
from src.segmentation import windowed_filter

filtered = windowed_filter(
    long_signal,
    algorithm_bandpass_butterworth,
    segment_length=44100 * 30,  # 30 seconds
    overlap=0.25,                # 25% overlap
    sr=44100,
    lowcut=20,
    highcut=1000
)
```

**How it Works:**
```
Signal: [========================================]
         ↓
Segment 1:  [----------]
Segment 2:      [----------]  ← 25% overlap
Segment 3:          [----------]
         ↓
Process each with fade in/out in overlap regions
         ↓
Reconstruct: [========================================]
```

**Benefits:**
- ✅ Prevents edge artifacts
- ✅ Memory efficient
- ✅ Parallelizable
- ✅ No quality loss with proper overlap

#### 2. **Adaptive Segment Sizing**
```python
from src.segmentation import adaptive_segment_length

# Automatically calculate optimal size
segment_len = adaptive_segment_length(
    signal_length=len(signal),
    sr=44100,
    max_memory_mb=500  # Target memory usage
)
```

**Calculates based on:**
- STFT memory requirements
- n_fft size
- Available RAM
- Recommended: 10-60 second segments

#### 3. **Enhanced Pipeline**
```python
from src.pipeline_segmented import process_single_call_segmented

result = process_single_call_segmented(
    audio_path='long_recording.wav',
    start_time=0,
    end_time=300,  # 5 minute call
    selection_id=1,
    use_segmentation=True,      # Enable auto-segmentation
    segment_duration=30.0        # 30 second segments
)
```

**Auto-detects:**
- If call > 30s → segments Butterworth & Notch filters
- If call < 30s → processes normally
- HPSS and Spectral Gating → always use internal windowing

---

## 📊 Windowing Comparison

| Algorithm | Window Type | Window Size | Overlap | Purpose |
|-----------|-------------|-------------|---------|---------|
| **HPSS STFT** | Hann | 4096 (~93ms) | 88% | Time-freq separation |
| **Spectral Gating** | Hann (internal) | 2048 (~46ms) | ~75% | Noise estimation |
| **Wiener** | Rectangular | 29 (~0.7ms) | Sliding | Local statistics |
| **Butterworth** | None* | Full signal | N/A | Frequency filtering |
| **Notch** | None* | Full signal | N/A | Harmonic removal |

*New segmented version uses 25% overlap with Hann fading

---

## 🔬 Technical Details

### STFT Windowing Math

**Forward:**
```
X[m,k] = Σ(n=0 to N-1) x[n] · w[n-mH] · e^(-j2πkn/N)

where:
  m = frame index
  k = frequency bin
  H = hop_length
  w = window function (Hann)
```

**Hann Window:**
```
w[n] = 0.5 - 0.5·cos(2πn/(N-1))
```

**Overlap-Add Reconstruction:**
```
                sum over m of ( X_m[n] · w[n] )
y[n] = ────────────────────────────────────────
                sum over m of ( w[n] )
```

### Why 25% Overlap for Segmentation?

**Trade-offs:**
| Overlap | Pro | Con |
|---------|-----|-----|
| 0% | Fast, no redundancy | Edge artifacts |
| 25% | Good artifact reduction | Minimal overhead |
| 50% | Smooth transitions | 2x computation |
| 75% | Very smooth | 4x computation |

**Recommendation**: 25% overlap = sweet spot
- Smooth enough to eliminate audible artifacts
- Only 33% computation overhead
- Standard in audio processing

---

## 💡 Best Practices

### 1. **Short Calls (< 30s)**
```python
# Use standard pipeline
from src.pipeline import process_single_call
result = process_single_call(...)
```
**Why**: Overhead of segmentation not worth it

### 2. **Long Calls (30s - 5min)**
```python
# Use segmented pipeline with default settings
from src.pipeline_segmented import process_single_call_segmented
result = process_single_call_segmented(
    ...,
    use_segmentation=True,
    segment_duration=30.0
)
```
**Why**: Good balance of efficiency and quality

### 3. **Very Long Calls (> 5min)**
```python
# Use adaptive segmentation
from src.segmentation import adaptive_segment_length, windowed_filter

segment_len = adaptive_segment_length(len(signal), sr, max_memory_mb=200)

filtered = windowed_filter(
    signal,
    algorithm_bandpass_butterworth,
    segment_length=segment_len,
    overlap=0.25,
    sr=sr
)
```
**Why**: Prevents memory issues

### 4. **Batch Processing Mixed Durations**
```python
# Auto-detect in pipeline
for _, row in df.iterrows():
    duration = row['End_time'] - row['Start_time']
    
    if duration > 30:
        result = process_single_call_segmented(...)
    else:
        result = process_single_call(...)
```

---

## 🎯 When Segmentation Matters Most

### ✅ **High Impact:**
1. **Memory-constrained systems** (< 8GB RAM)
2. **Very long recordings** (> 5 minutes)
3. **Batch processing** hundreds of files
4. **Real-time streaming** applications

### ⚠️ **Low Impact:**
1. **Short calls** (< 30 seconds) - overhead > benefit
2. **Desktop with 16+ GB RAM** - can process 5min files easily
3. **Already using STFT-based methods** (HPSS, spectral gating) - windowed internally

---

## 📖 References

- **Overlap-Add**: Smith, J.O. "Spectral Audio Signal Processing"
- **Hann Window**: Harris, F.J. (1978) "On the use of windows for harmonic analysis"
- **STFT**: Allen, J.B. (1977) "Short term spectral analysis"
