# 🎵 Acoustic Feature Extraction Libraries for Python

Complete guide to specialized libraries for audio/bioacoustic feature extraction.

---

## 📚 Top Libraries

### 1. **openSMILE** ⭐⭐⭐⭐⭐ (BEST for Speech/Bioacoustics)

**What**: Industry-standard audio feature extraction (used in emotion recognition, speech analysis)

**Features**: 6,000+ acoustic features including:
- Prosodic features (pitch, energy contours)
- Voice quality (jitter, shimmer, harmonicity)
- Spectral features (MFCC, LSP, formants)
- Temporal features (zero-crossing, attack time)

**Installation**:
```bash
pip install opensmile
```

**Usage**:
```python
import opensmile

# Use pre-configured feature sets
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,  # 6,373 features!
    feature_level=opensmile.FeatureLevel.Functionals
)

# Extract features from audio
features = smile.process_file('rumble.wav')
print(features.shape)  # (1, 6373) - one vector per file

# Or use for bioacoustics
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,  # 88 features
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors
)
features = smile.process_file('rumble.wav')
```

**Pre-configured sets**:
- `ComParE_2016`: 6,373 features (comprehensive)
- `eGeMAPSv02`: 88 features (efficient)
- `GeMAPSv01b`: 62 features (minimalistic)

**Pros**:
- ✅ Industry standard (used in papers)
- ✅ Huge feature set
- ✅ Fast C++ backend
- ✅ Pre-configured for different tasks

**Cons**:
- ⚠️ Can be overwhelming (6,000+ features)
- ⚠️ Less control over individual features

---

### 2. **pyAudioAnalysis** ⭐⭐⭐⭐ (Great for Music/General Audio)

**What**: Comprehensive audio analysis library with 34 standard features

**Features**:
- Short-term features (MFCCs, chroma, spectral)
- Mid-term statistics
- Beat tracking
- Segmentation
- Classification built-in

**Installation**:
```bash
pip install pyAudioAnalysis
```

**Usage**:
```python
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures

# Load audio
[Fs, x] = audioBasicIO.read_audio_file("rumble.wav")

# Extract short-term features (34 features per window)
features, feature_names = ShortTermFeatures.feature_extraction(
    x, Fs, 
    window=1.0 * Fs,  # 1 second window
    step=0.5 * Fs     # 0.5 second hop
)

print(f"Features shape: {features.shape}")  # (34, num_windows)
print(f"Feature names: {feature_names}")

# Features include:
# - Zero crossing rate
# - Energy, Energy entropy
# - Spectral: centroid, spread, entropy, flux, rolloff
# - MFCCs (13)
# - Chroma vector (12)
# - Chroma deviation
```

**34 Features extracted**:
1. Zero Crossing Rate
2. Energy
3. Energy Entropy
4. Spectral Centroid
5. Spectral Spread
6. Spectral Entropy
7. Spectral Flux
8. Spectral Rolloff
9-21. MFCCs (13)
22-33. Chroma Vector (12)
34. Chroma Deviation

**Pros**:
- ✅ Easy to use
- ✅ Good documentation
- ✅ Built-in classification
- ✅ Music-specific features

**Cons**:
- ⚠️ Fewer features than openSMILE
- ⚠️ Less active development

---

### 3. **Essentia** ⭐⭐⭐⭐⭐ (BEST for Music, Research-Grade)

**What**: C++ library from Music Technology Group (Barcelona), Python bindings

**Features**: 200+ algorithms including:
- Low-level (spectral, temporal)
- Rhythm (beat tracking, tempo)
- Tonal (key, chords, pitch)
- High-level (mood, genre)

**Installation**:
```bash
pip install essentia
```

**Usage**:
```python
import essentia
import essentia.standard as es

# Load audio
audio = es.MonoLoader(filename='rumble.wav', sampleRate=44100)()

# Extract features
# Spectral features
spec = es.Spectrum(size=2048)
spectral_centroid = es.Centroid()
spectral_rolloff = es.RollOff()

# MFCC
mfcc = es.MFCC(
    numberCoefficients=20,
    inputSize=1025,
    sampleRate=44100,
    lowFrequencyBound=10,
    highFrequencyBound=300
)

# Extract per frame
for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512):
    spectrum = spec(frame)
    mfcc_bands, mfcc_coeffs = mfcc(spectrum)
    
    centroid = spectral_centroid(spectrum)
    rolloff = spectral_rolloff(spectrum)

# Or use high-level extractors
features, features_frames = es.MusicExtractor(
    lowlevelStats=['mean', 'stdev'],
    rhythmStats=['mean', 'stdev']
)(filename='rumble.wav')

print(features.descriptorNames())  # 100s of features!
```

**Pros**:
- ✅ Extremely comprehensive
- ✅ Very fast (C++ backend)
- ✅ Research-grade quality
- ✅ Actively maintained

**Cons**:
- ⚠️ Steeper learning curve
- ⚠️ Music-focused (but works for bioacoustics)

---

### 4. **torchopenl3** / **openl3** ⭐⭐⭐⭐ (Deep Learning Embeddings)

**What**: Pre-trained deep learning embeddings for audio

**Features**:
- 512-dim or 6144-dim embeddings
- Trained on AudioSet (millions of audio samples)
- Transfer learning ready

**Installation**:
```bash
pip install openl3
# or
pip install torchopenl3  # PyTorch version
```

**Usage**:
```python
import openl3
import soundfile as sf

# Load audio
audio, sr = sf.read('rumble.wav')

# Extract embeddings
emb, ts = openl3.get_audio_embedding(
    audio, sr,
    content_type="env",  # 'env' for environmental sounds, 'music' for music
    embedding_size=512,   # or 6144
    hop_size=0.5          # seconds
)

print(emb.shape)  # (num_windows, 512)

# Use as features for classification
from sklearn.svm import SVC
clf = SVC()
clf.fit(emb, labels)
```

**Pros**:
- ✅ Pre-trained (no training needed!)
- ✅ State-of-the-art representations
- ✅ Works well with small datasets
- ✅ Transfer learning

**Cons**:
- ⚠️ Less interpretable than handcrafted features
- ⚠️ Larger memory footprint

---

### 5. **madmom** ⭐⭐⭐ (Music Information Retrieval)

**What**: Audio signal processing library focused on music

**Features**:
- Beat tracking
- Tempo estimation  
- Onset detection
- Chord recognition
- Spectral features

**Installation**:
```bash
pip install madmom
```

**Usage**:
```python
import madmom

# Beat tracking
proc = madmom.features.beats.RNNBeatProcessor()
act = proc('rumble.wav')

beat_tracker = madmom.features.beats.BeatTrackingProcessor(fps=100)
beats = beat_tracker(act)

# Spectral features
spec = madmom.audio.spectrogram.Spectrogram('rumble.wav', fps=100)
log_spec = madmom.audio.spectrogram.LogarithmicSpectrogram('rumble.wav')
```

**Pros**:
- ✅ Excellent for rhythm analysis
- ✅ Deep learning models included

**Cons**:
- ⚠️ Music-specific
- ⚠️ Limited low-level features

---

### 6. **Kaldi** ⭐⭐⭐⭐⭐ (Speech Recognition Features)

**What**: Industry-standard speech recognition toolkit (via pykaldi)

**Features**:
- MFCC, PLP, Filterbanks
- Pitch extraction
- Voice activity detection
- Speaker recognition features

**Installation**:
```bash
pip install kaldi-python
# or use kaldi_io for simpler interface
pip install kaldi_io
```

**Usage**:
```python
from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.matrix import Vector

# MFCC configuration
mfcc_opts = MfccOptions()
mfcc_opts.frame_opts.frame_length_ms = 25.0
mfcc_opts.frame_opts.frame_shift_ms = 10.0
mfcc_opts.num_ceps = 20

# Extract MFCCs
mfcc = Mfcc(mfcc_opts)
features = mfcc.compute_features(waveform, sample_rate, vtln_warp=1.0)
```

**Pros**:
- ✅ Industry-standard for speech
- ✅ Optimized for low-frequency
- ✅ Robust to noise

**Cons**:
- ⚠️ Complex installation
- ⚠️ Steep learning curve

---

### 7. **VGGish** / **YAMNet** ⭐⭐⭐⭐ (Google's Audio Embeddings)

**What**: Pre-trained audio classification models from Google

**Installation**:
```bash
# VGGish
pip install tensorflow tensorflow_hub

# YAMNet
pip install tensorflow tensorflow_hub
```

**VGGish Usage**:
```python
import tensorflow_hub as hub
import numpy as np

# Load pre-trained VGGish
vggish = hub.load('https://tfhub.dev/google/vggish/1')

# Extract embeddings (128-dim)
embeddings = vggish(audio_waveform)  # (num_windows, 128)
```

**YAMNet Usage** (more recent):
```python
import tensorflow_hub as hub

yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
scores, embeddings, spectrogram = yamnet(audio_waveform)

print(embeddings.shape)  # (num_windows, 1024)
```

**Pros**:
- ✅ Pre-trained on AudioSet (2M+ samples)
- ✅ Very strong representations
- ✅ Easy to use

**Cons**:
- ⚠️ Requires TensorFlow
- ⚠️ Less control over features

---

## 🎯 Comparison Table

| Library | # Features | Speed | Ease of Use | Best For |
|---------|-----------|-------|-------------|----------|
| **openSMILE** | 6,373 | ⚡⚡⚡ Fast | ⭐⭐⭐ Medium | Speech, Emotion, Bioacoustics |
| **pyAudioAnalysis** | 34 | ⚡⚡ Medium | ⭐⭐⭐⭐⭐ Easy | General audio, Music |
| **Essentia** | 200+ | ⚡⚡⚡ Fast | ⭐⭐⭐ Medium | Music, Research |
| **openl3** | 512/6144 | ⚡⚡ Medium | ⭐⭐⭐⭐ Easy | Transfer learning |
| **Kaldi** | Custom | ⚡⚡⚡ Fast | ⭐⭐ Hard | Speech recognition |
| **VGGish/YAMNet** | 128/1024 | ⚡⚡ Medium | ⭐⭐⭐⭐⭐ Easy | Deep embeddings |
| **librosa** (baseline) | Flexible | ⚡⚡ Medium | ⭐⭐⭐⭐ Easy | Baseline, Custom |

---

## 🐘 **Recommendation for Elephant Rumbles**

### **Option 1: openSMILE (Best Overall)**

```python
import opensmile
import pandas as pd

# Use eGeMAPSv02 (88 features, designed for paralinguistics)
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

# Process all cleaned rumbles
features_list = []
labels = []

for audio_file in audio_files:
    features = smile.process_file(audio_file)
    features_list.append(features)
    labels.append(get_label(audio_file))

# Convert to DataFrame
features_df = pd.concat(features_list, ignore_index=True)

# Train classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(features_df, labels)
```

**Why**: 
- ✅ Designed for paralinguistics (emotion, speaker traits)
- ✅ Works well for low-frequency vocalizations
- ✅ 88 features is manageable
- ✅ Fast extraction

---

### **Option 2: openl3 (For Small Datasets)**

```python
import openl3
import soundfile as sf
import numpy as np

# Extract embeddings for all files
all_embeddings = []
all_labels = []

for audio_file in audio_files:
    audio, sr = sf.read(audio_file)
    
    # Extract 512-dim embeddings
    emb, ts = openl3.get_audio_embedding(
        audio, sr,
        content_type="env",  # Environmental sounds
        embedding_size=512,
        hop_size=1.0  # 1 second windows
    )
    
    # Aggregate windows (mean)
    file_embedding = np.mean(emb, axis=0)
    
    all_embeddings.append(file_embedding)
    all_labels.append(get_label(audio_file))

# Train
from sklearn.svm import SVC
clf = SVC(kernel='rbf')
clf.fit(all_embeddings, all_labels)
```

**Why**:
- ✅ Pre-trained (no feature engineering)
- ✅ Works with < 100 samples
- ✅ Transfer learning

---

### **Option 3: Combine Both (Hybrid)**

```python
# Extract both handcrafted AND deep features
import opensmile
import openl3
import numpy as np

def extract_hybrid_features(audio_file):
    # 1. openSMILE features
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02)
    handcrafted = smile.process_file(audio_file).values[0]  # (88,)
    
    # 2. openl3 embeddings
    audio, sr = sf.read(audio_file)
    deep_emb, _ = openl3.get_audio_embedding(audio, sr, embedding_size=512)
    deep = np.mean(deep_emb, axis=0)  # (512,)
    
    # 3. Combine
    combined = np.concatenate([handcrafted, deep])  # (600,)
    
    return combined

# Use for all files
features = [extract_hybrid_features(f) for f in audio_files]
```

**Why**:
- ✅ Best of both worlds
- ✅ Handcrafted = interpretable
- ✅ Deep = powerful patterns
- ✅ Usually best performance

---

## 📦 Integration with Your Pipeline

Let me update your neural network to use these libraries:

```python
# In neural_classifier.py, replace _extract_window_features

def _extract_window_features_opensmile(self, signal: np.ndarray, sr: int):
    """Extract features using openSMILE."""
    import opensmile
    import tempfile
    import soundfile as sf
    
    # Save window to temp file (openSMILE needs file)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, signal, sr)
        
        # Extract eGeMAPSv02 features (88 features)
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors
        )
        features = smile.process_file(tmp.name)
        
        os.unlink(tmp.name)
    
    # Return as numpy array
    return features.values[0].astype(np.float32)


def _extract_window_features_openl3(self, signal: np.ndarray, sr: int):
    """Extract features using openl3."""
    import openl3
    
    # Extract embeddings
    emb, _ = openl3.get_audio_embedding(
        signal, sr,
        content_type="env",
        embedding_size=512,
        hop_size=0.1  # Dense sampling
    )
    
    # Aggregate (mean)
    features = np.mean(emb, axis=0)
    
    return features.astype(np.float32)
```

---

## 🎓 Summary

**For Elephant Rumbles:**

1. **Best single library**: **openSMILE** (eGeMAPSv02)
   - 88 features
   - Designed for vocalizations
   - Fast & interpretable

2. **Best for small data**: **openl3**
   - Pre-trained embeddings
   - Transfer learning
   - No feature engineering

3. **Best performance**: **Hybrid** (openSMILE + openl3)
   - Combine both
   - 600 features total
   - Best accuracy

4. **Already works well**: **Your current implementation**
   - 39 custom features
   - Elephant-optimized
   - Good performance

**Install them all:**
```bash
pip install opensmile openl3 pyAudioAnalysis essentia
```

You now have access to 6,000+ acoustic features from world-class libraries! 🎵🐘
