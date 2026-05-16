# AI Learning for Elephant Rumbles

Complete guide to using machine learning on your denoised elephant rumbles.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Feature Extraction](#feature-extraction)
4. [Classical ML](#classical-ml-random-forest-svm)
5. [Deep Learning](#deep-learning)
6. [Unsupervised Learning](#unsupervised-learning)
7. [Use Cases](#use-cases)
8. [Model Zoo](#model-zoo)

---

## Overview

### What Can You Do?

**Classification Tasks:**
- 📊 **Rumble Type Classification** - Contact vs greeting vs mating calls
- 🐘 **Individual Identification** - Which elephant made the call
- 📍 **Geographic Classification** - Which herd/region
- 😊 **Emotional State** - Stress vs calm vs excitement

**Detection Tasks:**
- 🎯 **Rumble Detection** - Is this audio a rumble or noise?
- ⏱️ **Temporal Segmentation** - Find exact start/end of calls
- 🔍 **Call Unit Detection** - Identify subunits within calls

**Unsupervised Tasks:**
- 🗂️ **Call Clustering** - Discover natural groupings
- 🔬 **Anomaly Detection** - Find unusual/rare calls
- 📐 **Feature Learning** - Extract meaningful representations

---

## Quick Start

### Step 1: Install AI Dependencies

```bash
# Core ML
pip install scikit-learn

# Deep Learning (choose one)
pip install torch torchvision  # PyTorch
pip install tensorflow         # TensorFlow

# Visualization
pip install umap-learn  # Better than t-SNE for large datasets
```

### Step 2: Prepare Labels File

Create `data/labels.json`:

```json
{
  "selection_001_airplane_02_cleaned.wav": 0,
  "selection_002_vehicle_03_cleaned.wav": 1,
  "selection_003_generator_01_cleaned.wav": 0,
  ...
}
```

Where labels are:
- **0**: Contact call
- **1**: Greeting call
- **2**: Mating call
- etc.

### Step 3: Extract Features & Train

```python
from src.ai_training import RumbleTrainer

# Initialize
trainer = RumbleTrainer(
    audio_dir='outputs/audio',
    labels_file='data/labels.json',
    output_dir='models'
)

# Extract features from all cleaned audio
features, labels = trainer.extract_all_features()

# Train Random Forest
rf_results = trainer.train_random_forest(n_estimators=200)

# Train Deep Learning (optional)
cnn_results = trainer.train_cnn_pytorch(epochs=100, batch_size=16)
```

**Done!** Models saved to `models/` directory.

---

## Feature Extraction

### What Features Are Extracted?

```python
from src.ai_features import extract_rumble_features
import librosa

# Load cleaned signal
signal, sr = librosa.load('outputs/audio/selection_001_cleaned.wav')

# Extract all features
features = extract_rumble_features(signal, sr)

print(f"Duration: {features.duration:.2f}s")
print(f"Fundamental freq: {features.fundamental_freq:.1f} Hz")
print(f"Harmonic ratio: {features.harmonic_energy_ratio:.3f}")
print(f"MFCCs: {features.mfcc.shape}")  # (20, n_frames)
print(f"Mel spec: {features.mel_spectrogram.shape}")  # (128, n_frames)
```

### Feature Categories

| Category | Features | Use Case |
|----------|----------|----------|
| **Temporal** | Duration, ZCR, Energy, RMS | Call dynamics |
| **Spectral** | Centroid, Rolloff, Bandwidth, Contrast | Timbral quality |
| **Low-Frequency** | Fundamental F0, Harmonic ratio | Elephant-specific |
| **MFCCs** | 20 coefficients + deltas | Pattern recognition |
| **Mel Spectrogram** | 128 mel bands | Deep learning input |
| **Chroma** | 12 pitch classes | Harmonic content |

### Feature Vector for Classical ML

```python
from src.ai_features import features_to_vector

# Convert to flat vector
vector = features_to_vector(
    features,
    include_temporal=True,
    include_spectral=True,
    include_mfcc=True
)

print(f"Feature vector size: {len(vector)}")  # ~100 dimensions
# Use with Random Forest, SVM, etc.
```

---

## Classical ML (Random Forest, SVM)

### Random Forest

**Best for:**
- Small datasets (< 1000 samples)
- Feature importance analysis
- Fast training
- Interpretable results

```python
from src.ai_training import RumbleTrainer

trainer = RumbleTrainer('outputs/audio', 'data/labels.json')
features, labels = trainer.extract_all_features()

# Train
results = trainer.train_random_forest(n_estimators=200)

# Results
print(f"Test accuracy: {results['test_acc']:.3f}")
print(f"Top feature: {results['feature_importance'].argmax()}")

# Use model
from src.ai_features import extract_rumble_features, features_to_vector
import librosa

new_signal, sr = librosa.load('new_rumble.wav')
new_features = extract_rumble_features(new_signal, sr)
new_vector = features_to_vector(new_features)

prediction = results['model'].predict([new_vector])[0]
print(f"Predicted class: {prediction}")
```

### SVM (Support Vector Machine)

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Extract features
trainer = RumbleTrainer('outputs/audio', 'data/labels.json')
X, y = trainer.extract_all_features()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# Scale features (important for SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_train_scaled, y_train)

# Evaluate
print(f"Accuracy: {svm.score(X_test_scaled, y_test):.3f}")
```

---

## Deep Learning

### CNN for Spectrogram Classification

**Best for:**
- Medium/large datasets (> 500 samples)
- Complex patterns in time-frequency
- Higher accuracy potential

```python
from src.ai_training import RumbleTrainer

trainer = RumbleTrainer('outputs/audio', 'data/labels.json')

# Train CNN
results = trainer.train_cnn_pytorch(
    epochs=100,
    batch_size=16
)

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(results['history']['train_loss'], label='Train')
plt.plot(results['history']['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(results['history']['train_acc'], label='Train')
plt.plot(results['history']['val_acc'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training Accuracy')

plt.tight_layout()
plt.show()
```

### LSTM for Temporal Modeling

**Best for:**
- Sequence classification
- Temporal pattern recognition
- Variable-length calls

```python
import torch
from src.ai_models import create_rumble_lstm_pytorch
from src.ai_features import extract_rumble_features

# Create model
model = create_rumble_lstm_pytorch(
    input_dim=20,    # 20 MFCCs
    hidden_dim=128,
    num_layers=2,
    num_classes=5
)

# Prepare data (MFCCs from multiple calls)
mfccs_list = []
labels_list = []

for audio_file in audio_files:
    signal, sr = librosa.load(audio_file)
    features = extract_rumble_features(signal, sr)
    mfccs_list.append(features.mfcc.T)  # (time, features)
    labels_list.append(get_label(audio_file))

# Pad sequences to same length
from torch.nn.utils.rnn import pad_sequence

mfccs_padded = pad_sequence(
    [torch.FloatTensor(m) for m in mfccs_list],
    batch_first=True
)
labels_tensor = torch.LongTensor(labels_list)

# Train (similar to CNN)
```

### Transfer Learning with Pre-trained Models

```python
# Use VGGish (Google's audio model)
import torch
import torchaudio

# Download VGGish
vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
vggish.eval()

# Extract embeddings
with torch.no_grad():
    embeddings = vggish.forward(audio_signal)

# Use embeddings with classifier
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(embeddings, labels)
```

---

## Unsupervised Learning

### Autoencoder for Feature Learning

```python
from src.ai_models import create_autoencoder_keras
from src.ai_dataset import RumbleDataset

# Load data
dataset = RumbleDataset('outputs/audio')
specs, _ = dataset.get_mel_spectrograms(fixed_length=200)

# Create autoencoder
autoencoder = create_autoencoder_keras(
    input_shape=(128, 200),
    latent_dim=64
)

# Train
autoencoder.fit(
    specs[..., np.newaxis],  # Add channel dim
    specs[..., np.newaxis],  # Reconstruct input
    epochs=100,
    batch_size=16,
    validation_split=0.2
)

# Extract learned features
encoder = autoencoder.get_layer('encoder')
latent_features = encoder.predict(specs[..., np.newaxis])

# Cluster latent features
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(latent_features)

print(f"Discovered {len(np.unique(clusters))} call types")
```

### t-SNE / UMAP Visualization

```python
from src.ai_training import RumbleTrainer

trainer = RumbleTrainer('outputs/audio', 'data/labels.json')
trainer.extract_all_features()

# Visualize with t-SNE
trainer.visualize_features(save=True)

# Or use UMAP (better for large datasets)
import umap
import matplotlib.pyplot as plt

reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(trainer.feature_vectors)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=trainer.labels,
    cmap='tab10',
    alpha=0.6
)
plt.colorbar(scatter)
plt.title('UMAP Projection of Rumble Features')
plt.savefig('models/umap_features.png', dpi=150)
```

---

## Use Cases

### 1. Individual Elephant Identification

```python
# Label files with elephant IDs
labels = {
    "selection_001_cleaned.wav": "Ellie",
    "selection_002_cleaned.wav": "Dumbo",
    "selection_003_cleaned.wav": "Ellie",
    ...
}

# Convert names to numbers
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
numeric_labels = encoder.fit_transform(list(labels.values()))

# Train classifier
trainer = RumbleTrainer('outputs/audio')
features, _ = trainer.extract_all_features()

from sklearn.svm import SVC
clf = SVC(kernel='rbf')
clf.fit(features, numeric_labels)

# Predict new call
new_features = extract_features('unknown_call.wav')
elephant_id = encoder.inverse_transform([clf.predict([new_features])])[0]
print(f"This call is from: {elephant_id}")
```

### 2. Call Type Classification

```python
# Contact call (0), Greeting (1), Alarm (2), Mating (3)

results = trainer.train_random_forest()

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(results['true_labels'], results['predictions'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Call Type Classification')
plt.show()
```

### 3. Anomaly Detection

```python
# Find unusual/rare rumbles

from sklearn.ensemble import IsolationForest

# Train on normal calls
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(features)

# Predict
anomaly_scores = iso_forest.decision_function(features)
anomalies = iso_forest.predict(features)

# Get anomalous calls
anomaly_indices = np.where(anomalies == -1)[0]
print(f"Found {len(anomaly_indices)} anomalous calls:")
for idx in anomaly_indices:
    print(f"  - {dataset.audio_files[idx].name}")
```

---

## Model Zoo

### Pre-trained Models (Future)

We're building a repository of pre-trained models:

```python
# Coming soon!
from elephant_rumble_models import load_pretrained

# Load model trained on 10,000 African elephant calls
model = load_pretrained('african_elephant_classifier_v1')

prediction = model.predict('my_rumble.wav')
```

## Next Steps

1. **Collect More Data** - More samples = better models
2. **Label Quality** - Ensure accurate, consistent labels
3. **Hyperparameter Tuning** - Grid search for optimal parameters
4. **Ensemble Methods** - Combine multiple models
5. **Active Learning** - Prioritize labeling difficult examples

---

## References

- **MFCCs**: Logan, B. (2000). "Mel Frequency Cepstral Coefficients for Music Modeling"
- **CNNs for Audio**: Hershey et al. (2017). "CNN Architectures for Large-Scale Audio Classification"
- **Transfer Learning**: Gemmeke et al. (2017). "Audio Set: An ontology and human-labeled dataset"

---

📧 Questions? Open an issue on GitHub!
