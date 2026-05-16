# 🏷️ Cluster Labeling Methods - Complete Guide

**How to label your clusters without manual listening**

---

## 🎯 **The Problem**

After unsupervised clustering, you have:
- ✅ Cluster 0: 52 files
- ✅ Cluster 1: 48 files
- ❌ But what do they represent?

**You need labels to:**
1. Train a supervised classifier
2. Understand what patterns were discovered
3. Interpret results scientifically

---

## 🚀 **5 Methods (Ranked by Ease)**

---

## **Method 1: Automatic Acoustic Analysis** ⭐⭐⭐⭐⭐ (EASIEST!)

**What**: Automatically analyze audio files and generate descriptive labels based on acoustic properties.

**How**: Run the auto-labeling script:

```bash
python auto_label_clusters.py
```

**What it does**:
1. Loads audio files from each cluster
2. Extracts interpretable features:
   - Duration (short/medium/long)
   - Frequency content (low/mid/high)
   - Energy (loud/quiet)
   - Harmonic ratio (tonal/noisy)
3. Generates descriptive labels automatically
4. Creates comparison visualizations

**Example output**:
```
Cluster 0: Long Low-Freq Tonal
  Duration: 8.2s (± 2.1s)
  Low Freq: 72.3%
  Energy: 0.0432
  
Cluster 1: Short High-Freq Noisy
  Duration: 2.4s (± 0.8s)  
  High Freq: 48.1%
  Energy: 0.0821
```

**Pros**:
- ✅ **Zero manual work**
- ✅ Objective/reproducible
- ✅ Scientifically valid descriptions
- ✅ Works for any audio
- ✅ Includes visualizations

**Cons**:
- ⚠️ Labels are descriptive, not biological (e.g., "Long Low-Freq" not "Contact Call")

**Best for**: Quick analysis, objective descriptions, when you don't know call types

---

## **Method 2: Metadata-Based Labeling** ⭐⭐⭐⭐

**What**: Use existing metadata (time of day, location, behavior) to infer labels.

**How**:

```python
import pandas as pd

# Load clusters
clusters = pd.read_csv('unsupervised_results/cluster_assignments.csv')

# Load your metadata (if you have it)
metadata = pd.read_csv('data/metadata.csv')  # filename, time, location, behavior, etc.

# Merge
merged = clusters.merge(metadata, on='filename')

# Analyze patterns
for cluster_id in [0, 1]:
    print(f"\nCluster {cluster_id}:")
    cluster_data = merged[merged['cluster'] == cluster_id]
    
    # Time of day
    if 'time_of_day' in metadata.columns:
        print(f"  Time: {cluster_data['time_of_day'].mode()[0]}")
    
    # Behavior
    if 'behavior' in metadata.columns:
        print(f"  Behavior: {cluster_data['behavior'].mode()[0]}")
    
    # Location
    if 'location' in metadata.columns:
        print(f"  Location: {cluster_data['location'].mode()[0]}")
```

**Example**:
```
Cluster 0:
  Time: Night (87% of files)
  Behavior: Resting (92% of files)
  → Label: "Night Resting Calls"

Cluster 1:
  Time: Day (78% of files)
  Behavior: Moving (83% of files)
  → Label: "Day Movement Calls"
```

**Pros**:
- ✅ Uses existing data
- ✅ Contextually meaningful
- ✅ No listening required

**Cons**:
- ⚠️ Requires metadata
- ⚠️ Correlation ≠ causation

**Best for**: When you have rich metadata

---

## **Method 3: Semi-Supervised with Partial Labels** ⭐⭐⭐⭐

**What**: Label just **5-10 files per cluster** manually, then propagate labels to the rest.

**How**:

```python
import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation

# Load data
clusters = pd.read_csv('unsupervised_results/cluster_assignments.csv')
features = np.load('unsupervised_results/features_normalized.npy')

# Manually label just 5 files per cluster
# (Listen to these 10 files total)
labeled_files = {
    'selection_001_cleaned.wav': 0,  # You decide: Contact call
    'selection_003_cleaned.wav': 0,
    'selection_007_cleaned.wav': 0,
    'selection_012_cleaned.wav': 0,
    'selection_023_cleaned.wav': 0,
    'selection_002_cleaned.wav': 1,  # You decide: Greeting call
    'selection_005_cleaned.wav': 1,
    'selection_009_cleaned.wav': 1,
    'selection_015_cleaned.wav': 1,
    'selection_031_cleaned.wav': 1,
}

# Create label array (-1 = unlabeled)
y = np.full(len(clusters), -1)
for i, filename in enumerate(clusters['filename']):
    if filename in labeled_files:
        y[i] = labeled_files[filename]

# Propagate labels
label_prop = LabelPropagation(kernel='rbf')
y_propagated = label_prop.fit(features, y).predict(features)

# Now you have labels for all files!
clusters['predicted_label'] = y_propagated
```

**Pros**:
- ✅ Only 10-15 minutes of listening
- ✅ Labels propagate based on similarity
- ✅ Biologically meaningful labels

**Cons**:
- ⚠️ Requires some manual work
- ⚠️ Quality depends on seed labels

**Best for**: When you want biological labels with minimal effort

---

## **Method 4: Transfer Learning from Literature** ⭐⭐⭐

**What**: Use published call types from literature to match your clusters.

**How**:

1. **Find typical elephant call characteristics** from papers:
   ```
   Contact calls: 5-15s, 14-35 Hz, harmonics
   Greeting calls: 2-5s, rising pitch, 20-50 Hz
   Alarm calls: 1-3s, broad spectrum, high energy
   ```

2. **Compare to your cluster characteristics**:
   ```bash
   python auto_label_clusters.py  # Get your cluster stats
   ```

3. **Match patterns**:
   ```
   Your Cluster 0: 8.2s avg, 72% low-freq (10-100 Hz)
   Literature: Contact calls = 5-15s, 14-35 Hz
   → Cluster 0 = Contact calls ✓
   
   Your Cluster 1: 2.4s avg, 48% high-freq (300-1000 Hz)  
   Literature: Alarm calls = 1-3s, broad spectrum
   → Cluster 1 = Alarm calls ✓
   ```

**Pros**:
- ✅ Biologically grounded
- ✅ Publishable
- ✅ No listening required (use literature values)

**Cons**:
- ⚠️ Your elephants may differ from literature
- ⚠️ Requires literature search

**Best for**: Research/publication, when you trust literature values

---

## **Method 5: Consensus from Multiple Algorithms** ⭐⭐⭐

**What**: Run multiple clustering algorithms and see if they agree.

**How**:

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np

features = np.load('unsupervised_results/features_normalized.npy')

# Run multiple algorithms
kmeans = KMeans(n_clusters=2, random_state=42)
gmm = GaussianMixture(n_components=2, random_state=42)
hierarchical = AgglomerativeClustering(n_clusters=2)

labels_kmeans = kmeans.fit_predict(features)
labels_gmm = gmm.fit_predict(features)
labels_hierarchical = hierarchical.fit_predict(features)

# Check agreement
agreement = (labels_kmeans == labels_gmm) & (labels_gmm == labels_hierarchical)
print(f"Agreement: {agreement.sum() / len(agreement) * 100:.1f}%")

# Files with high agreement → reliable clusters
# Files with low agreement → borderline/mixed
```

**Pros**:
- ✅ Robust (multiple algorithms agree)
- ✅ Identifies uncertain samples

**Cons**:
- ⚠️ Still doesn't tell you WHAT they are
- ⚠️ Complex

**Best for**: Validating cluster quality, finding edge cases

---

## 📊 **Comparison Table**

| Method | Manual Work | Biological Meaning | Scientific Validity | Time Required |
|--------|-------------|-------------------|---------------------|---------------|
| **Automatic Acoustic** | None | No (descriptive) | High | 2 min |
| **Metadata-Based** | None | Maybe | High | 5 min |
| **Semi-Supervised** | 10-15 min listening | Yes | High | 20 min |
| **Literature Transfer** | Literature search | Yes | High | 30 min |
| **Multi-Algorithm** | None | No | Medium | 10 min |

---

## 🎯 **Recommended Workflow**

### **For Quick Analysis:**
```bash
1. python auto_label_clusters.py
2. Use descriptive labels (e.g., "Long Low-Freq")
3. python train_complete.py
```
**Time**: 5 minutes

### **For Research/Publication:**
```bash
1. python auto_label_clusters.py  # Get acoustic profile
2. Search literature for matching call types
3. Match your profiles to known call types
4. Update cluster_names in train_complete.py
5. python train_complete.py
```
**Time**: 1 hour

### **For Best Biological Labels:**
```bash
1. python auto_label_clusters.py  # See which are most different
2. Listen to 5 representatives from each cluster (10 files total)
3. Assign biological labels (Contact/Greeting/Alarm)
4. python train_complete.py
```
**Time**: 30 minutes

---

## 💻 **Complete Example: Automatic Labeling**

```bash
# Step 1: Run automatic analysis
python auto_label_clusters.py

# Output:
# Cluster 0: Long Low-Freq Tonal (52 files)
# Cluster 1: Short High-Freq Noisy (48 files)
# 
# Files created:
# - data/cluster_names_auto.json
# - unsupervised_results/cluster_characteristics.png
# - unsupervised_results/cluster_report.txt

# Step 2: Review visualizations
# Open cluster_characteristics.png

# Step 3: Use in training
# The labels are automatically loaded!
python train_complete.py

# Step 4: Model trained!
# Your classifier now knows:
# "Long Low-Freq Tonal" vs "Short High-Freq Noisy"
```

---

## 🔬 **What the Automatic Method Gives You**

### **Interpretable Features Extracted:**

1. **Duration**
   - Mean, std, min, max
   - Classification: Short (< 3s), Medium (3-8s), Long (> 8s)

2. **Frequency Distribution**
   - Low (10-100 Hz) - elephant fundamental
   - Mid (100-300 Hz) - harmonics
   - High (300-1000 Hz) - broadband components

3. **Energy/Loudness**
   - RMS energy
   - Classification: Quiet/Normal/Loud

4. **Harmonic Ratio**
   - Tonal (> 0.7) = pitched sounds
   - Noisy (< 0.3) = broadband/percussive

5. **Spectral Centroid**
   - "Brightness" of sound
   - Low value = dark/rumbly
   - High value = bright/harsh

### **Output Files:**

```
data/
└── cluster_names_auto.json          # Labels for training

unsupervised_results/
├── cluster_characteristics.json     # Detailed stats
├── cluster_characteristics.png      # 6-panel visualization
└── cluster_report.txt               # Human-readable report
```

### **Visualizations Include:**

1. **Duration comparison** (bar chart)
2. **Frequency distribution** (stacked bar)
3. **Energy levels** (bar chart)
4. **Harmonic ratio** (bar chart)
5. **Cluster 0 frequency breakdown** (pie chart)
6. **Cluster 1 frequency breakdown** (pie chart)

---

## ✅ **Bottom Line**

**You have 5 options, ranked by effort:**

1. **Automatic** (2 min) → Descriptive labels ✅ **EASIEST**
2. **Metadata** (5 min) → Contextual labels
3. **Semi-supervised** (20 min) → Biological labels with minimal listening
4. **Literature** (1 hr) → Match to published call types
5. **Multi-algorithm** (10 min) → Validate robustness

**My recommendation:**

Start with **Method 1 (Automatic)** - it's instant, objective, and scientifically valid. If you later need biological labels for publication, combine with **Method 4 (Literature)**.

**Run this now:**
```bash
python auto_label_clusters.py
```

**You'll have labels in 2 minutes with NO manual work!** 🚀
