# 🐘 Elephant Rumble Analysis - Quick Reference

**Complete workflow from raw audio to discovered patterns (no labels needed!)**

---

## 📋 Complete Workflow (3 Steps)

### **Step 1: Denoise Your Rumbles**

```bash
python main.py --csv data/calls.csv --audio data/audio
```

**Output**: `outputs/audio/*.wav` (cleaned rumbles)

---

### **Step 2: Discover Patterns (Unsupervised)**

```bash
python analyze_rumbles_unsupervised.py --audio outputs/audio
```

**What it does**:
- ✅ Extracts 88 acoustic features per rumble (openSMILE)
- ✅ Auto-detects optimal clusters (e.g., 4 call types)
- ✅ Groups similar rumbles together
- ✅ Finds rare/unusual calls (10% anomalies)
- ✅ Creates visualizations

**Output**:
```
unsupervised_results/
├── cluster_assignments.csv    # Which rumble → which cluster
├── anomalies.csv               # Unusual rumbles
├── visualization.png           # 2D scatter plot
├── optimal_clusters.png        # Metrics for choosing k
└── features_normalized.npy     # Feature matrix
```

---

### **Step 3: Interpret Results**

**A. Check the visualization:**
```bash
open unsupervised_results/visualization.png
```
- Left plot: Clusters (different colors = different call types)
- Right plot: Anomalies (red = unusual)

**B. Listen to cluster representatives:**
```python
import pandas as pd

clusters = pd.read_csv('unsupervised_results/cluster_assignments.csv')

# Listen to 5 rumbles from each cluster
for cluster_id in clusters['cluster'].unique():
    files = clusters[clusters['cluster'] == cluster_id]['filename'].head(5)
    print(f"\nCluster {cluster_id}:")
    for f in files:
        print(f"  - outputs/audio/{f}")
        # Play these files to understand the pattern!
```

**C. Investigate anomalies:**
```python
anomalies = pd.read_csv('unsupervised_results/anomalies.csv')
unusual = anomalies[anomalies['is_anomaly'] == True]

print("Unusual rumbles to investigate:")
for _, row in unusual.iterrows():
    print(f"  - {row['filename']} (score: {row['anomaly_score']:.3f})")
```

---

## 🎯 Common Use Cases

### **Use Case 1: "I have 100 rumbles, what call types are there?"**

```bash
# Auto-detect everything
python analyze_rumbles_unsupervised.py --audio outputs/audio
```

**Result**: Discovers 4 clusters, you listen and realize:
- Cluster 0 = Contact calls
- Cluster 1 = Greeting calls
- Cluster 2 = Alarm calls
- Cluster 3 = Mixed/transitional

---

### **Use Case 2: "Find unusual/rare vocalizations"**

```bash
# Focus on anomaly detection
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --anomaly-only \
    --contamination 0.05  # Find top 5% most unusual
```

**Check**: `unsupervised_results/anomalies.csv`

---

### **Use Case 3: "I have < 50 rumbles (small dataset)"**

```bash
# Use pre-trained deep embeddings (better for small data)
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --features openl3
```

**Why**: openl3 uses transfer learning from millions of sounds

---

### **Use Case 4: "I want to try different clustering methods"**

```bash
# K-Means (default)
python analyze_rumbles_unsupervised.py --audio outputs/audio --method kmeans

# Gaussian Mixture Model (soft clustering)
python analyze_rumbles_unsupervised.py --audio outputs/audio --method gmm

# DBSCAN (auto-detects number of clusters)
python analyze_rumbles_unsupervised.py --audio outputs/audio --method dbscan

# Hierarchical (creates dendrogram)
python analyze_rumbles_unsupervised.py --audio outputs/audio --method hierarchical
```

---

### **Use Case 5: "I think there are 6 call types"**

```bash
# Force specific number of clusters
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --clusters 6
```

---

## 🔧 Feature Extraction Options

### **openSMILE (BEST for vocalizations)**
```bash
--features opensmile
```
- 88 features
- Designed for paralinguistics
- Includes pitch, jitter, shimmer, formants
- **Recommended for elephant rumbles**

### **openl3 (BEST for small datasets)**
```bash
--features openl3
```
- 512-dim deep embeddings
- Pre-trained on AudioSet
- Works with < 50 samples
- No feature engineering

### **Custom (Elephant-specific)**
```bash
--features custom
```
- 39 features
- Optimized for 10-300 Hz range
- MFCCs, spectral, temporal, low-freq features

### **Hybrid (BEST accuracy)**
```bash
--features hybrid
```
- 551 features (custom + openl3)
- Combines handcrafted + deep learning
- Usually 3-5% better accuracy

---

## 📊 Output Files Explained

### **cluster_assignments.csv**
```csv
filename,cluster
selection_001_cleaned.wav,0
selection_002_cleaned.wav,1
selection_003_cleaned.wav,0
```
**Use**: Group files by cluster, find representatives

### **anomalies.csv**
```csv
filename,is_anomaly,anomaly_score
selection_047_cleaned.wav,True,-0.128
selection_023_cleaned.wav,False,0.234
```
**Use**: Find unusual calls (lower score = more anomalous)

### **optimal_clusters.png**
Four plots showing metrics for choosing k:
- Silhouette (higher better)
- Elbow (look for bend)
- Davies-Bouldin (lower better)
- BIC (lower better)

### **visualization.png**
Two scatter plots:
- Left: Clusters (colors show groups)
- Right: Anomalies (red = unusual)

---

## 🐛 Troubleshooting

### **"ModuleNotFoundError: No module named 'opensmile'"**

```bash
pip install opensmile
```

### **"ModuleNotFoundError: No module named 'umap'"**

```bash
pip install umap-learn
```

Or use PCA instead:
```bash
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --dim-reduction pca
```

### **"All rumbles in one cluster"**

**Problem**: Features not capturing variation

**Solution 1**: Try different features
```bash
--features openl3  # Instead of opensmile
```

**Solution 2**: Check if denoising removed too much
```bash
# Re-run with gentler denoising
```

### **"Too many tiny clusters"**

**Solution**: Force fewer clusters
```bash
--clusters 4  # Manual override
```

### **"Clusters don't make sense"**

**Solution**: Try different dimensionality reduction
```bash
--dim-reduction umap  # Usually best
# or
--dim-reduction tsne  # Good for visualization
```

---

## 🚀 After Clustering: Create Labels

**Once you've listened to clusters and understand the patterns:**

```python
import pandas as pd
import json

# Load cluster assignments
clusters = pd.read_csv('unsupervised_results/cluster_assignments.csv')

# Map clusters to semantic labels (based on what you heard)
label_mapping = {
    0: 0,  # Cluster 0 = Contact calls
    1: 1,  # Cluster 1 = Greeting calls
    2: 2,  # Cluster 2 = Alarm calls
    3: 3   # Cluster 3 = Mixed
}

# Create labels.json for supervised learning
labels = dict(zip(
    clusters['filename'],
    clusters['cluster'].map(label_mapping)
))

with open('data/labels.json', 'w') as f:
    json.dump(labels, f, indent=2)

print("✅ Created labels.json")
print("Now you can train supervised models!")
```

---

## 📚 Full Documentation

- **[UNSUPERVISED_GUIDE.md](docs/UNSUPERVISED_GUIDE.md)** - Complete unsupervised learning tutorial
- **[FEATURE_LIBRARIES.md](docs/FEATURE_LIBRARIES.md)** - 7 acoustic feature extraction libraries
- **[NEURAL_NETWORK_GUIDE.md](docs/NEURAL_NETWORK_GUIDE.md)** - Supervised learning with windowing
- **[WINDOWING.md](docs/WINDOWING.md)** - DSP windowing theory

---

## ⚡ Quick Commands Reference

```bash
# === DENOISING ===
python main.py --csv data/calls.csv --audio data/audio

# === UNSUPERVISED LEARNING ===
# Basic
python analyze_rumbles_unsupervised.py --audio outputs/audio

# With openSMILE (recommended)
python analyze_rumbles_unsupervised.py --audio outputs/audio --features opensmile

# Small dataset
python analyze_rumbles_unsupervised.py --audio outputs/audio --features openl3

# Specific clusters
python analyze_rumbles_unsupervised.py --audio outputs/audio --clusters 5

# Anomaly focus
python analyze_rumbles_unsupervised.py --audio outputs/audio --anomaly-only

# === FEATURE COMPARISON ===
python src/advanced_features.py outputs/audio/selection_001_cleaned.wav
```

---

## 🎓 Interpretation Tips

### **What do clusters mean?**

Clusters group **acoustically similar** rumbles. This could indicate:

1. **Call Type** (contact vs greeting vs alarm)
2. **Individual Signature** (different elephants)
3. **Context** (calm vs excited vs stressed)
4. **Recording Quality** (clean vs noisy)

**To know which**: Listen to representatives from each cluster!

### **What do anomalies mean?**

Anomalies are the most **unusual** rumbles. Could be:

✅ **Novel vocalizations** → Exciting discovery!  
✅ **Rare behaviors** → Worth investigating  
⚠️ **Recording artifacts** → Filter out  
⚠️ **Mixed calls** → Transitional states  

**Always listen to anomalies manually!**

---

## 🎯 Success Metrics

**Good clustering**:
- ✅ Silhouette score > 0.5
- ✅ Clear visual separation in plots
- ✅ Representative rumbles sound similar
- ✅ Makes acoustic sense when you listen

**Poor clustering**:
- ❌ All rumbles in one cluster
- ❌ Many tiny clusters (< 5% each)
- ❌ Representatives sound random
- ❌ Doesn't match what you hear

**If clustering fails**: Try different features or methods!

---

## 💡 Pro Tips

1. **Start simple**: Use default settings first
2. **Listen, listen, listen**: Algorithms find patterns, you interpret them
3. **Try multiple methods**: K-Means vs GMM vs DBSCAN
4. **Compare features**: openSMILE vs openl3 vs custom
5. **Investigate anomalies**: They're often the most interesting!
6. **Use UMAP**: Usually better than t-SNE or PCA
7. **Save everything**: Re-run is fast but features take time
8. **Document findings**: Note what each cluster represents

---

**You're now ready to discover acoustic patterns in elephant rumbles! 🐘🔍**
