# 🔍 Unsupervised Learning for Elephant Rumbles (No Labels Required!)

Complete guide to discovering patterns in your rumbles **without any manual labeling**.

---

## 🎯 What You'll Discover

### **1. Natural Rumble Groups (Clusters)**
- Different call types naturally group together
- Contact calls vs greeting calls vs alarm calls
- Geographic variations
- Individual elephant signatures

### **2. Rare/Unusual Calls (Anomalies)**
- Find the 5-10% most unusual rumbles
- Novel vocalizations
- Recording artifacts
- Mixed calls

### **3. Acoustic Patterns**
- Which features distinguish different rumbles
- Temporal patterns
- Frequency characteristics

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Extract features from your rumbles
# 2. Find optimal number of clusters
# 3. Visualize groupings

python analyze_rumbles_unsupervised.py --audio outputs/audio
```

**That's it!** Results saved to `unsupervised_results/`

---

## 📊 What Happens

### Pipeline Overview

```
Cleaned Rumbles (N files)
         ↓
┌─────────────────────────────┐
│  Feature Extraction         │  openSMILE: 88 features per rumble
│  (openSMILE/openl3)         │  OR openl3: 512-dim embeddings
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  Normalize Features         │  Mean=0, Std=1
│  (StandardScaler)           │  Ensures fair comparison
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  Dimensionality Reduction   │  88D → 2D for visualization
│  (UMAP/PCA/t-SNE)           │  Preserves structure
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  Find Optimal Clusters      │  Test 2-10 clusters
│  (Silhouette, Elbow, BIC)  │  Pick best automatically
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  Cluster Rumbles            │  K-Means / GMM / DBSCAN
│                             │  Groups similar calls
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  Detect Anomalies           │  Isolation Forest
│                             │  Find unusual rumbles
└─────────────────────────────┘
         ↓
    Visualizations + CSV Reports
```

---

## 🛠️ Usage Examples

### **Example 1: Auto-Detect Everything**

```bash
python analyze_rumbles_unsupervised.py --audio outputs/audio
```

**Output**:
```
✅ Extracted features: (100, 88)
   Files: 100
   Features per file: 88

📐 Reducing dimensionality with UMAP...
   Reduced to 2D: (100, 2)

🔍 Finding optimal number of clusters (2-10)...
   → Recommended: 4 clusters

🎯 Clustering with KMEANS (k=4)...
   Cluster 0: 28 rumbles (28.0%)
   Cluster 1: 35 rumbles (35.0%)
   Cluster 2: 22 rumbles (22.0%)
   Cluster 3: 15 rumbles (15.0%)

🔍 Detecting anomalies...
   Found 10 anomalous rumbles (10.0%)
```

---

### **Example 2: Use openSMILE Features (Best for Bioacoustics)**

```bash
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --features opensmile \
    --clusters 5 \
    --method gmm
```

**Why openSMILE**:
- 88 features designed for vocalizations
- Includes prosodic features (pitch contours, energy)
- Voice quality metrics (jitter, shimmer)
- Works great for low-frequency rumbles

---

### **Example 3: Use Deep Learning Embeddings**

```bash
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --features openl3 \
    --dim-reduction umap
```

**Why openl3**:
- Pre-trained on millions of sounds
- 512-dim embeddings
- No feature engineering
- Good for small datasets (< 100 files)

---

### **Example 4: Focus on Anomaly Detection**

```bash
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --anomaly-only \
    --contamination 0.05  # Expect 5% anomalies
```

**Finds**:
- Rare calls
- Recording artifacts
- Mixed vocalizations
- Novel patterns

---

## 📈 Understanding the Results

### **1. optimal_clusters.png**

Four plots showing cluster quality metrics:

**Silhouette Score** (top-left):
- Range: -1 to 1
- Higher is better
- > 0.5 = good clustering
- Peak shows optimal k

**Elbow Method** (top-right):
- Look for "elbow" bend
- Where adding clusters stops helping
- Subjective but useful

**Davies-Bouldin** (bottom-left):
- Lower is better
- Measures cluster separation
- Minimum shows optimal k

**BIC** (bottom-right):
- Bayesian Information Criterion
- Lower is better
- Balances fit and complexity

**Example interpretation**:
```
If Silhouette peaks at k=4, Elbow bends at k=4, 
and BIC minimizes at k=4 → Use 4 clusters!
```

---

### **2. visualization.png**

Two scatter plots:

**Left: Clusters**
- Each color = one rumble type
- Red stars = cluster centers
- Tight groups = distinct call types
- Overlap = gradual transitions

**Right: Anomalies**
- Green = normal
- Yellow = borderline
- Red = anomalous
- Check red points manually

---

### **3. cluster_assignments.csv**

```csv
filename,cluster
selection_001_airplane_02_cleaned.wav,0
selection_002_vehicle_03_cleaned.wav,1
selection_003_generator_01_cleaned.wav,0
...
```

**Use this to**:
- Group rumbles by cluster
- Listen to representative examples
- Export clusters for further analysis

---

### **4. anomalies.csv**

```csv
filename,is_anomaly,anomaly_score
selection_001_cleaned.wav,False,0.234
selection_047_cleaned.wav,True,-0.128
...
```

**Lower score = more anomalous**

**Listen to anomalies!** They might be:
- Novel call types
- Mixed vocalizations
- Recording errors
- Rare behaviors

---

## 🔬 Clustering Methods

### **K-Means** (Default)
```bash
--method kmeans
```
**Pros**: Fast, simple, works well  
**Cons**: Assumes spherical clusters  
**Best for**: Most cases

### **Gaussian Mixture Model (GMM)**
```bash
--method gmm
```
**Pros**: Soft clustering, probabilistic  
**Cons**: Slower  
**Best for**: Overlapping call types

### **DBSCAN**
```bash
--method dbscan
```
**Pros**: Auto-detects k, finds noise  
**Cons**: Sensitive to parameters  
**Best for**: When k is unknown

### **Hierarchical**
```bash
--method hierarchical
```
**Pros**: Creates dendrogram  
**Cons**: Slow on large datasets  
**Best for**: < 200 files

---

## 🎨 Dimensionality Reduction

### **UMAP** (Recommended)
```bash
--dim-reduction umap
```
**Pros**: Preserves global + local structure  
**Cons**: Requires `pip install umap-learn`  
**Best for**: Most visualizations

### **t-SNE**
```bash
--dim-reduction tsne
```
**Pros**: Beautiful visualizations  
**Cons**: Slow, loses global structure  
**Best for**: Final publication figures

### **PCA**
```bash
--dim-reduction pca
```
**Pros**: Fast, deterministic  
**Cons**: Linear, may miss patterns  
**Best for**: Quick exploration

---

## 💡 Interpreting Clusters

### **What do the clusters mean?**

Clusters group rumbles that are **acoustically similar**. This could be:

1. **Call Type**
   - Cluster 0: Contact calls (low freq, long)
   - Cluster 1: Greetings (mid freq, rising)
   - Cluster 2: Alarms (high freq, short)

2. **Individual Signature**
   - Cluster 0: Ellie's calls
   - Cluster 1: Dumbo's calls
   - Cluster 2: Jumbo's calls

3. **Context**
   - Cluster 0: Calm/resting
   - Cluster 1: Excited/moving
   - Cluster 2: Stressed/alarmed

4. **Recording Quality**
   - Cluster 0: Clean recordings
   - Cluster 1: Noisy recordings
   - Cluster 2: Distant calls

**To interpret**: Listen to ~5 rumbles from each cluster!

---

## 🔍 Next Steps After Clustering

### **1. Label the Clusters**

```python
# Listen to representative rumbles
import pandas as pd

clusters = pd.read_csv('unsupervised_results/cluster_assignments.csv')

for cluster_id in clusters['cluster'].unique():
    files = clusters[clusters['cluster'] == cluster_id]['filename'].head(5)
    print(f"\nCluster {cluster_id}:")
    for f in files:
        print(f"  - {f}")
        # Listen to these files!
```

After listening, you might decide:
- Cluster 0 = Contact calls
- Cluster 1 = Greeting calls
- Cluster 2 = Alarm calls

### **2. Use as Labels for Supervised Learning**

```python
# Create labels.json from clusters
clusters = pd.read_csv('unsupervised_results/cluster_assignments.csv')

# Map clusters to semantic labels
mapping = {
    0: 0,  # Contact
    1: 1,  # Greeting
    2: 2,  # Alarm
}

labels = dict(zip(clusters['filename'], 
                  clusters['cluster'].map(mapping)))

import json
with open('data/labels.json', 'w') as f:
    json.dump(labels, f, indent=2)

# Now train supervised model!
```

### **3. Investigate Anomalies**

```python
anomalies = pd.read_csv('unsupervised_results/anomalies.csv')
unusual = anomalies[anomalies['is_anomaly'] == True]

print(f"Found {len(unusual)} unusual rumbles:")
for _, row in unusual.iterrows():
    print(f"  {row['filename']}: score={row['anomaly_score']:.3f}")
```

**Could be**:
- Novel vocalizations → exciting discovery!
- Recording artifacts → filter out
- Rare behaviors → investigate further

---

## 📊 Advanced: Feature Importance

**Which features distinguish the clusters?**

```python
from src.unsupervised_learning import UnsupervisedRumbleAnalyzer
import numpy as np
import pandas as pd

analyzer = UnsupervisedRumbleAnalyzer('outputs/audio', 'opensmile')
analyzer.extract_all_features()
analyzer.cluster(n_clusters=4)

# Get cluster centers
centers = []
for i in range(4):
    mask = analyzer.cluster_labels == i
    center = analyzer.features_normalized[mask].mean(axis=0)
    centers.append(center)

centers = np.array(centers)

# Find most variable features across clusters
variance = centers.var(axis=0)
top_features = np.argsort(variance)[-10:][::-1]

print("Top 10 discriminative features:")
for i, feat_idx in enumerate(top_features):
    print(f"{i+1}. Feature {feat_idx}: variance={variance[feat_idx]:.4f}")
```

---

## 🎯 Real-World Examples

### **Example: 100 Rumbles from Single Herd**

**Expectation**: 3-5 clusters
- Different call types
- Maybe individual signatures

**Run**:
```bash
python analyze_rumbles_unsupervised.py --audio outputs/audio
```

**Result**: 4 clusters
- Cluster 0 (35%): Long, low-frequency → Contact calls
- Cluster 1 (28%): Medium duration, rising → Greetings
- Cluster 2 (22%): Short, high-energy → Alarms
- Cluster 3 (15%): Variable → Mixed/transitional

---

### **Example: 200 Rumbles from Multiple Herds**

**Expectation**: More clusters (6-8)
- Call types × geographic variations

**Run**:
```bash
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --max-clusters 15
```

**Result**: 7 clusters
- Clusters 0-2: Call types (contact/greeting/alarm)
- Clusters 3-6: Geographic signatures

---

## 🐛 Troubleshooting

### **"Too few features" or "All in one cluster"**

**Problem**: Not enough variation between rumbles  
**Solution**: Check if denoising removed too much

```bash
# Try gentler denoising
# Or use more features
--features hybrid  # 551 features instead of 88
```

### **"Too many tiny clusters"**

**Problem**: Over-clustering  
**Solution**: Force fewer clusters

```bash
--clusters 4  # Manual override
```

### **"Clusters don't make sense"**

**Problem**: Features not capturing important variation  
**Solution**: Try different feature extractor

```bash
# Try openl3 instead of openSMILE
--features openl3
```

---

## 📚 Summary

**You can now**:
✅ Discover natural rumble groupings **without labels**  
✅ Find optimal number of clusters automatically  
✅ Detect rare/unusual calls  
✅ Visualize acoustic relationships  
✅ Use clusters as labels for supervised learning  

**One command**:
```bash
python analyze_rumbles_unsupervised.py --audio outputs/audio
```

**Outputs**: Clusters, anomalies, visualizations, CSV reports! 🐘🔍
