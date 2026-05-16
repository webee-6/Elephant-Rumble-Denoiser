# 🐛 Troubleshooting Guide

Common issues and solutions for the elephant rumble unsupervised learning pipeline.

---

## ❌ Error: "GMM failed because some components have ill-defined empirical covariance"

### **Problem:**
```
ValueError: Fitting the mixture model failed because some components 
have ill-defined empirical covariance (for instance caused by singleton 
or collapsed samples). Try to decrease the number of components, 
increase reg_covar, or scale the input data.
```

### **Cause:**
This happens when using **high-dimensional features** (like openl3's 512 dimensions) with Gaussian Mixture Models (GMM). The covariance matrix becomes singular or ill-conditioned.

### **Solution (Already Fixed in Latest Version):**

The latest version automatically:
1. ✅ **Applies PCA** to reduce 512D → ~50-100D (preserving 95% variance)
2. ✅ **Uses diagonal covariance** instead of full covariance for GMM
3. ✅ **Adds regularization** (`reg_covar=1e-6`)
4. ✅ **Falls back to K-Means** if GMM still fails

### **Manual Workaround (if needed):**

**Option 1: Use K-Means instead of GMM**
```bash
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --features openl3 \
    --method kmeans  # Instead of gmm
```

**Option 2: Use fewer features**
```bash
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --features opensmile  # 88 dims instead of 512
```

**Option 3: Use custom features**
```bash
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --features custom  # Only 39 dims
```

---

## ❌ Error: "ModuleNotFoundError: No module named 'opensmile'"

### **Problem:**
```
ModuleNotFoundError: No module named 'opensmile'
```

### **Solution:**
```bash
pip install opensmile
```

If installation fails, use an alternative feature extractor:
```bash
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --features custom  # Uses librosa only
```

---

## ❌ Error: "ModuleNotFoundError: No module named 'umap'"

### **Problem:**
```
ModuleNotFoundError: No module named 'umap'
```

### **Solution:**
```bash
pip install umap-learn
```

Or use an alternative dimensionality reduction:
```bash
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --dim-reduction pca  # Instead of umap
```

---

## ❌ Error: "All rumbles assigned to one cluster"

### **Problem:**
All rumbles get assigned to the same cluster.

### **Causes:**
1. Features aren't capturing variation
2. Denoising removed too much signal
3. Rumbles are genuinely very similar

### **Solutions:**

**Try different features:**
```bash
# Try openl3 (deep embeddings)
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --features openl3

# Or hybrid (551 features)
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --features hybrid
```

**Force more clusters:**
```bash
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --clusters 4  # Manual override
```

**Check denoising:**
Listen to cleaned files - did denoising remove too much?

---

## ❌ Error: "Too many tiny clusters"

### **Problem:**
DBSCAN creates 20+ clusters with 1-2 rumbles each.

### **Solution:**

**Use K-Means instead:**
```bash
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --method kmeans \
    --clusters 4
```

**Or adjust DBSCAN parameters** (in code):
```python
# In src/unsupervised_learning.py, line ~360
clusterer = DBSCAN(
    eps=1.0,  # Increase from 0.5
    min_samples=5  # Increase from 3
)
```

---

## ❌ Warning: "Variance explained: 45%"

### **Problem:**
```
⚙️ High-dimensional features detected (512D)
   Applying PCA for clustering stability...
   Reduced 512D → 50D
   Variance explained: 45.2%
```

### **Meaning:**
PCA could only preserve 45% of variance with 50 components. This means features are very spread out.

### **Solutions:**

**Not usually a problem!** Clustering can still work well.

But if clusters don't make sense:

**Option 1: Use more features**
```bash
--features hybrid  # 551 features
```

**Option 2: Skip PCA** (in code):
```python
# Comment out PCA section in extract_all_features()
```

**Option 3: Use fewer samples per file**
For very diverse datasets, clustering might work better with custom features.

---

## ❌ Error: "Silhouette score is negative"

### **Problem:**
```
Silhouette Score: -0.123 (higher better, >0.5 = good)
⚠️ Clusters may overlap
```

### **Meaning:**
Clusters are poorly separated - rumbles assigned to wrong clusters.

### **Solutions:**

**Try fewer clusters:**
```bash
--clusters 3  # Instead of auto-detect
```

**Try different features:**
```bash
--features opensmile  # May capture different patterns
```

**Check if rumbles are actually clusterable:**
```bash
# Visualize without clustering
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --anomaly-only  # Skip clustering
```

If visualization shows no clear groups, rumbles might be genuinely similar.

---

## ❌ Error: "LinAlgError: SVD did not converge"

### **Problem:**
PCA or other matrix operations fail.

### **Solution:**

**Already fixed in latest version** with float64 conversion.

If still occurs:
```bash
# Use simpler features
--features custom
```

---

## ⚠️ Performance: "Very slow (> 10 minutes)"

### **Problem:**
Feature extraction or clustering taking too long.

### **Causes & Solutions:**

**1. High-dimensional features (openl3: 512D)**
```bash
# Use openSMILE instead (faster)
--features opensmile
```

**2. UMAP dimensionality reduction (slow for large datasets)**
```bash
# Use PCA instead (much faster)
--dim-reduction pca
```

**3. Many files (> 500)**
- This is normal, be patient
- Feature extraction: ~1-2 sec/file
- Clustering: ~10-30 sec

**4. GMM clustering (slower than K-Means)**
```bash
# Use K-Means
--method kmeans
```

---

## ⚠️ Clusters don't make acoustic sense

### **Problem:**
Clusters don't correspond to call types you expected.

### **This is normal!**

Clusters represent **acoustic similarity**, which might be:
- Call types (what you expected)
- Individual signatures (different elephants)
- Recording quality (SNR differences)
- Context (calm vs stressed)
- Geographic variations

### **What to do:**

1. **Listen to representatives** from each cluster
2. **Note what's similar** (duration? pitch? energy?)
3. **Relabel clusters** based on what you hear
4. **It's discovery!** You might find patterns you didn't expect

---

## ❌ Error: "No rumbles found in outputs/audio"

### **Problem:**
```
📂 Found 0 rumble files
```

### **Solution:**

**Check directory path:**
```bash
ls outputs/audio/*.wav
```

**Make sure denoising ran first:**
```bash
python main.py --csv data/calls.csv --audio data/audio
```

**Check file extensions:**
Script looks for `.wav` files only.

---

## 🔧 General Debugging Tips

### **1. Start simple:**
```bash
python analyze_rumbles_unsupervised.py \
    --audio outputs/audio \
    --features custom \
    --method kmeans \
    --dim-reduction pca
```

### **2. Check feature extraction worked:**
```bash
ls unsupervised_results/features_raw.npy
# Should exist if feature extraction succeeded
```

### **3. Visualize raw data:**
```python
import numpy as np
import matplotlib.pyplot as plt

features = np.load('unsupervised_results/features_normalized.npy')
print(f"Features shape: {features.shape}")
print(f"NaN values: {np.isnan(features).sum()}")
print(f"Inf values: {np.isinf(features).sum()}")

# Plot feature distributions
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(features.flatten(), bins=50)
plt.title('Feature distribution')
plt.subplot(1, 2, 2)
plt.boxplot(features[:, :20])
plt.title('First 20 features')
plt.show()
```

### **4. Test with subset:**
```bash
# Copy just 20 files for testing
mkdir test_audio
cp outputs/audio/*.wav test_audio/ | head -20

python analyze_rumbles_unsupervised.py --audio test_audio
```

---

## 📊 Expected Behavior

### **Normal output:**
```
✅ Extracted features: (100, 88)
   Files: 100
   Features per file: 88

✅ Features normalized (mean=0, std=1)

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

### **Good metrics:**
- Silhouette: > 0.3 (excellent if > 0.5)
- Cluster sizes: 10-40% each (roughly balanced)
- Anomalies: 5-15%

### **Bad metrics:**
- Silhouette: < 0.1
- One cluster: 90%+
- Anomalies: < 1% or > 50%

---

## 🆘 Still Having Issues?

1. **Check Python version:** Requires Python 3.8+
2. **Update packages:**
   ```bash
   pip install --upgrade numpy scikit-learn librosa
   ```
3. **Try minimal example:**
   ```bash
   python examples/unsupervised_analysis_example.py
   ```
4. **Check logs:** Look for warnings in output
5. **Validate data:** Make sure WAV files are valid
   ```bash
   file outputs/audio/*.wav | head -5
   ```

---

**Most issues are fixed in the latest version!** Make sure you're using the updated `elephant_denoiser.tar.gz`. 🐘✨
