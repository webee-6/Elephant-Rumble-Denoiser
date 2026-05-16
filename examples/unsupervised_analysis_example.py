"""
Complete Example: Unsupervised Rumble Analysis
===============================================

This script demonstrates the full workflow from denoised audio
to discovered patterns without any manual labeling.

Run this after you've denoised your rumbles with main.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("ELEPHANT RUMBLE UNSUPERVISED ANALYSIS")
print("="*70)

# ============================================================================
# STEP 1: Initialize Analyzer
# ============================================================================
print("\n" + "="*70)
print("STEP 1: INITIALIZE ANALYZER")
print("="*70)

from src.unsupervised_learning import UnsupervisedRumbleAnalyzer

analyzer = UnsupervisedRumbleAnalyzer(
    audio_dir='outputs/audio',
    feature_extractor='opensmile',  # BEST for vocalizations
    output_dir='unsupervised_results'
)

print(f"Loaded {len(analyzer.audio_files)} rumble files")

# ============================================================================
# STEP 2: Extract Acoustic Features
# ============================================================================
print("\n" + "="*70)
print("STEP 2: EXTRACT BIOACOUSTIC FEATURES")
print("="*70)

# Extract 88 features per rumble using openSMILE
features = analyzer.extract_all_features()

print(f"\nFeature matrix shape: {features.shape}")
print(f"  - Rows: {features.shape[0]} rumbles")
print(f"  - Columns: {features.shape[1]} features")
print(f"\nFeatures include:")
print("  • Prosodic (pitch contours, energy)")
print("  • Voice quality (jitter, shimmer)")
print("  • Spectral (formants, MFCCs)")
print("  • Temporal (ZCR, RMS)")

# ============================================================================
# STEP 3: Reduce to 2D for Visualization
# ============================================================================
print("\n" + "="*70)
print("STEP 3: DIMENSIONALITY REDUCTION")
print("="*70)

# Reduce 88D → 2D using UMAP (preserves structure)
embeddings = analyzer.reduce_dimensionality(method='umap', n_components=2)

print(f"Reduced {features.shape[1]}D → 2D for visualization")

# ============================================================================
# STEP 4: Find Optimal Number of Clusters
# ============================================================================
print("\n" + "="*70)
print("STEP 4: FIND OPTIMAL CLUSTERS")
print("="*70)

# Test 2-10 clusters, use multiple metrics
optimal_k = analyzer.find_optimal_clusters(max_k=10)

print(f"\nOptimal number of clusters: {optimal_k}")
print(f"See: unsupervised_results/optimal_clusters.png")

# ============================================================================
# STEP 5: Cluster the Rumbles
# ============================================================================
print("\n" + "="*70)
print("STEP 5: CLUSTER RUMBLES")
print("="*70)

# K-Means clustering
labels = analyzer.cluster(n_clusters=optimal_k, method='kmeans')

# Show cluster statistics
print(f"\nCluster distribution:")
for cluster_id in np.unique(labels):
    count = np.sum(labels == cluster_id)
    percentage = count / len(labels) * 100
    print(f"  Cluster {cluster_id}: {count:3d} rumbles ({percentage:5.1f}%)")

# ============================================================================
# STEP 6: Detect Anomalies
# ============================================================================
print("\n" + "="*70)
print("STEP 6: ANOMALY DETECTION")
print("="*70)

# Find the 10% most unusual rumbles
anomalies = analyzer.detect_anomalies(contamination=0.10, method='isolation_forest')

n_anomalies = np.sum(anomalies == -1)
print(f"\n🔍 Found {n_anomalies} anomalous rumbles")

# List top 5 most anomalous
anomaly_df = pd.read_csv('unsupervised_results/anomalies.csv')
top_anomalies = anomaly_df[anomaly_df['is_anomaly']].nsmallest(5, 'anomaly_score')

print(f"\nTop 5 most unusual rumbles:")
for i, (_, row) in enumerate(top_anomalies.iterrows(), 1):
    print(f"  {i}. {row['filename']} (score: {row['anomaly_score']:.3f})")

# ============================================================================
# STEP 7: Visualize Results
# ============================================================================
print("\n" + "="*70)
print("STEP 7: CREATE VISUALIZATIONS")
print("="*70)

analyzer.visualize(show_clusters=True, show_anomalies=True, method='umap')

print(f"\n📊 Visualization saved!")
print(f"   Left plot: Clusters (different colors)")
print(f"   Right plot: Anomalies (red = unusual)")

# ============================================================================
# STEP 8: Analyze Each Cluster
# ============================================================================
print("\n" + "="*70)
print("STEP 8: ANALYZE CLUSTERS")
print("="*70)

analyzer.analyze_clusters()

# ============================================================================
# STEP 9: Interpret Results
# ============================================================================
print("\n" + "="*70)
print("STEP 9: NEXT STEPS FOR INTERPRETATION")
print("="*70)

print("\n📋 What to do next:")
print("\n1. LISTEN to representative rumbles from each cluster:")

cluster_df = pd.read_csv('unsupervised_results/cluster_assignments.csv')

for cluster_id in sorted(cluster_df['cluster'].unique()):
    files = cluster_df[cluster_df['cluster'] == cluster_id]['filename'].head(3)
    print(f"\n   Cluster {cluster_id} representatives:")
    for f in files:
        print(f"      - outputs/audio/{f}")

print("\n2. INVESTIGATE anomalies:")
print(f"   Check: unsupervised_results/anomalies.csv")
print(f"   Listen to unusual rumbles - could be:")
print(f"      • Novel vocalizations")
print(f"      • Rare behaviors")
print(f"      • Recording artifacts")

print("\n3. LABEL the clusters:")
print(f"   After listening, decide what each cluster represents:")
print(f"      • Call types (contact/greeting/alarm)?")
print(f"      • Individual signatures?")
print(f"      • Contextual variations?")

print("\n4. CREATE supervised labels:")
print(f"   Use cluster assignments as initial labels:")

print("""
   # Example: Map clusters to call types
   import json
   
   label_mapping = {
       0: 0,  # Contact calls
       1: 1,  # Greeting calls
       2: 2,  # Alarm calls
       3: 3   # Mixed
   }
   
   clusters = pd.read_csv('unsupervised_results/cluster_assignments.csv')
   labels = dict(zip(
       clusters['filename'],
       clusters['cluster'].map(label_mapping)
   ))
   
   with open('data/labels.json', 'w') as f:
       json.dump(labels, f, indent=2)
""")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\nSummary:")
print(f"   Rumbles analyzed: {len(analyzer.audio_files)}")
print(f"   Features extracted: {features.shape[1]}")
print(f"   Clusters discovered: {len(np.unique(labels))}")
print(f"   Anomalies found: {n_anomalies}")

print(f"\nResults saved to: unsupervised_results/")
print(f"   cluster_assignments.csv")
print(f"   anomalies.csv")
print(f"   visualization.png")
print(f"   optimal_clusters.png")
print(f"   features_normalized.npy")

print(f"\nKey insight:")
print(f"   Your rumbles naturally group into {len(np.unique(labels))} distinct patterns!")
print(f"   Listen to representatives to understand what they represent.")
