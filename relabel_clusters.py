#!/usr/bin/env python3
"""
Improved Automatic Cluster Labeling
====================================

Better handling of similar clusters with relative comparisons.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

print("="*70)
print("IMPROVED AUTOMATIC CLUSTER LABELING")
print("="*70)

# Load cluster assignments
clusters_df = pd.read_csv('unsupervised_results/cluster_assignments.csv')

# Load characteristics from auto_label_clusters.py
with open('unsupervised_results/cluster_characteristics.json', 'r') as f:
    cluster_chars = json.load(f)

print(f"\n Loaded {len(clusters_df)} files in {len(cluster_chars)} clusters")

# ============================================================================
# IMPROVED LABELING STRATEGY
# ============================================================================

print("\n Analyzing cluster differences...")

# Convert keys to int and collect all stats
clusters = {}
for key, stats in cluster_chars.items():
    cluster_id = int(key)
    clusters[cluster_id] = stats

# Find what distinguishes each cluster
all_durations = [c['duration']['mean'] for c in clusters.values()]
all_energies = [c['energy']['mean'] for c in clusters.values()]
all_low_freq = [c['frequency_distribution']['low'] for c in clusters.values()]
all_harmonics = [c['harmonic_ratio']['mean'] for c in clusters.values()]

# Calculate percentiles for relative labeling
duration_25 = np.percentile(all_durations, 25)
duration_75 = np.percentile(all_durations, 75)
energy_25 = np.percentile(all_energies, 25)
energy_75 = np.percentile(all_energies, 75)
lowfreq_50 = np.percentile(all_low_freq, 50)

print("\n📊 Cluster Thresholds (Relative):")
print(f"   Duration: Short<{duration_25:.1f}s, Long>{duration_75:.1f}s")
print(f"   Energy: Quiet<{energy_25:.3f}, Loud>{energy_75:.3f}")
print(f"   Low-freq: Below {lowfreq_50:.1%}")

# Remove outlier clusters (only 1-2 files)
cluster_counts = clusters_df['cluster'].value_counts()
outlier_threshold = max(3, len(clusters_df) * 0.02)  # At least 3 files or 2%

outliers = []
main_clusters = {}

for cluster_id, stats in clusters.items():
    count = cluster_counts.get(cluster_id, 0)
    if count < outlier_threshold:
        outliers.append(cluster_id)
        print(f"\n Cluster {cluster_id}: Only {count} files - Marked as OUTLIER")
    else:
        main_clusters[cluster_id] = stats

# ============================================================================
# GENERATE IMPROVED LABELS
# ============================================================================

def generate_improved_label(cluster_id, stats, is_outlier=False):
    """Generate label with relative comparisons."""
    
    if is_outlier:
        return "Outlier/Anomaly"
    
    labels = []
    
    # Relative duration
    duration = stats['duration']['mean']
    if duration < duration_25:
        labels.append("Short")
    elif duration > duration_75:
        labels.append("Long")
    # Don't add "Medium" - too generic
    
    # Relative low-frequency content
    low_freq = stats['frequency_distribution']['low']
    if low_freq > lowfreq_50:
        labels.append("Low-Freq-Rich")
    else:
        labels.append("Broader-Spectrum")
    
    # Relative energy
    energy = stats['energy']['mean']
    if energy > energy_75:
        labels.append("High-Energy")
    elif energy < energy_25:
        labels.append("Low-Energy")
    
    # If no distinguishing features, use cluster number
    if not labels:
        labels.append(f"Type-{chr(65+cluster_id)}")  # A, B, C, D
    
    return " ".join(labels)

# Generate labels
auto_labels = {}

print("\n Generated Labels:\n")
for cluster_id, stats in clusters.items():
    is_outlier = cluster_id in outliers
    label = generate_improved_label(cluster_id, stats, is_outlier)
    auto_labels[cluster_id] = label
    
    count = cluster_counts.get(cluster_id, 0)
    print(f"Cluster {cluster_id}: {label}")
    print(f"  Files: {count}")
    print(f"  Duration: {stats['duration']['mean']:.2f}s")
    print(f"  Low-freq: {stats['frequency_distribution']['low']*100:.1f}%")
    print(f"  Energy: {stats['energy']['mean']:.4f}")
    print()

# ============================================================================
# HANDLE VERY SIMILAR CLUSTERS
# ============================================================================

# Check if labels are too similar
unique_labels = set(auto_labels.values())
if len(unique_labels) < len(main_clusters) * 0.5:  # Less than half unique
    print(" Warning: Clusters are very similar!")
    print("   Adding cluster IDs to distinguish them...\n")
    
    # Add cluster letter suffix to make them unique
    for cluster_id in auto_labels.keys():
        if cluster_id not in outliers:
            letter = chr(65 + cluster_id)  # A, B, C, D
            auto_labels[cluster_id] = f"{auto_labels[cluster_id]}-Type{letter}"

# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

print("="*70)
print(" CLUSTER COMPARISON")
print("="*70)

if len(main_clusters) >= 2:
    print("\n Key Differences Between Main Clusters:\n")
    
    cluster_ids = sorted(main_clusters.keys())
    for i in range(len(cluster_ids) - 1):
        c1 = cluster_ids[i]
        c2 = cluster_ids[i + 1]
        
        s1 = clusters[c1]
        s2 = clusters[c2]
        
        print(f"Cluster {c1} vs Cluster {c2}:")
        
        # Duration
        dur_diff = abs(s1['duration']['mean'] - s2['duration']['mean'])
        if dur_diff > 1.0:
            longer = c1 if s1['duration']['mean'] > s2['duration']['mean'] else c2
            print(f"  • Cluster {longer} is {dur_diff:.1f}s LONGER")
        
        # Energy
        energy_diff = abs(s1['energy']['mean'] - s2['energy']['mean'])
        if energy_diff > 0.03:
            louder = c1 if s1['energy']['mean'] > s2['energy']['mean'] else c2
            print(f"  • Cluster {louder} is {energy_diff*100:.1f}% LOUDER")
        
        # Low-frequency
        freq_diff = abs(s1['frequency_distribution']['low'] - s2['frequency_distribution']['low'])
        if freq_diff > 0.1:
            more_low = c1 if s1['frequency_distribution']['low'] > s2['frequency_distribution']['low'] else c2
            print(f"  • Cluster {more_low} has {freq_diff*100:.1f}% MORE low-frequency content")
        
        if dur_diff < 1.0 and energy_diff < 0.03 and freq_diff < 0.1:
            print(f"  • Very similar clusters! Consider merging.")
        
        print()

# ============================================================================
# SAVE IMPROVED LABELS
# ============================================================================

print("="*70)
print(" SAVING IMPROVED LABELS")
print("="*70)

Path('data').mkdir(exist_ok=True)

# Save labels
with open('data/cluster_names_auto.json', 'w') as f:
    json.dump(auto_labels, f, indent=2)

print(f"\n Saved: data/cluster_names_auto.json")

# Print final summary
print("\n" + "="*70)
print(" LABELING COMPLETE!")
print("="*70)

print(f"\n Final Labels:")
for cluster_id in sorted(auto_labels.keys()):
    count = cluster_counts.get(cluster_id, 0)
    print(f"  Cluster {cluster_id}: {auto_labels[cluster_id]} ({count} files)")

if outliers:
    print(f"\n Outlier clusters: {outliers}")
    print("   These have too few files for reliable training.")
    print("   Consider removing them or merging with nearest cluster.")

print("\n Next Step:")
print("   python train_complete.py")
print("="*70)
