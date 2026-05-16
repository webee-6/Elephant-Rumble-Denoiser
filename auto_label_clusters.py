#!/usr/bin/env python3
"""
Automatic Cluster Labeling via Acoustic Feature Analysis
=========================================================

Automatically generates descriptive labels for clusters based on 
their acoustic characteristics - NO manual listening required!

Usage:
    python auto_label_clusters.py
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print(" AUTOMATIC CLUSTER LABELING")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n Loading data...")

# Load clusters
clusters_df = pd.read_csv('unsupervised_results/cluster_assignments.csv')

# Load features
features_raw = np.load('unsupervised_results/features_raw.npy')

# Check which features we have
if Path('unsupervised_results/features_pca.npy').exists():
    print(" PCA features detected - using raw features for interpretation")
    features = features_raw
else:
    features = np.load('unsupervised_results/features_normalized.npy')

print(f" Loaded {len(clusters_df)} files with {features.shape[1]} features")

# ============================================================================
# DEFINE FEATURE MEANINGS (for interpretation)
# ============================================================================

# For openl3/PCA features, we can't interpret individual dimensions
# But we CAN analyze the AUDIO FILES directly!

print("\n Analyzing acoustic characteristics per cluster...")

# We'll extract interpretable features from the raw audio
import librosa

def extract_interpretable_features(audio_path, sr=44100):
    """Extract human-interpretable acoustic features."""
    
    try:
        signal, sr = librosa.load(audio_path, sr=sr)
        
        features = {}
        
        # Duration
        features['duration'] = len(signal) / sr
        
        # Energy/Loudness
        features['rms_energy'] = float(np.sqrt(np.mean(signal**2)))
        
        # Frequency characteristics
        D = np.abs(librosa.stft(signal))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Spectral centroid (brightness)
        spec_cent = librosa.feature.spectral_centroid(y=signal, sr=sr)
        features['spectral_centroid'] = float(np.mean(spec_cent))
        
        # Low-frequency energy (elephant rumbles: 10-100 Hz)
        low_mask = (freqs >= 10) & (freqs <= 100)
        low_energy = np.sum(D[low_mask, :] ** 2)
        
        # Mid-frequency energy (100-300 Hz)
        mid_mask = (freqs >= 100) & (freqs <= 300)
        mid_energy = np.sum(D[mid_mask, :] ** 2)
        
        # High-frequency energy (300-1000 Hz)
        high_mask = (freqs >= 300) & (freqs <= 1000)
        high_energy = np.sum(D[high_mask, :] ** 2)
        
        total_energy = low_energy + mid_energy + high_energy
        
        features['low_freq_ratio'] = low_energy / (total_energy + 1e-10)
        features['mid_freq_ratio'] = mid_energy / (total_energy + 1e-10)
        features['high_freq_ratio'] = high_energy / (total_energy + 1e-10)
        
        # Pitch variation (using zero-crossing rate as proxy)
        zcr = librosa.feature.zero_crossing_rate(signal)
        features['pitch_variation'] = float(np.std(zcr))
        
        # Harmonic vs noise
        harmonic, percussive = librosa.effects.hpss(signal)
        h_energy = np.sum(harmonic ** 2)
        p_energy = np.sum(percussive ** 2)
        features['harmonic_ratio'] = h_energy / (h_energy + p_energy + 1e-10)
        
        return features
    
    except Exception as e:
        print(f" Error processing file: {e}")
        return None

# ============================================================================
# ANALYZE EACH CLUSTER
# ============================================================================

cluster_characteristics = {}

for cluster_id in sorted(clusters_df['cluster'].unique()):
    print(f"\n{'='*70}")
    print(f"ANALYZING CLUSTER {cluster_id}")
    print('='*70)
    
    cluster_files = clusters_df[clusters_df['cluster'] == cluster_id]['filename']
    
    print(f"Files in cluster: {len(cluster_files)}")
    print(f"Analyzing {min(20, len(cluster_files))} files...")
    
    # Extract features from sample files
    cluster_features = []
    for filename in cluster_files.head(20):  # Analyze first 20
        audio_path = f"outputs/audio/{filename}"
        if Path(audio_path).exists():
            feats = extract_interpretable_features(audio_path)
            if feats:
                cluster_features.append(feats)
    
    if not cluster_features:
        print(" No valid audio files found")
        continue
    
    # Convert to DataFrame
    cluster_df = pd.DataFrame(cluster_features)
    
    # Calculate statistics
    stats = {
        'duration': {
            'mean': float(cluster_df['duration'].mean()),
            'std': float(cluster_df['duration'].std()),
            'min': float(cluster_df['duration'].min()),
            'max': float(cluster_df['duration'].max())
        },
        'energy': {
            'mean': float(cluster_df['rms_energy'].mean()),
            'std': float(cluster_df['rms_energy'].std())
        },
        'spectral_centroid': {
            'mean': float(cluster_df['spectral_centroid'].mean()),
            'std': float(cluster_df['spectral_centroid'].std())
        },
        'frequency_distribution': {
            'low': float(cluster_df['low_freq_ratio'].mean()),
            'mid': float(cluster_df['mid_freq_ratio'].mean()),
            'high': float(cluster_df['high_freq_ratio'].mean())
        },
        'harmonic_ratio': {
            'mean': float(cluster_df['harmonic_ratio'].mean()),
            'std': float(cluster_df['harmonic_ratio'].std())
        },
        'pitch_variation': {
            'mean': float(cluster_df['pitch_variation'].mean()),
            'std': float(cluster_df['pitch_variation'].std())
        }
    }
    
    cluster_characteristics[int(cluster_id)] = stats  # Convert to Python int
    
    # Print analysis
    print(f"\n📊 Acoustic Profile:")
    print(f"   Duration: {stats['duration']['mean']:.2f}s (± {stats['duration']['std']:.2f}s)")
    print(f"   Range: {stats['duration']['min']:.2f}s - {stats['duration']['max']:.2f}s")
    print(f"\n   Energy: {stats['energy']['mean']:.4f}")
    print(f"   Spectral Centroid: {stats['spectral_centroid']['mean']:.1f} Hz")
    print(f"\n   Frequency Distribution:")
    print(f"     Low (10-100 Hz):    {stats['frequency_distribution']['low']*100:.1f}%")
    print(f"     Mid (100-300 Hz):   {stats['frequency_distribution']['mid']*100:.1f}%")
    print(f"     High (300-1000 Hz): {stats['frequency_distribution']['high']*100:.1f}%")
    print(f"\n   Harmonic Ratio: {stats['harmonic_ratio']['mean']:.2f}")
    print(f"   Pitch Variation: {stats['pitch_variation']['mean']:.4f}")

# ============================================================================
# GENERATE AUTOMATIC LABELS
# ============================================================================

print("\n" + "="*70)
print(" GENERATING AUTOMATIC LABELS")
print("="*70)

def generate_label(cluster_id, stats):
    """Generate descriptive label based on acoustic characteristics."""
    
    labels = []
    
    # Duration classification
    duration = stats['duration']['mean']
    if duration < 3:
        labels.append("Short")
    elif duration < 8:
        labels.append("Medium")
    else:
        labels.append("Long")
    
    # Frequency classification
    freq_dist = stats['frequency_distribution']
    if freq_dist['low'] > 0.6:
        labels.append("Low-Freq")
    elif freq_dist['high'] > 0.3:
        labels.append("High-Freq")
    else:
        labels.append("Mid-Freq")
    
    # Energy classification
    energy = stats['energy']['mean']
    # This is relative, but we can use thresholds
    if energy > 0.1:
        labels.append("Loud")
    elif energy < 0.01:
        labels.append("Quiet")
    
    # Harmonic vs noisy
    harmonic = stats['harmonic_ratio']['mean']
    if harmonic > 0.7:
        labels.append("Tonal")
    elif harmonic < 0.3:
        labels.append("Noisy")
    
    return " ".join(labels)

# Generate labels
auto_labels = {}
for cluster_id, stats in cluster_characteristics.items():
    label = generate_label(cluster_id, stats)
    auto_labels[int(cluster_id)] = label  # Convert to Python int
    
    print(f"\nCluster {cluster_id}: {label}")
    print(f"  Characteristics: {stats['duration']['mean']:.1f}s, "
          f"{stats['frequency_distribution']['low']*100:.0f}% low-freq, "
          f"{stats['energy']['mean']:.3f} energy")

# ============================================================================
# COMPARE CLUSTERS
# ============================================================================

print("\n" + "="*70)
print(" CLUSTER COMPARISON")
print("="*70)

# Find most distinctive features
print("\n Most Distinctive Characteristics:\n")

clusters = list(cluster_characteristics.keys())
if len(clusters) >= 2:
    c0, c1 = clusters[0], clusters[1]
    
    # Duration difference
    dur_diff = abs(cluster_characteristics[c0]['duration']['mean'] - 
                   cluster_characteristics[c1]['duration']['mean'])
    print(f"Duration difference: {dur_diff:.2f}s")
    
    # Frequency distribution difference
    freq_diff_low = abs(cluster_characteristics[c0]['frequency_distribution']['low'] -
                        cluster_characteristics[c1]['frequency_distribution']['low'])
    print(f"Low-frequency content difference: {freq_diff_low*100:.1f}%")
    
    # Energy difference
    energy_diff = abs(cluster_characteristics[c0]['energy']['mean'] -
                     cluster_characteristics[c1]['energy']['mean'])
    print(f"Energy difference: {energy_diff:.4f}")
    
    print("\n Key Differences:")
    if dur_diff > 2:
        print(f"  • Cluster {c0} is {'LONGER' if cluster_characteristics[c0]['duration']['mean'] > cluster_characteristics[c1]['duration']['mean'] else 'SHORTER'} than Cluster {c1}")
    
    if freq_diff_low > 0.2:
        print(f"  • Cluster {c0} has {'MORE' if cluster_characteristics[c0]['frequency_distribution']['low'] > cluster_characteristics[c1]['frequency_distribution']['low'] else 'LESS'} low-frequency content")
    
    if energy_diff > 0.02:
        print(f"  • Cluster {c0} is {'LOUDER' if cluster_characteristics[c0]['energy']['mean'] > cluster_characteristics[c1]['energy']['mean'] else 'QUIETER'}")

# ============================================================================
# VISUALIZE CLUSTER DIFFERENCES
# ============================================================================

print("\n Creating visualizations...")

# Prepare data for plotting
plot_data = []
for cluster_id, stats in cluster_characteristics.items():
    plot_data.append({
        'Cluster': f"Cluster {cluster_id}\n({auto_labels[cluster_id]})",
        'Duration (s)': stats['duration']['mean'],
        'Low Freq %': stats['frequency_distribution']['low'] * 100,
        'Mid Freq %': stats['frequency_distribution']['mid'] * 100,
        'High Freq %': stats['frequency_distribution']['high'] * 100,
        'Energy': stats['energy']['mean'],
        'Harmonic Ratio': stats['harmonic_ratio']['mean']
    })

plot_df = pd.DataFrame(plot_data)

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Duration
axes[0, 0].bar(plot_df['Cluster'], plot_df['Duration (s)'], color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
axes[0, 0].set_title('Average Duration', fontweight='bold')
axes[0, 0].set_ylabel('Seconds')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Frequency distribution (stacked bar)
freq_cols = ['Low Freq %', 'Mid Freq %', 'High Freq %']
plot_df[freq_cols].plot(kind='bar', stacked=True, ax=axes[0, 1], 
                        color=['#3498db', '#2ecc71', '#e74c3c'])
axes[0, 1].set_title('Frequency Distribution', fontweight='bold')
axes[0, 1].set_ylabel('Percentage')
axes[0, 1].set_xticklabels(plot_df['Cluster'], rotation=0)
axes[0, 1].legend(title='Frequency Band')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Energy
axes[0, 2].bar(plot_df['Cluster'], plot_df['Energy'], color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
axes[0, 2].set_title('Average Energy', fontweight='bold')
axes[0, 2].set_ylabel('RMS Energy')
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Harmonic ratio
axes[1, 0].bar(plot_df['Cluster'], plot_df['Harmonic Ratio'], color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
axes[1, 0].set_title('Harmonic Ratio', fontweight='bold')
axes[1, 0].set_ylabel('Ratio')
axes[1, 0].set_ylim([0, 1])
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Frequency distribution pie for each cluster
for idx, (cluster_id, stats) in enumerate(cluster_characteristics.items()):
    if idx < 2:  # Show first 2 clusters
        ax = axes[1, idx + 1]
        sizes = [
            stats['frequency_distribution']['low'] * 100,
            stats['frequency_distribution']['mid'] * 100,
            stats['frequency_distribution']['high'] * 100
        ]
        ax.pie(sizes, labels=['Low (10-100 Hz)', 'Mid (100-300 Hz)', 'High (300-1000 Hz)'],
               autopct='%1.1f%%', colors=['#3498db', '#2ecc71', '#e74c3c'])
        ax.set_title(f'Cluster {cluster_id} - {auto_labels[cluster_id]}', fontweight='bold')

plt.tight_layout()
plt.savefig('unsupervised_results/cluster_characteristics.png', dpi=150, bbox_inches='tight')
plt.show()

print(" Saved: unsupervised_results/cluster_characteristics.png")

# ============================================================================
# SAVE LABELS
# ============================================================================

print("\n" + "="*70)
print(" SAVING AUTOMATIC LABELS")
print("="*70)

# Save for use in training
cluster_names = {}
for cluster_id, label in auto_labels.items():
    cluster_names[int(cluster_id)] = label  # Convert to Python int

# Save to JSON
Path('data').mkdir(exist_ok=True)
with open('data/cluster_names_auto.json', 'w') as f:
    json.dump(cluster_names, f, indent=2)

print(f"\n Saved: data/cluster_names_auto.json")

# Save detailed characteristics
with open('unsupervised_results/cluster_characteristics.json', 'w') as f:
    json.dump(cluster_characteristics, f, indent=2)

print(f" Saved: unsupervised_results/cluster_characteristics.json")

# Create human-readable report
report_lines = []
report_lines.append("="*70)
report_lines.append("AUTOMATIC CLUSTER LABELING REPORT")
report_lines.append("="*70)
report_lines.append("")

for cluster_id, label in auto_labels.items():
    stats = cluster_characteristics[cluster_id]
    report_lines.append(f"Cluster {cluster_id}: {label}")
    report_lines.append("-" * 70)
    report_lines.append(f"  Duration: {stats['duration']['mean']:.2f}s (± {stats['duration']['std']:.2f}s)")
    report_lines.append(f"  Range: {stats['duration']['min']:.2f}s - {stats['duration']['max']:.2f}s")
    report_lines.append(f"  Energy: {stats['energy']['mean']:.4f}")
    report_lines.append(f"  Spectral Centroid: {stats['spectral_centroid']['mean']:.1f} Hz")
    report_lines.append(f"  Frequency Distribution:")
    report_lines.append(f"    - Low (10-100 Hz):    {stats['frequency_distribution']['low']*100:.1f}%")
    report_lines.append(f"    - Mid (100-300 Hz):   {stats['frequency_distribution']['mid']*100:.1f}%")
    report_lines.append(f"    - High (300-1000 Hz): {stats['frequency_distribution']['high']*100:.1f}%")
    report_lines.append(f"  Harmonic Ratio: {stats['harmonic_ratio']['mean']:.2f}")
    report_lines.append("")

with open('unsupervised_results/cluster_report.txt', 'w') as f:
    f.write('\n'.join(report_lines))

print(f" Saved: unsupervised_results/cluster_report.txt")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print(" AUTOMATIC LABELING COMPLETE!")
print("="*70)

print("\n Generated Labels:")
for cluster_id, label in auto_labels.items():
    count = (clusters_df['cluster'] == cluster_id).sum()
    print(f"  Cluster {cluster_id}: {label} ({count} files)")

print("\n Files Created:")
print("   data/cluster_names_auto.json")
print("   unsupervised_results/cluster_characteristics.json ")
print("   unsupervised_results/cluster_characteristics.png ")
print("   unsupervised_results/cluster_report.txt ")

