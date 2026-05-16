#!/usr/bin/env python3
"""
Unsupervised Learning for Elephant Rumbles

Discovers natural patterns and clusters without labels.

Usage:
    python analyze_rumbles_unsupervised.py --audio outputs/audio
    python analyze_rumbles_unsupervised.py --audio outputs/audio --features opensmile
    python analyze_rumbles_unsupervised.py --audio outputs/audio --features opensmile --clusters 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.unsupervised_learning import UnsupervisedRumbleAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description='Unsupervised analysis of elephant rumbles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect everything
    python analyze_rumbles_unsupervised.py --audio outputs/audio

    # Use openSMILE features (recommended)
    python analyze_rumbles_unsupervised.py \\
        --audio outputs/audio \\
        --features opensmile

    # Use deep learning embeddings
    python analyze_rumbles_unsupervised.py \\
        --audio outputs/audio \\
        --features openl3

    # Specify number of clusters
    python analyze_rumbles_unsupervised.py \\
        --audio outputs/audio \\
        --clusters 4 \\
        --method gmm

    # Focus on anomaly detection
    python analyze_rumbles_unsupervised.py \\
        --audio outputs/audio \\
        --anomaly-only \\
        --contamination 0.05
        """
    )
    
    parser.add_argument(
        '--audio',
        required=True,
        help='Directory with cleaned audio files'
    )
    
    parser.add_argument(
        '--features',
        default='opensmile',
        choices=['opensmile', 'openl3', 'custom', 'hybrid'],
        help='Feature extraction method (default: opensmile)'
    )
    
    parser.add_argument(
        '--clusters',
        type=int,
        default=None,
        help='Number of clusters (if None, auto-detect)'
    )
    
    parser.add_argument(
        '--max-clusters',
        type=int,
        default=10,
        help='Maximum clusters to test for auto-detection (default: 10)'
    )
    
    parser.add_argument(
        '--method',
        default='kmeans',
        choices=['kmeans', 'gmm', 'hierarchical', 'dbscan'],
        help='Clustering method (default: kmeans)'
    )
    
    parser.add_argument(
        '--dim-reduction',
        default='umap',
        choices=['pca', 'tsne', 'umap'],
        help='Dimensionality reduction for visualization (default: umap)'
    )
    
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.1,
        help='Expected fraction of anomalies (default: 0.1)'
    )
    
    parser.add_argument(
        '--anomaly-only',
        action='store_true',
        help='Only run anomaly detection, skip clustering'
    )
    
    parser.add_argument(
        '--output',
        default='unsupervised_results',
        help='Output directory (default: unsupervised_results)'
    )
    
    args = parser.parse_args()
    
    # Validate
    if not Path(args.audio).exists():
        print(f" Audio directory not found: {args.audio}")
        sys.exit(1)
    
    # Print configuration
    print("="*70)
    print("🐘 UNSUPERVISED RUMBLE ANALYSIS")
    print("="*70)
    print(f"Audio directory:    {args.audio}")
    print(f"Feature extractor:  {args.features}")
    print(f"Clustering method:  {args.method}")
    print(f"Dim reduction:      {args.dim_reduction}")
    print(f"Output:             {args.output}")
    print("="*70)
    
    # Create analyzer
    analyzer = UnsupervisedRumbleAnalyzer(
        audio_dir=args.audio,
        feature_extractor=args.features,
        output_dir=args.output
    )
    
    # Extract features
    print("\n" + "="*70)
    print("STEP 1: FEATURE EXTRACTION")
    print("="*70)
    features = analyzer.extract_all_features()
    
    # Reduce dimensionality
    print("\n" + "="*70)
    print("STEP 2: DIMENSIONALITY REDUCTION")
    print("="*70)
    embeddings = analyzer.reduce_dimensionality(method=args.dim_reduction)
    
    if not args.anomaly_only:
        # Find optimal clusters (if not specified)
        if args.clusters is None:
            print("\n" + "="*70)
            print("STEP 3: OPTIMAL CLUSTER DETECTION")
            print("="*70)
            optimal_k = analyzer.find_optimal_clusters(max_k=args.max_clusters)
        else:
            optimal_k = args.clusters
            print(f"\n✓ Using specified cluster count: {optimal_k}")
        
        # Cluster
        print("\n" + "="*70)
        print("STEP 4: CLUSTERING")
        print("="*70)
        labels = analyzer.cluster(n_clusters=optimal_k, method=args.method)
    
    # Anomaly detection
    print("\n" + "="*70)
    print("STEP 5: ANOMALY DETECTION")
    print("="*70)
    anomalies = analyzer.detect_anomalies(contamination=args.contamination)
    
    # Visualize
    print("\n" + "="*70)
    print("STEP 6: VISUALIZATION")
    print("="*70)
    analyzer.visualize(
        show_clusters=not args.anomaly_only,
        show_anomalies=True,
        method=args.dim_reduction
    )
    
    # Analyze clusters
    if not args.anomaly_only:
        print("\n" + "="*70)
        print("STEP 7: CLUSTER ANALYSIS")
        print("="*70)
        analyzer.analyze_clusters()
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f" Results saved to: {args.output}/")
    print(f"\nGenerated files:")
    print(f"  - features_raw.npy              # Raw features")
    print(f"  - features_normalized.npy       # Normalized features")
    print(f"  - cluster_assignments.csv       # Cluster for each rumble")
    print(f"  - anomalies.csv                 # Anomaly scores")
    print(f"  - optimal_clusters.png          # Cluster optimization plot")
    print(f"  - visualization.png             # Main visualization")
    
    print(f"\n📊 Quick Summary:")
    if not args.anomaly_only and analyzer.cluster_labels is not None:
        unique_clusters = len(set(analyzer.cluster_labels)) - (1 if -1 in analyzer.cluster_labels else 0)
        print(f"  Clusters discovered: {unique_clusters}")
    
    anomaly_count = sum(anomalies == -1)
    print(f"  Anomalies found: {anomaly_count} ({anomaly_count/len(anomalies)*100:.1f}%)")

if __name__ == "__main__":
    main()
