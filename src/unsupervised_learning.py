"""
Unsupervised Learning Pipeline for Elephant Rumble Analysis

No labels required! Discovers natural patterns and clusters in rumbles.

Features:
- Clustering (K-Means, DBSCAN, Hierarchical, GMM)
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Anomaly detection
- Pattern discovery
- Automated optimal cluster detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm.auto import tqdm
import warnings
import json

warnings.filterwarnings('ignore')

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# Anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor


class UnsupervisedRumbleAnalyzer:
    """
    Complete unsupervised analysis of elephant rumbles.
    
    Workflow:
    1. Extract acoustic features from all rumbles
    2. Normalize features
    3. Reduce dimensionality (PCA/UMAP)
    4. Find optimal number of clusters
    5. Cluster rumbles
    6. Detect anomalies
    7. Visualize and interpret
    """
    
    def __init__(self, 
                 audio_dir: str,
                 feature_extractor='opensmile',
                 output_dir: str = 'unsupervised_results'):
        """
        Initialize analyzer.
        
        Args:
            audio_dir: Directory with cleaned WAV files
            feature_extractor: 'opensmile', 'openl3', 'custom', or 'hybrid'
            output_dir: Where to save results
        """
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.audio_files = sorted(list(self.audio_dir.glob('*.wav')))
        print(f"📂 Found {len(self.audio_files)} rumble files")
        
        # Initialize feature extractor
        from src.advanced_features import AdvancedFeatureExtractor
        self.extractor = AdvancedFeatureExtractor(backend=feature_extractor)
        
        # Storage
        self.features = None
        self.features_normalized = None
        self.feature_names = None
        self.filenames = []
        
        # Results
        self.clusters = None
        self.cluster_labels = None
        self.anomaly_scores = None
        self.embeddings_2d = None
        self.embeddings_3d = None
    
    def extract_all_features(self):
        """Extract features from all rumble files."""
        print(f"\n🔬 Extracting acoustic features ({self.extractor.backend})...")
        
        features_list = []
        
        for audio_file in tqdm(self.audio_files, desc="Extracting features"):
            try:
                import librosa
                signal, sr = librosa.load(audio_file, sr=None)
                
                # Extract features
                features = self.extractor.extract(signal, sr)
                features_list.append(features)
                self.filenames.append(audio_file.name)
                
            except Exception as e:
                print(f"⚠️ Failed to process {audio_file.name}: {e}")
        
        # Convert to array
        self.features = np.array(features_list)
        
        print(f"✅ Extracted features: {self.features.shape}")
        print(f"   Files: {len(self.filenames)}")
        print(f"   Features per file: {self.features.shape[1]}")
        
        # Normalize features (important for clustering!)
        scaler = StandardScaler()
        self.features_normalized = scaler.fit_transform(self.features)
        
        print(f"✅ Features normalized (mean=0, std=1)")
        
        # Apply PCA if features are very high-dimensional (helps with clustering stability)
        if self.features.shape[1] > 100:
            from sklearn.decomposition import PCA
            print(f"\n⚙️ High-dimensional features detected ({self.features.shape[1]}D)")
            print(f"   Applying PCA for clustering stability...")
            
            # Reduce to explain 95% variance (or max 100 dims)
            n_components = min(100, self.features.shape[0] - 1)
            pca = PCA(n_components=n_components, random_state=42)
            self.features_normalized_pca = pca.fit_transform(self.features_normalized)
            
            variance_explained = pca.explained_variance_ratio_.sum()
            print(f"   Reduced {self.features.shape[1]}D → {pca.n_components_}D")
            print(f"   Variance explained: {variance_explained:.1%}")
            
            # Use PCA-reduced features for clustering
            self.features_for_clustering = self.features_normalized_pca
        else:
            self.features_for_clustering = self.features_normalized
        
        # Save features
        np.save(self.output_dir / 'features_raw.npy', self.features)
        np.save(self.output_dir / 'features_normalized.npy', self.features_normalized)
        if hasattr(self, 'features_normalized_pca'):
            np.save(self.output_dir / 'features_pca.npy', self.features_normalized_pca)
        
        with open(self.output_dir / 'filenames.json', 'w') as f:
            json.dump(self.filenames, f, indent=2)
        
        return self.features_normalized
    
    def reduce_dimensionality(self, 
                             method: str = 'umap',
                             n_components: int = 2):
        """
        Reduce to 2D/3D for visualization.
        
        Args:
            method: 'pca', 'tsne', or 'umap'
            n_components: 2 or 3
        """
        print(f"\n📐 Reducing dimensionality with {method.upper()}...")
        
        if self.features_normalized is None:
            self.extract_all_features()
        
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, random_state=42)
            embeddings = reducer.fit_transform(self.features_normalized)
            
            # Explained variance
            var_explained = reducer.explained_variance_ratio_
            print(f"   Variance explained: {var_explained[:n_components].sum():.2%}")
        
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(30, len(self.features_normalized) - 1),
                random_state=42,
                n_iter=1000
            )
            embeddings = reducer.fit_transform(self.features_normalized)
        
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=min(15, len(self.features_normalized) - 1),
                    min_dist=0.1,
                    random_state=42
                )
                embeddings = reducer.fit_transform(self.features_normalized)
            except ImportError:
                print("⚠️ UMAP not installed. Install with: pip install umap-learn")
                print("   Falling back to PCA...")
                return self.reduce_dimensionality(method='pca', n_components=n_components)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if n_components == 2:
            self.embeddings_2d = embeddings
        elif n_components == 3:
            self.embeddings_3d = embeddings
        
        print(f"✅ Reduced to {n_components}D: {embeddings.shape}")
        
        # Save
        np.save(self.output_dir / f'embeddings_{method}_{n_components}d.npy', embeddings)
        
        return embeddings
    
    def find_optimal_clusters(self, 
                             max_k: int = 10,
                             methods: List[str] = ['silhouette', 'elbow', 'bic']) -> int:
        """
        Automatically determine optimal number of clusters.
        
        Args:
            max_k: Maximum number of clusters to try
            methods: Evaluation methods to use
        
        Returns:
            Optimal number of clusters
        """
        print(f"\n🔍 Finding optimal number of clusters (2-{max_k})...")
        
        if self.features_normalized is None:
            self.extract_all_features()
        
        # Use PCA-reduced features if available, otherwise normalized features
        features_to_use = self.features_for_clustering
        
        # Ensure max_k doesn't exceed number of samples
        max_k = min(max_k, len(features_to_use) - 1)
        
        results = {
            'k': list(range(2, max_k + 1)),
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': [],
            'inertia': [],
            'bic': []
        }
        
        for k in tqdm(range(2, max_k + 1), desc="Testing cluster counts"):
            # K-Means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_to_use)
            
            # Silhouette (higher is better, range: -1 to 1)
            sil = silhouette_score(features_to_use, labels)
            results['silhouette'].append(sil)
            
            # Davies-Bouldin (lower is better)
            db = davies_bouldin_score(features_to_use, labels)
            results['davies_bouldin'].append(db)
            
            # Calinski-Harabasz (higher is better)
            ch = calinski_harabasz_score(features_to_use, labels)
            results['calinski_harabasz'].append(ch)
            
            # Inertia (for elbow method, lower is better)
            results['inertia'].append(kmeans.inertia_)
            
            # BIC (for GMM, lower is better)
            # Add regularization to prevent numerical issues with high-dimensional data
            try:
                gmm = GaussianMixture(
                    n_components=k, 
                    random_state=42,
                    reg_covar=1e-6,  # Regularization for numerical stability
                    covariance_type='diag'  # Diagonal covariance for high-dim data
                )
                gmm.fit(features_to_use.astype(np.float64))  # Use float64 for stability
                results['bic'].append(gmm.bic(features_to_use.astype(np.float64)))
            except Exception as e:
                # If GMM fails, use a penalty value based on k-means inertia
                print(f"⚠️ GMM failed for k={k}, using K-Means BIC approximation")
                # Approximate BIC from K-Means: BIC ≈ inertia + k*log(n)*d
                n_samples, n_features = features_to_use.shape
                bic_approx = kmeans.inertia_ + k * np.log(n_samples) * n_features
                results['bic'].append(bic_approx)
        
        # Plot metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Silhouette
        axes[0, 0].plot(results['k'], results['silhouette'], 'o-', linewidth=2)
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].set_title('Silhouette Score (higher is better)')
        axes[0, 0].grid(True, alpha=0.3)
        best_sil = results['k'][np.argmax(results['silhouette'])]
        axes[0, 0].axvline(best_sil, color='red', linestyle='--', label=f'Best: {best_sil}')
        axes[0, 0].legend()
        
        # Elbow (inertia)
        axes[0, 1].plot(results['k'], results['inertia'], 'o-', linewidth=2)
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Inertia')
        axes[0, 1].set_title('Elbow Method (look for bend)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Davies-Bouldin
        axes[1, 0].plot(results['k'], results['davies_bouldin'], 'o-', linewidth=2)
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Davies-Bouldin Score')
        axes[1, 0].set_title('Davies-Bouldin Index (lower is better)')
        axes[1, 0].grid(True, alpha=0.3)
        best_db = results['k'][np.argmin(results['davies_bouldin'])]
        axes[1, 0].axvline(best_db, color='red', linestyle='--', label=f'Best: {best_db}')
        axes[1, 0].legend()
        
        # BIC
        axes[1, 1].plot(results['k'], results['bic'], 'o-', linewidth=2)
        axes[1, 1].set_xlabel('Number of Clusters')
        axes[1, 1].set_ylabel('BIC')
        axes[1, 1].set_title('BIC (lower is better)')
        axes[1, 1].grid(True, alpha=0.3)
        best_bic = results['k'][np.argmin(results['bic'])]
        axes[1, 1].axvline(best_bic, color='red', linestyle='--', label=f'Best: {best_bic}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'optimal_clusters.png', dpi=150, bbox_inches='tight')
        print(f"📊 Saved cluster optimization plot")
        
        # Recommend based on silhouette (most reliable)
        optimal_k = best_sil
        
        print(f"\n✅ Optimal clusters detected:")
        print(f"   Silhouette method: {best_sil}")
        print(f"   Davies-Bouldin: {best_db}")
        print(f"   BIC: {best_bic}")
        print(f"\n   → Recommended: {optimal_k} clusters")
        
        # Save results
        pd.DataFrame(results).to_csv(self.output_dir / 'cluster_metrics.csv', index=False)
        
        return optimal_k
    
    def cluster(self, 
               n_clusters: Optional[int] = None,
               method: str = 'kmeans') -> np.ndarray:
        """
        Cluster rumbles into groups.
        
        Args:
            n_clusters: Number of clusters (if None, auto-detect)
            method: 'kmeans', 'gmm', 'hierarchical', or 'dbscan'
        
        Returns:
            Cluster labels
        """
        if self.features_normalized is None:
            self.extract_all_features()
        
        # Use PCA-reduced features if available
        features_to_use = self.features_for_clustering
        
        # Auto-detect optimal clusters
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters()
        
        print(f"\n🎯 Clustering with {method.upper()} (k={n_clusters})...")
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
            labels = clusterer.fit_predict(features_to_use)
        
        elif method == 'gmm':
            # Use regularization and diagonal covariance for high-dimensional data
            try:
                clusterer = GaussianMixture(
                    n_components=n_clusters, 
                    random_state=42,
                    reg_covar=1e-6,  # Regularization
                    covariance_type='diag',  # Diagonal covariance (works better for high-dim)
                    max_iter=200,
                    n_init=5
                )
                labels = clusterer.fit_predict(features_to_use.astype(np.float64))
            except Exception as e:
                print(f"⚠️ GMM failed: {e}")
                print(f"   Falling back to K-Means...")
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
                labels = clusterer.fit_predict(features_to_use)
        
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(features_to_use)
        
        elif method == 'dbscan':
            # DBSCAN auto-determines number of clusters
            clusterer = DBSCAN(eps=0.5, min_samples=3)
            labels = clusterer.fit_predict(features_to_use)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"   DBSCAN found {n_clusters} clusters")
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.cluster_labels = labels
        
        # Statistics
        unique_labels = np.unique(labels)
        print(f"\n✅ Clustering complete:")
        for label in unique_labels:
            count = np.sum(labels == label)
            percentage = count / len(labels) * 100
            cluster_name = f"Noise" if label == -1 else f"Cluster {label}"
            print(f"   {cluster_name}: {count} rumbles ({percentage:.1f}%)")
        
        # Save
        cluster_df = pd.DataFrame({
            'filename': self.filenames,
            'cluster': labels
        })
        cluster_df.to_csv(self.output_dir / 'cluster_assignments.csv', index=False)
        print(f"💾 Saved cluster assignments")
        
        return labels
    
    def detect_anomalies(self, 
                        contamination: float = 0.1,
                        method: str = 'isolation_forest') -> np.ndarray:
        """
        Detect unusual/rare rumbles.
        
        Args:
            contamination: Expected fraction of outliers (0.01-0.5)
            method: 'isolation_forest', 'lof', or 'elliptic'
        
        Returns:
            Anomaly labels (-1 = anomaly, 1 = normal)
        """
        print(f"\n🔍 Detecting anomalies with {method}...")
        
        if self.features_normalized is None:
            self.extract_all_features()
        
        # Use PCA-reduced features if available
        features_to_use = self.features_for_clustering
        
        if method == 'isolation_forest':
            detector = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
        elif method == 'lof':
            detector = LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=min(20, len(features_to_use) - 1)
            )
        elif method == 'elliptic':
            detector = EllipticEnvelope(
                contamination=contamination,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Detect
        anomaly_labels = detector.fit_predict(features_to_use)
        
        # Get anomaly scores if available
        if hasattr(detector, 'score_samples'):
            self.anomaly_scores = detector.score_samples(features_to_use)
        else:
            self.anomaly_scores = None
        
        # Statistics
        n_anomalies = np.sum(anomaly_labels == -1)
        print(f"✅ Found {n_anomalies} anomalous rumbles ({n_anomalies/len(anomaly_labels)*100:.1f}%)")
        
        # Save
        anomaly_df = pd.DataFrame({
            'filename': self.filenames,
            'is_anomaly': anomaly_labels == -1,
            'anomaly_score': self.anomaly_scores if self.anomaly_scores is not None else np.nan
        })
        anomaly_df.to_csv(self.output_dir / 'anomalies.csv', index=False)
        
        # List anomalies
        if n_anomalies > 0:
            print(f"\n🚨 Anomalous rumbles:")
            anomalies = anomaly_df[anomaly_df['is_anomaly']]
            for idx, row in anomalies.iterrows():
                print(f"   - {row['filename']}")
        
        return anomaly_labels
    
    def visualize(self, 
                 show_clusters: bool = True,
                 show_anomalies: bool = True,
                 method: str = 'umap'):
        """
        Create comprehensive visualizations.
        
        Args:
            show_clusters: Color by cluster
            show_anomalies: Highlight anomalies
            method: 'pca', 'tsne', or 'umap'
        """
        print(f"\n📊 Creating visualizations...")
        
        # Reduce to 2D if not done
        if self.embeddings_2d is None:
            self.reduce_dimensionality(method=method, n_components=2)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Clusters
        if show_clusters and self.cluster_labels is not None:
            scatter = axes[0].scatter(
                self.embeddings_2d[:, 0],
                self.embeddings_2d[:, 1],
                c=self.cluster_labels,
                cmap='tab10',
                alpha=0.6,
                s=100,
                edgecolors='black',
                linewidth=0.5
            )
            axes[0].set_title(f'Rumble Clusters ({method.upper()})', fontsize=14, fontweight='bold')
            
            # Add cluster centers
            unique_clusters = np.unique(self.cluster_labels)
            for cluster in unique_clusters:
                if cluster == -1:
                    continue  # Skip noise in DBSCAN
                mask = self.cluster_labels == cluster
                center = self.embeddings_2d[mask].mean(axis=0)
                axes[0].scatter(center[0], center[1], 
                              marker='*', s=500, c='red', 
                              edgecolors='black', linewidth=2,
                              zorder=10)
                axes[0].text(center[0], center[1], f'C{cluster}',
                           fontsize=12, fontweight='bold',
                           ha='center', va='center')
            
            plt.colorbar(scatter, ax=axes[0], label='Cluster')
        else:
            axes[0].scatter(
                self.embeddings_2d[:, 0],
                self.embeddings_2d[:, 1],
                alpha=0.6,
                s=100,
                edgecolors='black',
                linewidth=0.5
            )
            axes[0].set_title(f'Rumble Distribution ({method.upper()})', fontsize=14, fontweight='bold')
        
        axes[0].set_xlabel(f'{method.upper()} 1', fontsize=12)
        axes[0].set_ylabel(f'{method.upper()} 2', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Anomalies
        if show_anomalies and self.anomaly_scores is not None:
            # Normalize scores to 0-1 for coloring
            scores_norm = (self.anomaly_scores - self.anomaly_scores.min()) / \
                         (self.anomaly_scores.max() - self.anomaly_scores.min())
            
            scatter = axes[1].scatter(
                self.embeddings_2d[:, 0],
                self.embeddings_2d[:, 1],
                c=scores_norm,
                cmap='RdYlGn_r',  # Red = anomalous, Green = normal
                alpha=0.6,
                s=100,
                edgecolors='black',
                linewidth=0.5
            )
            axes[1].set_title('Anomaly Scores', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=axes[1], label='Anomaly Score')
        else:
            axes[1].scatter(
                self.embeddings_2d[:, 0],
                self.embeddings_2d[:, 1],
                alpha=0.6,
                s=100,
                edgecolors='black',
                linewidth=0.5
            )
            axes[1].set_title(f'Rumble Distribution', fontsize=14, fontweight='bold')
        
        axes[1].set_xlabel(f'{method.upper()} 1', fontsize=12)
        axes[1].set_ylabel(f'{method.upper()} 2', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualization.png', dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization")
        
        plt.show()
    
    def analyze_clusters(self):
        """Analyze characteristics of each cluster."""
        if self.cluster_labels is None:
            print("⚠️ Run cluster() first")
            return
        
        print(f"\n📊 Cluster Analysis:")
        print("="*70)
        
        unique_clusters = np.unique(self.cluster_labels)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                cluster_name = "Noise"
            else:
                cluster_name = f"Cluster {cluster_id}"
            
            mask = self.cluster_labels == cluster_id
            cluster_features = self.features[mask]
            
            print(f"\n{cluster_name}:")
            print(f"  Size: {mask.sum()} rumbles ({mask.sum()/len(mask)*100:.1f}%)")
            
            # Representative rumbles (closest to cluster center)
            center = cluster_features.mean(axis=0)
            distances = np.linalg.norm(cluster_features - center, axis=1)
            closest_idx = np.argsort(distances)[:3]
            
            print(f"  Representative rumbles (closest to center):")
            for i, idx in enumerate(closest_idx):
                file_idx = np.where(mask)[0][idx]
                print(f"    {i+1}. {self.filenames[file_idx]}")


if __name__ == "__main__":
    # Example usage
    analyzer = UnsupervisedRumbleAnalyzer(
        audio_dir='outputs/audio',
        feature_extractor='opensmile',  # or 'openl3', 'custom', 'hybrid'
        output_dir='unsupervised_results'
    )
    
    # Extract features
    features = analyzer.extract_all_features()
    
    # Find optimal clusters
    optimal_k = analyzer.find_optimal_clusters(max_k=10)
    
    # Cluster
    labels = analyzer.cluster(n_clusters=optimal_k, method='kmeans')
    
    # Detect anomalies
    anomalies = analyzer.detect_anomalies(contamination=0.1)
    
    # Visualize
    analyzer.visualize(method='umap')
    
    # Analyze
    analyzer.analyze_clusters()
