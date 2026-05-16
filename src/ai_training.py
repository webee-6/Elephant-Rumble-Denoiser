"""
End-to-end training pipeline for elephant rumble learning.

Workflow:
1. Extract features from cleaned audio
2. Prepare datasets
3. Train models
4. Evaluate and save
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src.ai_features import (
    extract_rumble_features,
    features_to_vector,
    create_feature_dataframe
)
from src.ai_dataset import RumbleDataset


class RumbleTrainer:
    """
    Complete training pipeline for rumble learning.
    """
    
    def __init__(self,
                 audio_dir: str,
                 labels_file: Optional[str] = None,
                 output_dir: str = 'models'):
        """
        Initialize trainer.
        
        Args:
            audio_dir: Directory with cleaned WAV files
            labels_file: JSON file with labels
            output_dir: Where to save models
        """
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load dataset
        self.dataset = RumbleDataset(
            audio_dir=str(audio_dir),
            labels_file=labels_file
        )
        
        self.features = None
        self.labels = None
        self.feature_vectors = None
    
    def extract_all_features(self, save: bool = True):
        """
        Extract features from all audio files.
        
        Args:
            save: Save features to disk
        """
        print("🔬 Extracting features from all rumbles...")
        
        features_list = []
        labels_list = []
        
        for i in tqdm(range(len(self.dataset))):
            data = self.dataset[i]
            
            # Get features (already extracted in dataset)
            import librosa
            signal = data['audio']
            sr = data['sr']
            
            features = extract_rumble_features(signal, sr)
            features_list.append(features)
            
            if 'label' in data:
                labels_list.append(data['label'])
        
        self.features = features_list
        
        if labels_list:
            self.labels = np.array(labels_list)
        
        # Create feature vectors for classical ML
        self.feature_vectors = np.array([
            features_to_vector(f) for f in features_list
        ])
        
        print(f"✅ Extracted {len(features_list)} feature sets")
        print(f"   Feature vector shape: {self.feature_vectors.shape}")
        
        if save:
            # Save features
            np.save(self.output_dir / 'feature_vectors.npy', self.feature_vectors)
            if self.labels is not None:
                np.save(self.output_dir / 'labels.npy', self.labels)
            print(f"   Saved to {self.output_dir}")
        
        return self.feature_vectors, self.labels
    
    def train_random_forest(self,
                          n_estimators: int = 100,
                          save_model: bool = True) -> Dict:
        """
        Train Random Forest classifier.
        
        Args:
            n_estimators: Number of trees
            save_model: Save trained model
        
        Returns:
            Dictionary with model and metrics
        """
        if self.feature_vectors is None:
            self.extract_all_features()
        
        if self.labels is None:
            print("⚠️ No labels available for supervised training")
            return None
        
        print("\n🌲 Training Random Forest...")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        import joblib
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.feature_vectors, self.labels,
            test_size=0.2, random_state=42, stratify=self.labels
        )
        
        # Train
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        # Evaluate
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        y_pred = clf.predict(X_test)
        
        print(f"\n✅ Random Forest Results:")
        print(f"   Train accuracy: {train_acc:.3f}")
        print(f"   Test accuracy: {test_acc:.3f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = clf.feature_importances_
        top_features = np.argsort(feature_importance)[-10:][::-1]
        
        print(f"\nTop 10 Most Important Features:")
        for i, idx in enumerate(top_features):
            print(f"   {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
        
        # Save model
        if save_model:
            model_path = self.output_dir / 'random_forest.joblib'
            joblib.dump(clf, model_path)
            print(f"\n💾 Model saved to {model_path}")
        
        return {
            'model': clf,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'true_labels': y_test
        }
    
    def train_cnn_pytorch(self,
                         epochs: int = 50,
                         batch_size: int = 16,
                         save_model: bool = True) -> Dict:
        """
        Train PyTorch CNN on spectrograms.
        
        Args:
            epochs: Training epochs
            batch_size: Batch size
            save_model: Save trained model
        
        Returns:
            Dictionary with model and history
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            if self.labels is None:
                print("⚠️ No labels available")
                return None
            
            print("\n🧠 Training PyTorch CNN...")
            
            # Get spectrograms
            specs, labels = self.dataset.get_mel_spectrograms(
                normalize=True,
                fixed_length=200
            )
            
            # Add channel dimension
            specs = specs[:, np.newaxis, :, :]  # (N, 1, freq, time)
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                specs, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Convert to tensors
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train = torch.LongTensor(y_train)
            y_test = torch.LongTensor(y_test)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Create model
            from src.ai_models import create_rumble_cnn_pytorch
            num_classes = len(np.unique(labels))
            model = create_rumble_cnn_pytorch(
                input_shape=(specs.shape[2], specs.shape[3]),
                num_classes=num_classes
            )
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
            
            for epoch in range(epochs):
                # Train
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += batch_y.size(0)
                    train_correct += predicted.eq(batch_y).sum().item()
                
                train_loss /= len(train_loader)
                train_acc = train_correct / train_total
                
                # Validate
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += batch_y.size(0)
                        val_correct += predicted.eq(batch_y).sum().item()
                
                val_loss /= len(test_loader)
                val_acc = val_correct / val_total
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}")
                    print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                    print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            print(f"\n✅ Training complete!")
            print(f"   Final val accuracy: {val_acc:.4f}")
            
            # Save model
            if save_model:
                model_path = self.output_dir / 'cnn_pytorch.pth'
                torch.save(model.state_dict(), model_path)
                print(f"💾 Model saved to {model_path}")
            
            return {
                'model': model,
                'history': history,
                'val_acc': val_acc
            }
        
        except ImportError:
            print("⚠️ PyTorch not installed. Install with: pip install torch")
            return None
    
    def visualize_features(self, save: bool = True):
        """
        Visualize extracted features with t-SNE.
        
        Args:
            save: Save visualization
        """
        if self.feature_vectors is None:
            self.extract_all_features(save=False)
        
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            
            print("\n📊 Creating t-SNE visualization...")
            
            # Reduce to 2D
            tsne = TSNE(n_components=2, random_state=42)
            features_2d = tsne.fit_transform(self.feature_vectors)
            
            # Plot
            plt.figure(figsize=(10, 8))
            
            if self.labels is not None:
                unique_labels = np.unique(self.labels)
                for label in unique_labels:
                    mask = self.labels == label
                    plt.scatter(
                        features_2d[mask, 0],
                        features_2d[mask, 1],
                        label=f'Class {label}',
                        alpha=0.6
                    )
                plt.legend()
            else:
                plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6)
            
            plt.title('t-SNE Visualization of Rumble Features')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.grid(True, alpha=0.3)
            
            if save:
                plt.savefig(self.output_dir / 'tsne_features.png', dpi=150, bbox_inches='tight')
                print(f"💾 Saved to {self.output_dir / 'tsne_features.png'}")
            
            plt.show()
        
        except ImportError:
            print("⚠️ scikit-learn not installed")


if __name__ == "__main__":
    # Example usage
    trainer = RumbleTrainer(
        audio_dir='outputs/audio',
        labels_file='data/labels.json',  # Create this file with your labels
        output_dir='models'
    )
    
    # Extract features
    features, labels = trainer.extract_all_features()
    
    # Visualize
    trainer.visualize_features()
    
    # Train Random Forest
    rf_results = trainer.train_random_forest(n_estimators=100)
    
    # Train PyTorch CNN
    # cnn_results = trainer.train_cnn_pytorch(epochs=50)
