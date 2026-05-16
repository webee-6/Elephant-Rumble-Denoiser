"""
End-to-End Neural Network Pipeline for Elephant Rumble Classification

Includes:
- Segmentation & windowing of long rumbles
- Acoustic feature extraction per window
- Neural network classification
- Temporal aggregation of predictions
"""

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm.auto import tqdm
import json

from src.segmentation import frame_signal, hann_window


class WindowedRumbleDataset(Dataset):
    """
    Dataset that segments rumbles into windows with overlap.
    Each window gets acoustic features extracted.
    """
    
    def __init__(self,
                 audio_dir: str,
                 labels_file: Optional[str] = None,
                 window_length_sec: float = 1.0,
                 hop_length_sec: float = 0.5,
                 sr: int = 44100,
                 normalize: bool = True):
        """
        Initialize windowed dataset.
        
        Args:
            audio_dir: Directory with cleaned WAV files
            labels_file: JSON with labels (optional)
            window_length_sec: Window size in seconds
            hop_length_sec: Hop size in seconds (overlap = window - hop)
            sr: Expected sample rate
            normalize: Normalize features per window
        """
        self.audio_dir = Path(audio_dir)
        self.audio_files = sorted(list(self.audio_dir.glob('*.wav')))
        self.window_length = int(window_length_sec * sr)
        self.hop_length = int(hop_length_sec * sr)
        self.sr = sr
        self.normalize = normalize
        
        # Load labels
        self.labels = {}
        if labels_file:
            with open(labels_file, 'r') as f:
                self.labels = json.load(f)
        
        # Pre-compute windows for all files
        print(f"🔬 Extracting windowed features from {len(self.audio_files)} files...")
        self._prepare_windows()
    
    def _prepare_windows(self):
        """Extract all windows from all audio files."""
        self.windows = []  # List of (features, label, file_idx, window_idx)
        
        for file_idx, audio_path in enumerate(tqdm(self.audio_files, desc="Processing files")):
            # Load audio
            signal, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            
            # Get label if available
            label = self.labels.get(audio_path.name, -1)
            
            # Create windows
            window_func = hann_window(self.window_length)
            
            num_windows = 1 + (len(signal) - self.window_length) // self.hop_length
            
            for win_idx in range(num_windows):
                start = win_idx * self.hop_length
                end = start + self.window_length
                
                if end > len(signal):
                    # Pad last window
                    window = np.pad(signal[start:], (0, end - len(signal)))
                else:
                    window = signal[start:end]
                
                # Apply window function
                windowed_signal = window * window_func
                
                # Extract features for this window
                features = self._extract_window_features(windowed_signal, sr)
                
                self.windows.append({
                    'features': features,
                    'label': label,
                    'file_idx': file_idx,
                    'window_idx': win_idx,
                    'filename': audio_path.name
                })
        
        print(f"✅ Created {len(self.windows)} windows from {len(self.audio_files)} files")
        print(f"   Window size: {self.window_length/self.sr:.2f}s")
        print(f"   Overlap: {(self.window_length - self.hop_length)/self.sr:.2f}s")
    
    def _extract_window_features(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract acoustic features from a single window.
        
        Features (39 total):
        - MFCCs (20)
        - Spectral features (7)
        - Temporal features (4)
        - Low-frequency features (8)
        
        Args:
            signal: Windowed audio signal
            sr: Sample rate
        
        Returns:
            Feature vector
        """
        features = []
        
        # === MFCCs (20 coefficients) ===
        mfcc = librosa.feature.mfcc(
            y=signal, 
            sr=sr, 
            n_mfcc=20,
            n_fft=2048,
            hop_length=512,
            fmin=10,
            fmax=300
        )
        # Take mean across time
        features.extend(np.mean(mfcc, axis=1))
        
        # === Spectral Features ===
        # Spectral centroid
        spec_cent = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=2048)[0]
        features.append(np.mean(spec_cent))
        
        # Spectral rolloff
        spec_roll = librosa.feature.spectral_rolloff(y=signal, sr=sr, n_fft=2048)[0]
        features.append(np.mean(spec_roll))
        
        # Spectral bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=2048)[0]
        features.append(np.mean(spec_bw))
        
        # Spectral flatness
        spec_flat = librosa.feature.spectral_flatness(y=signal, n_fft=2048)[0]
        features.append(np.mean(spec_flat))
        
        # Spectral contrast (mean of 7 bands)
        spec_contrast = librosa.feature.spectral_contrast(
            y=signal, sr=sr, n_fft=2048, fmin=10
        )
        features.extend(np.mean(spec_contrast, axis=1))  # 7 values
        
        # === Temporal Features ===
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(signal, frame_length=2048)[0]
        features.append(np.mean(zcr))
        
        # RMS energy
        rms = librosa.feature.rms(y=signal, frame_length=2048)[0]
        features.append(np.mean(rms))
        
        # Energy
        energy = np.sum(signal ** 2) / len(signal)
        features.append(energy)
        
        # Autocorrelation peak (periodicity indicator)
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        features.append(np.max(autocorr[1:min(len(autocorr), 1000)]) / autocorr[0])
        
        # === Low-Frequency Elephant-Specific Features ===
        # Compute power spectrum
        D = np.abs(librosa.stft(signal, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # Energy in different frequency bands
        bands = [
            (10, 25),   # Fundamental range
            (25, 50),   # First harmonic range
            (50, 100),  # Second harmonic range
            (100, 200), # Higher harmonics
            (10, 300),  # Total elephant range
        ]
        
        for low, high in bands:
            mask = (freqs >= low) & (freqs <= high)
            band_energy = np.sum(D[mask, :] ** 2)
            features.append(band_energy)
        
        # Harmonic-to-noise ratio estimate
        total_energy = np.sum(D ** 2)
        elephant_energy = features[-1]  # 10-300 Hz energy
        harmonic_ratio = elephant_energy / (total_energy + 1e-10)
        features.append(harmonic_ratio)
        
        # Spectral flux (change over time)
        spec_flux = np.mean(np.diff(D, axis=1) ** 2)
        features.append(spec_flux)
        
        # Dominant frequency in 10-100 Hz range
        low_freq_mask = (freqs >= 10) & (freqs <= 100)
        low_freq_spectrum = np.mean(D[low_freq_mask, :], axis=1)
        if len(low_freq_spectrum) > 0:
            dom_freq_idx = np.argmax(low_freq_spectrum)
            dom_freq = freqs[low_freq_mask][dom_freq_idx]
        else:
            dom_freq = 0
        features.append(dom_freq)
        
        features = np.array(features, dtype=np.float32)
        
        # Normalize
        if self.normalize and np.std(features) > 0:
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """Get a single window."""
        window = self.windows[idx]
        return {
            'features': torch.FloatTensor(window['features']),
            'label': window['label'],
            'file_idx': window['file_idx'],
            'window_idx': window['window_idx']
        }
    
    def get_file_windows(self, file_idx: int) -> List[Dict]:
        """Get all windows from a specific file."""
        return [w for w in self.windows if w['file_idx'] == file_idx]


class TemporalRumbleClassifier(nn.Module):
    """
    Neural network for rumble classification.
    
    Architecture:
    - Input: Acoustic features per window (39 features)
    - LSTM: Captures temporal patterns across windows
    - Attention: Focuses on important windows
    - Dense layers: Classification
    """
    
    def __init__(self,
                 input_dim: int = 39,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 4,
                 dropout: float = 0.3,
                 use_attention: bool = True):
        """
        Initialize classifier.
        
        Args:
            input_dim: Feature dimension (39)
            hidden_dim: LSTM hidden size
            num_layers: Number of LSTM layers
            num_classes: Number of call types
            dropout: Dropout rate
            use_attention: Use attention mechanism
        """
        super(TemporalRumbleClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x, return_attention=False):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, features) or (batch, features) for single window
            return_attention: Return attention weights
        
        Returns:
            logits: (batch, num_classes)
            attention_weights: (batch, seq_len) if return_attention=True
        """
        # Handle single window input
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention or mean pooling
        if self.use_attention and x.size(1) > 1:
            # Compute attention scores
            attn_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            
            # Weighted sum
            context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)
        else:
            # Mean pooling over time
            context = torch.mean(lstm_out, dim=1)
            attn_weights = None
        
        # Classification
        logits = self.classifier(context)
        
        if return_attention and attn_weights is not None:
            return logits, attn_weights.squeeze(-1)
        else:
            return logits


def collate_windows_from_same_file(batch):
    """
    Custom collate function to group windows from the same file.
    
    This allows the LSTM to see temporal context within a rumble.
    """
    # Group by file
    files = {}
    for item in batch:
        file_idx = item['file_idx']
        if file_idx not in files:
            files[file_idx] = []
        files[file_idx].append(item)
    
    # Create batches (one per file)
    batched_data = []
    for file_idx, windows in files.items():
        # Sort by window index
        windows = sorted(windows, key=lambda x: x['window_idx'])
        
        # Stack features
        features = torch.stack([w['features'] for w in windows])  # (seq_len, feat_dim)
        label = windows[0]['label']  # All windows have same label
        
        batched_data.append({
            'features': features.unsqueeze(0),  # (1, seq_len, feat_dim)
            'label': torch.LongTensor([label]),
            'file_idx': file_idx
        })
    
    return batched_data


class RumbleClassificationTrainer:
    """
    Complete training pipeline for windowed rumble classification.
    """
    
    def __init__(self,
                 audio_dir: str,
                 labels_file: str,
                 window_length_sec: float = 1.0,
                 hop_length_sec: float = 0.5,
                 output_dir: str = 'models'):
        """
        Initialize trainer.
        
        Args:
            audio_dir: Directory with cleaned audio
            labels_file: JSON with labels
            window_length_sec: Window size
            hop_length_sec: Hop size
            output_dir: Where to save models
        """
        self.audio_dir = audio_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create dataset
        print("📊 Creating windowed dataset...")
        self.dataset = WindowedRumbleDataset(
            audio_dir=audio_dir,
            labels_file=labels_file,
            window_length_sec=window_length_sec,
            hop_length_sec=hop_length_sec
        )
        
        # Determine number of classes
        labels = [w['label'] for w in self.dataset.windows if w['label'] != -1]
        self.num_classes = len(set(labels))
        print(f"   Number of classes: {self.num_classes}")
        
        self.model = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def create_dataloaders(self,
                          val_split: float = 0.2,
                          batch_size: int = 8,
                          use_file_batching: bool = True):
        """
        Create train/val dataloaders.
        
        Args:
            val_split: Validation split ratio
            batch_size: Batch size (number of files if file_batching=True)
            use_file_batching: Group windows by file for temporal modeling
        
        Returns:
            train_loader, val_loader
        """
        # Split by files (not windows) to avoid leakage
        num_files = len(self.dataset.audio_files)
        indices = np.random.permutation(num_files)
        
        split_idx = int(num_files * (1 - val_split))
        train_file_indices = set(indices[:split_idx])
        val_file_indices = set(indices[split_idx:])
        
        # Filter windows
        train_windows = [w for w in self.dataset.windows if w['file_idx'] in train_file_indices]
        val_windows = [w for w in self.dataset.windows if w['file_idx'] in val_file_indices]
        
        print(f"\n📊 Data split:")
        print(f"   Train files: {len(train_file_indices)}, windows: {len(train_windows)}")
        print(f"   Val files: {len(val_file_indices)}, windows: {len(val_windows)}")
        
        # Create subset datasets
        from torch.utils.data import Subset
        train_dataset = Subset(self.dataset, [i for i, w in enumerate(self.dataset.windows) if w in train_windows])
        val_dataset = Subset(self.dataset, [i for i, w in enumerate(self.dataset.windows) if w in val_windows])
        
        # Create loaders
        if use_file_batching:
            # Custom collation for temporal context
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_windows_from_same_file
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_windows_from_same_file
            )
        else:
            # Standard window-by-window
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train(self,
             epochs: int = 100,
             learning_rate: float = 0.001,
             batch_size: int = 8,
             early_stopping_patience: int = 15):
        """
        Train the model.
        
        Args:
            epochs: Number of epochs
            learning_rate: Learning rate
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
        
        Returns:
            Trained model
        """
        # Create model
        print("\n🧠 Creating model...")
        self.model = TemporalRumbleClassifier(
            input_dim=39,  # Feature dimension
            hidden_dim=128,
            num_layers=2,
            num_classes=self.num_classes,
            dropout=0.3,
            use_attention=True
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        print(f"   Device: {device}")
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(
            val_split=0.2,
            batch_size=batch_size,
            use_file_batching=True
        )
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        
        print(f"\n🚀 Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for file_batch in train_loader:
                for item in file_batch:
                    features = item['features'].to(device)  # (1, seq_len, feat_dim)
                    labels = item['label'].to(device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validate
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for file_batch in val_loader:
                    for item in file_batch:
                        features = item['features'].to(device)
                        labels = item['label'].to(device)
                        
                        outputs = self.model(features)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
                print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self.output_dir / 'best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
        
        print(f"\n✅ Training complete!")
        print(f"   Best val accuracy: {best_val_acc:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load(self.output_dir / 'best_model.pth'))
        
        return self.model
    
    def predict(self, audio_path: str) -> Dict:
        """
        Predict call type for a new rumble.
        
        Args:
            audio_path: Path to cleaned audio file
        
        Returns:
            Prediction dictionary with class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Load and window the audio
        signal, sr = librosa.load(audio_path, sr=self.dataset.sr)
        
        # Extract windows
        window_features = []
        num_windows = 1 + (len(signal) - self.dataset.window_length) // self.dataset.hop_length
        
        window_func = hann_window(self.dataset.window_length)
        
        for win_idx in range(num_windows):
            start = win_idx * self.dataset.hop_length
            end = start + self.dataset.window_length
            
            if end > len(signal):
                window = np.pad(signal[start:], (0, end - len(signal)))
            else:
                window = signal[start:end]
            
            windowed_signal = window * window_func
            features = self.dataset._extract_window_features(windowed_signal, sr)
            window_features.append(features)
        
        # Stack windows
        features_tensor = torch.FloatTensor(window_features).unsqueeze(0).to(device)  # (1, seq_len, feat_dim)
        
        # Predict
        with torch.no_grad():
            logits, attention = self.model(features_tensor, return_attention=True)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            predicted_class = int(torch.argmax(logits, dim=1)[0])
            attention_weights = attention[0].cpu().numpy() if attention is not None else None
        
        return {
            'predicted_class': predicted_class,
            'probabilities': probs.tolist(),
            'confidence': float(probs[predicted_class]),
            'num_windows': num_windows,
            'attention_weights': attention_weights.tolist() if attention_weights is not None else None
        }


if __name__ == "__main__":
    # Example usage
    trainer = RumbleClassificationTrainer(
        audio_dir='outputs/audio',
        labels_file='data/labels.json',
        window_length_sec=1.0,  # 1 second windows
        hop_length_sec=0.5,     # 50% overlap
        output_dir='models'
    )
    
    # Train
    model = trainer.train(
        epochs=100,
        learning_rate=0.001,
        batch_size=8,
        early_stopping_patience=15
    )
    
    # Predict on new rumble
    result = trainer.predict('outputs/audio/selection_001_cleaned.wav')
    print(f"\nPrediction: Class {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Probabilities: {result['probabilities']}")
