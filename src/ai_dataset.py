"""
Dataset preparation for deep learning on elephant rumbles.

Supports PyTorch and TensorFlow workflows.
"""

import numpy as np
import librosa
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
from tqdm.auto import tqdm

from src.ai_features import extract_rumble_features


class RumbleDataset:
    """
    Dataset class for elephant rumble learning.
    
    Prepares cleaned audio for:
    - Classification (rumble type, individual ID)
    - Detection (rumble vs non-rumble)
    - Segmentation (start/end point detection)
    - Feature learning (autoencoder, contrastive)
    """
    
    def __init__(self,
                 audio_dir: str,
                 labels_file: Optional[str] = None,
                 transform=None):
        """
        Initialize dataset.
        
        Args:
            audio_dir: Directory with cleaned WAV files
            labels_file: JSON file with labels (optional)
            transform: Data augmentation transform
        """
        self.audio_dir = Path(audio_dir)
        self.audio_files = sorted(list(self.audio_dir.glob('*.wav')))
        self.transform = transform
        
        # Load labels if provided
        self.labels = {}
        if labels_file:
            with open(labels_file, 'r') as f:
                self.labels = json.load(f)
        
        print(f"✅ Loaded {len(self.audio_files)} audio files")
        if self.labels:
            print(f"   With labels for {len(self.labels)} files")
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get item for PyTorch/TensorFlow training.
        
        Returns:
            Dictionary with:
            - 'audio': raw audio signal
            - 'mel_spec': mel spectrogram
            - 'mfcc': MFCC features
            - 'label': class label (if available)
            - 'filename': original filename
        """
        audio_path = self.audio_files[idx]
        
        # Load audio
        signal, sr = librosa.load(audio_path, sr=None)
        
        # Extract features
        features = extract_rumble_features(signal, sr)
        
        # Prepare data dict
        data = {
            'audio': signal,
            'sr': sr,
            'mel_spec': features.mel_spectrogram,
            'mfcc': features.mfcc,
            'filename': audio_path.name,
            'duration': features.duration,
            'fundamental_freq': features.fundamental_freq,
        }
        
        # Add label if available
        filename = audio_path.name
        if filename in self.labels:
            data['label'] = self.labels[filename]
        
        # Apply transform (augmentation)
        if self.transform:
            data = self.transform(data)
        
        return data
    
    def get_mel_spectrograms(self,
                            normalize: bool = True,
                            fixed_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all mel spectrograms as numpy array.
        
        Args:
            normalize: Normalize each spectrogram to [0, 1]
            fixed_length: Pad/crop to fixed time dimension
        
        Returns:
            (spectrograms, labels) arrays
        """
        specs = []
        labels = []
        
        for i in tqdm(range(len(self)), desc="Extracting spectrograms"):
            data = self[i]
            spec = data['mel_spec']
            
            # Normalize
            if normalize:
                spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
            
            # Pad or crop to fixed length
            if fixed_length:
                if spec.shape[1] < fixed_length:
                    # Pad with zeros
                    pad_width = fixed_length - spec.shape[1]
                    spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
                elif spec.shape[1] > fixed_length:
                    # Crop
                    spec = spec[:, :fixed_length]
            
            specs.append(spec)
            
            # Add label if available
            if 'label' in data:
                labels.append(data['label'])
        
        specs = np.array(specs)
        
        if labels:
            labels = np.array(labels)
        else:
            labels = None
        
        return specs, labels
    
    def save_preprocessed(self, output_path: str):
        """
        Save preprocessed features to disk for faster loading.
        
        Args:
            output_path: Path to save .npz file
        """
        all_data = []
        
        for i in tqdm(range(len(self)), desc="Preprocessing"):
            all_data.append(self[i])
        
        np.savez_compressed(output_path, data=all_data)
        print(f"✅ Saved preprocessed data to {output_path}")


def create_spectrogram_patches(mel_spec: np.ndarray,
                               patch_size: Tuple[int, int] = (64, 64),
                               stride: Tuple[int, int] = (32, 32)) -> List[np.ndarray]:
    """
    Create overlapping patches from spectrogram for CNNs.
    
    Args:
        mel_spec: Mel spectrogram (freq, time)
        patch_size: (freq_bins, time_frames)
        stride: (freq_stride, time_stride)
    
    Returns:
        List of patches
    """
    patches = []
    freq_size, time_size = patch_size
    freq_stride, time_stride = stride
    
    n_freq, n_time = mel_spec.shape
    
    for f in range(0, n_freq - freq_size + 1, freq_stride):
        for t in range(0, n_time - time_size + 1, time_stride):
            patch = mel_spec[f:f+freq_size, t:t+time_size]
            patches.append(patch)
    
    return patches


def augment_audio(signal: np.ndarray,
                 sr: int,
                 augmentation_type: str = 'time_stretch') -> np.ndarray:
    """
    Apply data augmentation to audio signal.
    
    Args:
        signal: Audio signal
        sr: Sample rate
        augmentation_type: 'time_stretch', 'pitch_shift', 'add_noise'
    
    Returns:
        Augmented signal
    """
    if augmentation_type == 'time_stretch':
        # Random time stretching (0.8x to 1.2x)
        rate = np.random.uniform(0.8, 1.2)
        augmented = librosa.effects.time_stretch(signal, rate=rate)
    
    elif augmentation_type == 'pitch_shift':
        # Random pitch shift (-2 to +2 semitones)
        n_steps = np.random.uniform(-2, 2)
        augmented = librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)
    
    elif augmentation_type == 'add_noise':
        # Add Gaussian noise
        noise_level = 0.005
        noise = np.random.randn(len(signal)) * noise_level
        augmented = signal + noise
    
    else:
        augmented = signal
    
    return augmented


def create_train_val_split(audio_files: List[Path],
                          val_ratio: float = 0.2,
                          random_seed: int = 42) -> Tuple[List[Path], List[Path]]:
    """
    Split audio files into train/validation sets.
    
    Args:
        audio_files: List of audio file paths
        val_ratio: Validation set ratio
        random_seed: Random seed for reproducibility
    
    Returns:
        (train_files, val_files)
    """
    np.random.seed(random_seed)
    
    indices = np.arange(len(audio_files))
    np.random.shuffle(indices)
    
    split_idx = int(len(audio_files) * (1 - val_ratio))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_files = [audio_files[i] for i in train_indices]
    val_files = [audio_files[i] for i in val_indices]
    
    return train_files, val_files


def prepare_for_pytorch(dataset: RumbleDataset) -> 'torch.utils.data.Dataset':
    """
    Wrap dataset for PyTorch DataLoader.
    
    Returns:
        PyTorch Dataset
    """
    try:
        import torch
        from torch.utils.data import Dataset as TorchDataset
        
        class PyTorchRumbleDataset(TorchDataset):
            def __init__(self, rumble_dataset):
                self.dataset = rumble_dataset
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                data = self.dataset[idx]
                
                # Convert to tensors
                return {
                    'mel_spec': torch.FloatTensor(data['mel_spec']),
                    'mfcc': torch.FloatTensor(data['mfcc']),
                    'audio': torch.FloatTensor(data['audio']),
                    'label': data.get('label', -1)
                }
        
        return PyTorchRumbleDataset(dataset)
    
    except ImportError:
        print("⚠️ PyTorch not installed. Install with: pip install torch")
        return None


def prepare_for_tensorflow(specs: np.ndarray,
                          labels: np.ndarray) -> 'tf.data.Dataset':
    """
    Create TensorFlow dataset from spectrograms.
    
    Args:
        specs: Spectrogram array (n_samples, freq, time)
        labels: Label array
    
    Returns:
        TensorFlow Dataset
    """
    try:
        import tensorflow as tf
        
        # Add channel dimension for CNN
        specs = specs[..., np.newaxis]
        
        dataset = tf.data.Dataset.from_tensor_slices((specs, labels))
        
        # Shuffle and batch
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(32)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    except ImportError:
        print("⚠️ TensorFlow not installed. Install with: pip install tensorflow")
        return None


if __name__ == "__main__":
    # Example usage
    
    # Create dataset
    dataset = RumbleDataset(
        audio_dir='outputs/audio',
        labels_file='data/labels.json'  # Optional
    )
    
    # Get single item
    sample = dataset[0]
    print(f"\nSample data keys: {sample.keys()}")
    print(f"Mel spec shape: {sample['mel_spec'].shape}")
    
    # Get all spectrograms
    specs, labels = dataset.get_mel_spectrograms(
        normalize=True,
        fixed_length=200  # 200 time frames
    )
    print(f"\nAll specs shape: {specs.shape}")
    
    # Save preprocessed
    dataset.save_preprocessed('outputs/preprocessed_rumbles.npz')
