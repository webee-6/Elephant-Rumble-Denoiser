"""
AI Feature Extraction for Elephant Rumble Learning

Extracts acoustic features from denoised signals for machine learning.
"""

import numpy as np
import librosa
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import pandas as pd

from config.config import CONFIG


@dataclass
class RumbleFeatures:
    """Container for extracted rumble features"""
    
    # Temporal features
    duration: float
    zero_crossing_rate: np.ndarray
    energy: np.ndarray
    rms: np.ndarray
    
    # Spectral features
    spectral_centroid: np.ndarray
    spectral_rolloff: np.ndarray
    spectral_bandwidth: np.ndarray
    spectral_contrast: np.ndarray
    spectral_flatness: np.ndarray
    
    # MFCCs (for pattern recognition)
    mfcc: np.ndarray  # Shape: (n_mfcc, n_frames)
    mfcc_delta: np.ndarray
    mfcc_delta2: np.ndarray
    
    # Low-frequency specific (elephant range)
    fundamental_freq: float
    harmonic_energy_ratio: float
    
    # Mel spectrogram (for deep learning)
    mel_spectrogram: np.ndarray
    
    # Chroma features (pitch class)
    chroma: np.ndarray
    
    # Metadata
    sample_rate: int
    n_frames: int


def extract_rumble_features(signal: np.ndarray,
                           sr: int,
                           n_mfcc: int = 20,
                           n_mels: int = 128,
                           fmin: float = 10,
                           fmax: float = 300) -> RumbleFeatures:
    """
    Extract comprehensive features for elephant rumble learning.
    
    Focuses on low-frequency features (10-300 Hz) relevant to elephant calls.
    
    Args:
        signal: Denoised audio signal
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of mel bands
        fmin: Minimum frequency (Hz) - elephant fundamental
        fmax: Maximum frequency (Hz) - elephant harmonics
    
    Returns:
        RumbleFeatures dataclass with all extracted features
    """
    
    # === Temporal Features ===
    duration = len(signal) / sr
    
    # Zero crossing rate (indicates periodicity)
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=2048, hop_length=512)[0]
    
    # Energy envelope
    energy = np.array([
        sum(abs(signal[i:i+2048]**2))
        for i in range(0, len(signal), 512)
    ])
    
    # RMS energy
    rms = librosa.feature.rms(y=signal, frame_length=2048, hop_length=512)[0]
    
    # === Spectral Features ===
    # Spectral centroid (brightness)
    spec_centroid = librosa.feature.spectral_centroid(
        y=signal, sr=sr, n_fft=4096, hop_length=512
    )[0]
    
    # Spectral rolloff (frequency below which X% of energy is contained)
    spec_rolloff = librosa.feature.spectral_rolloff(
        y=signal, sr=sr, n_fft=4096, hop_length=512, roll_percent=0.85
    )[0]
    
    # Spectral bandwidth
    spec_bandwidth = librosa.feature.spectral_bandwidth(
        y=signal, sr=sr, n_fft=4096, hop_length=512
    )[0]
    
    # Spectral contrast (peaks vs valleys in spectrum)
    spec_contrast = librosa.feature.spectral_contrast(
        y=signal, sr=sr, n_fft=4096, hop_length=512, fmin=fmin
    )
    
    # Spectral flatness (noise vs tonal)
    spec_flatness = librosa.feature.spectral_flatness(
        y=signal, n_fft=4096, hop_length=512
    )[0]
    
    # === MFCCs (Mel-Frequency Cepstral Coefficients) ===
    # Standard in speech/bioacoustics recognition
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=4096, hop_length=512,
        fmin=fmin, fmax=fmax
    )
    
    # Delta MFCCs (velocity)
    mfcc_delta = librosa.feature.delta(mfcc)
    
    # Delta-delta MFCCs (acceleration)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # === Low-Frequency Specific Features ===
    # Estimate fundamental frequency (pitch)
    f0 = estimate_fundamental_frequency(signal, sr, fmin=fmin, fmax=fmax)
    
    # Harmonic-to-noise ratio
    harmonic_ratio = compute_harmonic_energy_ratio(signal, sr, fmin=fmin, fmax=fmax)
    
    # === Mel Spectrogram (for deep learning) ===
    # Time-frequency representation optimized for perception
    mel_spec = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=4096, hop_length=512,
        n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    # Convert to dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # === Chroma Features ===
    # Pitch class representation
    chroma = librosa.feature.chroma_cqt(
        y=signal, sr=sr, hop_length=512, fmin=fmin
    )
    
    # Package into dataclass
    features = RumbleFeatures(
        duration=duration,
        zero_crossing_rate=zcr,
        energy=energy,
        rms=rms,
        spectral_centroid=spec_centroid,
        spectral_rolloff=spec_rolloff,
        spectral_bandwidth=spec_bandwidth,
        spectral_contrast=spec_contrast,
        spectral_flatness=spec_flatness,
        mfcc=mfcc,
        mfcc_delta=mfcc_delta,
        mfcc_delta2=mfcc_delta2,
        fundamental_freq=f0,
        harmonic_energy_ratio=harmonic_ratio,
        mel_spectrogram=mel_spec_db,
        chroma=chroma,
        sample_rate=sr,
        n_frames=mfcc.shape[1]
    )
    
    return features


def estimate_fundamental_frequency(signal: np.ndarray,
                                   sr: int,
                                   fmin: float = 10,
                                   fmax: float = 300) -> float:
    """
    Estimate fundamental frequency (F0) of elephant rumble.
    
    Uses autocorrelation method optimized for low frequencies.
    
    Args:
        signal: Audio signal
        sr: Sample rate
        fmin: Minimum F0 to search
        fmax: Maximum F0 to search
    
    Returns:
        Estimated F0 in Hz
    """
    # Use librosa's piptrack for pitch tracking
    pitches, magnitudes = librosa.piptrack(
        y=signal, sr=sr, n_fft=4096, hop_length=512,
        fmin=fmin, fmax=fmax, threshold=0.1
    )
    
    # Get pitch with highest magnitude
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    
    if len(pitch_values) == 0:
        return 0.0
    
    # Return median pitch (robust to outliers)
    return float(np.median(pitch_values))


def compute_harmonic_energy_ratio(signal: np.ndarray,
                                  sr: int,
                                  fmin: float = 10,
                                  fmax: float = 300) -> float:
    """
    Compute ratio of harmonic to total energy.
    
    Higher values indicate tonal rumble, lower values indicate noisy.
    
    Args:
        signal: Audio signal
        sr: Sample rate
        fmin: Low frequency bound
        fmax: High frequency bound
    
    Returns:
        Harmonic energy ratio (0-1)
    """
    # Compute STFT
    D = np.abs(librosa.stft(signal, n_fft=4096, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    
    # Energy in elephant frequency range
    mask = (freqs >= fmin) & (freqs <= fmax)
    elephant_energy = np.sum(D[mask, :] ** 2)
    
    # Total energy
    total_energy = np.sum(D ** 2)
    
    if total_energy == 0:
        return 0.0
    
    return float(elephant_energy / total_energy)


def features_to_vector(features: RumbleFeatures,
                      include_temporal: bool = True,
                      include_spectral: bool = True,
                      include_mfcc: bool = True) -> np.ndarray:
    """
    Convert RumbleFeatures to flat feature vector for classical ML.
    
    Args:
        features: RumbleFeatures object
        include_temporal: Include temporal features
        include_spectral: Include spectral features
        include_mfcc: Include MFCC features
    
    Returns:
        1D feature vector
    """
    vector = []
    
    if include_temporal:
        # Statistical aggregates of temporal features
        vector.extend([
            features.duration,
            np.mean(features.zero_crossing_rate),
            np.std(features.zero_crossing_rate),
            np.mean(features.energy),
            np.std(features.energy),
            np.mean(features.rms),
            np.std(features.rms),
        ])
    
    if include_spectral:
        # Statistical aggregates of spectral features
        vector.extend([
            np.mean(features.spectral_centroid),
            np.std(features.spectral_centroid),
            np.mean(features.spectral_rolloff),
            np.std(features.spectral_rolloff),
            np.mean(features.spectral_bandwidth),
            np.std(features.spectral_bandwidth),
            np.mean(features.spectral_flatness),
            np.std(features.spectral_flatness),
            features.fundamental_freq,
            features.harmonic_energy_ratio,
        ])
        # Mean of each spectral contrast band
        vector.extend(np.mean(features.spectral_contrast, axis=1))
    
    if include_mfcc:
        # Statistical aggregates of MFCCs
        vector.extend(np.mean(features.mfcc, axis=1))
        vector.extend(np.std(features.mfcc, axis=1))
        vector.extend(np.mean(features.mfcc_delta, axis=1))
        vector.extend(np.std(features.mfcc_delta, axis=1))
    
    return np.array(vector)


def create_feature_dataframe(features_list: List[RumbleFeatures],
                            labels: Optional[List] = None) -> pd.DataFrame:
    """
    Create pandas DataFrame from list of features for ML training.
    
    Args:
        features_list: List of RumbleFeatures objects
        labels: Optional list of labels (rumble type, individual ID, etc.)
    
    Returns:
        DataFrame with one row per rumble
    """
    data = []
    
    for features in features_list:
        row = {
            # Temporal
            'duration': features.duration,
            'zcr_mean': np.mean(features.zero_crossing_rate),
            'zcr_std': np.std(features.zero_crossing_rate),
            'energy_mean': np.mean(features.energy),
            'energy_std': np.std(features.energy),
            'rms_mean': np.mean(features.rms),
            'rms_std': np.std(features.rms),
            
            # Spectral
            'spec_centroid_mean': np.mean(features.spectral_centroid),
            'spec_centroid_std': np.std(features.spectral_centroid),
            'spec_rolloff_mean': np.mean(features.spectral_rolloff),
            'spec_rolloff_std': np.std(features.spectral_rolloff),
            'spec_bandwidth_mean': np.mean(features.spectral_bandwidth),
            'spec_bandwidth_std': np.std(features.spectral_bandwidth),
            'spec_flatness_mean': np.mean(features.spectral_flatness),
            'spec_flatness_std': np.std(features.spectral_flatness),
            
            # Low-frequency
            'fundamental_freq': features.fundamental_freq,
            'harmonic_ratio': features.harmonic_energy_ratio,
        }
        
        # Add MFCC statistics
        for i in range(features.mfcc.shape[0]):
            row[f'mfcc_{i}_mean'] = np.mean(features.mfcc[i, :])
            row[f'mfcc_{i}_std'] = np.std(features.mfcc[i, :])
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    if labels is not None:
        df['label'] = labels
    
    return df


if __name__ == "__main__":
    # Example usage
    import librosa
    
    # Load example cleaned signal
    signal, sr = librosa.load('outputs/audio/selection_001_cleaned.wav', sr=None)
    
    print(f"Extracting features from {len(signal)/sr:.2f}s rumble...")
    features = extract_rumble_features(signal, sr)
    
    print(f"\n✅ Extracted features:")
    print(f"   Duration: {features.duration:.2f}s")
    print(f"   Fundamental freq: {features.fundamental_freq:.1f} Hz")
    print(f"   Harmonic ratio: {features.harmonic_energy_ratio:.3f}")
    print(f"   MFCC shape: {features.mfcc.shape}")
    print(f"   Mel spectrogram: {features.mel_spectrogram.shape}")
    
    # Convert to feature vector
    vec = features_to_vector(features)
    print(f"\n   Feature vector size: {len(vec)}")
