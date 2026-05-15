"""
Utilities for noise profile extraction and validation.
"""

import numpy as np
import librosa
from typing import Dict, Tuple

from config.config import CONFIG


def extract_noise_profile(audio: np.ndarray,
                         sr: int,
                         call_start: int,
                         call_end: int,
                         mode: str = 'adaptive') -> Tuple[np.ndarray, str]:
    """
    Extract noise profile segment from audio.
    
    Args:
        audio: Full audio signal
        sr: Sample rate
        call_start: Start sample of elephant call
        call_end: End sample of elephant call
        mode: 'before', 'after', or 'adaptive'
    
    Returns:
        (noise_profile_segment, source_description)
    """
    duration_samples = int(CONFIG.noise_duration_sec * sr)
    min_samples = int(0.5 * sr)
    
    if mode == 'before' or mode == 'adaptive':
        # Try to get noise before the call
        noise_start = max(0, call_start - duration_samples)
        noise_end = call_start
        
        if noise_end - noise_start >= min_samples:
            return audio[noise_start:noise_end], f'before_call ({(noise_end-noise_start)/sr:.1f}s)'
    
    if mode == 'after' or mode == 'adaptive':
        # Try to get noise after the call
        noise_start = call_end
        noise_end = min(len(audio), call_end + duration_samples)
        
        if noise_end - noise_start >= min_samples:
            return audio[noise_start:noise_end], f'after_call ({(noise_end-noise_start)/sr:.1f}s)'
    
    # Fallback: use end of file
    fallback_start = max(0, len(audio) - duration_samples)
    return audio[fallback_start:], f'end_of_file ({(len(audio)-fallback_start)/sr:.1f}s)'


def validate_noise_profile(noise_profile: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Check if noise profile contains low-frequency energy (potential elephant contamination).
    
    Returns:
        Dictionary with validation metrics
    """
    # Compute spectrum
    D = np.abs(librosa.stft(noise_profile, n_fft=CONFIG.n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=CONFIG.n_fft)
    
    # Energy in elephant fundamental range (10-30 Hz)
    fundamental_mask = (freqs >= 10) & (freqs <= 30)
    fundamental_energy = np.mean(D[fundamental_mask, :] ** 2)
    
    # Energy in full elephant range (10-300 Hz)
    elephant_mask = (freqs >= 10) & (freqs <= 300)
    elephant_energy = np.mean(D[elephant_mask, :] ** 2)
    
    # Total energy
    total_energy = np.mean(D ** 2)
    
    return {
        'fundamental_energy': fundamental_energy,
        'elephant_band_energy': elephant_energy,
        'total_energy': total_energy,
        'fundamental_ratio': fundamental_energy / (total_energy + 1e-10),
        'elephant_ratio': elephant_energy / (total_energy + 1e-10)
    }


def classify_noise_type(filename: str) -> str:
    """
    Classify noise type from filename.
    
    Returns:
        'airplane', 'vehicle', 'generator', or 'background'
    """
    filename_lower = filename.lower()
    
    if 'airplane' in filename_lower:
        return 'airplane'
    elif 'generator' in filename_lower:
        return 'generator'
    elif 'vehicle' in filename_lower or 'car' in filename_lower:
        return 'vehicle'
    elif 'background' in filename_lower:
        return 'background'
    else:
        return 'vehicle'  # Default
