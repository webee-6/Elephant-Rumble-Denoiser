"""
Visualization utilities for spectrograms and comparisons.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Optional
from datetime import datetime

from config.config import CONFIG


def create_bw_spectrogram(signal: np.ndarray,
                         sr: int,
                         title: str = 'Spectrogram',
                         save_path: Optional[str] = None,
                         show_colorbar: bool = False) -> str:
    """
    Generate black & white spectrogram.
    
    Args:
        signal: Audio signal
        sr: Sample rate
        title: Plot title
        save_path: Output path (if None, auto-generated)
        show_colorbar: Whether to show colorbar
    
    Returns:
        Path to saved spectrogram image
    """
    fig, ax = plt.subplots(figsize=CONFIG.spec_figsize)
    fig.patch.set_facecolor('white')
    
    # Compute STFT and convert to dB
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(signal, 
                           n_fft=CONFIG.n_fft, 
                           hop_length=CONFIG.hop_length)),
        ref=np.max
    )
    
    # Display spectrogram
    img = librosa.display.specshow(
        D,
        sr=sr,
        hop_length=CONFIG.hop_length,
        x_axis='time',
        y_axis='log',
        fmax=CONFIG.fmax_display,
        ax=ax,
        cmap=CONFIG.spec_colormap  # 'binary' for B/W
    )
    
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_ylim([10, CONFIG.fmax_display])
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)
    
    # Add reference lines for elephant frequency range
    ax.axhline(20, color='red', linestyle='--', linewidth=0.8, alpha=0.3, label='20 Hz')
    ax.axhline(300, color='blue', linestyle='--', linewidth=0.8, alpha=0.3, label='300 Hz')
    
    if show_colorbar:
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.ax.tick_params(labelsize=8)
    
    ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    
    # Save
    if save_path is None:
        save_path = f'outputs/spectrograms/spec_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    
    plt.savefig(save_path, dpi=CONFIG.spec_dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path


def create_comparison_plot(original: np.ndarray,
                          cleaned: np.ndarray,
                          sr: int,
                          title: str = 'Before/After Comparison',
                          save_path: Optional[str] = None) -> str:
    """
    Create side-by-side before/after comparison.
    
    Returns:
        Path to saved comparison image
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor('white')
    
    for ax, sig, label in zip(axes, [original, cleaned], ['BEFORE (Noisy)', 'AFTER (Cleaned)']):
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(sig, n_fft=CONFIG.n_fft, hop_length=CONFIG.hop_length)),
            ref=np.max
        )
        
        img = librosa.display.specshow(
            D, sr=sr, hop_length=CONFIG.hop_length,
            x_axis='time', y_axis='log',
            fmax=CONFIG.fmax_display, ax=ax,
            cmap=CONFIG.spec_colormap
        )
        
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_ylim([10, CONFIG.fmax_display])
        ax.axhline(20, color='red', linestyle='--', linewidth=0.8, alpha=0.3)
        ax.axhline(300, color='blue', linestyle='--', linewidth=0.8, alpha=0.3)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path is None:
        save_path = f'outputs/spectrograms/comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    
    plt.savefig(save_path, dpi=CONFIG.spec_dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path
