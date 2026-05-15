"""
Digital Signal Processing algorithms for elephant rumble denoising.
"""

import numpy as np
import librosa
import noisereduce as nr
from scipy.signal import butter, filtfilt, wiener, iirnotch
from typing import Tuple

from config.config import CONFIG


def algorithm_bandpass_butterworth(signal: np.ndarray, 
                                  sr: int,
                                  lowcut: float = None,
                                  highcut: float = None,
                                  order: int = None) -> np.ndarray:
    """
    Zero-phase Butterworth band-pass filter.
    
    Math: H(ω) = 1 / (1 + (ω/ωc)^(2n))
    Applied via filtfilt (forward-backward) for zero phase distortion.
    
    Args:
        signal: Input time-domain signal
        sr: Sample rate
        lowcut: Lower cutoff frequency (Hz)
        highcut: Upper cutoff frequency (Hz)
        order: Filter order
    
    Returns:
        Filtered signal
    """
    # Use config defaults if not provided
    lowcut = lowcut or CONFIG.bp_lowcut
    highcut = highcut or CONFIG.bp_highcut
    order = order or CONFIG.bp_order
    
    nyq = sr / 2.0
    low = max(lowcut / nyq, 0.001)   # Normalized, must be in (0, 1)
    high = min(highcut / nyq, 0.990)
    
    if high <= low:
        high = min(0.99, low + 0.1)
    
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    
    # Safety check
    if not np.isfinite(filtered).all():
        print('⚠️  Butterworth filter unstable, returning original')
        return signal
    
    return filtered


def algorithm_notch_generator_harmonics(signal: np.ndarray,
                                       sr: int,
                                       fundamental: float = None,
                                       n_harmonics: int = None,
                                       Q: float = None) -> np.ndarray:
    """
    Notch filter to remove generator harmonics (60Hz, 120Hz, 180Hz, ...).
    
    Args:
        signal: Input signal
        sr: Sample rate
        fundamental: Fundamental frequency (50Hz Europe, 60Hz US)
        n_harmonics: Number of harmonics to notch
        Q: Quality factor (higher = narrower notch)
    
    Returns:
        Filtered signal with generator harmonics removed
    """
    fundamental = fundamental or CONFIG.generator_fundamental
    n_harmonics = n_harmonics or CONFIG.generator_harmonics
    Q = Q or CONFIG.generator_notch_q
    
    filtered = signal.copy()
    
    for i in range(1, n_harmonics + 1):
        freq = fundamental * i
        if freq >= sr / 2:
            break
        
        # Design IIR notch filter
        b, a = iirnotch(freq, Q, sr)
        filtered = filtfilt(b, a, filtered)
    
    return filtered


def algorithm_spectral_gating(signal: np.ndarray,
                             noise_profile: np.ndarray,
                             sr: int,
                             noise_type: str = 'vehicle') -> np.ndarray:
    """
    Spectral gating using noise profile.
    
    Math: G(m,k) = max(1 - α·Φ_n(k)/|X(m,k)|², 0)
          Ŝ(m,k) = G(m,k) · X(m,k)
    
    Args:
        signal: Input signal (already band-passed)
        noise_profile: Pure noise segment for estimation
        sr: Sample rate
        noise_type: 'airplane', 'vehicle', 'generator', 'background'
    
    Returns:
        Denoised signal
    """
    # Get noise-specific parameters
    params = CONFIG.noise_params.get(noise_type, CONFIG.noise_params['vehicle'])
    
    # Ensure noise profile is long enough
    min_samples = int(0.5 * sr)
    if len(noise_profile) < min_samples:
        # Tile to extend
        repeats = int(np.ceil(sr / len(noise_profile))) + 1
        noise_profile = np.tile(noise_profile, repeats)[:sr]
    
    # Apply spectral gating
    denoised = nr.reduce_noise(
        y=signal,
        y_noise=noise_profile,
        sr=sr,
        stationary=params['stationary'],
        prop_decrease=params['prop_decrease'],
        freq_mask_smooth_hz=params['freq_mask_smooth_hz'],
        time_mask_smooth_ms=params['time_mask_smooth_ms']
    )
    
    return denoised


def algorithm_hpss(signal: np.ndarray,
                  sr: int,
                  kernel_size: int = None,
                  margin: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Harmonic-Percussive Source Separation.
    
    Math:
        H(m,k) = Median_time(|X(m,k)|)
        P(m,k) = Median_freq(|X(m,k)|)
        M_H = H^β / (H^β + P^β)
    
    Elephants produce HARMONIC rumbles (horizontal ridges in spectrogram).
    Machinery produces PERCUSSIVE noise (vertical spikes).
    
    Args:
        signal: Input signal
        sr: Sample rate
        kernel_size: Median filter size (odd number)
        margin: Separation margin (higher = harder separation)
    
    Returns:
        (harmonic_component, percussive_component)
    """
    kernel_size = kernel_size or CONFIG.hpss_kernel_size
    margin = margin or CONFIG.hpss_margin
    
    # STFT
    D = librosa.stft(signal, n_fft=CONFIG.n_fft, hop_length=CONFIG.hop_length)
    
    # Decompose
    D_h, D_p = librosa.decompose.hpss(D, kernel_size=kernel_size, margin=margin)
    
    # Inverse STFT
    harmonic = librosa.istft(D_h, hop_length=CONFIG.hop_length)
    percussive = librosa.istft(D_p, hop_length=CONFIG.hop_length)
    
    # Match length
    min_len = min(len(signal), len(harmonic))
    return harmonic[:min_len], percussive[:min_len]


def algorithm_wiener_filter(signal: np.ndarray,
                           window_size: int = None) -> np.ndarray:
    """
    Wiener filter for MSE-optimal noise reduction.
    
    Math: W(k) = Φ_s(k) / (Φ_s(k) + Φ_n(k))
    
    Minimizes E[|s(t) - ŝ(t)|²] using local statistics.
    
    Args:
        signal: Input signal
        window_size: Local estimation window (odd number)
    
    Returns:
        Filtered signal
    """
    window_size = window_size or CONFIG.wiener_window
    
    filtered = wiener(signal, mysize=window_size)
    
    # Safety check
    if not np.isfinite(filtered).all():
        print('⚠️  Wiener filter produced NaN/Inf, returning original')
        return signal
    
    return filtered
