"""
Advanced segmentation and windowing utilities for efficient processing of long recordings.
"""

import numpy as np
from typing import Tuple, Generator, Optional
from config.config import CONFIG


def overlap_add_segments(signal: np.ndarray,
                        segment_length: int = 44100 * 30,  # 30 seconds
                        overlap: float = 0.25) -> Generator[Tuple[int, int, np.ndarray], None, None]:
    """
    Generate overlapping segments for processing long audio.
    
    Uses overlap-add method to prevent edge artifacts.
    
    Args:
        signal: Full audio signal
        segment_length: Length of each segment in samples (default: 30s at 44.1kHz)
        overlap: Overlap ratio (0.0-0.5, default: 0.25 = 25%)
    
    Yields:
        (start_idx, end_idx, segment) tuples
    
    Example:
        >>> for start, end, seg in overlap_add_segments(long_audio):
        >>>     processed_seg = process(seg)
        >>>     # Reconstruct with proper overlap handling
    """
    hop_size = int(segment_length * (1 - overlap))
    
    for start in range(0, len(signal), hop_size):
        end = min(start + segment_length, len(signal))
        segment = signal[start:end]
        
        # Apply fade in/out to overlap regions for smooth transitions
        if overlap > 0 and start > 0:
            fade_length = int(segment_length * overlap)
            fade_in = np.linspace(0, 1, fade_length)
            segment[:fade_length] *= fade_in
        
        if overlap > 0 and end < len(signal):
            fade_length = int(segment_length * overlap)
            fade_out = np.linspace(1, 0, fade_length)
            segment[-fade_length:] *= fade_out
        
        yield start, end, segment


def windowed_filter(signal: np.ndarray,
                   filter_func,
                   segment_length: int = 44100 * 30,
                   overlap: float = 0.25,
                   **filter_kwargs) -> np.ndarray:
    """
    Apply a filter function to signal using segmentation with overlap-add.
    
    Prevents memory issues and edge artifacts on long recordings.
    
    Args:
        signal: Input signal
        filter_func: Filter function to apply (e.g., algorithm_bandpass_butterworth)
        segment_length: Segment size in samples
        overlap: Overlap ratio
        **filter_kwargs: Arguments to pass to filter_func
    
    Returns:
        Filtered signal reconstructed from segments
    
    Example:
        >>> from src.algorithms import algorithm_bandpass_butterworth
        >>> filtered = windowed_filter(
        >>>     long_signal, 
        >>>     algorithm_bandpass_butterworth,
        >>>     sr=44100,
        >>>     lowcut=20,
        >>>     highcut=1000
        >>> )
    """
    # For short signals, process directly
    if len(signal) < segment_length:
        return filter_func(signal, **filter_kwargs)
    
    # Initialize output
    output = np.zeros_like(signal)
    overlap_count = np.zeros_like(signal)  # Track overlaps for averaging
    
    # Process each segment
    for start, end, segment in overlap_add_segments(signal, segment_length, overlap):
        # Apply filter
        filtered_segment = filter_func(segment, **filter_kwargs)
        
        # Add to output with overlap
        output[start:end] += filtered_segment
        overlap_count[start:end] += 1
    
    # Average overlapping regions
    overlap_count[overlap_count == 0] = 1  # Avoid division by zero
    output /= overlap_count
    
    return output


def adaptive_segment_length(signal_length: int,
                           sr: int,
                           max_memory_mb: float = 500) -> int:
    """
    Calculate optimal segment length based on available memory.
    
    Args:
        signal_length: Total signal length in samples
        sr: Sample rate
        max_memory_mb: Maximum memory per segment in MB
    
    Returns:
        Optimal segment length in samples
    """
    # Estimate memory usage for STFT
    # Complex STFT: n_fft/2 * n_frames * 16 bytes (complex128)
    n_frames_per_second = sr / CONFIG.hop_length
    bytes_per_second = (CONFIG.n_fft / 2) * n_frames_per_second * 16
    mb_per_second = bytes_per_second / (1024 * 1024)
    
    max_seconds = max_memory_mb / mb_per_second
    max_samples = int(max_seconds * sr)
    
    # Ensure at least 10 seconds, at most 60 seconds
    segment_length = np.clip(max_samples, 10 * sr, 60 * sr)
    
    return segment_length


def hann_window(length: int) -> np.ndarray:
    """
    Generate Hann window for smooth transitions.
    
    Args:
        length: Window length in samples
    
    Returns:
        Hann window array
    """
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(length) / (length - 1))


def apply_window_smoothing(segments: list,
                          overlap_samples: int) -> np.ndarray:
    """
    Combine segments with smooth windowing in overlap regions.
    
    Args:
        segments: List of (start_idx, signal_segment) tuples
        overlap_samples: Number of overlapping samples between segments
    
    Returns:
        Combined signal with smooth transitions
    """
    if not segments:
        return np.array([])
    
    # Calculate total length
    last_start, last_seg = segments[-1]
    total_length = last_start + len(last_seg)
    
    output = np.zeros(total_length)
    window_sum = np.zeros(total_length)
    
    for start, segment in segments:
        end = start + len(segment)
        
        # Create window
        window = np.ones(len(segment))
        
        # Apply Hann window to overlap regions
        if overlap_samples > 0:
            # Fade in at start (except first segment)
            if start > 0:
                fade_len = min(overlap_samples, len(segment))
                window[:fade_len] = hann_window(fade_len * 2)[:fade_len]
            
            # Fade out at end (except last segment)
            if end < total_length:
                fade_len = min(overlap_samples, len(segment))
                window[-fade_len:] = hann_window(fade_len * 2)[fade_len:]
        
        # Add weighted segment
        output[start:end] += segment * window
        window_sum[start:end] += window
    
    # Normalize
    window_sum[window_sum == 0] = 1
    output /= window_sum
    
    return output


def frame_signal(signal: np.ndarray,
                frame_length: int,
                hop_length: int,
                window: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Frame a signal into overlapping windows (like librosa.util.frame).
    
    Args:
        signal: Input signal
        frame_length: Length of each frame
        hop_length: Number of samples between frames
        window: Window function to apply (if None, uses rectangular)
    
    Returns:
        2D array of shape (frame_length, n_frames)
    """
    n_frames = 1 + (len(signal) - frame_length) // hop_length
    
    # Pad signal if needed
    padded_length = frame_length + (n_frames - 1) * hop_length
    if len(signal) < padded_length:
        signal = np.pad(signal, (0, padded_length - len(signal)))
    
    # Create frame indices
    indices = np.arange(frame_length)[:, None] + np.arange(n_frames) * hop_length
    frames = signal[indices]
    
    # Apply window if provided
    if window is not None:
        frames = frames * window[:, None]
    
    return frames


def reconstruct_from_frames(frames: np.ndarray,
                           hop_length: int,
                           window: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Reconstruct signal from overlapping frames using overlap-add.
    
    Args:
        frames: 2D array of shape (frame_length, n_frames)
        hop_length: Hop size used in framing
        window: Window used in framing (for normalization)
    
    Returns:
        Reconstructed signal
    """
    frame_length, n_frames = frames.shape
    signal_length = frame_length + (n_frames - 1) * hop_length
    
    signal = np.zeros(signal_length)
    window_sum = np.zeros(signal_length)
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        signal[start:end] += frames[:, i]
        
        if window is not None:
            window_sum[start:end] += window
        else:
            window_sum[start:end] += 1
    
    # Normalize by window sum
    window_sum[window_sum == 0] = 1
    signal /= window_sum
    
    return signal


# Example usage:
if __name__ == "__main__":
    # Demo: Process long signal with segmentation
    import librosa
    from src.algorithms import algorithm_bandpass_butterworth
    
    # Simulate long recording (5 minutes)
    sr = 44100
    duration = 300  # 5 minutes
    long_signal = np.random.randn(sr * duration)
    
    print(f"Signal length: {len(long_signal) / sr:.1f} seconds")
    print(f"Samples: {len(long_signal):,}")
    
    # Calculate optimal segment length
    segment_len = adaptive_segment_length(len(long_signal), sr, max_memory_mb=500)
    print(f"Optimal segment: {segment_len / sr:.1f} seconds")
    
    # Process with windowing
    filtered = windowed_filter(
        long_signal,
        algorithm_bandpass_butterworth,
        segment_length=segment_len,
        overlap=0.25,
        sr=sr,
        lowcut=20,
        highcut=1000
    )
    
    print(f"Output length: {len(filtered)}")
    print("✅ Segmented processing complete!")
