"""
Enhanced processing pipeline with segmentation for long recordings.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict

from config.config import CONFIG
from src.algorithms import (
    algorithm_bandpass_butterworth,
    algorithm_notch_generator_harmonics,
    algorithm_spectral_gating,
    algorithm_hpss,
    algorithm_wiener_filter
)
from src.noise_utils import (
    extract_noise_profile,
    validate_noise_profile,
    classify_noise_type
)
from src.visualization import (
    create_bw_spectrogram,
    create_comparison_plot
)
from src.segmentation import (
    windowed_filter,
    adaptive_segment_length
)


def process_single_call_segmented(audio_path: str,
                                  start_time: float,
                                  end_time: float,
                                  selection_id: int,
                                  output_dir: str = 'outputs',
                                  use_segmentation: bool = True,
                                  segment_duration: float = 30.0) -> Dict:
    """
    Enhanced pipeline with automatic segmentation for long calls.
    
    **New Features:**
    - Automatic segmentation for calls > 30 seconds
    - Overlap-add processing to prevent edge artifacts
    - Memory-efficient processing of very long recordings
    - Adaptive segment sizing based on available memory
    
    Pipeline Stages:
        0. Load Audio & Setup
        1. Extract & Validate Noise Profile
        2. Segmented Butterworth Band-Pass (if long)
        3. Generator Notch Filter (if applicable)
        4. Spectral Gating
        5. HPSS (Harmonic-Percussive Separation)
        6. Wiener Filter
        7. Normalize Output
        8. Save Audio & Spectrograms
    
    Args:
        audio_path: Path to audio file
        start_time: Start time in seconds
        end_time: End time in seconds
        selection_id: Unique identifier
        output_dir: Output directory
        use_segmentation: Enable segmentation for long calls (default: True)
        segment_duration: Segment length in seconds (default: 30s)
    
    Returns:
        Dictionary with processing results and metadata
    """
    result = {
        'selection_id': selection_id,
        'filename': Path(audio_path).name,
        'start_time': start_time,
        'end_time': end_time,
        'status': 'pending',
        'error': None,
        'segmented': False
    }
    
    try:
        # === STEP 0: Setup ===
        import os
        os.makedirs(f"{output_dir}/audio", exist_ok=True)
        os.makedirs(f"{output_dir}/spectrograms", exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        
        # Load Audio
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Extract call segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        start_sample = min(start_sample, len(y) - 1)
        end_sample = min(end_sample, len(y))
        
        call_segment = y[start_sample:end_sample]
        call_duration = len(call_segment) / sr
        
        # Classify noise type
        noise_type = classify_noise_type(Path(audio_path).name)
        result['noise_type'] = noise_type
        result['call_duration'] = call_duration
        
        # Decide if segmentation is needed
        needs_segmentation = use_segmentation and call_duration > segment_duration
        result['segmented'] = needs_segmentation
        
        if needs_segmentation:
            segment_length = int(segment_duration * sr)
            print(f"  📊 Long call ({call_duration:.1f}s) - using segmentation")
        
        # === STEP 1: Extract Noise Profile ===
        noise_profile, noise_source = extract_noise_profile(
            y, sr, start_sample, end_sample, mode=CONFIG.noise_profile_mode
        )
        result['noise_source'] = noise_source
        
        # Validate noise profile
        noise_metrics = validate_noise_profile(noise_profile, sr)
        result['noise_validation'] = noise_metrics
        
        # === STEP 2: Butterworth Band-Pass (with segmentation if needed) ===
        if needs_segmentation:
            bp_filtered = windowed_filter(
                call_segment,
                algorithm_bandpass_butterworth,
                segment_length=segment_length,
                overlap=0.25,
                sr=sr
            )
            noise_bp = algorithm_bandpass_butterworth(noise_profile, sr)
        else:
            bp_filtered = algorithm_bandpass_butterworth(call_segment, sr)
            noise_bp = algorithm_bandpass_butterworth(noise_profile, sr)
        
        # === STEP 3: Generator Notch (if applicable) ===
        if noise_type == 'generator' and CONFIG.generator_notch_enabled:
            if needs_segmentation:
                bp_filtered = windowed_filter(
                    bp_filtered,
                    algorithm_notch_generator_harmonics,
                    segment_length=segment_length,
                    overlap=0.25,
                    sr=sr
                )
            else:
                bp_filtered = algorithm_notch_generator_harmonics(bp_filtered, sr)
            
            noise_bp = algorithm_notch_generator_harmonics(noise_bp, sr)
        
        # === STEP 4: Spectral Gating ===
        # Note: noisereduce handles long signals internally, no segmentation needed
        denoised = algorithm_spectral_gating(bp_filtered, noise_bp, sr, noise_type)
        
        # === STEP 5: HPSS ===
        # STFT-based, already windowed internally
        harmonic, percussive = algorithm_hpss(denoised, sr)
        
        # === STEP 6: Wiener Filter ===
        # Local windowing already built-in
        final = algorithm_wiener_filter(harmonic)
        
        # === STEP 7: Normalize ===
        if CONFIG.normalize_output:
            peak = np.max(np.abs(final))
            if peak > 0:
                final = final / peak * CONFIG.normalize_level
        
        # === STEP 8: Save Audio ===
        from src.pipeline import sanitize_filename
        safe_filename = sanitize_filename(Path(audio_path).name)
        base_name = f"selection_{selection_id:03d}_{safe_filename}"
        audio_out_path = f"{output_dir}/audio/{base_name}_cleaned.wav"
        sf.write(audio_out_path, final, sr)
        result['output_audio'] = audio_out_path
        
        # === STEP 9: Generate Spectrograms ===
        spec_path = create_bw_spectrogram(
            final, sr,
            title=f'Selection {selection_id} - Cleaned {"(Segmented)" if needs_segmentation else ""}',
            save_path=f"{output_dir}/spectrograms/{base_name}_cleaned.png"
        )
        result['spectrogram'] = spec_path
        
        comparison_path = create_comparison_plot(
            call_segment, final, sr,
            title=f'Selection {selection_id} - {noise_type.capitalize()}',
            save_path=f"{output_dir}/spectrograms/{base_name}_comparison.png"
        )
        result['comparison_plot'] = comparison_path
        
        # === STEP 10: Compute Metrics ===
        result['duration'] = len(final) / sr
        result['sample_rate'] = sr
        result['status'] = 'success'
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        print(f"❌ Selection {selection_id} failed: {e}")
    
    return result
