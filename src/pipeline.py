"""
Main processing pipeline for elephant rumble denoising.
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


def process_single_call(audio_path: str,
                       start_time: float,
                       end_time: float,
                       selection_id: int,
                       output_dir: str = 'outputs') -> Dict:
    """
    Process a single elephant call through the complete pipeline.
    
    Pipeline Stages:
        0. Load Audio & Extract Call Segment
        1. Extract & Validate Noise Profile
        2. Butterworth Band-Pass (20-1000 Hz)
        3. Generator Notch Filter (if applicable)
        4. Spectral Gating (Noise Subtraction)
        5. HPSS (Harmonic-Percussive Separation)
        6. Wiener Filter (MSE-Optimal)
        7. Normalize Output
        8. Save Audio
        9. Generate B/W Spectrograms
        10. Compute Metrics
    
    Args:
        audio_path: Path to audio file
        start_time: Start time in seconds
        end_time: End time in seconds
        selection_id: Unique identifier
        output_dir: Output directory
    
    Returns:
        Dictionary with processing results and metadata
    """
    result = {
        'selection_id': selection_id,
        'filename': Path(audio_path).name,
        'start_time': start_time,
        'end_time': end_time,
        'status': 'pending',
        'error': None
    }
    
    try:
        # === STEP 0: Load Audio ===
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Convert time to samples
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        start_sample = min(start_sample, len(y) - 1)
        end_sample = min(end_sample, len(y))
        
        call_segment = y[start_sample:end_sample]
        
        # Classify noise type
        noise_type = classify_noise_type(Path(audio_path).name)
        result['noise_type'] = noise_type
        
        # === STEP 1: Extract Noise Profile ===
        noise_profile, noise_source = extract_noise_profile(
            y, sr, start_sample, end_sample, mode=CONFIG.noise_profile_mode
        )
        result['noise_source'] = noise_source
        
        # Validate noise profile
        noise_metrics = validate_noise_profile(noise_profile, sr)
        result['noise_validation'] = noise_metrics
        
        # === STEP 2: Butterworth Band-Pass ===
        bp_filtered = algorithm_bandpass_butterworth(call_segment, sr)
        noise_bp = algorithm_bandpass_butterworth(noise_profile, sr)
        
        # === STEP 3: Generator Notch (if applicable) ===
        if noise_type == 'generator' and CONFIG.generator_notch_enabled:
            bp_filtered = algorithm_notch_generator_harmonics(bp_filtered, sr)
            noise_bp = algorithm_notch_generator_harmonics(noise_bp, sr)
        
        # === STEP 4: Spectral Gating ===
        denoised = algorithm_spectral_gating(bp_filtered, noise_bp, sr, noise_type)
        
        # === STEP 5: HPSS ===
        harmonic, percussive = algorithm_hpss(denoised, sr)
        
        # === STEP 6: Wiener Filter ===
        final = algorithm_wiener_filter(harmonic)
        
        # === STEP 7: Normalize ===
        if CONFIG.normalize_output:
            peak = np.max(np.abs(final))
            if peak > 0:
                final = final / peak * CONFIG.normalize_level
        
        # === STEP 8: Save Audio ===
        # Sanitize stem: replace spaces with underscores to avoid soundfile System errors
        safe_stem = Path(audio_path).stem.replace(' ', '_')
        base_name = f"selection_{selection_id:03d}_{safe_stem}"
        audio_dir = Path(output_dir) / 'audio'
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_out_path = str(audio_dir / f"{base_name}_cleaned.wav")
        sf.write(audio_out_path, final, sr)
        result['output_audio'] = audio_out_path
        
        # === STEP 9: Generate Spectrograms ===
        spec_dir = Path(output_dir) / 'spectrograms'
        spec_dir.mkdir(parents=True, exist_ok=True)
        # B/W spectrogram of cleaned signal
        spec_path = create_bw_spectrogram(
            final, sr,
            title=f'Selection {selection_id} - Cleaned',
            save_path=str(spec_dir / f"{base_name}_cleaned.png")
        )
        result['spectrogram'] = spec_path
        
        # Comparison plot
        comparison_path = create_comparison_plot(
            call_segment, final, sr,
            title=f'Selection {selection_id} - {noise_type.capitalize()}',
            save_path=str(spec_dir / f"{base_name}_comparison.png")
        )
        result['comparison_plot'] = comparison_path
        
        # === STEP 10: Compute Metrics ===
        result['duration'] = len(final) / sr
        result['sample_rate'] = sr
        result['status'] = 'success'
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        print(f" Selection {selection_id} failed: {e}")
    
    return result
