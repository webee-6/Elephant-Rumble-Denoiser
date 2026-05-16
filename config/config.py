"""
Configuration module for elephant rumble denoising pipeline.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Tuple


@dataclass
class PipelineConfig:
    """Central configuration for entire pipeline"""
    
    # ==== STFT Parameters ====
    n_fft: int = 4096          # FFT window size (higher = better frequency resolution)
    hop_length: int = 512      # Hop between frames (lower = better time resolution)
    fmax_display: int = 1000   # Max frequency for display (Hz)
    
    # ==== Butterworth Band-Pass Filter ====
    bp_lowcut: float = 20.0    # Lower cutoff (Hz) - below elephant range
    bp_highcut: float = 1000.0 # Upper cutoff (Hz) - above elephant harmonics
    bp_order: int = 2          # Filter order (2 = stable at low freq)
    
    # ==== Noise-Type Specific Parameters ====
    noise_params: Dict = None
    
    # ==== HPSS Parameters ====
    hpss_kernel_size: int = 31   # Median filter size (odd number, 11-51 typical)
    hpss_margin: float = 3.0     # Separation margin (1.0-5.0, higher = harder separation)
    
    # ==== Wiener Filter ====
    wiener_window: int = 29      # Local estimation window (odd number)
    
    # ==== Noise Profile Extraction ====
    noise_duration_sec: float = 2.0    # Duration of noise profile to extract
    noise_profile_mode: str = 'adaptive' # 'before', 'after', 'adaptive'
    
    # ==== Generator-Specific ====
    generator_notch_enabled: bool = True
    generator_fundamental: float = 60.0  # Hz (50 for Europe, 60 for US)
    generator_harmonics: int = 10        # Number of harmonics to notch
    generator_notch_q: float = 30.0      # Quality factor (higher = narrower)
    
    # ==== Output Settings ====
    output_sample_rate: int = 44100      # Keep original SR
    normalize_output: bool = True
    normalize_level: float = 0.7         # Peak normalization (0.0-1.0)
    
    # ==== Visualization ====
    spec_colormap: str = 'binary'        # 'binary' for B/W, 'gray' for grayscale
    spec_dpi: int = 150
    spec_figsize: Tuple[int, int] = (12, 4)
    
    def __post_init__(self):
        if self.noise_params is None:
            self.noise_params = {
                'airplane': {
                    'prop_decrease': 0.85,
                    'stationary': False,
                    'freq_mask_smooth_hz': 100,
                    'time_mask_smooth_ms': 50
                },
                'vehicle': {
                    'prop_decrease': 0.80,
                    'stationary': False,
                    'freq_mask_smooth_hz': 100,
                    'time_mask_smooth_ms': 50
                },
                'generator': {
                    'prop_decrease': 0.90,
                    'stationary': True,
                    'freq_mask_smooth_hz': 120,
                    'time_mask_smooth_ms': 40
                },
                'background': {
                    'prop_decrease': 0.75,
                    'stationary': False,
                    'freq_mask_smooth_hz': 100,
                    'time_mask_smooth_ms': 60
                }
            }
    
    def print_summary(self):
        """Print configuration summary"""
        print('Pipeline Configuration')
        print('='*60)
        print(f'STFT: n_fft={self.n_fft}, hop={self.hop_length}')
        print(f'Band-Pass: {self.bp_lowcut}-{self.bp_highcut} Hz (order={self.bp_order})')
        print(f'HPSS: kernel={self.hpss_kernel_size}, margin={self.hpss_margin}')
        print(f'Wiener: window={self.wiener_window}')
        print(f'Generator Notch: {"Enabled" if self.generator_notch_enabled else "Disabled"}')
        print('='*60)
        print('\nNoise-Type Parameters:')
        for noise_type, params in self.noise_params.items():
            print(f'  {noise_type:12s}: α={params["prop_decrease"]:.2f}, '
                  f'stationary={params["stationary"]}')


# Global config instance
CONFIG = PipelineConfig()
