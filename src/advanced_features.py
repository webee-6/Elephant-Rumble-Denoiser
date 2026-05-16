"""
Advanced feature extraction using specialized acoustic libraries.

Supports:
- openSMILE (6,373 features)
- openl3 (512/6144 deep embeddings)
- pyAudioAnalysis (34 features)
- Custom (39 elephant-specific features)
"""

import numpy as np
import librosa
from typing import Dict, Optional, List
import warnings

warnings.filterwarnings('ignore')


class AdvancedFeatureExtractor:
    """
    Feature extraction with multiple backend support.
    """
    
    def __init__(self, backend: str = 'custom'):
        """
        Initialize feature extractor.
        
        Args:
            backend: 'custom', 'opensmile', 'openl3', 'pyaudio', or 'hybrid'
        """
        self.backend = backend
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the selected backend."""
        if self.backend == 'opensmile':
            try:
                import opensmile
                self.smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.LowLevelDescriptors
                )
                print("✅ openSMILE initialized (88 features)")
            except ImportError:
                print("⚠️ openSMILE not installed. Install with: pip install opensmile")
                self.backend = 'custom'
        
        elif self.backend == 'openl3':
            try:
                import openl3
                self.openl3 = openl3
                print("✅ openl3 initialized (512 features)")
            except ImportError:
                print("⚠️ openl3 not installed. Install with: pip install openl3")
                self.backend = 'custom'
        
        elif self.backend == 'pyaudio':
            try:
                from pyAudioAnalysis import ShortTermFeatures
                self.pyaudio = ShortTermFeatures
                print("✅ pyAudioAnalysis initialized (34 features)")
            except ImportError:
                print("⚠️ pyAudioAnalysis not installed. Install with: pip install pyAudioAnalysis")
                self.backend = 'custom'
        
        elif self.backend == 'hybrid':
            print("✅ Hybrid mode: combining multiple backends")
        
        else:  # custom
            print("✅ Using custom elephant-specific features (39 features)")
    
    def extract(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract features from audio signal.
        
        Args:
            signal: Audio signal
            sr: Sample rate
        
        Returns:
            Feature vector
        """
        if self.backend == 'opensmile':
            return self._extract_opensmile(signal, sr)
        elif self.backend == 'openl3':
            return self._extract_openl3(signal, sr)
        elif self.backend == 'pyaudio':
            return self._extract_pyaudio(signal, sr)
        elif self.backend == 'hybrid':
            return self._extract_hybrid(signal, sr)
        else:  # custom
            return self._extract_custom(signal, sr)
    
    def _extract_opensmile(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Extract features using openSMILE eGeMAPSv02."""
        import tempfile
        import soundfile as sf
        import os
        
        # openSMILE requires file input
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save signal
            sf.write(tmp_path, signal, sr)
            
            # Extract features
            features_df = self.smile.process_file(tmp_path)
            features = features_df.values[0]
            
            return features.astype(np.float32)
        
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def _extract_openl3(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Extract deep embeddings using openl3."""
        # Extract embeddings
        emb, ts = self.openl3.get_audio_embedding(
            signal, sr,
            content_type="env",  # Environmental sounds
            embedding_size=512,
            hop_size=0.1  # Dense sampling
        )
        
        # Aggregate across time (mean)
        features = np.mean(emb, axis=0)
        
        return features.astype(np.float32)
    
    def _extract_pyaudio(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Extract features using pyAudioAnalysis."""
        # Ensure signal is in correct format
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        
        # Extract short-term features (34 features)
        features, names = self.pyaudio.feature_extraction(
            signal, sr,
            window=int(0.05 * sr),  # 50ms window
            step=int(0.025 * sr)    # 25ms step
        )
        
        # Aggregate across time (mean and std)
        features_mean = np.mean(features, axis=1)
        features_std = np.std(features, axis=1)
        
        combined = np.concatenate([features_mean, features_std])
        
        return combined.astype(np.float32)
    
    def _extract_custom(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract custom elephant-specific features (39 features).
        
        Same as in neural_classifier.py
        """
        features = []
        
        # === MFCCs (20) ===
        mfcc = librosa.feature.mfcc(
            y=signal, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512,
            fmin=10, fmax=300
        )
        features.extend(np.mean(mfcc, axis=1))
        
        # === Spectral Features (11) ===
        spec_cent = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=2048)[0]
        features.append(np.mean(spec_cent))
        
        spec_roll = librosa.feature.spectral_rolloff(y=signal, sr=sr, n_fft=2048)[0]
        features.append(np.mean(spec_roll))
        
        spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=2048)[0]
        features.append(np.mean(spec_bw))
        
        spec_flat = librosa.feature.spectral_flatness(y=signal, n_fft=2048)[0]
        features.append(np.mean(spec_flat))
        
        spec_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr, n_fft=2048, fmin=10)
        features.extend(np.mean(spec_contrast, axis=1))  # 7 values
        
        # === Temporal Features (4) ===
        zcr = librosa.feature.zero_crossing_rate(signal, frame_length=2048)[0]
        features.append(np.mean(zcr))
        
        rms = librosa.feature.rms(y=signal, frame_length=2048)[0]
        features.append(np.mean(rms))
        
        energy = np.sum(signal ** 2) / len(signal)
        features.append(energy)
        
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        features.append(np.max(autocorr[1:min(len(autocorr), 1000)]) / autocorr[0])
        
        # === Low-Frequency Elephant-Specific (4) ===
        D = np.abs(librosa.stft(signal, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # Energy in elephant range (10-300 Hz)
        elephant_mask = (freqs >= 10) & (freqs <= 300)
        elephant_energy = np.sum(D[elephant_mask, :] ** 2)
        total_energy = np.sum(D ** 2)
        harmonic_ratio = elephant_energy / (total_energy + 1e-10)
        features.append(harmonic_ratio)
        
        # Spectral flux
        spec_flux = np.mean(np.diff(D, axis=1) ** 2)
        features.append(spec_flux)
        
        # Dominant frequency
        low_freq_mask = (freqs >= 10) & (freqs <= 100)
        low_freq_spectrum = np.mean(D[low_freq_mask, :], axis=1)
        if len(low_freq_spectrum) > 0:
            dom_freq_idx = np.argmax(low_freq_spectrum)
            dom_freq = freqs[low_freq_mask][dom_freq_idx]
        else:
            dom_freq = 0
        features.append(dom_freq)
        
        # Energy ratio in fundamental band
        fund_mask = (freqs >= 10) & (freqs <= 25)
        fund_energy = np.sum(D[fund_mask, :] ** 2)
        features.append(fund_energy / (total_energy + 1e-10))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_hybrid(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract hybrid features: custom + openl3 (if available).
        
        Returns 39 + 512 = 551 features
        """
        # Custom features
        custom = self._extract_custom(signal, sr)
        
        # Try to add openl3
        try:
            import openl3
            emb, _ = openl3.get_audio_embedding(
                signal, sr,
                content_type="env",
                embedding_size=512,
                hop_size=0.1
            )
            deep = np.mean(emb, axis=0).astype(np.float32)
            
            # Combine
            return np.concatenate([custom, deep])
        
        except:
            # Fallback to custom only
            print("⚠️ openl3 not available, using custom features only")
            return custom
    
    @property
    def feature_dim(self) -> int:
        """Get feature dimensionality."""
        dims = {
            'custom': 39,
            'opensmile': 88,
            'openl3': 512,
            'pyaudio': 68,  # 34 * 2 (mean + std)
            'hybrid': 551   # 39 + 512
        }
        return dims.get(self.backend, 39)


def compare_extractors(audio_file: str):
    """
    Compare different feature extractors on same audio.
    """
    import soundfile as sf
    import time
    
    # Load audio
    signal, sr = sf.read(audio_file)
    print(f"Audio: {audio_file}")
    print(f"Duration: {len(signal)/sr:.2f}s")
    print("\n" + "="*60)
    
    backends = ['custom', 'opensmile', 'openl3', 'pyaudio', 'hybrid']
    results = {}
    
    for backend in backends:
        try:
            print(f"\n{backend.upper()}:")
            extractor = AdvancedFeatureExtractor(backend=backend)
            
            start = time.time()
            features = extractor.extract(signal, sr)
            elapsed = time.time() - start
            
            print(f"  Features: {len(features)}")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Sample values: {features[:5]}")
            
            results[backend] = {
                'features': features,
                'time': elapsed,
                'dim': len(features)
            }
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Backend':<15} {'Features':<10} {'Time (s)':<10} {'Speed'}")
    print("-"*60)
    for backend, res in results.items():
        features_per_sec = res['dim'] / res['time']
        print(f"{backend:<15} {res['dim']:<10} {res['time']:<10.3f} {features_per_sec:.0f} feat/s")
    
    return results


if __name__ == "__main__":
    # Example comparison
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = 'outputs/audio/selection_001_cleaned.wav'
    
    results = compare_extractors(audio_file)
