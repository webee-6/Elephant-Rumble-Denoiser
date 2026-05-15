"""
Unit tests for DSP algorithms.
"""

import unittest
import numpy as np
from src.algorithms import (
    algorithm_bandpass_butterworth,
    algorithm_notch_generator_harmonics,
    algorithm_hpss
)


class TestAlgorithms(unittest.TestCase):
    
    def setUp(self):
        """Create test signals"""
        self.sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(self.sr * duration))
        
        # Test signal: 50Hz + 200Hz + 500Hz
        self.signal = (np.sin(2 * np.pi * 50 * t) +
                      np.sin(2 * np.pi * 200 * t) +
                      np.sin(2 * np.pi * 500 * t))
    
    def test_bandpass_filter_shape(self):
        """Test that bandpass filter preserves signal shape"""
        filtered = algorithm_bandpass_butterworth(self.signal, self.sr)
        self.assertEqual(len(filtered), len(self.signal))
        self.assertTrue(np.isfinite(filtered).all())
    
    def test_bandpass_filter_removes_low_freq(self):
        """Test that bandpass removes frequencies below cutoff"""
        # Signal with 5Hz component (below 20Hz cutoff)
        t = np.linspace(0, 1.0, self.sr)
        low_freq_signal = np.sin(2 * np.pi * 5 * t)
        
        filtered = algorithm_bandpass_butterworth(low_freq_signal, self.sr)
        
        # Energy should be significantly reduced
        original_energy = np.sum(low_freq_signal ** 2)
        filtered_energy = np.sum(filtered ** 2)
        
        self.assertLess(filtered_energy, 0.1 * original_energy)
    
    def test_notch_filter_shape(self):
        """Test that notch filter preserves signal shape"""
        filtered = algorithm_notch_generator_harmonics(self.signal, self.sr)
        self.assertEqual(len(filtered), len(self.signal))
        self.assertTrue(np.isfinite(filtered).all())
    
    def test_hpss_shape(self):
        """Test that HPSS returns correct shapes"""
        harmonic, percussive = algorithm_hpss(self.signal, self.sr)
        
        self.assertGreater(len(harmonic), 0)
        self.assertGreater(len(percussive), 0)
        self.assertTrue(np.isfinite(harmonic).all())
        self.assertTrue(np.isfinite(percussive).all())


if __name__ == '__main__':
    unittest.main()
