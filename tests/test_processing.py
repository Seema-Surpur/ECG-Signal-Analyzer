import unittest
import numpy as np
from pathlib import Path
from src.image_processor import ECGImageProcessor
from src.signal_processor import SignalProcessor

class TestECGProcessing(unittest.TestCase):
    def setUp(self):
        self.img_processor = ECGImageProcessor()
        self.sig_processor = SignalProcessor()
        
    def test_signal_normalization(self):
        # Create sample data
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([1, 2, 1, 3, 1])
        
        # Normalize signal
        x_norm, y_norm = self.sig_processor.normalize_signal(x, y)
        
        # Check normalization
        self.assertEqual(x_norm[0], 0)  # x should start at 0
        self.assertTrue(np.all(y_norm >= 0) and np.all(y_norm <= 1))  # y should be normalized to [0,1]
        
    def test_signal_resampling(self):
        # Create sample data
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([1, 2, 1, 3, 1])
        sampling_rate = 2.0  # 2 Hz
        
        # Resample signal
        x_resampled, y_resampled = self.sig_processor.resample_signal(x, y, sampling_rate)
        
        # Check sampling rate
        expected_points = int((x[-1] - x[0]) * sampling_rate) + 1
        self.assertEqual(len(x_resampled), expected_points)

if __name__ == '__main__':
    unittest.main()