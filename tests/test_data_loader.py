"""
Test cases for data loading functionality
"""
import unittest
import torch
from data_provider.data_loader import UEAloader


class TestDataLoader(unittest.TestCase):
    """Test cases for UEAloader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.root_path = "./dataset/UWaveGestureLibrary"
    
    def test_dataset_loading(self):
        """Test that dataset can be loaded"""
        # Add your test cases here
        pass
    
    def test_data_normalization(self):
        """Test data normalization"""
        # Add your test cases here
        pass


if __name__ == "__main__":
    unittest.main()

