"""
Unit tests for LRW dataset downloader.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from validate.download_lrw import LRWDatasetDownloader


class TestLRWDatasetDownloader(unittest.TestCase):
    """Test cases for LRWDatasetDownloader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.downloader = LRWDatasetDownloader(self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test downloader initialization."""
        self.assertTrue(Path(self.test_dir).exists())
        self.assertEqual(self.downloader.expected_word_classes, 500)
        self.assertEqual(self.downloader.expected_frames_per_video, 29)
        self.assertEqual(self.downloader.splits, ['train', 'val', 'test'])
    
    def test_check_dataset_exists_empty_dir(self):
        """Test dataset existence check with empty directory."""
        exists = self.downloader.check_dataset_exists()
        self.assertFalse(exists)
    
    def test_check_dataset_exists_with_structure(self):
        """Test dataset existence check with proper structure."""
        # Create mock dataset structure
        word_dir = Path(self.test_dir) / "ABOUT"
        split_dir = word_dir / "train"
        split_dir.mkdir(parents=True)
        
        # Create a mock video file
        video_file = split_dir / "ABOUT_00001.mp4"
        video_file.touch()
        
        exists = self.downloader.check_dataset_exists()
        self.assertTrue(exists)
    
    def test_get_download_info(self):
        """Test getting download information."""
        info = self.downloader.get_download_info()
        
        self.assertIn('dataset_size', info)
        self.assertIn('num_videos', info)
        self.assertIn('num_word_classes', info)
        self.assertIn('download_url', info)
        self.assertIn('license_info', info)
        
        self.assertEqual(info['num_word_classes'], 500)
        self.assertIn('50 GB', info['dataset_size'])
    
    def test_verify_dataset_empty(self):
        """Test dataset verification with empty directory."""
        results = self.downloader.verify_dataset()
        
        self.assertEqual(results['total_videos'], 0)
        self.assertEqual(results['total_word_classes'], 0)
        self.assertFalse(results['is_complete'])
    
    def test_verify_dataset_with_videos(self):
        """Test dataset verification with mock videos."""
        # Create mock dataset structure
        for word_class in ['ABOUT', 'ABSOLUTELY']:
            for split in ['train', 'val']:
                split_dir = Path(self.test_dir) / word_class / split
                split_dir.mkdir(parents=True)
                
                # Create mock video files
                for i in range(3):
                    video_file = split_dir / f"{word_class}_{i:05d}.mp4"
                    video_file.write_text("mock video content")
        
        results = self.downloader.verify_dataset()
        
        self.assertEqual(results['total_word_classes'], 2)
        self.assertEqual(results['total_videos'], 12)  # 2 words * 2 splits * 3 videos
        self.assertEqual(results['split_counts']['train'], 6)
        self.assertEqual(results['split_counts']['val'], 6)
        self.assertEqual(len(results['corrupted_videos']), 0)
    
    def test_verify_dataset_with_corrupted_video(self):
        """Test dataset verification detects corrupted videos."""
        # Create mock dataset with one corrupted video
        word_dir = Path(self.test_dir) / "ABOUT"
        split_dir = word_dir / "train"
        split_dir.mkdir(parents=True)
        
        # Create normal video
        normal_video = split_dir / "ABOUT_00001.mp4"
        normal_video.write_text("mock video content")
        
        # Create corrupted (empty) video
        corrupted_video = split_dir / "ABOUT_00002.mp4"
        corrupted_video.touch()  # Empty file
        
        results = self.downloader.verify_dataset()
        
        self.assertEqual(results['total_videos'], 2)
        self.assertEqual(len(results['corrupted_videos']), 1)
        self.assertIn(str(corrupted_video), results['corrupted_videos'])
    
    def test_save_verification_report(self):
        """Test saving verification report to JSON."""
        results = {
            'total_videos': 100,
            'total_word_classes': 10,
            'is_complete': True
        }
        
        report_path = Path(self.test_dir) / "verification_report.json"
        self.downloader.save_verification_report(results, str(report_path))
        
        self.assertTrue(report_path.exists())
        
        # Verify JSON content
        import json
        with open(report_path) as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded['total_videos'], 100)
        self.assertEqual(loaded['total_word_classes'], 10)
        self.assertTrue(loaded['is_complete'])


if __name__ == '__main__':
    unittest.main()
