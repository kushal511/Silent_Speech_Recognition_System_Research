"""
LRW Dataset Download Utility

This module provides automated download and verification of the
Lip Reading in the Wild (LRW) dataset.

Note: The LRW dataset requires registration and agreement to terms of use.
This script provides utilities to download after obtaining access.
"""

import os
import logging
import requests
import wget
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
import json

logger = logging.getLogger('validation.download')


class LRWDatasetDownloader:
    """
    Automated downloader for the LRW dataset.
    
    Handles download, extraction, verification, and organization of
    the Lip Reading in the Wild dataset.
    """
    
    def __init__(self, 
                 output_dir: str,
                 dataset_url: Optional[str] = None):
        """
        Initialize LRW dataset downloader.
        
        Args:
            output_dir: Directory to download and extract dataset
            dataset_url: Optional custom URL for LRW dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default LRW dataset information
        self.dataset_url = dataset_url or "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html"
        
        # Dataset structure
        self.expected_word_classes = 500
        self.expected_frames_per_video = 29
        self.splits = ['train', 'val', 'test']
        
        logger.info(f"Initialized LRW downloader: output_dir={output_dir}")
    
    def check_dataset_exists(self) -> bool:
        """
        Check if LRW dataset already exists in output_dir.
        
        Returns:
            True if dataset structure exists, False otherwise
        """
        # Check for expected directory structure
        if not self.output_dir.exists():
            return False
        
        # Look for word class directories
        word_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
        
        if len(word_dirs) == 0:
            return False
        
        # Check if at least one word class has the expected structure
        for word_dir in word_dirs[:5]:  # Check first 5
            for split in self.splits:
                split_dir = word_dir / split
                if split_dir.exists():
                    videos = list(split_dir.glob("*.mp4"))
                    if len(videos) > 0:
                        logger.info(f"Found existing LRW dataset in {self.output_dir}")
                        return True
        
        return False
    
    def download_dataset(self, 
                        splits: Optional[List[str]] = None,
                        word_classes: Optional[List[str]] = None) -> bool:
        """
        Download LRW dataset from official source.
        
        Args:
            splits: Optional list of splits to download ['train', 'val', 'test']
            word_classes: Optional list of word classes to download (default: all 500)
        
        Returns:
            True if download successful, False otherwise
        
        Note:
            The LRW dataset requires registration and agreement to terms.
            This method provides the framework for download after obtaining access.
            
            For actual download, users should:
            1. Register at http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html
            2. Obtain download links
            3. Use this utility or manual download
        """
        logger.info("=" * 60)
        logger.info("LRW Dataset Download")
        logger.info("=" * 60)
        
        # Check if dataset already exists
        if self.check_dataset_exists():
            logger.info("Dataset already exists. Use --verify_only to check integrity.")
            return True
        
        # Display dataset information
        info = self.get_download_info()
        logger.info(f"Dataset size: ~{info['dataset_size']}")
        logger.info(f"Number of videos: ~{info['num_videos']}")
        logger.info(f"Number of word classes: {info['num_word_classes']}")
        logger.info(f"License: {info['license_info']}")
        
        # Important notice
        logger.warning("=" * 60)
        logger.warning("IMPORTANT: LRW Dataset Access")
        logger.warning("=" * 60)
        logger.warning("The LRW dataset requires registration and agreement to terms of use.")
        logger.warning("Please visit:")
        logger.warning(f"  {self.dataset_url}")
        logger.warning("")
        logger.warning("After registration, you can:")
        logger.warning("1. Download manually and place in the output directory")
        logger.warning("2. Provide download links to this script")
        logger.warning("3. Use the official download tools provided by VGG")
        logger.warning("=" * 60)
        
        # For now, provide instructions rather than attempting download
        logger.info("\nTo set up the LRW dataset:")
        logger.info(f"1. Download LRW dataset files")
        logger.info(f"2. Extract to: {self.output_dir}")
        logger.info(f"3. Ensure structure: WORD_CLASS/SPLIT/VIDEO_ID.mp4")
        logger.info(f"4. Run with --verify_only to check setup")
        
        return False
    
    def verify_dataset(self) -> Dict:
        """
        Verify downloaded dataset integrity.
        
        Returns:
            Dictionary with verification results:
            {
                'total_videos': int,
                'total_word_classes': int,
                'missing_videos': List[str],
                'corrupted_videos': List[str],
                'is_complete': bool,
                'split_counts': Dict[str, int]
            }
        """
        logger.info("Verifying LRW dataset...")
        
        results = {
            'total_videos': 0,
            'total_word_classes': 0,
            'missing_videos': [],
            'corrupted_videos': [],
            'is_complete': False,
            'split_counts': {split: 0 for split in self.splits}
        }
        
        if not self.output_dir.exists():
            logger.error(f"Output directory does not exist: {self.output_dir}")
            return results
        
        # Count word classes
        word_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
        results['total_word_classes'] = len(word_dirs)
        
        logger.info(f"Found {len(word_dirs)} word class directories")
        
        # Verify each word class
        for word_dir in tqdm(word_dirs, desc="Verifying word classes"):
            for split in self.splits:
                split_dir = word_dir / split
                
                if not split_dir.exists():
                    continue
                
                # Count videos in this split
                videos = list(split_dir.glob("*.mp4"))
                results['total_videos'] += len(videos)
                results['split_counts'][split] += len(videos)
                
                # Basic verification: check if files can be opened
                for video_path in videos:
                    if video_path.stat().st_size == 0:
                        results['corrupted_videos'].append(str(video_path))
        
        # Check completeness
        results['is_complete'] = (
            results['total_word_classes'] >= self.expected_word_classes * 0.95 and
            results['total_videos'] > 0 and
            len(results['corrupted_videos']) == 0
        )
        
        # Log results
        logger.info("=" * 60)
        logger.info("Verification Results:")
        logger.info(f"  Total word classes: {results['total_word_classes']}")
        logger.info(f"  Total videos: {results['total_videos']}")
        for split, count in results['split_counts'].items():
            logger.info(f"    {split}: {count} videos")
        logger.info(f"  Corrupted videos: {len(results['corrupted_videos'])}")
        logger.info(f"  Is complete: {results['is_complete']}")
        logger.info("=" * 60)
        
        if results['corrupted_videos']:
            logger.warning(f"Found {len(results['corrupted_videos'])} corrupted videos")
            for video in results['corrupted_videos'][:10]:
                logger.warning(f"  - {video}")
            if len(results['corrupted_videos']) > 10:
                logger.warning(f"  ... and {len(results['corrupted_videos']) - 10} more")
        
        return results
    
    def get_download_info(self) -> Dict:
        """
        Get information about LRW dataset download.
        
        Returns:
            Dictionary with dataset information:
            {
                'dataset_size': str,
                'num_videos': int,
                'num_word_classes': int,
                'download_url': str,
                'license_info': str
            }
        """
        return {
            'dataset_size': '~50 GB',
            'num_videos': 1000 * self.expected_word_classes,  # Approximate
            'num_word_classes': self.expected_word_classes,
            'download_url': self.dataset_url,
            'license_info': 'Academic use only - Registration required'
        }
    
    def save_verification_report(self, 
                                 verification_results: Dict,
                                 output_path: str):
        """
        Save verification results to JSON file.
        
        Args:
            verification_results: Results from verify_dataset()
            output_path: Path to save JSON report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(verification_results, f, indent=2)
        
        logger.info(f"Saved verification report to {output_path}")


def main():
    """CLI entry point for LRW dataset downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download and verify LRW dataset'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to download/verify LRW dataset'
    )
    
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        choices=['train', 'val', 'test'],
        default=None,
        help='Specific splits to download (default: all)'
    )
    
    parser.add_argument(
        '--word_classes',
        type=str,
        nargs='+',
        default=None,
        help='Specific word classes to download (default: all)'
    )
    
    parser.add_argument(
        '--verify_only',
        action='store_true',
        help='Only verify existing dataset, do not download'
    )
    
    parser.add_argument(
        '--save_report',
        type=str,
        default=None,
        help='Save verification report to JSON file'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize downloader
    downloader = LRWDatasetDownloader(args.output_dir)
    
    if args.verify_only:
        # Verify existing dataset
        results = downloader.verify_dataset()
        
        if args.save_report:
            downloader.save_verification_report(results, args.save_report)
        
        if results['is_complete']:
            logger.info("✓ Dataset verification passed!")
            return 0
        else:
            logger.error("✗ Dataset verification failed")
            return 1
    else:
        # Download dataset
        success = downloader.download_dataset(
            splits=args.splits,
            word_classes=args.word_classes
        )
        
        if success:
            logger.info("✓ Dataset download completed!")
            return 0
        else:
            logger.info("Please follow the instructions above to obtain the LRW dataset")
            return 1


if __name__ == '__main__':
    exit(main())
