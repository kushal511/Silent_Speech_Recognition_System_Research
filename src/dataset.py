"""
Dataset Discovery and Iteration Module

This module handles LRW dataset structure discovery, validation, and iteration.
It provides utilities to find all video files organized by word class and split.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VideoSample:
    """Represents a single video sample from the dataset."""
    video_path: Path
    word_class: str
    split: str  # 'train', 'val', 'test', or 'all' for flat structures
    video_id: str
    
    def __post_init__(self):
        """Validate the video sample."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if self.split not in ['train', 'val', 'test', 'all']:
            raise ValueError(f"Invalid split: {self.split}")


class LRWDataset:
    """
    LRW Dataset handler for discovering and iterating over video samples.
    
    Supports two directory structures:
    
    1. LRW structure (hierarchical):
        lrw_dataset/
        ├── WORD_CLASS_1/
        │   ├── train/
        │   │   ├── WORD_CLASS_1_00001.mp4
        │   │   └── ...
        │   ├── val/
        │   └── test/
        └── WORD_CLASS_2/
            └── ...
    
    2. GRID structure (flat):
        dataset/
        ├── s1/
        │   ├── video1.mpg
        │   ├── video2.mpg
        │   └── ...
        └── alignments/
            └── s1/
                ├── video1.align
                └── ...
    """
    
    def __init__(self, root_dir: str, video_dir: Optional[str] = None, 
                 video_extension: str = ".mp4", splits: Optional[List[str]] = None):
        """
        Initialize dataset handler.
        
        Args:
            root_dir: Root directory of dataset
            video_dir: Subdirectory containing videos (for flat structure)
            video_extension: Video file extension (.mp4, .mpg, etc.)
            splits: List of splits to include ['train', 'val', 'test'].
                   If None, includes all splits (or all videos for flat structure).
        """
        self.root_dir = Path(root_dir)
        self.video_dir = video_dir
        self.video_extension = video_extension
        self.splits = splits or ['train', 'val', 'test']
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")
        
        logger.info(f"Initializing dataset from: {self.root_dir}")
        if video_dir:
            logger.info(f"Video subdirectory: {video_dir}")
        logger.info(f"Video extension: {video_extension}")
        
        # Discover all video samples
        self.samples = self._discover_samples()
        logger.info(f"Discovered {len(self.samples)} video samples")
        
        # Build statistics
        self.stats = self._compute_statistics()
        self._log_statistics()
    
    def _discover_samples(self) -> List[VideoSample]:
        """
        Discover all video samples in the dataset.
        Supports both hierarchical (LRW) and flat (GRID) structures.
        
        Returns:
            List of VideoSample objects
        """
        # Check if using flat structure (video_dir specified)
        if self.video_dir:
            return self._discover_flat_structure()
        else:
            return self._discover_hierarchical_structure()
    
    def _discover_flat_structure(self) -> List[VideoSample]:
        """
        Discover samples in flat directory structure (e.g., GRID dataset).
        
        Returns:
            List of VideoSample objects
        """
        samples = []
        video_path = self.root_dir / self.video_dir
        
        if not video_path.exists():
            logger.error(f"Video directory not found: {video_path}")
            return samples
        
        # Find all video files with specified extension
        video_files = list(video_path.glob(f"*{self.video_extension}"))
        logger.info(f"Found {len(video_files)} video files in {video_path}")
        
        for video_file in video_files:
            video_id = video_file.stem
            
            try:
                sample = VideoSample(
                    video_path=video_file,
                    word_class="unknown",  # Flat structure doesn't have word classes
                    split="all",  # Flat structure doesn't have splits
                    video_id=video_id
                )
                samples.append(sample)
            except Exception as e:
                logger.error(f"Error creating sample for {video_file}: {e}")
        
        return samples
    
    def _discover_hierarchical_structure(self) -> List[VideoSample]:
        """
        Discover samples in hierarchical directory structure (LRW dataset).
        
        Returns:
            List of VideoSample objects
        """
        samples = []
        
        # Iterate through word class directories
        for word_dir in sorted(self.root_dir.iterdir()):
            if not word_dir.is_dir():
                continue
            
            word_class = word_dir.name
            
            # Iterate through splits
            for split in self.splits:
                split_dir = word_dir / split
                
                if not split_dir.exists():
                    logger.warning(f"Split directory not found: {split_dir}")
                    continue
                
                # Find all video files
                video_files = list(split_dir.glob(f"*{self.video_extension}"))
                
                for video_path in video_files:
                    video_id = video_path.stem
                    
                    try:
                        sample = VideoSample(
                            video_path=video_path,
                            word_class=word_class,
                            split=split,
                            video_id=video_id
                        )
                        samples.append(sample)
                    except Exception as e:
                        logger.error(f"Error creating sample for {video_path}: {e}")
        
        return samples
    
    def _compute_statistics(self) -> Dict:
        """
        Compute dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_samples': len(self.samples),
            'word_classes': set(),
            'splits': {}
        }
        
        for sample in self.samples:
            stats['word_classes'].add(sample.word_class)
            
            if sample.split not in stats['splits']:
                stats['splits'][sample.split] = 0
            stats['splits'][sample.split] += 1
        
        stats['num_word_classes'] = len(stats['word_classes'])
        stats['word_classes'] = sorted(stats['word_classes'])
        
        return stats
    
    def _log_statistics(self):
        """Log dataset statistics."""
        logger.info("=" * 60)
        logger.info("Dataset Statistics:")
        logger.info(f"  Total samples: {self.stats['total_samples']}")
        logger.info(f"  Word classes: {self.stats['num_word_classes']}")
        
        for split, count in self.stats['splits'].items():
            logger.info(f"  {split}: {count} samples")
        
        logger.info("=" * 60)
    
    def get_samples(self, split: Optional[str] = None, 
                   word_class: Optional[str] = None,
                   max_samples: Optional[int] = None) -> List[VideoSample]:
        """
        Get filtered list of samples.
        
        Args:
            split: Filter by split ('train', 'val', 'test')
            word_class: Filter by word class
            max_samples: Maximum number of samples to return
        
        Returns:
            Filtered list of VideoSample objects
        """
        samples = self.samples
        
        if split:
            samples = [s for s in samples if s.split == split]
        
        if word_class:
            samples = [s for s in samples if s.word_class == word_class]
        
        if max_samples:
            samples = samples[:max_samples]
        
        return samples
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> VideoSample:
        """Get sample by index."""
        return self.samples[idx]
    
    def __iter__(self):
        """Iterate over samples."""
        return iter(self.samples)


def validate_dataset_structure(root_dir: str) -> Tuple[bool, List[str]]:
    """
    Validate that the dataset follows expected LRW structure.
    
    Args:
        root_dir: Root directory of dataset
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    root_path = Path(root_dir)
    
    if not root_path.exists():
        issues.append(f"Root directory does not exist: {root_dir}")
        return False, issues
    
    if not root_path.is_dir():
        issues.append(f"Root path is not a directory: {root_dir}")
        return False, issues
    
    # Check for word class directories
    word_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    
    if len(word_dirs) == 0:
        issues.append("No word class directories found")
        return False, issues
    
    # Check structure of first few word directories
    for word_dir in word_dirs[:5]:
        for split in ['train', 'val', 'test']:
            split_dir = word_dir / split
            if split_dir.exists():
                videos = list(split_dir.glob("*.mp4"))
                if len(videos) == 0:
                    issues.append(f"No videos found in {split_dir}")
    
    is_valid = len(issues) == 0
    return is_valid, issues
