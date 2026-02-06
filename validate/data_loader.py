"""
Data Loader Module

Loads both original LRW videos and preprocessed outputs for validation.
Matches original videos to preprocessed clips and provides unified access.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple

logger = logging.getLogger('validation.data_loader')


class PreprocessedDataLoader:
    """
    Loads original LRW videos and preprocessed outputs for validation.
    
    This loader provides unified access to:
    - Original video frames (from LRW dataset)
    - Preprocessed mouth crops (from preprocessing output)
    - Lip landmarks (from preprocessing output)
    - Metadata (from preprocessing output)
    """
    
    def __init__(self, 
                 lrw_dataset_root: str,
                 preprocessed_root: str):
        """
        Initialize data loader with both original and preprocessed data roots.
        
        Args:
            lrw_dataset_root: Path to original LRW dataset (e.g., '/path/to/lrw_dataset')
            preprocessed_root: Path to preprocessing output (e.g., '../output')
        """
        self.lrw_root = Path(lrw_dataset_root)
        self.preprocessed_root = Path(preprocessed_root)
        
        if not self.lrw_root.exists():
            raise FileNotFoundError(f"LRW dataset not found: {lrw_dataset_root}")
        
        if not self.preprocessed_root.exists():
            raise FileNotFoundError(f"Preprocessed data not found: {preprocessed_root}")
        
        logger.info(f"Initialized data loader:")
        logger.info(f"  LRW dataset: {lrw_dataset_root}")
        logger.info(f"  Preprocessed: {preprocessed_root}")
        
        # Discover all preprocessed clips
        self.clip_paths = self._discover_clips()
        logger.info(f"  Found {len(self.clip_paths)} preprocessed clips")
    
    def _discover_clips(self) -> List[Path]:
        """
        Discover all preprocessed clip directories.
        
        Returns:
            List of paths to preprocessed clip directories
        """
        clip_paths = []
        
        for word_dir in self.preprocessed_root.iterdir():
            if not word_dir.is_dir():
                continue
            
            for split_dir in word_dir.iterdir():
                if not split_dir.is_dir():
                    continue
                
                for clip_dir in split_dir.iterdir():
                    if not clip_dir.is_dir():
                        continue
                    
                    # Check if this is a valid preprocessed clip
                    if (clip_dir / 'landmarks.npy').exists():
                        clip_paths.append(clip_dir)
        
        return sorted(clip_paths)
    
    def load_clip(self, clip_path: Path) -> Dict:
        """
        Load a single clip with both original and preprocessed data.
        
        Args:
            clip_path: Path to preprocessed clip directory
        
        Returns:
            Dictionary with:
                - original_frames: np.ndarray (29, H_orig, W_orig, C) from .mp4
                - original_video_path: str
                - mouth_frames: np.ndarray (29, 96, 96, C) from frames/*.png
                - lip_landmarks: np.ndarray (29, 20, 2) from landmarks.npy
                - metadata: dict from metadata.json
                - clip_id: str
                - word_class: str
                - split: str
                - bboxes: List[Tuple] from metadata
                - detection_flags: List[bool] from metadata
        """
        clip_path = Path(clip_path)
        
        # Parse clip information from path
        # Structure: preprocessed_root/WORD_CLASS/SPLIT/VIDEO_ID/
        video_id = clip_path.name
        split = clip_path.parent.name
        word_class = clip_path.parent.parent.name
        
        # Load preprocessed data
        mouth_frames = self._load_mouth_frames(clip_path)
        lip_landmarks = self._load_landmarks(clip_path)
        metadata = self._load_metadata(clip_path)
        
        # Find and load original video
        original_video_path = self.lrw_root / word_class / split / f"{video_id}.mp4"
        original_frames = self._load_original_video(original_video_path)
        
        # Extract bboxes and detection flags from metadata
        bboxes = metadata.get('roi_boxes', [])
        detection_flags = metadata.get('face_detected', [True] * 29)
        
        return {
            'original_frames': original_frames,
            'original_video_path': str(original_video_path),
            'mouth_frames': mouth_frames,
            'lip_landmarks': lip_landmarks,
            'metadata': metadata,
            'clip_id': video_id,
            'word_class': word_class,
            'split': split,
            'bboxes': bboxes,
            'detection_flags': detection_flags,
        }
    
    def _load_mouth_frames(self, clip_path: Path) -> np.ndarray:
        """
        Load preprocessed mouth frames.
        
        Args:
            clip_path: Path to clip directory
        
        Returns:
            Numpy array of shape (29, H, W, 3)
        """
        frames_dir = clip_path / 'frames'
        
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        
        # Load all frame files
        frame_files = sorted(frames_dir.glob('frame_*.png'))
        
        if len(frame_files) == 0:
            raise ValueError(f"No frames found in {frames_dir}")
        
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is None:
                raise ValueError(f"Failed to load frame: {frame_file}")
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        return np.array(frames, dtype=np.uint8)
    
    def _load_landmarks(self, clip_path: Path) -> np.ndarray:
        """
        Load lip landmarks.
        
        Args:
            clip_path: Path to clip directory
        
        Returns:
            Numpy array of shape (29, 20, 2)
        """
        landmarks_file = clip_path / 'landmarks.npy'
        
        if not landmarks_file.exists():
            raise FileNotFoundError(f"Landmarks file not found: {landmarks_file}")
        
        landmarks = np.load(landmarks_file)
        return landmarks
    
    def _load_metadata(self, clip_path: Path) -> Dict:
        """
        Load preprocessing metadata.
        
        Args:
            clip_path: Path to clip directory
        
        Returns:
            Metadata dictionary
        """
        metadata_file = clip_path / 'metadata.json'
        
        if not metadata_file.exists():
            logger.warning(f"Metadata file not found: {metadata_file}")
            return {}
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def _load_original_video(self, video_path: Path) -> Optional[np.ndarray]:
        """
        Load original video frames.
        
        Args:
            video_path: Path to original .mp4 file
        
        Returns:
            Numpy array of shape (29, H, W, 3) or None if video not found
        """
        if not video_path.exists():
            logger.warning(f"Original video not found: {video_path}")
            return None
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            cap.release()
            
            if len(frames) == 0:
                logger.error(f"No frames extracted from video: {video_path}")
                return None
            
            return np.array(frames, dtype=np.uint8)
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            return None
    
    def iter_clips(self, split: Optional[str] = None) -> Iterator[Dict]:
        """
        Iterate over all clips in dataset.
        
        Args:
            split: Optional filter for 'train', 'val', or 'test'
        
        Yields:
            Clip dictionaries as returned by load_clip()
        """
        for clip_path in self.clip_paths:
            # Filter by split if specified
            if split and clip_path.parent.name != split:
                continue
            
            try:
                clip_data = self.load_clip(clip_path)
                yield clip_data
            except Exception as e:
                logger.error(f"Error loading clip {clip_path}: {e}")
                continue
    
    def get_clip_paths(self, split: Optional[str] = None) -> List[Path]:
        """
        Get list of all preprocessed clip paths.
        
        Args:
            split: Optional filter for 'train', 'val', or 'test'
        
        Returns:
            List of paths to preprocessed clip directories
        """
        if split:
            return [p for p in self.clip_paths if p.parent.name == split]
        return self.clip_paths
    
    def verify_data_availability(self) -> Dict:
        """
        Verify that original videos exist for all preprocessed clips.
        
        Returns:
            Dictionary with:
                - total_preprocessed: int
                - matched_originals: int
                - missing_originals: List[str] (clip IDs without original videos)
        """
        logger.info("Verifying data availability...")
        
        total = len(self.clip_paths)
        matched = 0
        missing = []
        
        for clip_path in self.clip_paths:
            video_id = clip_path.name
            split = clip_path.parent.name
            word_class = clip_path.parent.parent.name
            
            original_video_path = self.lrw_root / word_class / split / f"{video_id}.mp4"
            
            if original_video_path.exists():
                matched += 1
            else:
                missing.append(f"{word_class}/{split}/{video_id}")
        
        results = {
            'total_preprocessed': total,
            'matched_originals': matched,
            'missing_originals': missing
        }
        
        logger.info(f"Data availability:")
        logger.info(f"  Total preprocessed clips: {total}")
        logger.info(f"  Matched original videos: {matched}")
        logger.info(f"  Missing original videos: {len(missing)}")
        
        if missing:
            logger.warning(f"Some original videos are missing. Visualizations will be limited.")
            if len(missing) <= 10:
                for m in missing:
                    logger.warning(f"  Missing: {m}")
        
        return results
    
    def __len__(self) -> int:
        """Return total number of clips."""
        return len(self.clip_paths)
