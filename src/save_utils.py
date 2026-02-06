"""
Save Utilities Module

This module handles saving preprocessed outputs including cropped frames,
landmarks, and metadata in a structured format for downstream training.
"""

import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import cv2

logger = logging.getLogger(__name__)


class OutputSaver:
    """
    Saves preprocessed video outputs in structured format.
    
    Creates directory structure and saves:
    - Cropped mouth frames (as images)
    - Lip landmarks (as numpy arrays)
    - Metadata (as JSON)
    """
    
    def __init__(self, output_root: str, frame_format: str = 'png'):
        """
        Initialize output saver.
        
        Args:
            output_root: Root directory for outputs
            frame_format: Image format for frames ('png', 'jpg')
        """
        self.output_root = Path(output_root)
        self.frame_format = frame_format
        
        logger.info(f"Initialized OutputSaver: root={output_root}, format={frame_format}")
    
    def create_output_structure(self, 
                               word_class: str,
                               split: str,
                               video_id: str) -> Path:
        """
        Create output directory structure for a video.
        
        Args:
            word_class: Word class name
            split: Dataset split ('train', 'val', 'test')
            video_id: Video identifier
        
        Returns:
            Path to video output directory
        """
        video_dir = self.output_root / word_class / split / video_id
        frames_dir = video_dir / 'frames'
        
        # Create directories
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        return video_dir
    
    def save_frames(self,
                   frames: List[np.ndarray],
                   output_dir: Path) -> bool:
        """
        Save cropped mouth frames to disk.
        
        Args:
            frames: List of cropped mouth frames
            output_dir: Directory to save frames
        
        Returns:
            True if successful, False otherwise
        """
        frames_dir = output_dir / 'frames'
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for frame_idx, frame in enumerate(frames):
                if frame is None:
                    logger.warning(f"Skipping None frame at index {frame_idx}")
                    continue
                
                frame_path = frames_dir / f"frame_{frame_idx:02d}.{self.frame_format}"
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                success = cv2.imwrite(str(frame_path), frame_bgr)
                
                if not success:
                    logger.error(f"Failed to save frame {frame_idx} to {frame_path}")
                    return False
            
            logger.debug(f"Saved {len(frames)} frames to {frames_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving frames: {e}")
            return False
    
    def save_landmarks(self,
                      landmarks: np.ndarray,
                      output_dir: Path) -> bool:
        """
        Save lip landmarks to disk as numpy array.
        
        Args:
            landmarks: Landmarks array (num_frames, num_points, 2)
            output_dir: Directory to save landmarks
        
        Returns:
            True if successful, False otherwise
        """
        try:
            landmarks_path = output_dir / 'landmarks.npy'
            np.save(landmarks_path, landmarks)
            
            logger.debug(f"Saved landmarks to {landmarks_path} (shape: {landmarks.shape})")
            return True
            
        except Exception as e:
            logger.error(f"Error saving landmarks: {e}")
            return False
    
    def save_metadata(self,
                     metadata: Dict,
                     output_dir: Path) -> bool:
        """
        Save preprocessing metadata to disk as JSON.
        
        Args:
            metadata: Metadata dictionary
            output_dir: Directory to save metadata
        
        Returns:
            True if successful, False otherwise
        """
        try:
            metadata_path = output_dir / 'metadata.json'
            
            # Convert numpy types to Python types for JSON serialization
            metadata_serializable = self._make_json_serializable(metadata)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata_serializable, f, indent=2)
            
            logger.debug(f"Saved metadata to {metadata_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            return False
    
    def save_video_output(self,
                         video_sample,
                         mouth_crops: List[np.ndarray],
                         lip_landmarks: np.ndarray,
                         processing_info: Dict) -> bool:
        """
        Save all outputs for a processed video.
        
        Args:
            video_sample: VideoSample object
            mouth_crops: List of cropped mouth frames
            lip_landmarks: Lip landmarks array (num_frames, num_points, 2)
            processing_info: Dictionary with processing information
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory structure
            output_dir = self.create_output_structure(
                video_sample.word_class,
                video_sample.split,
                video_sample.video_id
            )
            
            # Save frames
            frames_success = self.save_frames(mouth_crops, output_dir)
            
            if not frames_success:
                logger.error(f"Failed to save frames for {video_sample.video_id}")
                return False
            
            # Save landmarks
            landmarks_success = self.save_landmarks(lip_landmarks, output_dir)
            
            if not landmarks_success:
                logger.error(f"Failed to save landmarks for {video_sample.video_id}")
                return False
            
            # Prepare metadata
            metadata = {
                'video_path': str(video_sample.video_path),
                'word_class': video_sample.word_class,
                'split': video_sample.split,
                'video_id': video_sample.video_id,
                'num_frames': len(mouth_crops),
                'landmarks_shape': list(lip_landmarks.shape),
                'preprocessing_timestamp': datetime.now().isoformat(),
                **processing_info
            }
            
            # Save metadata
            metadata_success = self.save_metadata(metadata, output_dir)
            
            if not metadata_success:
                logger.error(f"Failed to save metadata for {video_sample.video_id}")
                return False
            
            logger.info(f"Successfully saved outputs for {video_sample.video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving video output: {e}")
            return False
    
    def _make_json_serializable(self, obj):
        """
        Convert numpy types to Python types for JSON serialization.
        
        Args:
            obj: Object to convert
        
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def check_output_exists(self,
                           word_class: str,
                           split: str,
                           video_id: str) -> bool:
        """
        Check if output already exists for a video.
        
        Args:
            word_class: Word class name
            split: Dataset split
            video_id: Video identifier
        
        Returns:
            True if output exists, False otherwise
        """
        video_dir = self.output_root / word_class / split / video_id
        
        if not video_dir.exists():
            return False
        
        # Check for required files
        frames_dir = video_dir / 'frames'
        landmarks_file = video_dir / 'landmarks.npy'
        metadata_file = video_dir / 'metadata.json'
        
        exists = (frames_dir.exists() and 
                 landmarks_file.exists() and 
                 metadata_file.exists())
        
        return exists


def save_failed_videos_list(failed_videos: List[Dict],
                            output_path: str) -> bool:
    """
    Save list of failed videos to JSON file.
    
    Args:
        failed_videos: List of dictionaries with failure information
        output_path: Path to save failed videos list
    
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(failed_videos, f, indent=2)
        
        logger.info(f"Saved {len(failed_videos)} failed videos to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving failed videos list: {e}")
        return False


def save_processing_summary(summary: Dict, output_path: str) -> bool:
    """
    Save processing summary statistics to JSON file.
    
    Args:
        summary: Summary statistics dictionary
        output_path: Path to save summary
    
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved processing summary to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving processing summary: {e}")
        return False
