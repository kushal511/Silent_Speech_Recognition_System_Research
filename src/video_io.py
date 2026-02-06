"""
Video I/O Module

This module handles video loading and frame extraction from LRW video files.
It provides robust frame extraction with error handling and validation.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class VideoReader:
    """
    Video reader for extracting frames from LRW video files.
    
    Handles video loading, frame extraction, and validation for the expected
    29-frame LRW format.
    """
    
    def __init__(self, expected_frames: int = 29):
        """
        Initialize video reader.
        
        Args:
            expected_frames: Expected number of frames per video (LRW standard is 29)
        """
        self.expected_frames = expected_frames
    
    def read_video(self, video_path: str) -> Tuple[Optional[np.ndarray], dict]:
        """
        Read all frames from a video file.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Tuple of:
                - frames: numpy array of shape (num_frames, height, width, 3) or None if failed
                - metadata: dict with video information and any errors
        """
        video_path = Path(video_path)
        metadata = {
            'video_path': str(video_path),
            'success': False,
            'num_frames': 0,
            'fps': 0,
            'resolution': (0, 0),
            'error': None
        }
        
        if not video_path.exists():
            metadata['error'] = f"Video file not found: {video_path}"
            logger.error(metadata['error'])
            return None, metadata
        
        try:
            # Open video file
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                metadata['error'] = f"Failed to open video: {video_path}"
                logger.error(metadata['error'])
                return None, metadata
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            metadata['fps'] = fps
            metadata['resolution'] = (height, width)
            
            # Extract all frames
            frames = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_idx += 1
            
            cap.release()
            
            # Validate frame count
            metadata['num_frames'] = len(frames)
            
            if len(frames) == 0:
                metadata['error'] = "No frames extracted from video"
                logger.error(f"{metadata['error']}: {video_path}")
                return None, metadata
            
            if len(frames) != self.expected_frames:
                logger.warning(
                    f"Frame count mismatch for {video_path}: "
                    f"expected {self.expected_frames}, got {len(frames)}"
                )
            
            # Convert to numpy array
            frames_array = np.array(frames, dtype=np.uint8)
            
            metadata['success'] = True
            logger.debug(
                f"Successfully read {len(frames)} frames from {video_path.name} "
                f"({height}x{width} @ {fps:.2f} fps)"
            )
            
            return frames_array, metadata
            
        except Exception as e:
            metadata['error'] = f"Error reading video: {str(e)}"
            logger.error(f"{metadata['error']}: {video_path}")
            return None, metadata
    
    def read_frame(self, video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        """
        Read a single frame from a video file.
        
        Args:
            video_path: Path to video file
            frame_idx: Index of frame to read (0-based)
        
        Returns:
            Frame as numpy array (height, width, 3) or None if failed
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error(f"Failed to read frame {frame_idx} from {video_path}")
                return None
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return frame_rgb
            
        except Exception as e:
            logger.error(f"Error reading frame {frame_idx} from {video_path}: {e}")
            return None


def validate_frames(frames: np.ndarray, expected_frames: int = 29) -> Tuple[bool, str]:
    """
    Validate extracted frames array.
    
    Args:
        frames: Numpy array of frames (num_frames, height, width, 3)
        expected_frames: Expected number of frames
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if frames is None:
        return False, "Frames array is None"
    
    if not isinstance(frames, np.ndarray):
        return False, f"Frames must be numpy array, got {type(frames)}"
    
    if frames.ndim != 4:
        return False, f"Frames must be 4D array (N, H, W, C), got shape {frames.shape}"
    
    if frames.shape[3] != 3:
        return False, f"Frames must have 3 channels (RGB), got {frames.shape[3]}"
    
    if frames.shape[0] != expected_frames:
        return False, f"Expected {expected_frames} frames, got {frames.shape[0]}"
    
    if frames.dtype != np.uint8:
        return False, f"Frames must be uint8, got {frames.dtype}"
    
    return True, ""


def save_frame(frame: np.ndarray, output_path: str) -> bool:
    """
    Save a single frame to disk.
    
    Args:
        frame: Frame as numpy array (height, width, 3) in RGB format
        output_path: Path to save frame
    
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        success = cv2.imwrite(str(output_path), frame_bgr)
        
        if not success:
            logger.error(f"Failed to save frame to {output_path}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving frame to {output_path}: {e}")
        return False
