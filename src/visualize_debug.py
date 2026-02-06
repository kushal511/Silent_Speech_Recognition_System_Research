"""
Visualization and Debugging Module

This module provides visualization utilities for debugging the preprocessing pipeline,
including landmark overlay, ROI boxes, and frame-by-frame inspection.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import random

logger = logging.getLogger(__name__)


class PreprocessingVisualizer:
    """
    Visualizes preprocessing results for debugging and quality control.
    
    Provides utilities to draw landmarks, ROI boxes, and create
    annotated frames for visual inspection.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save debug visualizations (optional)
        """
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Debug visualizations will be saved to {self.output_dir}")
    
    def draw_landmarks(self,
                      frame: np.ndarray,
                      landmarks: np.ndarray,
                      color: Tuple[int, int, int] = (0, 255, 0),
                      radius: int = 2) -> np.ndarray:
        """
        Draw landmarks on frame.
        
        Args:
            frame: Input frame (height, width, 3) in RGB
            landmarks: Landmarks array (num_points, 2)
            color: RGB color for landmarks
            radius: Radius of landmark points
        
        Returns:
            Annotated frame
        """
        if frame is None or landmarks is None or len(landmarks) == 0:
            return frame
        
        annotated = frame.copy()
        
        for point in landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(annotated, (x, y), radius, color, -1)
        
        return annotated
    
    def draw_roi_box(self,
                    frame: np.ndarray,
                    roi_box: Tuple[int, int, int, int],
                    color: Tuple[int, int, int] = (255, 0, 0),
                    thickness: int = 2) -> np.ndarray:
        """
        Draw ROI bounding box on frame.
        
        Args:
            frame: Input frame (height, width, 3) in RGB
            roi_box: Bounding box (x, y, w, h)
            color: RGB color for box
            thickness: Line thickness
        
        Returns:
            Annotated frame
        """
        if frame is None or roi_box is None:
            return frame
        
        annotated = frame.copy()
        x, y, w, h = roi_box
        
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        
        return annotated
    
    def visualize_frame_processing(self,
                                  frame: np.ndarray,
                                  lip_landmarks: Optional[np.ndarray],
                                  roi_box: Optional[Tuple[int, int, int, int]],
                                  frame_idx: int) -> np.ndarray:
        """
        Create comprehensive visualization of frame processing.
        
        Args:
            frame: Input frame
            lip_landmarks: Lip landmarks
            roi_box: ROI bounding box
            frame_idx: Frame index
        
        Returns:
            Annotated frame with landmarks and ROI box
        """
        annotated = frame.copy()
        
        # Draw lip landmarks
        if lip_landmarks is not None and len(lip_landmarks) > 0:
            annotated = self.draw_landmarks(annotated, lip_landmarks, 
                                          color=(0, 255, 0), radius=3)
        
        # Draw ROI box
        if roi_box is not None:
            annotated = self.draw_roi_box(annotated, roi_box, 
                                        color=(255, 0, 0), thickness=2)
        
        # Add frame index text
        cv2.putText(annotated, f"Frame {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return annotated
    
    def visualize_video_processing(self,
                                  frames: np.ndarray,
                                  landmark_results: List[dict],
                                  roi_results: List[dict],
                                  video_id: str,
                                  save: bool = True) -> List[np.ndarray]:
        """
        Visualize processing for all frames in a video.
        
        Args:
            frames: Video frames array
            landmark_results: Landmark detection results
            roi_results: ROI extraction results
            video_id: Video identifier
            save: Whether to save annotated frames
        
        Returns:
            List of annotated frames
        """
        annotated_frames = []
        
        for frame_idx, frame in enumerate(frames):
            lip_landmarks = landmark_results[frame_idx].get('lip_landmarks')
            roi_box = roi_results[frame_idx].get('roi_box')
            
            annotated = self.visualize_frame_processing(
                frame, lip_landmarks, roi_box, frame_idx
            )
            
            annotated_frames.append(annotated)
            
            # Save individual frame if requested
            if save and self.output_dir:
                frame_path = self.output_dir / f"{video_id}_frame_{frame_idx:02d}.png"
                self._save_frame(annotated, frame_path)
        
        logger.info(f"Created visualizations for {len(annotated_frames)} frames")
        
        return annotated_frames
    
    def create_comparison_grid(self,
                             original_frames: List[np.ndarray],
                             cropped_frames: List[np.ndarray],
                             num_samples: int = 5) -> np.ndarray:
        """
        Create side-by-side comparison grid of original and cropped frames.
        
        Args:
            original_frames: List of original frames
            cropped_frames: List of cropped mouth regions
            num_samples: Number of frames to include in grid
        
        Returns:
            Comparison grid image
        """
        if len(original_frames) == 0 or len(cropped_frames) == 0:
            logger.warning("Empty frame lists provided")
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Sample frames evenly
        indices = np.linspace(0, len(original_frames) - 1, num_samples, dtype=int)
        
        grid_rows = []
        
        for idx in indices:
            orig = original_frames[idx]
            crop = cropped_frames[idx]
            
            if crop is None:
                continue
            
            # Resize cropped frame to match original height for side-by-side display
            crop_resized = cv2.resize(crop, (orig.shape[1] // 2, orig.shape[0]))
            
            # Concatenate horizontally
            row = np.hstack([orig, crop_resized])
            grid_rows.append(row)
        
        if len(grid_rows) == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Stack rows vertically
        grid = np.vstack(grid_rows)
        
        return grid
    
    def _save_frame(self, frame: np.ndarray, path: Path) -> bool:
        """
        Save frame to disk.
        
        Args:
            frame: Frame in RGB format
            path: Output path
        
        Returns:
            True if successful
        """
        try:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(str(path), frame_bgr)
            return success
        except Exception as e:
            logger.error(f"Error saving frame to {path}: {e}")
            return False


def sample_and_visualize(output_root: str,
                        num_samples: int = 10,
                        save_dir: Optional[str] = None) -> None:
    """
    Sample random preprocessed videos and create visualizations.
    
    Useful for quality control and spot-checking preprocessing results.
    
    Args:
        output_root: Root directory of preprocessed outputs
        num_samples: Number of videos to sample
        save_dir: Directory to save visualizations
    """
    output_root = Path(output_root)
    
    if not output_root.exists():
        logger.error(f"Output root does not exist: {output_root}")
        return
    
    # Find all processed videos
    video_dirs = []
    for word_dir in output_root.iterdir():
        if not word_dir.is_dir():
            continue
        for split_dir in word_dir.iterdir():
            if not split_dir.is_dir():
                continue
            for video_dir in split_dir.iterdir():
                if video_dir.is_dir() and (video_dir / 'metadata.json').exists():
                    video_dirs.append(video_dir)
    
    if len(video_dirs) == 0:
        logger.warning("No processed videos found")
        return
    
    # Sample random videos
    sampled = random.sample(video_dirs, min(num_samples, len(video_dirs)))
    
    logger.info(f"Sampling {len(sampled)} videos for visualization")
    
    visualizer = PreprocessingVisualizer(save_dir)
    
    for video_dir in sampled:
        try:
            # Load frames
            frames_dir = video_dir / 'frames'
            frame_files = sorted(frames_dir.glob('*.png'))
            
            frames = []
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            # Load landmarks
            landmarks = np.load(video_dir / 'landmarks.npy')
            
            logger.info(f"Loaded {len(frames)} frames from {video_dir.name}")
            
            # Create visualization (simple grid of frames)
            if len(frames) > 0:
                grid = np.hstack(frames[:5])  # Show first 5 frames
                
                if save_dir:
                    save_path = Path(save_dir) / f"{video_dir.name}_grid.png"
                    frame_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(save_path), frame_bgr)
                    logger.info(f"Saved visualization to {save_path}")
        
        except Exception as e:
            logger.error(f"Error visualizing {video_dir}: {e}")


if __name__ == "__main__":
    # Example usage for debugging
    import sys
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        sample_and_visualize(output_dir, num_samples=20, save_dir="debug_output")
    else:
        print("Usage: python visualize_debug.py <output_dir>")
