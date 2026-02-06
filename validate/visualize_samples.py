"""
Visual Validation Module

Generates visual validation outputs including landmark overlays,
side-by-side comparisons, and annotated frames.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional
import random

logger = logging.getLogger('validation.visualize')


class VisualValidator:
    """
    Generates visual validation outputs for quality control.
    
    Creates:
    - Landmark overlays on original frames
    - Bounding box drawings
    - Side-by-side comparison panels
    """
    
    def __init__(self, 
                 output_dir: str,
                 landmark_color: Tuple[int, int, int] = (0, 255, 0),
                 bbox_color: Tuple[int, int, int] = (255, 0, 0),
                 marker_size: int = 3,
                 line_thickness: int = 2):
        """
        Initialize visual validator.
        
        Args:
            output_dir: Directory to save visualizations
            landmark_color: RGB color for landmarks
            bbox_color: RGB color for bounding boxes
            marker_size: Radius of landmark markers
            line_thickness: Thickness of lines
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.landmark_color = landmark_color
        self.bbox_color = bbox_color
        self.marker_size = marker_size
        self.line_thickness = line_thickness
        
        logger.info(f"Initialized VisualValidator: output_dir={output_dir}")
    
    def overlay_landmarks(self,
                         frame: np.ndarray,
                         landmarks: np.ndarray,
                         bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Overlay landmarks and bbox on frame.
        
        Args:
            frame: Input frame (H, W, 3) in RGB
            landmarks: Landmarks array (num_points, 2)
            bbox: Optional bounding box (x, y, w, h)
        
        Returns:
            Annotated frame
        """
        if frame is None:
            return None
        
        annotated = frame.copy()
        
        # Draw bounding box
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), 
                         self.bbox_color, self.line_thickness)
        
        # Draw landmarks
        if landmarks is not None and len(landmarks) > 0:
            for point in landmarks:
                x, y = int(point[0]), int(point[1])
                cv2.circle(annotated, (x, y), self.marker_size, 
                          self.landmark_color, -1)
            
            # Connect landmarks to show lip contours
            if len(landmarks) >= 20:
                # Outer lip contour (first 10 points)
                for i in range(9):
                    pt1 = tuple(landmarks[i].astype(int))
                    pt2 = tuple(landmarks[i + 1].astype(int))
                    cv2.line(annotated, pt1, pt2, self.landmark_color, 1)
                
                # Inner lip contour (last 10 points)
                for i in range(10, 19):
                    pt1 = tuple(landmarks[i].astype(int))
                    pt2 = tuple(landmarks[i + 1].astype(int))
                    cv2.line(annotated, pt1, pt2, self.landmark_color, 1)
        
        return annotated
    
    def create_side_by_side(self,
                           original_frame: np.ndarray,
                           mouth_roi: np.ndarray,
                           landmarks: np.ndarray,
                           bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Create side-by-side comparison panel.
        
        Args:
            original_frame: Original frame with annotations
            mouth_roi: Cropped mouth region
            landmarks: Lip landmarks
            bbox: Bounding box
        
        Returns:
            Side-by-side comparison image
        """
        if original_frame is None or mouth_roi is None:
            return None
        
        # Annotate original frame
        annotated_original = self.overlay_landmarks(original_frame, landmarks, bbox)
        
        # Resize mouth ROI to match original frame height for display
        target_height = original_frame.shape[0]
        aspect_ratio = mouth_roi.shape[1] / mouth_roi.shape[0]
        target_width = int(target_height * aspect_ratio)
        mouth_resized = cv2.resize(mouth_roi, (target_width, target_height))
        
        # Concatenate horizontally
        side_by_side = np.hstack([annotated_original, mouth_resized])
        
        return side_by_side
    
    def visualize_clip(self, 
                      clip_data: dict,
                      frame_indices: Optional[list] = None):
        """
        Generate complete visual validation for a clip.
        
        Args:
            clip_data: Clip dictionary from data loader
            frame_indices: Optional list of frame indices to visualize (default: [0, 14, 28])
        """
        clip_id = clip_data['clip_id']
        
        if frame_indices is None:
            frame_indices = [0, 14, 28]  # Beginning, middle, end
        
        original_frames = clip_data.get('original_frames')
        mouth_frames = clip_data.get('mouth_frames')
        landmarks = clip_data.get('lip_landmarks')
        bboxes = clip_data.get('bboxes', [])
        
        if original_frames is None:
            logger.warning(f"No original frames for {clip_id}, skipping visualization")
            return
        
        for frame_idx in frame_indices:
            if frame_idx >= len(original_frames):
                continue
            
            # Get frame data
            original_frame = original_frames[frame_idx]
            mouth_frame = mouth_frames[frame_idx] if mouth_frames is not None else None
            frame_landmarks = landmarks[frame_idx] if landmarks is not None else None
            bbox = bboxes[frame_idx] if frame_idx < len(bboxes) else None
            
            # Create overlay visualization
            overlay = self.overlay_landmarks(original_frame, frame_landmarks, bbox)
            if overlay is not None:
                overlay_path = self.output_dir / f"{clip_id}_frame{frame_idx:02d}_overlay.png"
                self._save_image(overlay, overlay_path)
            
            # Create side-by-side visualization
            if mouth_frame is not None and bbox is not None:
                sidebyside = self.create_side_by_side(
                    original_frame, mouth_frame, frame_landmarks, bbox
                )
                if sidebyside is not None:
                    sidebyside_path = self.output_dir / f"{clip_id}_frame{frame_idx:02d}_sidebyside.png"
                    self._save_image(sidebyside, sidebyside_path)
        
        logger.info(f"Generated visualizations for {clip_id}")
    
    def _save_image(self, image: np.ndarray, path: Path):
        """
        Save image to disk.
        
        Args:
            image: Image in RGB format
            path: Output path
        """
        try:
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), image_bgr)
        except Exception as e:
            logger.error(f"Error saving image to {path}: {e}")


def sample_clips_for_visualization(clip_paths: list,
                                   num_samples: int = 20,
                                   random_seed: int = 42) -> list:
    """
    Sample clips for visualization using deterministic random sampling.
    
    Args:
        clip_paths: List of clip paths
        num_samples: Number of clips to sample
        random_seed: Random seed for reproducibility
    
    Returns:
        List of sampled clip paths
    """
    random.seed(random_seed)
    num_samples = min(num_samples, len(clip_paths))
    sampled = random.sample(clip_paths, num_samples)
    
    logger.info(f"Sampled {num_samples} clips for visualization (seed={random_seed})")
    
    return sampled
