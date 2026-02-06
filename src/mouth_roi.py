"""
Mouth Region of Interest (ROI) Computation Module

This module computes robust mouth ROI bounding boxes from lip landmarks,
with padding, size constraints, and fallback strategies for edge cases.
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)


class MouthROIExtractor:
    """
    Extracts mouth region of interest from frames using lip landmarks.
    
    Computes bounding boxes around lip landmarks with configurable padding,
    size constraints, and aspect ratio enforcement.
    """
    
    def __init__(self,
                 padding_factor: float = 0.3,
                 min_size: int = 64,
                 max_size: int = 128,
                 target_size: Tuple[int, int] = (96, 96),
                 aspect_ratio: float = 1.0):
        """
        Initialize mouth ROI extractor.
        
        Args:
            padding_factor: Padding to add around lip landmarks (0.3 = 30%)
            min_size: Minimum ROI dimension in pixels
            max_size: Maximum ROI dimension in pixels
            target_size: Target size for cropped mouth region (height, width)
            aspect_ratio: Desired aspect ratio (width/height), 1.0 for square
        """
        self.padding_factor = padding_factor
        self.min_size = min_size
        self.max_size = max_size
        self.target_size = target_size
        self.aspect_ratio = aspect_ratio
        
        logger.info(
            f"Initialized MouthROIExtractor: "
            f"padding={padding_factor}, size=[{min_size}, {max_size}], "
            f"target={target_size}"
        )
    
    def compute_roi_box(self, 
                       lip_landmarks: np.ndarray,
                       frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        Compute bounding box for mouth ROI from lip landmarks.
        
        Args:
            lip_landmarks: Lip landmarks array (num_points, 2) with (x, y) coordinates
            frame_shape: Frame dimensions (height, width)
        
        Returns:
            Bounding box as (x, y, width, height) or None if computation fails
        """
        if lip_landmarks is None or len(lip_landmarks) == 0:
            logger.warning("No lip landmarks provided for ROI computation")
            return None
        
        try:
            # Get bounding box of lip landmarks
            x_coords = lip_landmarks[:, 0]
            y_coords = lip_landmarks[:, 1]
            
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            # Compute center and size
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            
            # Add padding
            width_padded = width * (1 + self.padding_factor)
            height_padded = height * (1 + self.padding_factor)
            
            # Enforce aspect ratio (make square if aspect_ratio=1.0)
            if self.aspect_ratio > 0:
                current_ratio = width_padded / height_padded
                
                if current_ratio > self.aspect_ratio:
                    # Width is larger, increase height
                    height_padded = width_padded / self.aspect_ratio
                else:
                    # Height is larger, increase width
                    width_padded = height_padded * self.aspect_ratio
            
            # Enforce size constraints
            size = max(width_padded, height_padded)
            size = max(size, self.min_size)
            size = min(size, self.max_size)
            
            # Make square
            width_final = size
            height_final = size
            
            # Compute top-left corner
            x = int(center_x - width_final / 2)
            y = int(center_y - height_final / 2)
            w = int(width_final)
            h = int(height_final)
            
            # Ensure box is within frame bounds
            frame_height, frame_width = frame_shape
            x = max(0, min(x, frame_width - w))
            y = max(0, min(y, frame_height - h))
            
            # Adjust size if box extends beyond frame
            if x + w > frame_width:
                w = frame_width - x
            if y + h > frame_height:
                h = frame_height - y
            
            logger.debug(f"Computed ROI box: x={x}, y={y}, w={w}, h={h}")
            
            return (x, y, w, h)
            
        except Exception as e:
            logger.error(f"Error computing ROI box: {e}")
            return None
    
    def crop_mouth_region(self,
                         frame: np.ndarray,
                         roi_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Crop mouth region from frame using ROI box.
        
        Args:
            frame: Input frame (height, width, 3)
            roi_box: Bounding box (x, y, width, height)
        
        Returns:
            Cropped mouth region or None if crop fails
        """
        if frame is None or roi_box is None:
            return None
        
        try:
            x, y, w, h = roi_box
            
            # Crop region
            cropped = frame[y:y+h, x:x+w]
            
            if cropped.size == 0:
                logger.warning("Cropped region is empty")
                return None
            
            # Resize to target size
            resized = cv2.resize(cropped, 
                               (self.target_size[1], self.target_size[0]),
                               interpolation=cv2.INTER_LINEAR)
            
            return resized
            
        except Exception as e:
            logger.error(f"Error cropping mouth region: {e}")
            return None
    
    def process_frame(self,
                     frame: np.ndarray,
                     lip_landmarks: np.ndarray) -> dict:
        """
        Process a single frame to extract mouth ROI.
        
        Args:
            frame: Input frame (height, width, 3)
            lip_landmarks: Lip landmarks (num_points, 2)
        
        Returns:
            Dictionary with:
                - roi_box: bounding box (x, y, w, h) or None
                - mouth_crop: cropped mouth region or None
                - success: bool indicating if crop succeeded
        """
        result = {
            'roi_box': None,
            'mouth_crop': None,
            'success': False
        }
        
        if frame is None or lip_landmarks is None or len(lip_landmarks) == 0:
            return result
        
        # Compute ROI box
        roi_box = self.compute_roi_box(lip_landmarks, frame.shape[:2])
        
        if roi_box is None:
            return result
        
        result['roi_box'] = roi_box
        
        # Crop mouth region
        mouth_crop = self.crop_mouth_region(frame, roi_box)
        
        if mouth_crop is None:
            return result
        
        result['mouth_crop'] = mouth_crop
        result['success'] = True
        
        return result
    
    def process_video_frames(self,
                            frames: np.ndarray,
                            landmark_results: List[dict]) -> List[dict]:
        """
        Process all frames in a video to extract mouth ROIs.
        
        Args:
            frames: Video frames (num_frames, height, width, 3)
            landmark_results: List of landmark detection results
        
        Returns:
            List of ROI extraction results (one dict per frame)
        """
        if len(frames) != len(landmark_results):
            logger.error(
                f"Frame count mismatch: {len(frames)} frames, "
                f"{len(landmark_results)} landmark results"
            )
            return []
        
        results = []
        
        for frame_idx, (frame, landmark_result) in enumerate(zip(frames, landmark_results)):
            lip_landmarks = landmark_result.get('lip_landmarks')
            
            roi_result = self.process_frame(frame, lip_landmarks)
            roi_result['frame_idx'] = frame_idx
            
            results.append(roi_result)
            
            if not roi_result['success']:
                logger.debug(f"Failed to extract mouth ROI for frame {frame_idx}")
        
        # Log summary
        num_success = sum(1 for r in results if r['success'])
        success_rate = num_success / len(results) if results else 0
        
        logger.info(
            f"Extracted mouth ROIs: {num_success}/{len(results)} "
            f"({success_rate:.1%} success rate)"
        )
        
        return results


def compute_median_roi_box(roi_results: List[dict]) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute median ROI box from multiple frames.
    
    Useful as a fallback when individual frame detection fails.
    
    Args:
        roi_results: List of ROI extraction results
    
    Returns:
        Median ROI box (x, y, w, h) or None if no valid boxes
    """
    valid_boxes = [r['roi_box'] for r in roi_results 
                   if r['success'] and r['roi_box'] is not None]
    
    if len(valid_boxes) == 0:
        logger.warning("No valid ROI boxes for median computation")
        return None
    
    # Compute median of each component
    boxes_array = np.array(valid_boxes)
    median_box = np.median(boxes_array, axis=0).astype(int)
    
    x, y, w, h = median_box
    
    logger.debug(f"Computed median ROI box: x={x}, y={y}, w={w}, h={h}")
    
    return (x, y, w, h)


def fill_missing_roi_boxes(roi_results: List[dict]) -> List[dict]:
    """
    Fill in missing ROI boxes using median box as fallback.
    
    Args:
        roi_results: List of ROI extraction results
    
    Returns:
        Updated results with filled ROI boxes
    """
    median_box = compute_median_roi_box(roi_results)
    
    if median_box is None:
        logger.warning("Cannot fill missing ROI boxes: no valid boxes found")
        return roi_results
    
    for result in roi_results:
        if not result['success'] or result['roi_box'] is None:
            result['roi_box'] = median_box
            result['fallback_used'] = True
            logger.debug(f"Using median ROI box for frame {result['frame_idx']}")
    
    return roi_results
