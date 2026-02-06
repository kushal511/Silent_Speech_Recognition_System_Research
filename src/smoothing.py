"""
Temporal Smoothing Module

This module provides temporal smoothing for bounding boxes and landmarks
to reduce jitter and ensure smooth transitions across video frames.
"""

import numpy as np
import logging
from typing import List, Optional
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


class TemporalSmoother:
    """
    Applies temporal smoothing to landmark sequences and ROI boxes.
    
    Reduces jitter and noise in detected landmarks and bounding boxes
    across video frames using Gaussian or moving average filtering.
    """
    
    def __init__(self,
                 window_size: int = 5,
                 method: str = 'gaussian',
                 sigma: float = 1.0):
        """
        Initialize temporal smoother.
        
        Args:
            window_size: Size of smoothing window (must be odd)
            method: Smoothing method ('gaussian' or 'moving_average')
            sigma: Standard deviation for Gaussian smoothing
        """
        if window_size % 2 == 0:
            window_size += 1
            logger.warning(f"Window size must be odd, adjusted to {window_size}")
        
        self.window_size = window_size
        self.method = method
        self.sigma = sigma
        
        logger.info(
            f"Initialized TemporalSmoother: "
            f"method={method}, window={window_size}, sigma={sigma}"
        )
    
    def smooth_landmarks(self, landmarks_sequence: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply temporal smoothing to a sequence of landmarks.
        
        Args:
            landmarks_sequence: List of landmark arrays, each (num_points, 2)
        
        Returns:
            Smoothed landmark sequence
        """
        if not landmarks_sequence or len(landmarks_sequence) == 0:
            logger.warning("Empty landmarks sequence provided")
            return landmarks_sequence
        
        # Filter out None values and track their positions
        valid_indices = [i for i, lm in enumerate(landmarks_sequence) if lm is not None]
        
        if len(valid_indices) == 0:
            logger.warning("No valid landmarks in sequence")
            return landmarks_sequence
        
        # Stack valid landmarks into array (num_frames, num_points, 2)
        valid_landmarks = [landmarks_sequence[i] for i in valid_indices]
        landmarks_array = np.array(valid_landmarks)
        
        num_frames, num_points, num_coords = landmarks_array.shape
        
        if num_frames < self.window_size:
            logger.debug(
                f"Sequence too short for smoothing "
                f"({num_frames} < {self.window_size}), returning original"
            )
            return landmarks_sequence
        
        try:
            # Apply smoothing along time axis (axis=0)
            smoothed_array = np.zeros_like(landmarks_array)
            
            if self.method == 'gaussian':
                # Gaussian smoothing
                for point_idx in range(num_points):
                    for coord_idx in range(num_coords):
                        smoothed_array[:, point_idx, coord_idx] = gaussian_filter1d(
                            landmarks_array[:, point_idx, coord_idx],
                            sigma=self.sigma,
                            mode='nearest'
                        )
            
            elif self.method == 'moving_average':
                # Moving average smoothing
                half_window = self.window_size // 2
                
                for t in range(num_frames):
                    start = max(0, t - half_window)
                    end = min(num_frames, t + half_window + 1)
                    smoothed_array[t] = np.mean(landmarks_array[start:end], axis=0)
            
            else:
                logger.warning(f"Unknown smoothing method: {self.method}, using original")
                return landmarks_sequence
            
            # Reconstruct full sequence with smoothed values
            smoothed_sequence = landmarks_sequence.copy()
            for i, idx in enumerate(valid_indices):
                smoothed_sequence[idx] = smoothed_array[i]
            
            logger.debug(f"Smoothed {num_frames} landmark frames")
            
            return smoothed_sequence
            
        except Exception as e:
            logger.error(f"Error smoothing landmarks: {e}")
            return landmarks_sequence
    
    def smooth_roi_boxes(self, roi_boxes: List[Optional[tuple]]) -> List[Optional[tuple]]:
        """
        Apply temporal smoothing to ROI bounding boxes.
        
        Args:
            roi_boxes: List of ROI boxes, each (x, y, w, h) or None
        
        Returns:
            Smoothed ROI boxes
        """
        if not roi_boxes or len(roi_boxes) == 0:
            logger.warning("Empty ROI boxes list provided")
            return roi_boxes
        
        # Filter out None values
        valid_indices = [i for i, box in enumerate(roi_boxes) if box is not None]
        
        if len(valid_indices) == 0:
            logger.warning("No valid ROI boxes in sequence")
            return roi_boxes
        
        # Stack valid boxes into array (num_frames, 4)
        valid_boxes = [roi_boxes[i] for i in valid_indices]
        boxes_array = np.array(valid_boxes, dtype=np.float32)
        
        num_frames = len(boxes_array)
        
        if num_frames < self.window_size:
            logger.debug(
                f"Sequence too short for smoothing "
                f"({num_frames} < {self.window_size}), returning original"
            )
            return roi_boxes
        
        try:
            # Apply smoothing to each component (x, y, w, h)
            smoothed_array = np.zeros_like(boxes_array)
            
            if self.method == 'gaussian':
                for i in range(4):
                    smoothed_array[:, i] = gaussian_filter1d(
                        boxes_array[:, i],
                        sigma=self.sigma,
                        mode='nearest'
                    )
            
            elif self.method == 'moving_average':
                half_window = self.window_size // 2
                
                for t in range(num_frames):
                    start = max(0, t - half_window)
                    end = min(num_frames, t + half_window + 1)
                    smoothed_array[t] = np.mean(boxes_array[start:end], axis=0)
            
            else:
                logger.warning(f"Unknown smoothing method: {self.method}, using original")
                return roi_boxes
            
            # Round to integers and convert back to tuples
            smoothed_array = np.round(smoothed_array).astype(int)
            
            # Reconstruct full sequence
            smoothed_boxes = roi_boxes.copy()
            for i, idx in enumerate(valid_indices):
                smoothed_boxes[idx] = tuple(smoothed_array[i])
            
            logger.debug(f"Smoothed {num_frames} ROI boxes")
            
            return smoothed_boxes
            
        except Exception as e:
            logger.error(f"Error smoothing ROI boxes: {e}")
            return roi_boxes
    
    def compute_smoothness_metric(self, sequence: np.ndarray) -> float:
        """
        Compute smoothness metric for a sequence (lower is smoother).
        
        Uses variance of frame-to-frame differences as smoothness measure.
        
        Args:
            sequence: Array of shape (num_frames, ...)
        
        Returns:
            Smoothness metric (variance of differences)
        """
        if sequence is None or len(sequence) < 2:
            return 0.0
        
        try:
            # Compute frame-to-frame differences
            diffs = np.diff(sequence, axis=0)
            
            # Compute variance of differences
            variance = np.var(diffs)
            
            return float(variance)
            
        except Exception as e:
            logger.error(f"Error computing smoothness metric: {e}")
            return 0.0


def smooth_landmark_results(landmark_results: List[dict],
                           smoother: TemporalSmoother) -> List[dict]:
    """
    Apply temporal smoothing to landmark detection results.
    
    Args:
        landmark_results: List of landmark detection results
        smoother: TemporalSmoother instance
    
    Returns:
        Updated results with smoothed landmarks
    """
    # Extract landmark sequences
    lip_landmarks_seq = [r.get('lip_landmarks') for r in landmark_results]
    full_landmarks_seq = [r.get('landmarks') for r in landmark_results]
    
    # Smooth sequences
    smoothed_lip = smoother.smooth_landmarks(lip_landmarks_seq)
    smoothed_full = smoother.smooth_landmarks(full_landmarks_seq)
    
    # Update results
    for i, result in enumerate(landmark_results):
        result['lip_landmarks'] = smoothed_lip[i]
        result['landmarks'] = smoothed_full[i]
        result['smoothed'] = True
    
    logger.info("Applied temporal smoothing to landmarks")
    
    return landmark_results


def smooth_roi_results(roi_results: List[dict],
                      smoother: TemporalSmoother) -> List[dict]:
    """
    Apply temporal smoothing to ROI extraction results.
    
    Args:
        roi_results: List of ROI extraction results
        smoother: TemporalSmoother instance
    
    Returns:
        Updated results with smoothed ROI boxes
    """
    # Extract ROI box sequence
    roi_boxes = [r.get('roi_box') for r in roi_results]
    
    # Smooth boxes
    smoothed_boxes = smoother.smooth_roi_boxes(roi_boxes)
    
    # Update results
    for i, result in enumerate(roi_results):
        result['roi_box'] = smoothed_boxes[i]
        result['smoothed'] = True
    
    logger.info("Applied temporal smoothing to ROI boxes")
    
    return roi_results
