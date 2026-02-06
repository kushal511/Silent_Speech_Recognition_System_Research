"""
Temporal Validator Module

Measures temporal stability of landmarks and bounding boxes across frames.
Detects jitter, drift, and sudden jumps.
"""

import numpy as np
import logging
from typing import Dict
from validate.validation_result import ValidationResult

logger = logging.getLogger('validation.temporal')


class TemporalValidator:
    """
    Validates temporal stability of landmarks and bounding boxes.
    
    Computes:
    - Bbox center displacement (mean, std)
    - Bbox area variance
    - Landmark motion magnitude
    """
    
    def __init__(self,
                 bbox_displacement_threshold: float = 10.0,
                 bbox_area_variance_threshold: float = 100.0,
                 landmark_motion_threshold: float = 15.0):
        """
        Initialize temporal validator.
        
        Args:
            bbox_displacement_threshold: Max std of bbox center displacement
            bbox_area_variance_threshold: Max variance of bbox area
            landmark_motion_threshold: Max mean landmark motion per frame
        """
        self.bbox_displacement_threshold = bbox_displacement_threshold
        self.bbox_area_variance_threshold = bbox_area_variance_threshold
        self.landmark_motion_threshold = landmark_motion_threshold
        
        logger.info(f"Initialized TemporalValidator: "
                   f"bbox_disp={bbox_displacement_threshold}, "
                   f"bbox_var={bbox_area_variance_threshold}, "
                   f"landmark_motion={landmark_motion_threshold}")
    
    def compute_bbox_stability(self, bboxes: list) -> Dict:
        """
        Compute bbox stability metrics.
        
        Args:
            bboxes: List of (x, y, w, h) tuples
        
        Returns:
            Dictionary with stability metrics
        """
        valid_bboxes = [b for b in bboxes if b is not None]
        
        if len(valid_bboxes) < 2:
            return {
                'center_displacement_mean': 0.0,
                'center_displacement_std': 0.0,
                'area_variance': 0.0,
                'is_stable': True
            }
        
        # Compute centers
        centers = np.array([(x + w/2, y + h/2) for x, y, w, h in valid_bboxes])
        
        # Compute frame-to-frame displacement
        displacements = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        
        # Compute areas
        areas = np.array([w * h for x, y, w, h in valid_bboxes])
        
        metrics = {
            'center_displacement_mean': float(np.mean(displacements)),
            'center_displacement_std': float(np.std(displacements)),
            'area_variance': float(np.var(areas)),
            'is_stable': (np.std(displacements) <= self.bbox_displacement_threshold and
                         np.var(areas) <= self.bbox_area_variance_threshold)
        }
        
        return metrics
    
    def compute_landmark_stability(self, landmarks: np.ndarray) -> Dict:
        """
        Compute landmark stability metrics.
        
        Args:
            landmarks: Landmarks array (num_frames, num_points, 2)
        
        Returns:
            Dictionary with stability metrics
        """
        if landmarks is None or len(landmarks) < 2:
            return {
                'motion_magnitude_mean': 0.0,
                'motion_magnitude_std': 0.0,
                'max_jump': 0.0,
                'is_stable': True
            }
        
        # Compute frame-to-frame motion for each landmark
        motion = np.linalg.norm(np.diff(landmarks, axis=0), axis=2)  # (num_frames-1, num_points)
        
        # Average motion across all landmarks per frame
        motion_per_frame = np.mean(motion, axis=1)
        
        # Max motion (jump) across all landmarks and frames
        max_jump = float(np.max(motion))
        
        metrics = {
            'motion_magnitude_mean': float(np.mean(motion_per_frame)),
            'motion_magnitude_std': float(np.std(motion_per_frame)),
            'max_jump': max_jump,
            'is_stable': np.mean(motion_per_frame) <= self.landmark_motion_threshold
        }
        
        return metrics
    
    def validate_clip(self, clip_data: Dict) -> ValidationResult:
        """
        Validate temporal stability for a clip.
        
        Args:
            clip_data: Clip dictionary from data loader
        
        Returns:
            ValidationResult with temporal stability assessment
        """
        clip_id = clip_data['clip_id']
        
        # Compute bbox stability
        bboxes = clip_data.get('bboxes', [])
        bbox_metrics = self.compute_bbox_stability(bboxes)
        
        # Compute landmark stability
        landmarks = clip_data.get('lip_landmarks')
        landmark_metrics = self.compute_landmark_stability(landmarks)
        
        # Combine metrics
        metrics = {
            **{f'bbox_{k}': v for k, v in bbox_metrics.items()},
            **{f'landmark_{k}': v for k, v in landmark_metrics.items()}
        }
        
        # Determine status
        flags = []
        is_stable = bbox_metrics['is_stable'] and landmark_metrics['is_stable']
        
        if not is_stable:
            status = 'WARN'
            if not bbox_metrics['is_stable']:
                flags.append('bbox_instability')
            if not landmark_metrics['is_stable']:
                flags.append('landmark_instability')
            logger.warning(f"Temporal validation WARN for {clip_id}: {flags}")
        else:
            status = 'PASS'
            logger.debug(f"Temporal validation PASSED for {clip_id}")
        
        return ValidationResult(
            clip_id=clip_id,
            status=status,
            validator_name='TemporalValidator',
            metrics=metrics,
            flags=flags
        )
