"""
Shape Validator Module

Verifies shape and type invariants for preprocessed data.
These are hard fail conditions - any violation indicates corrupted data.
"""

import numpy as np
import logging
from typing import Dict, Tuple
from validate.validation_result import ValidationResult

logger = logging.getLogger('validation.shapes')


class ShapeValidator:
    """
    Validates shape and type invariants for preprocessed clips.
    
    Checks:
    - mouth_frames.shape == (29, H, W, C)
    - lip_landmarks.shape == (29, K, 2)
    - No NaNs or infinities
    - Bbox coordinates within bounds
    """
    
    def __init__(self,
                 expected_frames: int = 29,
                 expected_landmarks: int = 20,
                 expected_roi_size: Tuple[int, int] = (96, 96)):
        """
        Initialize shape validator.
        
        Args:
            expected_frames: Expected number of frames (29 for LRW)
            expected_landmarks: Expected number of lip landmarks (20)
            expected_roi_size: Expected ROI size (height, width)
        """
        self.expected_frames = expected_frames
        self.expected_landmarks = expected_landmarks
        self.expected_roi_size = expected_roi_size
        
        logger.info(f"Initialized ShapeValidator: frames={expected_frames}, "
                   f"landmarks={expected_landmarks}, roi_size={expected_roi_size}")
    
    def validate_clip(self, clip_data: Dict) -> ValidationResult:
        """
        Validate shapes and types for a single clip.
        
        Args:
            clip_data: Clip dictionary from data loader
        
        Returns:
            ValidationResult with PASS/FAIL status and violations
        """
        clip_id = clip_data['clip_id']
        violations = []
        metrics = {}
        
        # Validate mouth frames shape
        mouth_frames = clip_data['mouth_frames']
        if mouth_frames is not None:
            expected_shape = (self.expected_frames, 
                            self.expected_roi_size[0], 
                            self.expected_roi_size[1], 
                            3)
            if mouth_frames.shape != expected_shape:
                violations.append(
                    f"mouth_frames shape mismatch: expected {expected_shape}, "
                    f"got {mouth_frames.shape}"
                )
            
            metrics['mouth_frames_shape'] = mouth_frames.shape
            
            # Check for NaN or infinity
            if np.any(np.isnan(mouth_frames)):
                violations.append("mouth_frames contains NaN values")
            if np.any(np.isinf(mouth_frames)):
                violations.append("mouth_frames contains infinity values")
        else:
            violations.append("mouth_frames is None")
        
        # Validate lip landmarks shape
        lip_landmarks = clip_data['lip_landmarks']
        if lip_landmarks is not None:
            expected_shape = (self.expected_frames, self.expected_landmarks, 2)
            if lip_landmarks.shape != expected_shape:
                violations.append(
                    f"lip_landmarks shape mismatch: expected {expected_shape}, "
                    f"got {lip_landmarks.shape}"
                )
            
            metrics['lip_landmarks_shape'] = lip_landmarks.shape
            
            # Check for NaN or infinity
            if np.any(np.isnan(lip_landmarks)):
                violations.append("lip_landmarks contains NaN values")
            if np.any(np.isinf(lip_landmarks)):
                violations.append("lip_landmarks contains infinity values")
        else:
            violations.append("lip_landmarks is None")
        
        # Validate bounding boxes
        bboxes = clip_data.get('bboxes', [])
        if bboxes and clip_data.get('original_frames') is not None:
            original_frames = clip_data['original_frames']
            frame_height, frame_width = original_frames.shape[1:3]
            
            for i, bbox in enumerate(bboxes):
                if bbox is None:
                    continue
                
                x, y, w, h = bbox
                
                # Check if bbox is within frame bounds
                if x < 0 or y < 0:
                    violations.append(f"Frame {i}: bbox has negative coordinates ({x}, {y})")
                
                if x + w > frame_width or y + h > frame_height:
                    violations.append(
                        f"Frame {i}: bbox extends beyond frame bounds "
                        f"({x}+{w} > {frame_width} or {y}+{h} > {frame_height})"
                    )
            
            metrics['num_bboxes'] = len(bboxes)
        
        # Determine status
        if len(violations) > 0:
            status = 'FAIL'
            logger.warning(f"Shape validation FAILED for {clip_id}: {len(violations)} violations")
        else:
            status = 'PASS'
            logger.debug(f"Shape validation PASSED for {clip_id}")
        
        return ValidationResult(
            clip_id=clip_id,
            status=status,
            validator_name='ShapeValidator',
            metrics=metrics,
            violations=violations
        )
