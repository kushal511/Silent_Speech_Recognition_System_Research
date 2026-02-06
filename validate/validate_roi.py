"""
ROI Validator Module

Checks ROI size consistency and detects over-tight or over-loose cropping.
"""

import numpy as np
import logging
from typing import Dict, Tuple
from validate.validation_result import ValidationResult

logger = logging.getLogger('validation.roi')


class ROIValidator:
    """
    Validates ROI size consistency and detects cropping issues.
    
    Checks:
    - ROI area ratio (relative to expected)
    - Over-tight cropping (< 0.7 * expected)
    - Over-loose cropping (> 1.5 * expected)
    - Area variance across frames
    """
    
    def __init__(self,
                 expected_roi_size: Tuple[int, int] = (96, 96),
                 tight_threshold: float = 0.7,
                 loose_threshold: float = 1.5,
                 variance_threshold: float = 50.0):
        """
        Initialize ROI validator.
        
        Args:
            expected_roi_size: Expected ROI size (height, width)
            tight_threshold: Ratio below which ROI is too tight
            loose_threshold: Ratio above which ROI is too loose
            variance_threshold: Max acceptable area variance
        """
        self.expected_roi_size = expected_roi_size
        self.expected_area = expected_roi_size[0] * expected_roi_size[1]
        self.tight_threshold = tight_threshold
        self.loose_threshold = loose_threshold
        self.variance_threshold = variance_threshold
        
        logger.info(f"Initialized ROIValidator: size={expected_roi_size}, "
                   f"tight<{tight_threshold}, loose>{loose_threshold}")
    
    def compute_roi_metrics(self, bboxes: list) -> Dict:
        """
        Compute ROI size metrics.
        
        Args:
            bboxes: List of (x, y, w, h) tuples
        
        Returns:
            Dictionary with ROI metrics
        """
        valid_bboxes = [b for b in bboxes if b is not None]
        
        if len(valid_bboxes) == 0:
            return {
                'mean_area': 0.0,
                'area_variance': 0.0,
                'area_ratio': 0.0,
                'is_consistent': False
            }
        
        # Compute areas
        areas = np.array([w * h for x, y, w, h in valid_bboxes])
        
        mean_area = float(np.mean(areas))
        area_variance = float(np.var(areas))
        area_ratio = mean_area / self.expected_area
        
        is_consistent = (
            area_ratio >= self.tight_threshold and
            area_ratio <= self.loose_threshold and
            area_variance <= self.variance_threshold
        )
        
        return {
            'mean_area': mean_area,
            'area_variance': area_variance,
            'area_ratio': area_ratio,
            'is_consistent': is_consistent
        }
    
    def validate_clip(self, clip_data: Dict) -> ValidationResult:
        """
        Validate ROI size and consistency.
        
        Args:
            clip_data: Clip dictionary from data loader
        
        Returns:
            ValidationResult with ROI assessment
        """
        clip_id = clip_data['clip_id']
        bboxes = clip_data.get('bboxes', [])
        
        metrics = self.compute_roi_metrics(bboxes)
        
        # Determine status and flags
        flags = []
        violations = []
        
        if metrics['area_ratio'] < self.tight_threshold:
            flags.append('roi_too_tight')
            violations.append(f"ROI too tight: ratio={metrics['area_ratio']:.2f} < {self.tight_threshold}")
        
        if metrics['area_ratio'] > self.loose_threshold:
            flags.append('roi_too_loose')
            violations.append(f"ROI too loose: ratio={metrics['area_ratio']:.2f} > {self.loose_threshold}")
        
        if metrics['area_variance'] > self.variance_threshold:
            flags.append('roi_inconsistent')
            violations.append(f"ROI variance too high: {metrics['area_variance']:.2f} > {self.variance_threshold}")
        
        if len(violations) > 0:
            status = 'WARN'
            logger.warning(f"ROI validation WARN for {clip_id}: {flags}")
        else:
            status = 'PASS'
            logger.debug(f"ROI validation PASSED for {clip_id}")
        
        return ValidationResult(
            clip_id=clip_id,
            status=status,
            validator_name='ROIValidator',
            metrics=metrics,
            violations=violations,
            flags=flags
        )
