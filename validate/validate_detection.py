"""
Detection Validator Module

Analyzes detection success rates and categorizes clip quality.
"""

import logging
from typing import Dict
from validate.validation_result import ValidationResult

logger = logging.getLogger('validation.detection')


class DetectionValidator:
    """
    Validates detection success rates and categorizes quality.
    
    Quality tiers:
    - good: detection_rate >= 0.90
    - ok: 0.80 <= detection_rate < 0.90
    - bad: detection_rate < 0.80
    """
    
    def __init__(self,
                 good_threshold: float = 0.90,
                 ok_threshold: float = 0.80):
        """
        Initialize detection validator.
        
        Args:
            good_threshold: Minimum rate for 'good' quality
            ok_threshold: Minimum rate for 'ok' quality
        """
        self.good_threshold = good_threshold
        self.ok_threshold = ok_threshold
        
        logger.info(f"Initialized DetectionValidator: good>={good_threshold}, ok>={ok_threshold}")
    
    def compute_detection_rate(self, metadata: Dict) -> float:
        """
        Compute detection success rate from metadata.
        
        Args:
            metadata: Metadata dictionary
        
        Returns:
            Detection rate (0.0 to 1.0)
        """
        detection_flags = metadata.get('face_detected', [])
        
        if len(detection_flags) == 0:
            return 0.0
        
        num_detected = sum(1 for flag in detection_flags if flag)
        rate = num_detected / len(detection_flags)
        
        return rate
    
    def categorize_quality(self, detection_rate: float) -> str:
        """
        Categorize quality based on detection rate.
        
        Args:
            detection_rate: Detection success rate
        
        Returns:
            Quality tier: 'good', 'ok', or 'bad'
        """
        if detection_rate >= self.good_threshold:
            return 'good'
        elif detection_rate >= self.ok_threshold:
            return 'ok'
        else:
            return 'bad'
    
    def validate_clip(self, clip_data: Dict) -> ValidationResult:
        """
        Validate detection metrics for a clip.
        
        Args:
            clip_data: Clip dictionary from data loader
        
        Returns:
            ValidationResult with quality assessment
        """
        clip_id = clip_data['clip_id']
        metadata = clip_data['metadata']
        
        # Compute detection rate
        detection_rate = self.compute_detection_rate(metadata)
        quality_tier = self.categorize_quality(detection_rate)
        
        metrics = {
            'detection_rate': detection_rate,
            'quality_tier': quality_tier
        }
        
        # Determine status
        flags = []
        if quality_tier == 'bad':
            status = 'FAIL'
            flags.append('low_detection_rate')
            logger.warning(f"Detection validation FAILED for {clip_id}: rate={detection_rate:.2%}")
        elif quality_tier == 'ok':
            status = 'WARN'
            flags.append('moderate_detection_rate')
            logger.info(f"Detection validation WARN for {clip_id}: rate={detection_rate:.2%}")
        else:
            status = 'PASS'
            logger.debug(f"Detection validation PASSED for {clip_id}: rate={detection_rate:.2%}")
        
        return ValidationResult(
            clip_id=clip_id,
            status=status,
            validator_name='DetectionValidator',
            metrics=metrics,
            flags=flags
        )
