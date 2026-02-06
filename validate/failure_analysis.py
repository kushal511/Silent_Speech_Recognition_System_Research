"""
Failure Analysis Module

Detects and categorizes specific failure modes in preprocessing outputs.
"""

import numpy as np
import logging
from typing import Dict, List
from validate.validation_result import FailureReport, ValidationResult

logger = logging.getLogger('validation.failure')


class FailureAnalyzer:
    """
    Analyzes validation results to detect specific failure modes.
    
    Detects:
    - Wrong face selection (multi-face frames)
    - Landmark drift
    - Over-tight or over-loose ROIs
    - Jitter from insufficient smoothing
    """
    
    def __init__(self):
        """Initialize failure analyzer."""
        logger.info("Initialized FailureAnalyzer")
    
    def detect_wrong_face(self, metadata: Dict) -> bool:
        """
        Detect if wrong face was selected (multi-face frames).
        
        Args:
            metadata: Clip metadata
        
        Returns:
            True if multi-face scenario detected
        """
        # Check if metadata indicates multiple faces detected
        multi_face_flags = metadata.get('multi_face_detected', [])
        
        if any(multi_face_flags):
            return True
        
        return False
    
    def detect_landmark_drift(self, landmarks: np.ndarray, threshold: float = 50.0) -> bool:
        """
        Detect landmark drift across frames.
        
        Args:
            landmarks: Landmarks array (num_frames, num_points, 2)
            threshold: Max acceptable cumulative drift
        
        Returns:
            True if drift detected
        """
        if landmarks is None or len(landmarks) < 2:
            return False
        
        # Compute cumulative displacement of landmark centroids
        centroids = np.mean(landmarks, axis=1)  # (num_frames, 2)
        displacements = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        cumulative_drift = np.sum(displacements)
        
        return cumulative_drift > threshold
    
    def detect_roi_issues(self, bboxes: list, 
                         tight_threshold: float = 0.7,
                         loose_threshold: float = 1.5,
                         expected_area: float = 9216.0) -> str:
        """
        Detect over-tight, over-loose, or inconsistent ROIs.
        
        Args:
            bboxes: List of bounding boxes
            tight_threshold: Ratio below which ROI is too tight
            loose_threshold: Ratio above which ROI is too loose
            expected_area: Expected ROI area
        
        Returns:
            Issue type: 'tight', 'loose', 'inconsistent', or ''
        """
        valid_bboxes = [b for b in bboxes if b is not None]
        
        if len(valid_bboxes) == 0:
            return ''
        
        areas = np.array([w * h for x, y, w, h in valid_bboxes])
        mean_area = np.mean(areas)
        area_ratio = mean_area / expected_area
        
        if area_ratio < tight_threshold:
            return 'tight'
        elif area_ratio > loose_threshold:
            return 'loose'
        elif np.var(areas) > 100.0:
            return 'inconsistent'
        
        return ''
    
    def detect_jitter(self, temporal_metrics: Dict, threshold: float = 10.0) -> bool:
        """
        Detect jitter from insufficient smoothing.
        
        Args:
            temporal_metrics: Temporal stability metrics
            threshold: Max acceptable displacement std
        
        Returns:
            True if jitter detected
        """
        bbox_disp_std = temporal_metrics.get('bbox_center_displacement_std', 0.0)
        
        return bbox_disp_std > threshold
    
    def analyze_clip(self, 
                    clip_data: Dict,
                    validation_results: List[ValidationResult]) -> FailureReport:
        """
        Analyze all failure modes for a clip.
        
        Args:
            clip_data: Clip dictionary from data loader
            validation_results: List of validation results for this clip
        
        Returns:
            FailureReport with detected failure modes
        """
        clip_id = clip_data['clip_id']
        failure_modes = []
        diagnostic_info = {}
        
        # Check for wrong face
        if self.detect_wrong_face(clip_data['metadata']):
            failure_modes.append('wrong_face_selection')
            diagnostic_info['wrong_face'] = 'Multiple faces detected in some frames'
        
        # Check for landmark drift
        if self.detect_landmark_drift(clip_data.get('lip_landmarks')):
            failure_modes.append('landmark_drift')
            diagnostic_info['landmark_drift'] = 'Excessive cumulative landmark displacement'
        
        # Check for ROI issues
        roi_issue = self.detect_roi_issues(clip_data.get('bboxes', []))
        if roi_issue:
            failure_modes.append(f'roi_{roi_issue}')
            diagnostic_info['roi_issue'] = f'ROI is {roi_issue}'
        
        # Check for jitter from temporal metrics
        temporal_result = next((r for r in validation_results 
                               if r.validator_name == 'TemporalValidator'), None)
        if temporal_result:
            if self.detect_jitter(temporal_result.metrics):
                failure_modes.append('jitter')
                diagnostic_info['jitter'] = 'High bbox displacement variance'
        
        # Determine severity
        if len(failure_modes) == 0:
            severity = 'LOW'
            recommended_action = 'No action needed'
        elif len(failure_modes) == 1:
            severity = 'MEDIUM'
            recommended_action = f'Review {failure_modes[0]} issue'
        else:
            severity = 'HIGH'
            recommended_action = 'Multiple issues detected - consider reprocessing'
        
        return FailureReport(
            clip_id=clip_id,
            failure_modes=failure_modes,
            severity=severity,
            diagnostic_info=diagnostic_info,
            recommended_action=recommended_action
        )
