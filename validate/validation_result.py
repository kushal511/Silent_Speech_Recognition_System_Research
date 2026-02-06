"""
Validation Result Data Models

Defines data structures for validation results, failure reports, and quality metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class ValidationResult:
    """
    Result from a single validator on a single clip.
    """
    clip_id: str
    status: str  # 'PASS', 'WARN', 'FAIL'
    validator_name: str
    
    # Metrics computed by validator
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Violations found (for FAIL status)
    violations: List[str] = field(default_factory=list)
    
    # Flags for specific issues
    flags: List[str] = field(default_factory=list)
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Validate status value."""
        if self.status not in ['PASS', 'WARN', 'FAIL']:
            raise ValueError(f"Invalid status: {self.status}")


@dataclass
class FailureReport:
    """
    Detailed failure report for a clip.
    """
    clip_id: str
    failure_modes: List[str]  # e.g., ['landmark_drift', 'jitter']
    severity: str  # 'LOW', 'MEDIUM', 'HIGH'
    diagnostic_info: Dict = field(default_factory=dict)
    recommended_action: str = ""
    
    def __post_init__(self):
        """Validate severity value."""
        if self.severity not in ['LOW', 'MEDIUM', 'HIGH']:
            raise ValueError(f"Invalid severity: {self.severity}")


@dataclass
class QualityMetrics:
    """
    Comprehensive quality metrics for a clip.
    """
    # Detection metrics
    detection_rate: float
    quality_tier: str  # 'good', 'ok', 'bad'
    
    # Temporal stability metrics
    bbox_center_displacement_mean: float = 0.0
    bbox_center_displacement_std: float = 0.0
    bbox_area_variance: float = 0.0
    landmark_motion_mean: float = 0.0
    landmark_motion_std: float = 0.0
    max_landmark_jump: float = 0.0
    
    # ROI consistency metrics
    roi_mean_area: float = 0.0
    roi_area_variance: float = 0.0
    roi_area_ratio: float = 1.0
    
    # Flags
    is_temporally_stable: bool = True
    is_roi_consistent: bool = True
    has_failures: bool = False
    
    def __post_init__(self):
        """Validate quality tier."""
        if self.quality_tier not in ['good', 'ok', 'bad']:
            raise ValueError(f"Invalid quality tier: {self.quality_tier}")
