"""
LRW Preprocessing Validation Pipeline

This package provides comprehensive quality control and validation for
Silent Speech Recognition (SSR) preprocessing outputs from the LRW dataset.

The validation pipeline verifies:
- Visual correctness (landmark overlays, ROI extraction)
- Numeric sanity (shapes, detection rates, temporal stability)
- Dataset-wide consistency (distributions, outliers, failure modes)

Modules:
- data_loader: Load original videos and preprocessed outputs
- validate_shapes: Shape and type invariant checks
- validate_detection: Detection success rate analysis
- validate_temporal: Temporal stability metrics
- validate_roi: ROI size and consistency checks
- visualize_samples: Visual validation utilities
- generate_gifs: Animated sequence generation
- failure_analysis: Failure mode detection
- dataset_analysis: Dataset-wide statistics
- report_generator: CSV and summary report generation
- download_lrw: LRW dataset download utility
- run_validation: CLI entry point
"""

__version__ = "1.0.0"
__author__ = "SSR Validation Team"

# Import key classes for convenience
from validate.data_loader import PreprocessedDataLoader
from validate.validate_shapes import ShapeValidator
from validate.validate_detection import DetectionValidator
from validate.validate_temporal import TemporalValidator
from validate.validate_roi import ROIValidator

__all__ = [
    'PreprocessedDataLoader',
    'ShapeValidator',
    'DetectionValidator',
    'TemporalValidator',
    'ROIValidator',
]
