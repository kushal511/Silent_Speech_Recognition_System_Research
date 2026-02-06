# Design Document

## Overview

The validation pipeline is a comprehensive quality control system for SSR preprocessing outputs. It operates as a standalone module that reads both the **original LRW videos** and the **preprocessed outputs**, performing three levels of validation: visual correctness, numeric sanity, and dataset-wide consistency. The system is designed to be automated, reproducible, and suitable for research lab review.

**Dataset Context**: The validation pipeline operates on two data sources:

1. **Original LRW Dataset** (for reference and comparison):
```
lrw_dataset/                     # Original LRW videos
├── WORD_CLASS/
│   ├── train/
│   │   ├── VIDEO_ID.mp4         # Original 29-frame video
```

2. **Preprocessed Outputs** (to be validated):
```
output/                          # Preprocessing pipeline output
├── WORD_CLASS/
│   ├── train/
│   │   ├── VIDEO_ID/
│   │   │   ├── frames/          # 29 mouth ROI frames (96x96 PNG)
│   │   │   ├── landmarks.npy    # (29, 20, 2) lip landmarks
│   │   │   └── metadata.json    # Preprocessing metadata
```

By having access to both the original videos and preprocessed outputs, the validation pipeline can:
- Overlay landmarks on **original full-resolution frames** (not just cropped mouths)
- Verify that mouth ROIs are correctly extracted from the original frames
- Compare bounding boxes against original frame dimensions
- Generate side-by-side visualizations showing original → preprocessed transformation
- Detect if the wrong region was cropped or if landmarks are misaligned

The validation pipeline provides confidence that preprocessing outputs are stable, accurate, and ready for GPU-based model training. It detects common failure modes, generates diagnostic visualizations, and produces structured reports for systematic review.

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│              Original LRW Dataset (Reference)                │
│                                                              │
│  lrw_dataset/                                               │
│  ├── WORD_CLASS_1/                                          │
│  │   ├── train/                                             │
│  │   │   ├── VIDEO_ID_1.mp4  (29 frames, 256x256)          │
│  │   │   ├── VIDEO_ID_2.mp4                                 │
│  │   │   └── ...                                             │
│  │   ├── val/                                                │
│  │   └── test/                                               │
│  └── WORD_CLASS_2/                                          │
│      └── ...                                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ (Used for reference/comparison)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Preprocessing Pipeline (Already Complete)            │
│                   run_preprocess.py                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Preprocessed Data (To Be Validated)             │
│                                                              │
│  output/                                                     │
│  ├── WORD_CLASS_1/                                          │
│  │   ├── train/                                             │
│  │   │   ├── VIDEO_ID_1/                                    │
│  │   │   │   ├── frames/                                    │
│  │   │   │   │   ├── frame_00.png  (96x96 mouth crop)      │
│  │   │   │   │   ├── frame_01.png                           │
│  │   │   │   │   └── ... (29 frames total)                  │
│  │   │   │   ├── landmarks.npy  (29, 20, 2) lip landmarks   │
│  │   │   │   └── metadata.json  (bboxes, detection flags)   │
│  │   │   └── VIDEO_ID_2/                                    │
│  │   │       └── ...                                         │
│  │   ├── val/                                                │
│  │   └── test/                                               │
│  └── WORD_CLASS_2/                                          │
│      └── ...                                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Validation Pipeline (This Spec)                 │
│                   run_validation.py                          │
│                                                              │
│  Inputs:                                                     │
│  • Original LRW videos (lrw_dataset/)                       │
│  • Preprocessed outputs (output/)                           │
│                                                              │
│  Processing:                                                 │
│  1. Load original video frames + preprocessed data          │
│  2. Overlay landmarks on ORIGINAL frames (full resolution)  │
│  3. Verify mouth ROI extraction from original frames        │
│  4. Run validators (shape, detection, temporal, ROI)        │
│  5. Generate visualizations (original + bbox + landmarks)   │
│  6. Create side-by-side: original → cropped comparison      │
│  7. Analyze failures (detect specific failure modes)        │
│  8. Compute dataset statistics (distributions, outliers)    │
│  9. Generate reports (CSV summaries, failure reports)       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Validation Results (Output)                     │
│                                                              │
│  validation_results/                                         │
│  ├── validation_summary.csv      (per-clip metrics)         │
│  ├── failure_report.csv          (flagged clips)            │
│  ├── summary_statistics.json     (dataset aggregates)       │
│  ├── visualizations/              (sample clips)            │
│  │   ├── clip_001_original_overlay.png  (landmarks on orig) │
│  │   ├── clip_001_sidebyside.png  (orig → cropped)          │
│  │   ├── clip_001_mouth_sequence.gif  (cropped animation)   │
│  │   └── clip_001_original_sequence.gif  (orig with bbox)   │
│  ├── distributions/               (histograms)              │
│  │   ├── roi_size_distribution.png                          │
│  │   ├── detection_rate_distribution.png                    │
│  │   └── temporal_stability_distribution.png                │
│  └── logs/                                                   │
│      └── validation_YYYYMMDD_HHMMSS.log                     │
└─────────────────────────────────────────────────────────────┘
```

**Key Points**:
1. **Inputs**: 
   - Original LRW videos from `lrw_dataset/` (for reference and visualization)
   - Preprocessed data from `output/` (to be validated)
2. **Processing**: Read-only validation (no modification of either input)
3. **Output**: Validation reports and visualizations in separate `validation_results/` directory

**Benefits of Using Original Videos**:
- Can overlay landmarks on full-resolution original frames
- Can verify mouth ROI is correctly positioned in original frame
- Can create better visualizations showing original → preprocessed transformation
- Can detect if wrong face was selected in multi-face scenarios
- Can verify bounding boxes are reasonable relative to original frame size

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Preprocessed LRW Data                       │
│  (mouth_frames, lip_landmarks, metadata per clip)           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Validation Pipeline Entry Point                 │
│                  (run_validation.py)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Level 1   │  │   Level 2   │  │   Level 3   │
│   Visual    │  │   Numeric   │  │  Dataset    │
│ Validation  │  │  Validation │  │   Analysis  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Report Generator                          │
│  (validation_summary.csv, failure_report.csv, visuals)      │
└─────────────────────────────────────────────────────────────┘
```

### Module Structure

```
validate/
├── __init__.py                 # Package initialization
├── config.yaml                 # Validation thresholds and parameters
├── download_lrw.py             # LRW dataset download utility
├── validate_shapes.py          # Shape and type invariant checks
├── validate_temporal.py        # Temporal stability metrics
├── validate_detection.py       # Detection success rate analysis
├── validate_roi.py             # ROI size and consistency checks
├── visualize_samples.py        # Visual validation utilities
├── generate_gifs.py            # Animated sequence generation
├── failure_analysis.py         # Failure mode detection
├── dataset_analysis.py         # Dataset-wide statistics
├── report_generator.py         # CSV and summary report generation
└── run_validation.py           # CLI entry point
```

## Components and Interfaces

### 1. Data Loader (`validate/data_loader.py`)

**Purpose**: Load both original LRW videos and preprocessed outputs for validation.

**Data Sources**: 
1. **Original LRW Dataset**: Raw videos for reference and visualization
2. **Preprocessed Outputs**: Mouth crops, landmarks, and metadata to be validated

**Data Import Strategy**:
1. Scan the preprocessed output directory to discover all processed clips
2. For each clip, load:
   - **Original video frames** (29 frames, full resolution) from `lrw_dataset/WORD_CLASS/SPLIT/VIDEO_ID.mp4`
   - **Preprocessed mouth frames** (29 frames, 96x96) from `output/WORD_CLASS/SPLIT/VIDEO_ID/frames/`
   - **Landmarks** from `output/WORD_CLASS/SPLIT/VIDEO_ID/landmarks.npy`
   - **Metadata** from `output/WORD_CLASS/SPLIT/VIDEO_ID/metadata.json`
3. Match original videos to preprocessed outputs using word_class, split, and video_id
4. Organize data into a standardized dictionary format for validation

**Interface**:
```python
class PreprocessedDataLoader:
    def __init__(self, 
                 lrw_dataset_root: str,
                 preprocessed_root: str):
        """
        Initialize loader with both original and preprocessed data roots.
        
        Args:
            lrw_dataset_root: Path to original LRW dataset (e.g., '/path/to/lrw_dataset')
            preprocessed_root: Path to preprocessing output (e.g., '../output')
        
        The loader will:
        1. Scan preprocessed_root to find all processed clips
        2. Match each to its original video in lrw_dataset_root
        3. Validate that both sources exist for each clip
        """
        
    def load_clip(self, clip_path: str) -> Dict:
        """
        Load a single clip with both original and preprocessed data.
        
        Args:
            clip_path: Path to preprocessed clip directory 
                      (e.g., 'output/ABOUT/train/ABOUT_00001')
        
        Returns:
            {
                # Original data (from LRW dataset)
                'original_frames': np.ndarray,    # (29, H_orig, W_orig, C) from .mp4
                'original_video_path': str,       # Path to original .mp4
                
                # Preprocessed data (from preprocessing output)
                'mouth_frames': np.ndarray,       # (29, 96, 96, C) from frames/*.png
                'lip_landmarks': np.ndarray,      # (29, 20, 2) from landmarks.npy
                'metadata': dict,                 # From metadata.json
                
                # Identifiers
                'clip_id': str,                   # e.g., 'ABOUT_00001'
                'word_class': str,                # e.g., 'ABOUT'
                'split': str,                     # e.g., 'train'
                
                # Derived info
                'bboxes': List[Tuple],            # (29,) bounding boxes from metadata
                'detection_flags': List[bool],    # (29,) detection success from metadata
            }
        """
        
    def iter_clips(self, split: Optional[str] = None) -> Iterator[Dict]:
        """
        Iterate over all clips in dataset.
        
        Args:
            split: Optional filter for 'train', 'val', or 'test'
        
        Yields:
            Clip dictionaries as returned by load_clip()
        """
        
    def get_clip_paths(self) -> List[str]:
        """
        Get list of all preprocessed clip paths.
        
        Returns:
            List of paths to preprocessed clip directories
        """
        
    def verify_data_availability(self) -> Dict:
        """
        Verify that original videos exist for all preprocessed clips.
        
        Returns:
            {
                'total_preprocessed': int,
                'matched_originals': int,
                'missing_originals': List[str],  # Clip IDs without original videos
            }
        """
```

**Error Handling**:
- If original video is missing for a preprocessed clip, log warning and skip visualization (but continue other validation)
- If preprocessed data is corrupted, log error and mark clip as FAIL
- If video cannot be read, log error with details

**Note**: Having access to original videos enables:
- Landmark overlay on full-resolution frames
- Verification that mouth ROI is correctly positioned
- Side-by-side original → preprocessed visualizations
- Detection of wrong face selection in multi-face frames

### 2. Shape Validator (`validate/validate_shapes.py`)

**Purpose**: Verify shape and type invariants (hard fail conditions).

**Interface**:
```python
class ShapeValidator:
    def __init__(self, expected_frames: int = 29, 
                 expected_landmarks: int = 20,
                 expected_roi_size: Tuple[int, int] = (96, 96)):
        """Initialize with expected dimensions."""
        
    def validate_clip(self, clip_data: Dict) -> ValidationResult:
        """
        Validate shapes and types for a single clip.
        
        Checks:
        - mouth_frames.shape == (29, H, W, C)
        - lip_landmarks.shape == (29, K, 2)
        - No NaNs or infinities
        - Bbox coordinates within bounds
        
        Returns:
            ValidationResult with status (PASS/FAIL) and violations
        """
```

### 3. Temporal Validator (`validate/validate_temporal.py`)

**Purpose**: Measure temporal stability of landmarks and bounding boxes.

**Interface**:
```python
class TemporalValidator:
    def __init__(self, 
                 bbox_displacement_threshold: float = 10.0,
                 bbox_area_variance_threshold: float = 100.0,
                 landmark_motion_threshold: float = 15.0):
        """Initialize with stability thresholds."""
        
    def compute_bbox_stability(self, bboxes: np.ndarray) -> Dict:
        """
        Compute bbox stability metrics.
        
        Returns:
            {
                'center_displacement_mean': float,
                'center_displacement_std': float,
                'area_variance': float,
                'is_stable': bool
            }
        """
        
    def compute_landmark_stability(self, landmarks: np.ndarray) -> Dict:
        """
        Compute landmark stability metrics.
        
        Returns:
            {
                'motion_magnitude_mean': float,
                'motion_magnitude_std': float,
                'max_jump': float,
                'is_stable': bool
            }
        """
        
    def validate_clip(self, clip_data: Dict) -> ValidationResult:
        """Validate temporal stability for a clip."""
```

### 4. Detection Validator (`validate/validate_detection.py`)

**Purpose**: Analyze detection success rates and categorize quality.

**Interface**:
```python
class DetectionValidator:
    def __init__(self,
                 good_threshold: float = 0.90,
                 ok_threshold: float = 0.80):
        """Initialize with quality tier thresholds."""
        
    def compute_detection_rate(self, metadata: Dict) -> float:
        """Compute detection success rate from metadata."""
        
    def categorize_quality(self, detection_rate: float) -> str:
        """Categorize as 'good', 'ok', or 'bad'."""
        
    def validate_clip(self, clip_data: Dict) -> ValidationResult:
        """Validate detection metrics for a clip."""
```

### 5. ROI Validator (`validate/validate_roi.py`)

**Purpose**: Check ROI size consistency and detect over-tight/loose cropping.

**Interface**:
```python
class ROIValidator:
    def __init__(self,
                 expected_roi_size: Tuple[int, int] = (96, 96),
                 tight_threshold: float = 0.7,
                 loose_threshold: float = 1.5,
                 variance_threshold: float = 50.0):
        """Initialize with ROI size thresholds."""
        
    def compute_roi_metrics(self, bboxes: np.ndarray) -> Dict:
        """
        Compute ROI size metrics.
        
        Returns:
            {
                'mean_area': float,
                'area_variance': float,
                'area_ratio': float,  # relative to expected
                'is_consistent': bool
            }
        """
        
    def validate_clip(self, clip_data: Dict) -> ValidationResult:
        """Validate ROI size and consistency."""
```

### 6. Visual Validator (`validate/visualize_samples.py`)

**Purpose**: Generate visual validation outputs.

**Interface**:
```python
class VisualValidator:
    def __init__(self, output_dir: str):
        """Initialize with output directory for visualizations."""
        
    def overlay_landmarks(self, 
                         frame: np.ndarray,
                         landmarks: np.ndarray,
                         bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Overlay landmarks and bbox on frame."""
        
    def create_side_by_side(self,
                           original_frame: np.ndarray,
                           mouth_roi: np.ndarray,
                           landmarks: np.ndarray,
                           bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Create side-by-side comparison panel."""
        
    def visualize_clip(self, clip_data: Dict, output_path: str):
        """Generate complete visual validation for a clip."""
```

### 7. GIF Generator (`validate/generate_gifs.py`)

**Purpose**: Create animated sequences of mouth ROIs.

**Interface**:
```python
class GIFGenerator:
    def __init__(self, fps: int = 25, loop: int = 0):
        """Initialize GIF generator with frame rate and loop settings."""
        
    def create_mouth_sequence_gif(self,
                                 mouth_frames: np.ndarray,
                                 output_path: str):
        """Create GIF from 29 mouth frames."""
        
    def create_annotated_sequence_gif(self,
                                     original_frames: np.ndarray,
                                     landmarks: np.ndarray,
                                     bboxes: np.ndarray,
                                     output_path: str):
        """Create GIF with landmarks and bbox overlays."""
```

### 8. Failure Analyzer (`validate/failure_analysis.py`)

**Purpose**: Detect and categorize specific failure modes.

**Interface**:
```python
class FailureAnalyzer:
    def __init__(self):
        """Initialize failure mode detectors."""
        
    def detect_wrong_face(self, metadata: Dict) -> bool:
        """Detect if wrong face was selected (multi-face frames)."""
        
    def detect_landmark_drift(self, landmarks: np.ndarray) -> bool:
        """Detect landmark drift across frames."""
        
    def detect_roi_issues(self, bboxes: np.ndarray) -> str:
        """Detect over-tight, over-loose, or inconsistent ROIs."""
        
    def detect_jitter(self, temporal_metrics: Dict) -> bool:
        """Detect jitter from insufficient smoothing."""
        
    def analyze_clip(self, clip_data: Dict, 
                    validation_results: List[ValidationResult]) -> FailureReport:
        """Analyze all failure modes for a clip."""
```

### 9. Dataset Analyzer (`validate/dataset_analysis.py`)

**Purpose**: Compute dataset-wide statistics and distributions.

**Interface**:
```python
class DatasetAnalyzer:
    def __init__(self, output_dir: str):
        """Initialize with output directory for plots."""
        
    def compute_distributions(self, 
                            all_results: List[ValidationResult]) -> Dict:
        """
        Compute dataset-wide distributions.
        
        Returns distributions for:
        - ROI width/height
        - Detection success rates
        - Bbox motion variance
        - Landmark motion statistics
        """
        
    def plot_distributions(self, distributions: Dict):
        """Generate histogram plots for all distributions."""
        
    def identify_outliers(self, distributions: Dict) -> List[str]:
        """Identify outlier clips in distribution tails."""
```

### 10. Report Generator (`validate/report_generator.py`)

**Purpose**: Generate structured validation reports.

**Interface**:
```python
class ReportGenerator:
    def __init__(self, output_dir: str):
        """Initialize with output directory for reports."""
        
    def generate_summary_csv(self, 
                            results: List[ValidationResult],
                            output_path: str):
        """
        Generate validation_summary.csv with per-clip metrics.
        
        Columns:
        - clip_id, word_class, split
        - status (PASS/WARN/FAIL)
        - detection_rate, quality_tier
        - bbox_stability, landmark_stability
        - roi_consistency
        - failure_modes
        """
        
    def generate_failure_report(self,
                               failures: List[FailureReport],
                               output_path: str):
        """
        Generate failure_report.csv with flagged clips.
        
        Columns:
        - clip_id, failure_type, severity
        - diagnostic_info, recommended_action
        """
        
    def generate_summary_statistics(self,
                                   results: List[ValidationResult]) -> Dict:
        """Compute aggregate statistics for the dataset."""
```

### 11. LRW Dataset Downloader (`validate/download_lrw.py`)

**Purpose**: Automated download and setup of the LRW dataset.

**Interface**:
```python
class LRWDatasetDownloader:
    def __init__(self, 
                 output_dir: str,
                 dataset_url: Optional[str] = None):
        """
        Initialize LRW dataset downloader.
        
        Args:
            output_dir: Directory to download and extract dataset
            dataset_url: Optional custom URL for LRW dataset
        """
        
    def check_dataset_exists(self) -> bool:
        """Check if LRW dataset already exists in output_dir."""
        
    def download_dataset(self, 
                        splits: Optional[List[str]] = None,
                        word_classes: Optional[List[str]] = None) -> bool:
        """
        Download LRW dataset from official source.
        
        Args:
            splits: Optional list of splits to download ['train', 'val', 'test']
            word_classes: Optional list of word classes to download (default: all 500)
        
        Returns:
            True if download successful, False otherwise
        
        Process:
        1. Check if dataset already exists
        2. Download dataset files (with progress bars)
        3. Verify file integrity (checksums)
        4. Extract archives to output_dir
        5. Organize into expected structure
        """
        
    def verify_dataset(self) -> Dict:
        """
        Verify downloaded dataset integrity.
        
        Returns:
            {
                'total_videos': int,
                'missing_videos': List[str],
                'corrupted_videos': List[str],
                'is_complete': bool
            }
        """
        
    def get_download_info(self) -> Dict:
        """
        Get information about LRW dataset download.
        
        Returns:
            {
                'dataset_size': str,  # e.g., "~50 GB"
                'num_videos': int,
                'num_word_classes': int,
                'download_url': str,
                'license_info': str
            }
        """
```

**CLI Interface**:
```bash
# Download entire LRW dataset
python -m validate.download_lrw --output_dir /path/to/lrw_dataset

# Download specific splits
python -m validate.download_lrw \
    --output_dir /path/to/lrw_dataset \
    --splits train val

# Download specific word classes
python -m validate.download_lrw \
    --output_dir /path/to/lrw_dataset \
    --word_classes ABOUT ABSOLUTELY ABUSE

# Verify existing dataset
python -m validate.download_lrw \
    --output_dir /path/to/lrw_dataset \
    --verify_only
```

**Note**: The downloader will:
- Check for existing dataset before downloading
- Display progress bars for downloads
- Verify file integrity
- Handle network errors with retry logic
- Provide clear error messages
- Respect LRW dataset license terms

## Data Models

### ValidationResult

```python
@dataclass
class ValidationResult:
    clip_id: str
    status: str  # 'PASS', 'WARN', 'FAIL'
    validator_name: str
    
    # Metrics
    metrics: Dict[str, float]
    
    # Violations
    violations: List[str]
    
    # Flags
    flags: List[str]
    
    # Timestamp
    timestamp: str
```

### FailureReport

```python
@dataclass
class FailureReport:
    clip_id: str
    failure_modes: List[str]  # e.g., ['landmark_drift', 'jitter']
    severity: str  # 'LOW', 'MEDIUM', 'HIGH'
    diagnostic_info: Dict
    recommended_action: str
```

### QualityMetrics

```python
@dataclass
class QualityMetrics:
    # Detection
    detection_rate: float
    quality_tier: str  # 'good', 'ok', 'bad'
    
    # Temporal stability
    bbox_center_displacement_mean: float
    bbox_center_displacement_std: float
    bbox_area_variance: float
    landmark_motion_mean: float
    landmark_motion_std: float
    max_landmark_jump: float
    
    # ROI consistency
    roi_mean_area: float
    roi_area_variance: float
    roi_area_ratio: float
    
    # Flags
    is_temporally_stable: bool
    is_roi_consistent: bool
    has_failures: bool
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Shape invariance preservation

*For any* preprocessed clip, the mouth_frames shape must be exactly (29, H, W, C) and lip_landmarks shape must be exactly (29, K, 2), with no NaN or infinity values.

**Validates: Requirements 2.1, 2.2, 2.3**

### Property 2: Detection rate categorization consistency

*For any* clip with detection_rate >= 0.90, the quality tier must be "good"; for detection_rate in [0.80, 0.90), tier must be "ok"; for detection_rate < 0.80, tier must be "bad".

**Validates: Requirements 3.2, 3.3, 3.4**

### Property 3: Temporal stability threshold enforcement

*For any* clip, if bbox center displacement std exceeds threshold OR landmark motion exceeds threshold, the clip must be flagged as temporally unstable.

**Validates: Requirements 4.4, 4.5**

### Property 4: Bounding box containment

*For any* frame in any clip, all bounding box coordinates must lie within the frame dimensions (0 <= x < width, 0 <= y < height).

**Validates: Requirements 2.4**

### Property 5: ROI size ratio bounds

*For any* clip, if ROI area ratio < 0.7 OR > 1.5 relative to expected area, the clip must be flagged for ROI size issues.

**Validates: Requirements 13.2, 13.3**

### Property 6: Failure mode detection completeness

*For any* clip marked as FAIL, at least one specific failure mode (wrong_face, landmark_drift, roi_issue, or jitter) must be identified and logged.

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**

### Property 7: Visualization determinism

*For any* given random seed and sample size, the set of clips selected for visualization must be identical across multiple runs.

**Validates: Requirements 1.5**

### Property 8: Report completeness

*For any* validation run, the output must include validation_summary.csv, failure_report.csv, and debug visualizations directory, all containing data for the same set of processed clips.

**Validates: Requirements 7.1, 7.2, 7.3**

### Property 9: Read-only guarantee

*For any* validation run, no files in the input preprocessed data directory may be modified, created, or deleted.

**Validates: Requirements 9.2, 9.3, 9.5**

### Property 10: Threshold configurability

*For any* validation run, changing threshold values in the configuration file must result in different validation outcomes for clips near the threshold boundaries.

**Validates: Requirements 15.2, 15.3, 15.4**

## Error Handling

### Input Validation Errors

- **Missing files**: Log error, skip clip, continue processing
- **Corrupted data**: Log error with details, mark clip as FAIL
- **Invalid shapes**: Mark as HARD FAIL, include in failure report
- **Missing metadata**: Use default values where possible, flag as incomplete

### Processing Errors

- **Visualization failures**: Log error, skip visualization, continue validation
- **GIF generation failures**: Log error, continue with other outputs
- **Metric computation errors**: Log error, mark metric as unavailable

### Output Errors

- **Disk space issues**: Fail gracefully with clear error message
- **Permission errors**: Fail with actionable error message
- **CSV write errors**: Retry once, then fail with error details

## Testing Strategy

### Unit Tests

- Test each validator independently with synthetic data
- Test metric computation functions with known inputs
- Test failure mode detection with crafted failure cases
- Test report generation with mock validation results

### Integration Tests

- Test complete validation pipeline on small sample dataset
- Verify all output files are generated correctly
- Verify deterministic sampling with fixed seeds
- Test error handling with intentionally corrupted inputs

### Property-Based Tests

Property-based tests will use the `hypothesis` library for Python to generate random test cases and verify correctness properties hold across all inputs.

**Configuration**: Each property test will run a minimum of 100 iterations.

**Tagging**: Each property-based test will include a comment explicitly referencing the correctness property from the design document using the format: `# Feature: validation-pipeline, Property {number}: {property_text}`

### Visual Validation Tests

- Manually inspect sample visualizations for correctness
- Verify landmark overlays align with actual lip positions
- Verify side-by-side panels show correct correspondence
- Verify GIFs play smoothly and show all 29 frames

## Configuration

### Validation Thresholds (`validate/config.yaml`)

```yaml
# Shape validation
expected_frames: 29
expected_landmarks: 20
expected_roi_size: [96, 96]

# Detection quality tiers
detection_thresholds:
  good: 0.90
  ok: 0.80

# Temporal stability thresholds
temporal_stability:
  bbox_displacement_std_threshold: 10.0
  bbox_area_variance_threshold: 100.0
  landmark_motion_threshold: 15.0
  max_jump_threshold: 30.0

# ROI consistency thresholds
roi_validation:
  tight_threshold: 0.7
  loose_threshold: 1.5
  area_variance_threshold: 50.0

# Visualization settings
visualization:
  num_samples: 20
  random_seed: 42
  gif_fps: 25
  landmark_color: [0, 255, 0]
  bbox_color: [255, 0, 0]
  marker_size: 3

# Processing settings
processing:
  batch_size: 100
  num_workers: 4
  skip_existing: true

# Output settings
output:
  save_visualizations: true
  save_gifs: true
  save_reports: true
  debug_mode: false
```

## Performance Considerations

### Memory Management

- Process clips in batches to limit memory usage
- Release clip data after validation to free memory
- Use generators for iterating over large datasets

### Computation Optimization

- Vectorize metric computations using NumPy
- Parallelize clip processing using multiprocessing
- Cache expensive computations (e.g., distributions)

### I/O Optimization

- Batch write operations to reduce I/O overhead
- Use efficient image formats (PNG for lossless, JPEG for lossy)
- Compress GIFs to reduce file size

### Expected Performance

- **Validation speed**: 10-20 clips/second (single core)
- **Full LRW validation**: 2-5 hours (with 4 workers)
- **Memory usage**: < 2 GB peak
- **Storage**: ~1-2 GB for visualizations (20 samples)

## Deployment and Usage

### Installation

```bash
cd validate
pip install -r requirements.txt
```

### Basic Usage

```bash
# Validate entire dataset
python run_validation.py \
    --lrw_dataset /path/to/lrw_dataset \
    --preprocessed_dir ../output \
    --output_dir validation_results

# Validate specific split
python run_validation.py \
    --lrw_dataset /path/to/lrw_dataset \
    --preprocessed_dir ../output \
    --output_dir validation_results \
    --split train

# Validate with custom sample size
python run_validation.py \
    --lrw_dataset /path/to/lrw_dataset \
    --preprocessed_dir ../output \
    --output_dir validation_results \
    --num_samples 50 \
    --random_seed 123
```

### Output Structure

```
validation_results/
├── validation_summary.csv
├── failure_report.csv
├── summary_statistics.json
├── visualizations/
│   ├── clip_001_overlay.png
│   ├── clip_001_sidebyside.png
│   ├── clip_001_sequence.gif
│   └── ...
├── distributions/
│   ├── roi_size_distribution.png
│   ├── detection_rate_distribution.png
│   ├── temporal_stability_distribution.png
│   └── ...
└── logs/
    └── validation_YYYYMMDD_HHMMSS.log
```

## Integration with Preprocessing Pipeline

The validation pipeline is designed to work seamlessly with the existing preprocessing pipeline:

1. **Input compatibility**: Reads the exact output format produced by preprocessing
2. **Independent operation**: Runs separately without modifying preprocessing code
3. **Feedback loop**: Validation results can inform preprocessing parameter tuning
4. **Quality gates**: Can be used to filter clips before training

## Success Criteria

After validation, the system should provide clear answers to:

1. **Are shapes correct?** All clips pass shape validation
2. **Are detections successful?** >95% of clips are "good" or "ok" quality
3. **Are landmarks stable?** >90% of clips pass temporal stability checks
4. **Are ROIs consistent?** >95% of clips have consistent ROI sizes
5. **What are the failure modes?** Clear breakdown of failure types and counts
6. **Is data GPU-ready?** Overall PASS rate >95% with clear failure explanations

The validation pipeline enables confident statements like:

> "The preprocessing pipeline produces stable, accurate mouth ROIs and lip landmarks suitable for word-level and sentence-level Silent Speech Recognition models. Validation shows 97% PASS rate with remaining failures due to extreme head poses and occlusions, which are expected edge cases in the LRW dataset."
