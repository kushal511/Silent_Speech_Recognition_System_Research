# Requirements Document

## Introduction

This document specifies the requirements for a comprehensive validation and quality control (QC) pipeline for Silent Speech Recognition (SSR) preprocessing outputs. The validation system will verify that preprocessed LRW dataset outputs (mouth ROI frames, lip landmarks, and metadata) are correct, stable, and suitable for downstream GPU-based model training.

The validation pipeline operates in read-only mode, analyzing saved preprocessing artifacts without modification. It provides automated, reproducible quality assessment at visual, numeric, and dataset-wide levels.

## Glossary

- **SSR**: Silent Speech Recognition - recognizing speech from visual lip movements only
- **LRW**: Lip Reading in the Wild dataset - 500-word vocabulary dataset with 29-frame clips
- **Mouth ROI**: Region of Interest containing the mouth area, cropped from original frames
- **Lip Landmarks**: 20 key points around the mouth contour (x, y coordinates)
- **Temporal Stability**: Consistency of measurements across consecutive video frames
- **Detection Success Rate**: Ratio of frames with successful face/landmark detection
- **Validation System**: The complete QC pipeline that verifies preprocessing correctness
- **Quality Tier**: Classification of clips as good/ok/bad based on quality metrics
- **Failure Mode**: Specific type of preprocessing error (e.g., landmark drift, wrong face)

## Requirements

### Requirement 1

**User Story:** As a research engineer, I want to visually verify that mouth ROIs are correctly extracted, so that I can confirm the preprocessing pipeline is working as intended.

#### Acceptance Criteria

1. WHEN the validation system processes a clip THEN the system SHALL overlay lip landmarks on original frames with visible markers
2. WHEN landmarks are overlaid THEN the system SHALL draw the mouth bounding box on the original frame
3. WHEN generating visual outputs THEN the system SHALL create side-by-side comparison panels showing original frame with annotations and cropped mouth ROI
4. WHEN creating temporal visualizations THEN the system SHALL generate animated sequences (GIF or MP4) showing all 29 mouth frames
5. WHEN sampling clips for visualization THEN the system SHALL use deterministic random sampling with configurable seed

### Requirement 2

**User Story:** As a research engineer, I want automated numeric validation of preprocessing outputs, so that I can detect errors without manual inspection of every clip.

#### Acceptance Criteria

1. WHEN validating a clip THEN the system SHALL verify mouth_frames shape equals (29, H, W, C)
2. WHEN validating a clip THEN the system SHALL verify lip_landmarks shape equals (29, K, 2)
3. WHEN checking numeric validity THEN the system SHALL detect and flag any NaN or infinity values
4. WHEN validating bounding boxes THEN the system SHALL verify all bbox coordinates lie within frame bounds
5. WHEN a shape or type invariant fails THEN the system SHALL mark the clip as HARD FAIL and log the specific violation

### Requirement 3

**User Story:** As a research engineer, I want to measure detection success rates per clip, so that I can identify low-quality samples that may need reprocessing or exclusion.

#### Acceptance Criteria

1. WHEN computing detection metrics THEN the system SHALL calculate success_ratio as valid_frames divided by 29
2. WHEN categorizing clip quality THEN the system SHALL classify clips with success_ratio >= 0.90 as "good"
3. WHEN categorizing clip quality THEN the system SHALL classify clips with success_ratio between 0.80 and 0.90 as "ok"
4. WHEN categorizing clip quality THEN the system SHALL classify clips with success_ratio < 0.80 as "bad"
5. WHEN detection rate is below threshold THEN the system SHALL flag the clip for manual review

### Requirement 4

**User Story:** As a research engineer, I want to measure temporal stability of landmarks and bounding boxes, so that I can detect jitter, drift, or sudden jumps that indicate preprocessing errors.

#### Acceptance Criteria

1. WHEN computing temporal metrics THEN the system SHALL calculate mean and standard deviation of bbox center displacement across frames
2. WHEN computing temporal metrics THEN the system SHALL calculate variance of bbox area across frames
3. WHEN computing temporal metrics THEN the system SHALL calculate per-landmark motion magnitude across consecutive frames
4. WHEN detecting instability THEN the system SHALL flag clips with bbox center displacement std > threshold
5. WHEN detecting instability THEN the system SHALL flag clips with sudden landmark jumps exceeding motion threshold

### Requirement 5

**User Story:** As a research engineer, I want dataset-wide distribution analysis, so that I can understand the overall quality and identify systematic issues.

#### Acceptance Criteria

1. WHEN analyzing the dataset THEN the system SHALL generate histograms of mouth ROI width and height distributions
2. WHEN analyzing the dataset THEN the system SHALL generate histograms of detection success ratio distributions
3. WHEN analyzing the dataset THEN the system SHALL generate histograms of bbox motion variance distributions
4. WHEN analyzing the dataset THEN the system SHALL generate histograms of landmark motion statistics
5. WHEN distributions are computed THEN the system SHALL identify and report outliers in the tail regions

### Requirement 6

**User Story:** As a research engineer, I want automatic detection of specific failure modes, so that I can understand why certain clips fail and improve the preprocessing pipeline.

#### Acceptance Criteria

1. WHEN analyzing failures THEN the system SHALL detect and count wrong face selection errors
2. WHEN analyzing failures THEN the system SHALL detect and count landmark drift errors
3. WHEN analyzing failures THEN the system SHALL detect and count over-tight or over-loose ROI errors
4. WHEN analyzing failures THEN the system SHALL detect and count jitter caused by insufficient smoothing
5. WHEN a failure mode is detected THEN the system SHALL log the failure type, clip ID, and diagnostic information

### Requirement 7

**User Story:** As a research engineer, I want structured validation reports, so that I can review results systematically and share findings with my advisor.

#### Acceptance Criteria

1. WHEN validation completes THEN the system SHALL save a validation_summary.csv with per-clip metrics
2. WHEN validation completes THEN the system SHALL save a failure_report.csv with all flagged clips and failure reasons
3. WHEN validation completes THEN the system SHALL save debug visualizations to a structured output directory
4. WHEN generating reports THEN the system SHALL include clear PASS/WARN/FAIL status for each clip
5. WHEN generating reports THEN the system SHALL compute and display dataset-wide aggregate statistics

### Requirement 8

**User Story:** As a research engineer, I want a command-line interface for validation, so that I can easily run validation on different datasets and configure parameters.

#### Acceptance Criteria

1. WHEN running validation THEN the system SHALL accept lrw_dataset parameter specifying original LRW video location
2. WHEN running validation THEN the system SHALL accept preprocessed_dir parameter specifying preprocessed data location
3. WHEN running validation THEN the system SHALL accept output_dir parameter specifying where to save validation results
4. WHEN running validation THEN the system SHALL accept num_samples parameter to control visualization sample size
5. WHEN running validation THEN the system SHALL accept random_seed parameter for deterministic sampling
6. WHEN validation starts THEN the system SHALL log configuration parameters and dataset statistics
7. WHEN validation starts THEN the system SHALL verify that original videos exist for preprocessed clips

### Requirement 9

**User Story:** As a research engineer, I want validation to be CPU-only and read-only, so that it can run on any machine without modifying preprocessing outputs.

#### Acceptance Criteria

1. WHEN validation runs THEN the system SHALL NOT require GPU acceleration
2. WHEN validation runs THEN the system SHALL NOT modify any preprocessing output files
3. WHEN validation runs THEN the system SHALL only read from the preprocessed data directory
4. WHEN validation runs THEN the system SHALL write outputs only to the designated validation output directory
5. WHEN validation completes THEN the system SHALL leave all input data unchanged

### Requirement 10

**User Story:** As a research engineer, I want clear documentation on validation interpretation, so that I can confidently determine if preprocessing outputs are ready for GPU training.

#### Acceptance Criteria

1. WHEN documentation is provided THEN the system SHALL explain how to run validation with example commands
2. WHEN documentation is provided THEN the system SHALL explain how to interpret validation metrics and thresholds
3. WHEN documentation is provided THEN the system SHALL explain what each failure mode indicates
4. WHEN documentation is provided THEN the system SHALL provide clear criteria for "GPU-ready" status
5. WHEN documentation is provided THEN the system SHALL include troubleshooting guidance for common issues

### Requirement 11

**User Story:** As a research engineer, I want to validate landmark accuracy by visual overlay, so that I can confirm landmarks align with actual lip positions.

#### Acceptance Criteria

1. WHEN overlaying landmarks THEN the system SHALL draw each of the 20 lip landmarks as distinct colored points
2. WHEN overlaying landmarks THEN the system SHALL connect landmarks to show outer and inner lip contours
3. WHEN overlaying landmarks THEN the system SHALL use sufficient marker size for visibility at original frame resolution
4. WHEN generating overlays THEN the system SHALL preserve original frame colors and contrast
5. WHEN landmarks are misaligned THEN the system SHALL flag the clip for manual review based on landmark spread metrics

### Requirement 12

**User Story:** As a research engineer, I want to detect multi-face frames, so that I can identify cases where the wrong face was selected during preprocessing.

#### Acceptance Criteria

1. WHEN analyzing metadata THEN the system SHALL check if multiple faces were detected in any frame
2. WHEN multiple faces are detected THEN the system SHALL flag the clip as potential wrong-face-selection
3. WHEN wrong face is suspected THEN the system SHALL include the clip in the failure report with diagnostic info
4. WHEN generating visualizations for multi-face clips THEN the system SHALL highlight the selected face region
5. WHEN counting failure modes THEN the system SHALL report the total number of multi-face cases

### Requirement 13

**User Story:** As a research engineer, I want to measure ROI size consistency, so that I can detect over-tight or over-loose cropping.

#### Acceptance Criteria

1. WHEN validating ROI size THEN the system SHALL compute the ratio of ROI area to expected area
2. WHEN ROI is too tight THEN the system SHALL flag clips where ROI area < 0.7 * expected_area
3. WHEN ROI is too loose THEN the system SHALL flag clips where ROI area > 1.5 * expected_area
4. WHEN ROI size varies excessively THEN the system SHALL flag clips with ROI area variance > threshold
5. WHEN generating reports THEN the system SHALL include ROI size statistics in the summary

### Requirement 14

**User Story:** As a research engineer, I want batch processing with progress tracking, so that I can validate large datasets efficiently.

#### Acceptance Criteria

1. WHEN processing multiple clips THEN the system SHALL display a progress bar showing completion percentage
2. WHEN processing multiple clips THEN the system SHALL log processing speed (clips per second)
3. WHEN processing multiple clips THEN the system SHALL support resumption by skipping already-validated clips
4. WHEN processing multiple clips THEN the system SHALL handle errors gracefully and continue with remaining clips
5. WHEN batch processing completes THEN the system SHALL report total time and throughput statistics

### Requirement 15

**User Story:** As a research engineer, I want configurable quality thresholds, so that I can adjust validation strictness based on my research requirements.

#### Acceptance Criteria

1. WHEN validation runs THEN the system SHALL load quality thresholds from a configuration file
2. WHEN thresholds are configurable THEN the system SHALL allow customization of detection_success_threshold
3. WHEN thresholds are configurable THEN the system SHALL allow customization of temporal_stability_threshold
4. WHEN thresholds are configurable THEN the system SHALL allow customization of landmark_motion_threshold
5. WHEN thresholds are changed THEN the system SHALL log the active threshold values in the validation report

### Requirement 16

**User Story:** As a research engineer, I want an automated script to download the LRW dataset, so that I can obtain the original videos without manual download steps.

#### Acceptance Criteria

1. WHEN the download script runs THEN the system SHALL download LRW dataset files from the official source
2. WHEN downloading THEN the system SHALL verify file integrity using checksums or file sizes
3. WHEN downloading THEN the system SHALL display progress bars showing download status
4. WHEN download completes THEN the system SHALL extract video files to the specified directory
5. WHEN download fails THEN the system SHALL provide clear error messages and retry options
6. WHEN dataset already exists THEN the system SHALL skip downloading and verify existing files
7. WHEN running validation without dataset THEN the system SHALL detect missing dataset and prompt user to run download script
