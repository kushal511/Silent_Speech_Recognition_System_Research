# LRW Preprocessing Validation Pipeline

## Overview

This validation pipeline provides comprehensive quality control for Silent Speech Recognition (SSR) preprocessing outputs from the Lip Reading in the Wild (LRW) dataset.

The pipeline verifies:
- ✅ **Visual correctness** - Landmark overlays, side-by-side comparisons, animated GIFs
- ✅ **Numeric sanity** - Shape validation, detection rates, temporal stability
- ✅ **Dataset-wide consistency** - Distributions, outliers, failure modes
- ✅ **GPU-readiness** - Confirms data is ready for model training

## Installation

```bash
cd validate
pip install -r requirements.txt
```

## Quick Start

### 1. Download LRW Dataset (if needed)

The LRW dataset requires registration at Oxford VGG. After obtaining access:

```bash
# Verify existing dataset
python -m validate.download_lrw --output_dir /path/to/lrw_dataset --verify_only

# Or follow instructions to download manually
python -m validate.download_lrw --output_dir /path/to/lrw_dataset
```

### 2. Run Validation

```bash
python run_validation.py \
    --lrw_dataset /path/to/lrw_dataset \
    --preprocessed_dir ../output \
    --output_dir validation_results
```

### 3. Review Results

```bash
# View summary statistics
cat validation_results/summary_statistics.json

# Check validation summary
head validation_results/validation_summary.csv

# Review failures (if any)
cat validation_results/failure_report.csv
```

## Usage Examples

### Validate Specific Split

```bash
python run_validation.py \
    --lrw_dataset /path/to/lrw_dataset \
    --preprocessed_dir ../output \
    --output_dir validation_results \
    --split train
```

### Test with Small Sample

```bash
python run_validation.py \
    --lrw_dataset /path/to/lrw_dataset \
    --preprocessed_dir ../output \
    --output_dir validation_results \
    --max_clips 100 \
    --num_samples 10
```

### Custom Configuration

```bash
python run_validation.py \
    --lrw_dataset /path/to/lrw_dataset \
    --preprocessed_dir ../output \
    --output_dir validation_results \
    --config custom_config.yaml
```

## Output Structure

```
validation_results/
├── validation_summary.csv          # Per-clip metrics and status
├── failure_report.csv              # Clips with issues
├── summary_statistics.json         # Aggregate statistics
├── visualizations/                 # Sample visualizations
│   ├── CLIP_ID_frame00_overlay.png
│   ├── CLIP_ID_frame00_sidebyside.png
│   └── CLIP_ID_mouth_sequence.gif
└── logs/
    └── validation_YYYYMMDD_HHMMSS.log
```

## Validation Metrics

### Shape Validation
- Verifies mouth_frames shape (29, 96, 96, 3)
- Verifies lip_landmarks shape (29, 20, 2)
- Detects NaN/infinity values
- Checks bbox bounds

### Detection Quality
- Computes detection success rate
- Categorizes clips: good (≥90%), ok (80-90%), bad (<80%)

### Temporal Stability
- Measures bbox center displacement
- Computes bbox area variance
- Tracks landmark motion magnitude
- Detects jitter and drift

### ROI Consistency
- Checks ROI size ratio
- Detects over-tight cropping (<70% expected)
- Detects over-loose cropping (>150% expected)
- Measures area variance

## Interpreting Results

### PASS Status
- All validators passed
- Data is ready for GPU training
- No action needed

### WARN Status
- Some quality issues detected
- Review specific metrics
- Consider reprocessing if many warnings

### FAIL Status
- Critical issues found
- Review failure_report.csv
- Reprocess affected clips

### Pass Rate Threshold
- **≥95% PASS**: Data is GPU-ready ✓
- **<95% PASS**: Review failures before training

## Configuration

Edit `config.yaml` to customize:

```yaml
# Detection quality tiers
detection_thresholds:
  good: 0.90
  ok: 0.80

# Temporal stability thresholds
temporal_stability:
  bbox_displacement_std_threshold: 10.0
  landmark_motion_threshold: 15.0

# Visualization settings
visualization:
  num_samples: 20
  random_seed: 42
  
# Output settings
output:
  save_visualizations: true
  save_gifs: true
```

## Troubleshooting

### Dataset Not Found
```
Error: LRW dataset not found

Solution:
1. Download LRW dataset from Oxford VGG
2. Run: python -m validate.download_lrw --output_dir /path/to/lrw
3. Or place dataset manually in expected structure
```

### Preprocessed Data Missing
```
Error: Preprocessed data not found

Solution:
Run preprocessing pipeline first:
python run_preprocess.py --input_dir /path/to/lrw --output_dir output
```

### Low Pass Rate
```
Warning: Pass rate < 95%

Solution:
1. Review failure_report.csv for specific issues
2. Check preprocessing parameters
3. Reprocess clips with issues
4. Adjust validation thresholds if needed
```

## Performance

- **Processing speed**: 10-20 clips/second (single core)
- **Full LRW validation**: 2-5 hours (with 4 workers)
- **Memory usage**: <2 GB peak
- **Storage**: ~1-2 GB for visualizations (20 samples)

## Architecture

The validation pipeline consists of:

1. **Data Loader** - Loads original videos + preprocessed outputs
2. **Validators** - Shape, detection, temporal, ROI checks
3. **Visual Validator** - Landmark overlays, side-by-side panels
4. **GIF Generator** - Animated sequences for temporal review
5. **Failure Analyzer** - Detects specific failure modes
6. **Report Generator** - CSV summaries and statistics

## Documentation

See `.kiro/specs/validation-pipeline/` for:
- **requirements.md** - Complete requirements specification
- **design.md** - Architecture and design decisions
- **tasks.md** - Implementation task breakdown

## Citation

If you use this validation pipeline in your research, please cite the LRW dataset:

```bibtex
@inproceedings{chung2017lip,
  title={Lip reading in the wild},
  author={Chung, Joon Son and Zisserman, Andrew},
  booktitle={Asian Conference on Computer Vision},
  pages={87--103},
  year={2017},
  organization={Springer}
}
```

## License

MIT License - See LICENSE file for details.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the specification documents
3. Check validation logs in `validation_results/logs/`

---

**The validation pipeline ensures your preprocessing outputs are correct, stable, and ready for GPU-based SSR model training.**
