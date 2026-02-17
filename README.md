# 🎙️ Silent Speech Recognition Preprocessing Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)

**A robust, CPU-based preprocessing pipeline for Silent Speech Recognition using the Lip Reading in the Wild (LRW) dataset**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Results](#-results)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Pipeline Stages](#-pipeline-stages)
- [Configuration](#-configuration)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)

---

## 🌟 Overview

Silent Speech Recognition (SSR) aims to recognize spoken words from visual information alone—specifically, lip movements—without audio. This project implements a **complete preprocessing pipeline** that transforms raw video clips into model-ready mouth region sequences for downstream deep learning tasks.

### 🎯 Project Goals

- Extract mouth regions of interest (ROIs) from video frames
- Detect and track facial landmarks across temporal sequences
- Apply temporal smoothing to reduce jitter
- Generate consistent, high-quality training data for SSR models
- Provide comprehensive validation and quality control

---

## ✨ Features

### Core Capabilities

- 🎥 **Robust Video Processing**: Handles various video formats (.mp4, .mpg) with error recovery
- 👤 **Accurate Face Detection**: MediaPipe Face Mesh or dlib for precise facial landmark detection
- 📍 **Exact Landmark Extraction**: Targets actual lip boundaries (upper and lower lips separated)
- 🎯 **ROI Computation**: Intelligent mouth region extraction based on exact lip boundaries
- 💾 **Structured Output**: Organized data format ready for PyTorch/TensorFlow
- ⚡ **Multiprocessing**: Parallel processing for faster throughput
- 🔍 **Quality Control**: Comprehensive validation and smoke testing
- 📊 **Detailed Logging**: Track processing statistics and failures

### Technical Highlights

- **CPU-Only**: No GPU required for preprocessing
- **Accurate Detection**: MediaPipe/dlib for precise lip boundary targeting
- **Production-Ready**: Tested on 1000+ videos with high success rate
- **Configurable**: YAML-based configuration for easy experimentation
- **Resumable**: Skip already processed videos automatically
- **Validated**: Comprehensive smoke tests and output verification

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Raw LRW Video (29 frames)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Frame Extraction│
                    └────────┬───────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Face Detection │
                    │ (MediaPipe/dlib)│
                    └────────┬───────┘
                             │
                             ▼
                    ┌────────────────┐
                    │   Landmark     │
                    │  Extraction    │
                    │ (Exact Boundaries)│
                    └────────┬───────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  Lip Landmark  │
                    │   Selection    │
                    │ (Upper & Lower)│
                    └────────┬───────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  Mouth ROI     │
                    │  Computation   │
                    │ (From Exact    │
                    │  Boundaries)   │
                    └────────┬───────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  Crop & Resize │
                    │   (96×96 RGB)  │
                    └────────┬───────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│              Preprocessed Output (Ready for Training)          │
│  • Mouth frames: 29 × 96×96 RGB images                        │
│  • Lip landmarks: 29 × N × 2 coordinates (exact boundaries)   │
│  • Metadata: Processing statistics & quality metrics          │
└────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB+ RAM
- 10GB+ disk space (for processed data)

### Step 1: Clone the Repository

```bash
git clone https://github.com/kushal511/Silent_Speech_Recognition_System.git
cd Silent_Speech_Recognition_System
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python3 verify_smoke_test_setup.py
```

Expected output:
```
✓ Python 3.x.x
✓ opencv-python
✓ numpy
✓ scipy
✓ PyYAML
✓ tqdm
✓ imageio
✓ SETUP COMPLETE
```

---

## ⚡ Quick Start

### 1. Test with Sample Data

Run the fast smoke test (30-60 seconds):

```bash
python3 run_fast_smoke_test.py test_lrw_dataset/data
```

### 2. Process Your Dataset

```bash
python3 run_preprocess.py \
    --input_dir lrw_dataset/data \
    --output_dir processed_lrw \
    --num_workers 4
```

### 3. Validate Outputs

```bash
python3 validate/run_validation.py \
    --data_dir processed_lrw \
    --output_dir validation_results
```

### 4. Load Preprocessed Data

```python
import numpy as np
from pathlib import Path

# Load mouth frames
frames_dir = Path("processed_lrw/WORD_CLASS/train/VIDEO_ID/frames")
frames = [np.array(Image.open(f)) for f in sorted(frames_dir.glob("*.png"))]

# Load landmarks
landmarks = np.load("processed_lrw/WORD_CLASS/train/VIDEO_ID/landmarks.npy")

print(f"Frames shape: {np.array(frames).shape}")  # (29, 96, 96, 3)
print(f"Landmarks shape: {landmarks.shape}")      # (29, 20, 2)
```

---

## 📖 Usage

### Basic Preprocessing

Process entire dataset with default settings:

```bash
python3 run_preprocess.py \
    --input_dir lrw_dataset/data \
    --output_dir processed_lrw
```

### Process Specific Split

Process only training data:

```bash
python3 run_preprocess.py \
    --input_dir lrw_dataset/data \
    --output_dir processed_lrw \
    --split train
```

### Debug Mode

Enable visualizations and detailed logging:

```bash
python3 run_preprocess.py \
    --input_dir lrw_dataset/data \
    --output_dir processed_lrw \
    --debug
```

### Custom Configuration

Use custom config file:

```bash
python3 run_preprocess.py \
    --input_dir lrw_dataset/data \
    --output_dir processed_lrw \
    --config custom_config.yaml
```

### Parallel Processing

Utilize multiple CPU cores:

```bash
python3 run_preprocess.py \
    --input_dir lrw_dataset/data \
    --output_dir processed_lrw \
    --num_workers 8
```

---

## 🔧 Pipeline Stages

### Stage 1: Video Loading
- Reads video files using OpenCV
- Extracts all frames as RGB arrays
- Validates frame count (expected: 29 frames)
- Handles corrupt/missing videos gracefully

### Stage 2: Face Detection & Landmark Extraction
- Uses MediaPipe Face Mesh (primary) or dlib (fallback)
- Detects faces and extracts precise facial landmarks
- Targets exact facial feature boundaries
- Provides high-quality landmark coordinates

### Stage 3: Lip Landmark Selection
- Extracts lip-specific landmarks from full face landmarks
- Separates upper and lower lip boundaries correctly
- MediaPipe: Uses specific indices for outer/inner lip contours
- dlib: Uses points 48-67 for complete lip region

### Stage 4: ROI Computation
- Calculates bounding box from exact lip boundary landmarks
- Adds 30% padding around mouth
- Enforces size constraints (64-128 pixels)
- Maintains square aspect ratio (1:1)

### Stage 5: Mouth Cropping
- Extracts mouth region from each frame
- Resizes to consistent 96×96 pixels
- Preserves RGB color information
- Handles edge cases (partial faces)

### Stage 6: Output Saving
- Saves frames as PNG images
- Saves landmarks as NumPy arrays (.npy)
- Saves metadata as JSON
- Organizes by word class and split

---

## ⚙️ Configuration

The project includes two configuration files:

### config.yaml (Testing/Demo)
For testing and demos with sample GRID data (s1 directory):
```yaml
dataset:
  video_dir: "s1"  # Flat structure for test data
  video_extension: ".mpg"
```

Use with:
```bash
python3 demo_multiple_frames.py
python3 run_smoke_test.py lrw_dataset/data
```

### config_lrw.yaml (Production)
For processing the complete LRW dataset:
```yaml
dataset:
  video_dir: null  # Hierarchical structure (WORD_CLASS/SPLIT/)
  video_extension: ".mp4"
```

Use with:
```bash
python3 run_preprocess.py \
    --input_dir /path/to/lrw \
    --output_dir processed_lrw \
    --config config_lrw.yaml
```

### Key Configuration Options

```yaml
# Face Detection (MediaPipe/dlib for accurate lip boundaries)
face_detection:
  confidence_threshold: 0.5  # Minimum detection confidence
  model_selection: 0         # MediaPipe model (0 or 1)
  
# Mouth ROI
mouth_roi:
  padding_factor: 0.3        # 30% padding around lips
  target_size: [96, 96]      # Output dimensions
  min_size: 64               # Minimum ROI size
  max_size: 128              # Maximum ROI size
  
# Processing
processing:
  num_workers: 4             # Parallel workers
  skip_existing: true        # Resume capability
  max_videos: null           # Limit for testing
```

**Note**: Temporal smoothing has been removed from the pipeline as it's not required. Each frame is processed independently.

---

## 📊 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Face Detection** | MediaPipe/dlib accurate detection |
| **Landmark Accuracy** | Targets exact lip boundaries |
| **Processing Speed** | 2-5 seconds/video |
| **Success Rate** | High (tested on 1000+ videos) |
| **Output Quality** | 96×96 RGB, no artifacts |
| **Memory Usage** | ~500 MB per worker |

### Output Format

```
processed_lrw/
├── WORD_CLASS_1/
│   ├── train/
│   │   ├── VIDEO_001/
│   │   │   ├── frames/
│   │   │   │   ├── frame_00.png  # 96×96 RGB
│   │   │   │   ├── frame_01.png
│   │   │   │   └── ... (29 frames)
│   │   │   ├── landmarks.npy      # (29, N, 2) - exact boundaries
│   │   │   └── metadata.json
│   │   └── VIDEO_002/
│   ├── val/
│   └── test/
└── WORD_CLASS_2/
    └── ...
```

### Quality Assurance

- ✅ All frames validated for correct dimensions
- ✅ Landmarks target exact upper and lower lip boundaries
- ✅ Landmarks checked for NaN/infinity values
- ✅ ROI boxes verified within frame bounds
- ✅ Visual inspection via debug images

---

## 📁 Project Structure

```
Silent_Speech_Recognition_System/
├── 📄 README.md                    # This file
├── 📄 LICENSE                      # MIT License
├── ⚙️ config.yaml                  # Configuration
├── 📋 requirements.txt             # Dependencies
│
├── 🐍 Python Scripts
│   ├── run_preprocess.py           # Main preprocessing pipeline
│   ├── run_smoke_test.py           # Comprehensive testing
│   ├── run_fast_smoke_test.py      # Quick validation
│   ├── verify_smoke_test_setup.py  # Dependency checker
│   ├── test_lrw_loading.py         # Dataset loading demo
│   └── example_usage.py            # Code examples
│
├── 📦 src/                         # Core modules
│   ├── __init__.py
│   ├── dataset.py                  # Dataset discovery
│   ├── video_io.py                 # Video loading
│   ├── face_landmarks.py           # Face detection
│   ├── mouth_roi.py                # ROI extraction
│   ├── smoothing.py                # Temporal smoothing
│   ├── save_utils.py               # Output saving
│   ├── visualize_debug.py          # Visualization
│   └── smoke_test_utils.py         # Testing utilities
│
└── 🔍 validate/                    # Validation pipeline
    ├── run_validation.py           # Main validator
    ├── validate_shapes.py          # Shape checking
    ├── validate_detection.py       # Detection quality
    ├── validate_temporal.py        # Temporal consistency
    ├── validate_roi.py             # ROI quality
    └── visualize_samples.py        # Visual QC
```

---


## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Kushal Adhyaru

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 🔗 Resources

- **LRW Dataset**: [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)
- **OpenCV Documentation**: [opencv.org](https://opencv.org/)
- **Python Documentation**: [python.org](https://www.python.org/)

---

---

## 🙏 Acknowledgments

- Oxford VGG for the LRW dataset
- OpenCV community for computer vision tools
- Python scientific computing community (NumPy, SciPy)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for the Silent Speech Recognition community

</div>
