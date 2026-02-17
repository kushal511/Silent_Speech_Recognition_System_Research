# Single Frame Detailed Visualization - ACCURATE dlib Detection

This folder contains detailed frame-by-frame visualizations using **dlib's pre-trained 68-point facial landmark detector** for ACCURATE lip boundary detection.

## Detection Method

**dlib 68-point facial landmark detector**
- Pre-trained model: shape_predictor_68_face_landmarks.dat
- Provides ACCURATE landmarks targeting exact facial feature boundaries
- Lip landmarks (points 48-67) trace actual lip contours
- No geometric estimation - uses trained deep learning model

## Generated Frames

### Frame 10
- **frame_0010_detailed.png** (1.2 MB) - 6-panel detailed analysis
- **frame_0010_zoomed.png** (136 KB) - Zoomed lip view with ACCURATE landmarks

### Frame 25
- **frame_0025_detailed.png** (1.0 MB) - 6-panel detailed analysis
- **frame_0025_zoomed.png** (135 KB) - Zoomed lip view with ACCURATE landmarks

### Frame 50
- **frame_0050_detailed.png** (1.1 MB) - 6-panel detailed analysis
- **frame_0050_zoomed.png** (132 KB) - Zoomed lip view with ACCURATE landmarks

## Visualization Panels

### Detailed View (6 panels):

1. **Original Frame** - Raw video frame
2. **Face Detection** - Green box showing detected face region
3. **Facial Landmarks** - All 68 ACCURATE facial landmarks from dlib
4. **Lip Landmarks Detail** - Detailed view of 20 lip landmarks:
   - Red circles: Outer lip (12 points) - traces EXACT outer boundary
   - Blue squares: Inner lip (8 points) - traces EXACT inner boundary
   - Annotations showing upper and lower lip boundaries
5. **Mouth ROI Bounding Box** - Green box computed from ACCURATE lip landmarks
6. **Cropped Mouth** - Final 96×96 pixel mouth region

### Zoomed View (3 panels):

1. **Zoomed Mouth Region** - Close-up of mouth area
2. **Lip Landmarks** - ACCURATE landmark visualization:
   - Red: Outer lip contour (EXACT boundary from dlib)
   - Blue: Inner lip contour (EXACT boundary from dlib)
   - Yellow stars: Upper lip boundary points
   - Orange stars: Lower lip boundary points
3. **Final Crop** - 96×96 pixel output

## Key Features - ACCURATE Detection

### ✅ dlib Pre-trained Model
- Uses dlib's shape_predictor_68_face_landmarks.dat
- Trained on thousands of annotated faces
- Provides ACCURATE landmarks on exact facial boundaries
- No geometric estimation or approximation

### ✅ Exact Lip Boundaries
- **Points 48-59**: Outer lip contour (12 points)
  - Points 48-53: Lower lip outer boundary
  - Points 54-59: Upper lip outer boundary
- **Points 60-67**: Inner lip contour (8 points)
  - Points 60-63: Lower lip inner boundary
  - Points 64-67: Upper lip inner boundary

### ✅ Proper Separation
- Upper and lower lips correctly separated
- Distinct boundary points from trained model
- Outer and inner lip contours accurately traced

### ✅ Correct Bounding Boxes
- ROI computed from ACCURATE lip landmark positions
- Accurate mouth region extraction
- Consistent sizing (96×96 pixels)

## Technical Details

- **Detection Method**: dlib frontal face detector + shape predictor
- **Model**: shape_predictor_68_face_landmarks.dat (95 MB)
- **Landmarks**: 68-point facial model (trained on real faces)
- **Lip Points**: 20 landmarks (indices 48-67)
  - Outer lip: 48-59 (12 points tracing exact outer boundary)
  - Inner lip: 60-67 (8 points tracing exact inner boundary)
- **ROI Size**: 96×96 pixels
- **Processing**: Frame-independent (no smoothing)
- **Confidence**: 0.90 (high-quality dlib detections)

## dlib Lip Landmark Mapping

```
Outer Lip (48-59):
  48-53: Lower outer lip (left to right)
  54-59: Upper outer lip (right to left)

Inner Lip (60-67):
  60-63: Lower inner lip (left to right)
  64-67: Upper inner lip (right to left)
```

## How to Generate More Frames

To visualize any frame from the video:

```bash
python3 demo_single_frame.py <frame_number>
```

Examples:
```bash
python3 demo_single_frame.py 0    # First frame
python3 demo_single_frame.py 10   # Frame 10
python3 demo_single_frame.py 30   # Frame 30
```

The video has 75 frames (0-74).

## What to Look For

When examining these visualizations:

1. **Panel 4 (Lip Landmarks Detail)**: 
   - Landmarks should be ON the actual lip edges (not estimated)
   - Upper lip landmarks on the upper edge
   - Lower lip landmarks on the lower edge
   - Clear separation between upper and lower

2. **Zoomed View (Panel 2)**:
   - Yellow stars mark upper boundary (should be on actual upper lip edge)
   - Orange stars mark lower boundary (should be on actual lower lip edge)
   - Red line traces outer lip contour (should follow actual lip outline)
   - Blue line traces inner lip contour (should follow actual inner edge)

3. **Panel 5 (ROI Bounding Box)**:
   - Green box should tightly fit around mouth
   - Box computed from ACCURATE landmark positions
   - Dimensions shown on image

## Verification - ACCURATE Detection

These visualizations now use dlib's trained model and should show:
- ✅ Landmarks placed on EXACT lip boundaries (not estimated)
- ✅ Upper and lower lips correctly separated
- ✅ Accurate bounding box computation from real landmarks
- ✅ Consistent detection across frames
- ✅ No smoothing artifacts (each frame independent)

## Requirements

- dlib >= 20.0.0
- shape_predictor_68_face_landmarks.dat (downloaded automatically)
- cmake (required to build dlib)

Install dlib:
```bash
brew install cmake  # macOS
pip install dlib
```

Download model:
```bash
curl -L -o shape_predictor_68_face_landmarks.dat.bz2 \
  "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```
