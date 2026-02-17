#!/usr/bin/env python3
"""
Multi-Frame Pipeline Visualization Demo

This script demonstrates the pipeline on MULTIPLE frames from the video,
showing accurate lip boundary detection and ROI computation across the entire sequence.

Features:
- Uses MediaPipe/dlib for accurate facial landmark detection
- Targets exact upper and lower lip boundaries
- No temporal smoothing (each frame processed independently)
- Visualizes bounding boxes and mouth crops

Usage:
    python3 demo_multiple_frames.py
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import pipeline modules
from src.video_io import VideoReader
from src.face_landmarks import FaceLandmarkExtractor
from src.mouth_roi import MouthROIExtractor

# Create output directory
OUTPUT_DIR = Path("demo_output_multiple")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("MULTI-FRAME PIPELINE DEMONSTRATION")
print("Using MediaPipe/dlib for Accurate Lip Boundary Detection")
print("=" * 80)
print()

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Select video
VIDEO_PATH = "lrw_dataset/data/s1/prwq3s.mpg"
video_path = Path(VIDEO_PATH)

if not video_path.exists():
    print(f"❌ Video not found: {VIDEO_PATH}")
    exit(1)

print(f"📹 Video: {video_path.name}")
print(f"📁 Output: {OUTPUT_DIR}")
print()

# ============================================================================
# LOAD ALL FRAMES
# ============================================================================
print("Loading video...")
video_reader = VideoReader(expected_frames=29)
frames, metadata = video_reader.read_video(video_path)

if frames is None:
    print(f"❌ Failed to load video")
    exit(1)

print(f"✓ Loaded {len(frames)} frames")
print()

# ============================================================================
# PROCESS ALL FRAMES
# ============================================================================
print("Processing all frames through pipeline...")
print("• Using accurate landmark detection (MediaPipe/dlib)")
print("• Targeting exact lip boundaries")
print("• No temporal smoothing applied")
print()
landmark_extractor = FaceLandmarkExtractor(
    confidence_threshold=config['face_detection']['confidence_threshold'],
    model_selection=config['face_detection']['model_selection'],
    lip_indices=config['landmarks']['lip_indices_mediapipe']
)

roi_extractor = MouthROIExtractor(
    padding_factor=config['mouth_roi']['padding_factor'],
    min_size=config['mouth_roi']['min_size'],
    max_size=config['mouth_roi']['max_size'],
    target_size=tuple(config['mouth_roi']['target_size']),
    aspect_ratio=config['mouth_roi']['aspect_ratio']
)

# Process all frames
landmark_results = landmark_extractor.process_video_frames(frames)
roi_results = roi_extractor.process_video_frames(frames, landmark_results)

# Extract mouth crops
mouth_crops = []
for frame, roi_result in zip(frames, roi_results):
    if roi_result['roi_box'] is not None:
        crop = roi_extractor.crop_mouth_region(frame, roi_result['roi_box'])
        mouth_crops.append(crop)
    else:
        mouth_crops.append(None)

valid_crops = [c for c in mouth_crops if c is not None]

print(f"✓ Processed {len(frames)} frames")
print(f"✓ Valid mouth crops: {len(valid_crops)}/{len(frames)}")
print()

# ============================================================================
# VISUALIZATION 1: GRID OF FRAMES WITH BOUNDING BOXES
# ============================================================================
print("Creating multi-frame visualization with bounding boxes...")

# Select 12 frames evenly distributed
num_samples = 12
frame_indices = np.linspace(0, len(frames)-1, num_samples, dtype=int)

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

for idx, frame_idx in enumerate(frame_indices):
    ax = axes[idx]
    frame = frames[frame_idx]
    
    # Display frame
    ax.imshow(frame)
    
    # Draw face bounding box (if landmarks available)
    if landmark_results[frame_idx]['landmarks'] is not None:
        landmarks = landmark_results[frame_idx]['landmarks']
        x_min, x_max = int(np.min(landmarks[:, 0])), int(np.max(landmarks[:, 0]))
        y_min, y_max = int(np.min(landmarks[:, 1])), int(np.max(landmarks[:, 1]))
        
        face_bbox = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        ax.add_patch(face_bbox)
    
    # Draw ROI bounding box
    if roi_results[frame_idx]['roi_box'] is not None:
        x, y, w, h = roi_results[frame_idx]['roi_box']
        roi_bbox = patches.Rectangle(
            (x, y), w, h,
            linewidth=3, edgecolor='red', facecolor='none'
        )
        ax.add_patch(roi_bbox)
    
    ax.set_title(f'Frame {frame_idx}', fontsize=12, fontweight='bold')
    ax.axis('off')

plt.suptitle(f'Pipeline Processing: {num_samples} Sample Frames\nGreen = Face Detection | Red = Mouth ROI', 
             fontsize=16, fontweight='bold')
plt.tight_layout()

output_path = OUTPUT_DIR / "multi_frame_bounding_boxes.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"💾 Saved: {output_path}")
print()

# ============================================================================
# VISUALIZATION 2: TEMPORAL SEQUENCE OF MOUTH CROPS
# ============================================================================
print("Creating temporal sequence of mouth crops...")

# Show 16 mouth crops
num_crops = 16
crop_indices = np.linspace(0, len(valid_crops)-1, num_crops, dtype=int)

fig, axes = plt.subplots(2, 8, figsize=(24, 6))
axes = axes.flatten()

for idx, crop_idx in enumerate(crop_indices):
    ax = axes[idx]
    if crop_idx < len(valid_crops) and valid_crops[crop_idx] is not None:
        ax.imshow(valid_crops[crop_idx])
        ax.set_title(f'Frame {crop_idx}', fontsize=10)
    ax.axis('off')

plt.suptitle('Temporal Sequence: 16 Mouth Crops (96×96 pixels)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()

output_path = OUTPUT_DIR / "temporal_mouth_sequence.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"💾 Saved: {output_path}")
print()

# ============================================================================
# VISUALIZATION 3: SIDE-BY-SIDE COMPARISON (6 FRAMES)
# ============================================================================
print("Creating side-by-side comparison...")

num_comparisons = 6
comparison_indices = np.linspace(0, len(frames)-1, num_comparisons, dtype=int)

fig, axes = plt.subplots(num_comparisons, 2, figsize=(12, 18))

for row, frame_idx in enumerate(comparison_indices):
    # Left: Original with bounding boxes
    axes[row, 0].imshow(frames[frame_idx])
    
    # Draw landmarks
    if landmark_results[frame_idx]['lip_landmarks'] is not None:
        lip_landmarks = landmark_results[frame_idx]['lip_landmarks']
        axes[row, 0].scatter(lip_landmarks[:, 0], lip_landmarks[:, 1], 
                           c='red', s=30, alpha=0.8)
    
    # Draw ROI box
    if roi_results[frame_idx]['roi_box'] is not None:
        x, y, w, h = roi_results[frame_idx]['roi_box']
        roi_bbox = patches.Rectangle(
            (x, y), w, h,
            linewidth=3, edgecolor='lime', facecolor='none'
        )
        axes[row, 0].add_patch(roi_bbox)
    
    axes[row, 0].set_title(f'Frame {frame_idx}: Original + ROI', fontsize=11, fontweight='bold')
    axes[row, 0].axis('off')
    
    # Right: Cropped mouth
    if frame_idx < len(valid_crops) and valid_crops[frame_idx] is not None:
        axes[row, 1].imshow(valid_crops[frame_idx])
        axes[row, 1].set_title(f'Frame {frame_idx}: Mouth Crop (96×96)', fontsize=11, fontweight='bold')
    axes[row, 1].axis('off')

plt.suptitle('Side-by-Side Comparison: Original vs Cropped', 
             fontsize=16, fontweight='bold')
plt.tight_layout()

output_path = OUTPUT_DIR / "side_by_side_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"💾 Saved: {output_path}")
print()

# ============================================================================
# VISUALIZATION 4: ROI TRACKING ACROSS ALL FRAMES
# ============================================================================
print("Creating ROI tracking visualization...")

# Show middle frame with all ROI boxes overlaid
middle_frame_idx = len(frames) // 2
middle_frame = frames[middle_frame_idx]

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.imshow(middle_frame)

# Draw ROI boxes from multiple frames (every 5th frame)
colors = plt.cm.rainbow(np.linspace(0, 1, len(frames)//5))
for idx, frame_idx in enumerate(range(0, len(frames), 5)):
    if roi_results[frame_idx]['roi_box'] is not None:
        x, y, w, h = roi_results[frame_idx]['roi_box']
        roi_bbox = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor=colors[idx], facecolor='none', alpha=0.6
        )
        ax.add_patch(roi_bbox)

ax.set_title('ROI Tracking Across All Frames\n(Each colored box = different frame)', 
             fontsize=16, fontweight='bold')
ax.axis('off')

output_path = OUTPUT_DIR / "roi_tracking_all_frames.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"💾 Saved: {output_path}")
print()

# ============================================================================
# VISUALIZATION 5: STATISTICS SUMMARY
# ============================================================================
print("Creating statistics summary...")

face_detection_rate = sum(1 for r in landmark_results if r['face_detected']) / len(landmark_results)
roi_success_rate = sum(1 for r in roi_results if r['success']) / len(roi_results)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top-left: First frame with annotations
axes[0, 0].imshow(frames[0])
if roi_results[0]['roi_box'] is not None:
    x, y, w, h = roi_results[0]['roi_box']
    roi_bbox = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='lime', facecolor='none')
    axes[0, 0].add_patch(roi_bbox)
axes[0, 0].set_title('First Frame', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

# Top-right: Middle frame with annotations
axes[0, 1].imshow(frames[middle_frame_idx])
if roi_results[middle_frame_idx]['roi_box'] is not None:
    x, y, w, h = roi_results[middle_frame_idx]['roi_box']
    roi_bbox = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='lime', facecolor='none')
    axes[0, 1].add_patch(roi_bbox)
axes[0, 1].set_title('Middle Frame', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

# Bottom-left: Last frame with annotations
axes[1, 0].imshow(frames[-1])
if roi_results[-1]['roi_box'] is not None:
    x, y, w, h = roi_results[-1]['roi_box']
    roi_bbox = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='lime', facecolor='none')
    axes[1, 0].add_patch(roi_bbox)
axes[1, 0].set_title('Last Frame', fontsize=14, fontweight='bold')
axes[1, 0].axis('off')

# Bottom-right: Statistics
axes[1, 1].axis('off')
stats_text = f"""
PROCESSING STATISTICS

Video: {video_path.name}
Total Frames: {len(frames)}

Detection Method:
  • MediaPipe Face Mesh / dlib
  • Targets exact lip boundaries
  • No temporal smoothing

Face Detection:
  • Detected: {sum(1 for r in landmark_results if r['face_detected'])}/{len(frames)}
  • Rate: {face_detection_rate:.1%}

Landmark Extraction:
  • Accurate boundary detection
  • Upper & lower lips separated
  • Success rate: {face_detection_rate:.1%}

ROI Computation:
  • Based on exact lip boundaries
  • Valid ROIs: {sum(1 for r in roi_results if r['success'])}/{len(frames)}
  • Success rate: {roi_success_rate:.1%}

Mouth Crops:
  • Valid crops: {len(valid_crops)}/{len(frames)}
  • Output size: 96×96 RGB
  • Success rate: {len(valid_crops)/len(frames):.1%}

Overall Success: {len(valid_crops)/len(frames):.1%}
"""

axes[1, 1].text(0.1, 0.5, stats_text, fontsize=13, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.suptitle('Processing Statistics: First, Middle, and Last Frames', 
             fontsize=16, fontweight='bold')
plt.tight_layout()

output_path = OUTPUT_DIR / "statistics_summary.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"💾 Saved: {output_path}")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 80)
print("MULTI-FRAME DEMONSTRATION COMPLETE")
print("=" * 80)
print()
print(f"✓ Processed ALL {len(frames)} frames independently")
print(f"✓ Used accurate lip boundary detection")
print(f"✓ Generated 5 comprehensive visualizations")
print()
print("Generated files:")
print(f"  1. multi_frame_bounding_boxes.png    - 12 frames with bounding boxes")
print(f"  2. temporal_mouth_sequence.png       - 16 mouth crops in sequence")
print(f"  3. side_by_side_comparison.png       - 6 frames: original vs cropped")
print(f"  4. roi_tracking_all_frames.png       - ROI tracking across all frames")
print(f"  5. statistics_summary.png            - Complete statistics")
print()
print(f"📁 All files saved to: {OUTPUT_DIR}/")
print()
print("=" * 80)
print("Key Features:")
print("• Accurate landmark detection targeting exact lip boundaries")
print("• Upper and lower lips correctly separated")
print("• No temporal smoothing - each frame independent")
print("=" * 80)
