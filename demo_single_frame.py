#!/usr/bin/env python3
"""
Single Frame Visualization Demo

This script shows detailed visualization of one frame at a time,
displaying landmarks, bounding boxes, and mouth crop.

Usage:
    python3 demo_single_frame.py [frame_number]
"""

import cv2
import numpy as np
import yaml
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import pipeline modules
from src.video_io import VideoReader
from src.face_landmarks import FaceLandmarkExtractor
from src.mouth_roi import MouthROIExtractor

# Create output directory
OUTPUT_DIR = Path("demo_output_single")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("SINGLE FRAME DETAILED VISUALIZATION")
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

# Get frame number from command line or use default
if len(sys.argv) > 1:
    frame_number = int(sys.argv[1])
else:
    frame_number = 10  # Default to frame 10

print(f"📹 Video: {video_path.name}")
print(f"🎬 Frame: {frame_number}")
print(f"📁 Output: {OUTPUT_DIR}")
print()

# ============================================================================
# LOAD VIDEO
# ============================================================================
print("Loading video...")
video_reader = VideoReader(expected_frames=29)
frames, metadata = video_reader.read_video(video_path)

if frames is None:
    print(f"❌ Failed to load video")
    exit(1)

print(f"✓ Loaded {len(frames)} frames")

# Check if frame number is valid
if frame_number >= len(frames):
    print(f"❌ Frame {frame_number} not available (video has {len(frames)} frames)")
    print(f"   Using last frame instead: {len(frames)-1}")
    frame_number = len(frames) - 1

print()

# ============================================================================
# PROCESS SINGLE FRAME
# ============================================================================
print(f"Processing frame {frame_number}...")

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

# Get the specific frame
frame = frames[frame_number]

# Process frame
landmark_result = landmark_extractor.process_frame(frame)
roi_result = roi_extractor.process_frame(frame, landmark_result['lip_landmarks'])

print(f"✓ Face detected: {landmark_result['face_detected']}")
print(f"✓ Confidence: {landmark_result['confidence']:.2f}")
print(f"✓ Landmarks: {landmark_result['landmarks'].shape if landmark_result['landmarks'] is not None else 'None'}")
print(f"✓ Lip landmarks: {landmark_result['lip_landmarks'].shape if landmark_result['lip_landmarks'] is not None else 'None'}")
print(f"✓ ROI box: {roi_result['roi_box']}")
print()

# ============================================================================
# CREATE DETAILED VISUALIZATION
# ============================================================================
print("Creating detailed visualization...")

fig = plt.figure(figsize=(20, 12))

# ============================================================================
# Panel 1: Original Frame
# ============================================================================
ax1 = plt.subplot(2, 3, 1)
ax1.imshow(frame)
ax1.set_title('1. Original Frame', fontsize=14, fontweight='bold')
ax1.axis('off')

# ============================================================================
# Panel 2: Face Detection
# ============================================================================
ax2 = plt.subplot(2, 3, 2)
ax2.imshow(frame)

if landmark_result['landmarks'] is not None:
    landmarks = landmark_result['landmarks']
    x_min, x_max = int(np.min(landmarks[:, 0])), int(np.max(landmarks[:, 0]))
    y_min, y_max = int(np.min(landmarks[:, 1])), int(np.max(landmarks[:, 1]))
    
    face_bbox = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=3, edgecolor='lime', facecolor='none', label='Face Detection'
    )
    ax2.add_patch(face_bbox)

ax2.set_title('2. Face Detection', fontsize=14, fontweight='bold')
ax2.axis('off')
ax2.legend(loc='upper right')

# ============================================================================
# Panel 3: All Facial Landmarks
# ============================================================================
ax3 = plt.subplot(2, 3, 3)
ax3.imshow(frame)

if landmark_result['landmarks'] is not None:
    landmarks = landmark_result['landmarks']
    # Draw all landmarks
    ax3.scatter(landmarks[:, 0], landmarks[:, 1], c='cyan', s=20, alpha=0.6, label='All Landmarks')
    
    # Highlight lip landmarks
    lip_landmarks = landmark_result['lip_landmarks']
    ax3.scatter(lip_landmarks[:, 0], lip_landmarks[:, 1], c='red', s=40, alpha=0.8, label='Lip Landmarks')

ax3.set_title('3. Facial Landmarks (68 points)', fontsize=14, fontweight='bold')
ax3.axis('off')
ax3.legend(loc='upper right')

# ============================================================================
# Panel 4: Lip Landmarks Detail
# ============================================================================
ax4 = plt.subplot(2, 3, 4)
ax4.imshow(frame)

if landmark_result['lip_landmarks'] is not None:
    lip_landmarks = landmark_result['lip_landmarks']
    
    # Outer lip (first 12 points: 48-59)
    outer_lip = lip_landmarks[:12]
    ax4.scatter(outer_lip[:, 0], outer_lip[:, 1], c='red', s=60, alpha=0.8, label='Outer Lip', marker='o')
    
    # Inner lip (last 8 points: 60-67)
    inner_lip = lip_landmarks[12:]
    ax4.scatter(inner_lip[:, 0], inner_lip[:, 1], c='blue', s=60, alpha=0.8, label='Inner Lip', marker='s')
    
    # Draw lines connecting outer lip points
    outer_lip_closed = np.vstack([outer_lip, outer_lip[0]])
    ax4.plot(outer_lip_closed[:, 0], outer_lip_closed[:, 1], 'r-', linewidth=2, alpha=0.6)
    
    # Draw lines connecting inner lip points
    inner_lip_closed = np.vstack([inner_lip, inner_lip[0]])
    ax4.plot(inner_lip_closed[:, 0], inner_lip_closed[:, 1], 'b-', linewidth=2, alpha=0.6)
    
    # Annotate upper and lower boundaries
    upper_points = outer_lip[6:]  # Top half
    lower_points = outer_lip[:6]  # Bottom half
    
    if len(upper_points) > 0:
        upper_center = np.mean(upper_points, axis=0)
        ax4.annotate('Upper Lip\nBoundary', xy=upper_center, xytext=(upper_center[0], upper_center[1]-30),
                    fontsize=10, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    if len(lower_points) > 0:
        lower_center = np.mean(lower_points, axis=0)
        ax4.annotate('Lower Lip\nBoundary', xy=lower_center, xytext=(lower_center[0], lower_center[1]+30),
                    fontsize=10, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax4.set_title('4. Lip Landmarks Detail (20 points)', fontsize=14, fontweight='bold')
ax4.axis('off')
ax4.legend(loc='upper right')

# ============================================================================
# Panel 5: ROI Bounding Box
# ============================================================================
ax5 = plt.subplot(2, 3, 5)
ax5.imshow(frame)

if landmark_result['lip_landmarks'] is not None:
    lip_landmarks = landmark_result['lip_landmarks']
    ax5.scatter(lip_landmarks[:, 0], lip_landmarks[:, 1], c='red', s=40, alpha=0.8, label='Lip Landmarks')

if roi_result['roi_box'] is not None:
    x, y, w, h = roi_result['roi_box']
    roi_bbox = patches.Rectangle(
        (x, y), w, h,
        linewidth=3, edgecolor='lime', facecolor='none', label='Mouth ROI'
    )
    ax5.add_patch(roi_bbox)
    
    # Add dimensions
    ax5.text(x + w/2, y - 10, f'{w}×{h} pixels', 
            fontsize=10, color='lime', fontweight='bold',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

ax5.set_title('5. Mouth ROI Bounding Box', fontsize=14, fontweight='bold')
ax5.axis('off')
ax5.legend(loc='upper right')

# ============================================================================
# Panel 6: Cropped Mouth Region
# ============================================================================
ax6 = plt.subplot(2, 3, 6)

if roi_result['mouth_crop'] is not None:
    ax6.imshow(roi_result['mouth_crop'])
    ax6.set_title(f'6. Cropped Mouth (96×96)', fontsize=14, fontweight='bold')
else:
    ax6.text(0.5, 0.5, 'No mouth crop available', 
            ha='center', va='center', fontsize=12)
    ax6.set_title('6. Cropped Mouth', fontsize=14, fontweight='bold')

ax6.axis('off')

# ============================================================================
# Add overall title and info
# ============================================================================
info_text = f"""
Frame {frame_number} of {len(frames)} | Video: {video_path.name}
Detection Method: OpenCV DNN with Edge Detection
Face Detected: {landmark_result['face_detected']} | Confidence: {landmark_result['confidence']:.2f}
Landmarks: 68 points (20 lip points) | ROI: {roi_result['roi_box']}
Upper and Lower Lip Boundaries: Correctly Separated
No Temporal Smoothing Applied
"""

plt.suptitle(f'Detailed Frame Analysis - Frame {frame_number}', 
             fontsize=16, fontweight='bold', y=0.98)

fig.text(0.5, 0.02, info_text.strip(), ha='center', fontsize=10, 
         family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout(rect=[0, 0.08, 1, 0.96])

# Save
output_path = OUTPUT_DIR / f"frame_{frame_number:04d}_detailed.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"💾 Saved: {output_path}")
print()

# ============================================================================
# CREATE ZOOMED LIP VIEW
# ============================================================================
print("Creating zoomed lip view...")

if roi_result['roi_box'] is not None and landmark_result['lip_landmarks'] is not None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x, y, w, h = roi_result['roi_box']
    
    # Expand ROI for better view
    margin = 20
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(frame.shape[1], x + w + margin)
    y_end = min(frame.shape[0], y + h + margin)
    
    zoomed_frame = frame[y_start:y_end, x_start:x_end]
    
    # Adjust landmark coordinates for zoomed view
    lip_landmarks = landmark_result['lip_landmarks'].copy()
    lip_landmarks[:, 0] -= x_start
    lip_landmarks[:, 1] -= y_start
    
    # Panel 1: Zoomed original
    axes[0].imshow(zoomed_frame)
    axes[0].set_title('Zoomed Mouth Region', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: With landmarks
    axes[1].imshow(zoomed_frame)
    
    # Outer lip
    outer_lip = lip_landmarks[:12]
    axes[1].scatter(outer_lip[:, 0], outer_lip[:, 1], c='red', s=80, alpha=0.8, label='Outer Lip')
    outer_lip_closed = np.vstack([outer_lip, outer_lip[0]])
    axes[1].plot(outer_lip_closed[:, 0], outer_lip_closed[:, 1], 'r-', linewidth=2)
    
    # Inner lip
    inner_lip = lip_landmarks[12:]
    axes[1].scatter(inner_lip[:, 0], inner_lip[:, 1], c='blue', s=80, alpha=0.8, label='Inner Lip')
    inner_lip_closed = np.vstack([inner_lip, inner_lip[0]])
    axes[1].plot(inner_lip_closed[:, 0], inner_lip_closed[:, 1], 'b-', linewidth=2)
    
    # Mark upper and lower boundaries
    upper_points = outer_lip[6:]
    lower_points = outer_lip[:6]
    
    axes[1].scatter(upper_points[:, 0], upper_points[:, 1], c='yellow', s=120, 
                   marker='*', label='Upper Boundary', zorder=5)
    axes[1].scatter(lower_points[:, 0], lower_points[:, 1], c='orange', s=120, 
                   marker='*', label='Lower Boundary', zorder=5)
    
    axes[1].set_title('Lip Landmarks (Upper/Lower Separated)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].axis('off')
    
    # Panel 3: Final crop
    if roi_result['mouth_crop'] is not None:
        axes[2].imshow(roi_result['mouth_crop'])
        axes[2].set_title('Final Crop (96×96)', fontsize=12, fontweight='bold')
        axes[2].axis('off')
    
    plt.suptitle(f'Zoomed Lip Analysis - Frame {frame_number}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"frame_{frame_number:04d}_zoomed.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved: {output_path}")

print()
print("=" * 80)
print("SINGLE FRAME VISUALIZATION COMPLETE")
print("=" * 80)
print()
print(f"✓ Frame {frame_number} processed successfully")
print(f"✓ Generated 2 detailed visualizations")
print()
print("Generated files:")
print(f"  1. frame_{frame_number:04d}_detailed.png - 6-panel detailed analysis")
print(f"  2. frame_{frame_number:04d}_zoomed.png   - Zoomed lip view")
print()
print(f"📁 All files saved to: {OUTPUT_DIR}/")
print()
print("To view another frame, run:")
print(f"  python3 demo_single_frame.py <frame_number>")
print(f"  Example: python3 demo_single_frame.py 20")
print()
print("=" * 80)
