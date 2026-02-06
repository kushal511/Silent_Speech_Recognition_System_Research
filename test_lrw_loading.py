#!/usr/bin/env python3
"""
Test LRW Dataset Loading

This script demonstrates that the LRW dataset can be loaded and processed
successfully through the complete pipeline.
"""

from src.dataset import LRWDataset
from src.video_io import VideoReader
from src.face_landmarks import FaceLandmarkExtractor
from src.mouth_roi import MouthROIExtractor
import numpy as np

def test_dataset_loading():
    """Test loading LRW dataset."""
    print("=" * 80)
    print("LRW DATASET LOADING TEST")
    print("=" * 80)
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = LRWDataset(
        root_dir='lrw_dataset/data',
        video_dir='s1',
        video_extension='.mpg'
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} videos")
    print()
    
    # Show first few samples
    print("First 5 videos:")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"  {i+1}. {sample.video_id} - {sample.video_path.name}")
    
    print()
    return dataset


def test_video_loading(dataset):
    """Test loading individual videos."""
    print("Testing video loading...")
    print()
    
    reader = VideoReader()
    
    # Test first video
    sample = dataset[0]
    frames, metadata = reader.read_video(sample.video_path)
    
    if frames is not None:
        print(f"✓ Video loaded: {sample.video_id}")
        print(f"  Frames: {len(frames)}")
        print(f"  Shape: {frames[0].shape}")
        print(f"  FPS: {metadata['fps']:.2f}")
        print(f"  Resolution: {metadata['resolution']}")
        print()
        return frames
    else:
        print(f"✗ Failed to load video: {metadata.get('error')}")
        return None


def test_pipeline(frames):
    """Test complete preprocessing pipeline."""
    print("Testing preprocessing pipeline...")
    print()
    
    # Initialize components
    landmark_extractor = FaceLandmarkExtractor(confidence_threshold=0.5)
    roi_extractor = MouthROIExtractor(target_size=(96, 96))
    
    # Extract landmarks
    print("  [1/3] Extracting landmarks...")
    landmark_results = landmark_extractor.process_video_frames(frames)
    face_rate = sum(1 for r in landmark_results if r['face_detected']) / len(landmark_results)
    print(f"        ✓ Face detection rate: {face_rate:.1%}")
    
    # Extract ROIs
    print("  [2/3] Computing mouth ROIs...")
    roi_results = roi_extractor.process_video_frames(frames, landmark_results)
    roi_rate = sum(1 for r in roi_results if r['success']) / len(roi_results)
    print(f"        ✓ ROI extraction rate: {roi_rate:.1%}")
    
    # Extract mouth crops
    print("  [3/3] Extracting mouth crops...")
    mouth_crops = []
    for frame, roi_result in zip(frames, roi_results):
        if roi_result['roi_box'] is not None:
            crop = roi_extractor.crop_mouth_region(frame, roi_result['roi_box'])
            if crop is not None:
                mouth_crops.append(crop)
    
    print(f"        ✓ Mouth crops: {len(mouth_crops)}/{len(frames)}")
    
    if mouth_crops:
        crops_array = np.array(mouth_crops)
        print(f"        ✓ Output shape: {crops_array.shape}")
    
    print()
    return mouth_crops


def main():
    """Main test function."""
    # Test dataset loading
    dataset = test_dataset_loading()
    
    # Test video loading
    frames = test_video_loading(dataset)
    
    if frames is None:
        print("✗ Test failed at video loading stage")
        return
    
    # Test pipeline
    mouth_crops = test_pipeline(frames)
    
    if mouth_crops:
        print("=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        print()
        print("The LRW dataset can be loaded and processed successfully!")
        print()
        print("Next steps:")
        print("  1. Run smoke test:")
        print("     python3 run_fast_smoke_test.py lrw_dataset/data")
        print()
        print("  2. Run full preprocessing:")
        print("     python3 run_preprocess.py \\")
        print("         --input_dir lrw_dataset/data \\")
        print("         --output_dir processed_lrw \\")
        print("         --num_workers 4")
        print()
    else:
        print("✗ Test failed at pipeline stage")


if __name__ == '__main__':
    main()
