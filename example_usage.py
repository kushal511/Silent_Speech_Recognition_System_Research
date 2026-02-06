"""
Example Usage Script

This script demonstrates how to use the preprocessing pipeline
and provides a simple test case for development.
"""

import numpy as np
from pathlib import Path

# Example 1: Process a single video programmatically
def example_single_video():
    """Example of processing a single video."""
    from src.video_io import VideoReader
    from src.face_landmarks import FaceLandmarkExtractor
    from src.mouth_roi import MouthROIExtractor
    from src.smoothing import TemporalSmoother, smooth_landmark_results, smooth_roi_results
    
    # Initialize components
    video_reader = VideoReader(expected_frames=29)
    landmark_extractor = FaceLandmarkExtractor(confidence_threshold=0.5)
    roi_extractor = MouthROIExtractor(target_size=(96, 96))
    smoother = TemporalSmoother(window_size=5, method='gaussian')
    
    # Load video
    video_path = "path/to/video.mp4"
    frames, metadata = video_reader.read_video(video_path)
    
    if frames is not None:
        print(f"Loaded {len(frames)} frames")
        
        # Extract landmarks
        landmark_results = landmark_extractor.process_video_frames(frames)
        
        # Apply smoothing
        landmark_results = smooth_landmark_results(landmark_results, smoother)
        
        # Extract mouth ROIs
        roi_results = roi_extractor.process_video_frames(frames, landmark_results)
        roi_results = smooth_roi_results(roi_results, smoother)
        
        # Get cropped mouths
        mouth_crops = []
        for frame, roi_result in zip(frames, roi_results):
            if roi_result['roi_box']:
                crop = roi_extractor.crop_mouth_region(frame, roi_result['roi_box'])
                mouth_crops.append(crop)
        
        print(f"Extracted {len(mouth_crops)} mouth crops")


# Example 2: Load preprocessed data for training
def example_load_preprocessed():
    """Example of loading preprocessed data."""
    output_dir = Path("output/ABOUT/train/ABOUT_00001")
    
    if output_dir.exists():
        # Load frames
        frames_dir = output_dir / "frames"
        frame_files = sorted(frames_dir.glob("*.png"))
        print(f"Found {len(frame_files)} frames")
        
        # Load landmarks
        landmarks = np.load(output_dir / "landmarks.npy")
        print(f"Landmarks shape: {landmarks.shape}")
        
        # Load metadata
        import json
        with open(output_dir / "metadata.json") as f:
            metadata = json.load(f)
        print(f"Metadata: {metadata['word_class']}, {metadata['num_frames']} frames")


# Example 3: Create a simple PyTorch dataset
def example_pytorch_dataset():
    """Example PyTorch dataset for preprocessed LRW data."""
    try:
        import torch
        from torch.utils.data import Dataset
        import cv2
        
        class LRWPreprocessedDataset(Dataset):
            def __init__(self, output_root, split='train', transform=None):
                self.output_root = Path(output_root)
                self.split = split
                self.transform = transform
                self.samples = self._load_samples()
            
            def _load_samples(self):
                samples = []
                for word_dir in self.output_root.iterdir():
                    if not word_dir.is_dir():
                        continue
                    
                    split_dir = word_dir / self.split
                    if not split_dir.exists():
                        continue
                    
                    for video_dir in split_dir.iterdir():
                        if (video_dir / "landmarks.npy").exists():
                            samples.append({
                                'video_dir': video_dir,
                                'word_class': word_dir.name
                            })
                
                return samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sample = self.samples[idx]
                video_dir = sample['video_dir']
                
                # Load frames
                frames_dir = video_dir / "frames"
                frame_files = sorted(frames_dir.glob("*.png"))
                
                frames = []
                for frame_file in frame_files:
                    frame = cv2.imread(str(frame_file))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                
                frames = np.array(frames)  # (T, H, W, C)
                
                # Load landmarks
                landmarks = np.load(video_dir / "landmarks.npy")
                
                # Convert to tensors
                frames_tensor = torch.from_numpy(frames).float() / 255.0
                landmarks_tensor = torch.from_numpy(landmarks).float()
                
                # Transpose to (C, T, H, W) for 3D convolutions
                frames_tensor = frames_tensor.permute(3, 0, 1, 2)
                
                return frames_tensor, landmarks_tensor, sample['word_class']
        
        # Usage
        dataset = LRWPreprocessedDataset("output", split="train")
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            frames, landmarks, label = dataset[0]
            print(f"Frames shape: {frames.shape}")
            print(f"Landmarks shape: {landmarks.shape}")
            print(f"Label: {label}")
    
    except ImportError:
        print("PyTorch not installed, skipping example")


if __name__ == "__main__":
    print("LRW Preprocessing Pipeline - Example Usage")
    print("=" * 60)
    
    print("\nExample 1: Single video processing")
    print("See example_single_video() function")
    
    print("\nExample 2: Load preprocessed data")
    print("See example_load_preprocessed() function")
    
    print("\nExample 3: PyTorch dataset")
    print("See example_pytorch_dataset() function")
    
    print("\n" + "=" * 60)
    print("To run the full pipeline, use:")
    print("python run_preprocess.py --input_dir /path/to/lrw --output_dir output")
