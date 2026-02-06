"""
Smoke Test Utilities for LRW Preprocessing Pipeline

This module provides utilities for comprehensive smoke testing of the
preprocessing pipeline on manually downloaded LRW dataset subsets.
"""

import json
import csv
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

# Import pipeline modules
from src.video_io import VideoReader
from src.face_landmarks import FaceLandmarkExtractor, interpolate_missing_landmarks
from src.mouth_roi import MouthROIExtractor, fill_missing_roi_boxes
from src.smoothing import TemporalSmoother, smooth_landmark_results, smooth_roi_results
from src.save_utils import OutputSaver

logger = logging.getLogger(__name__)


# ============================================================================
# CONSOLE OUTPUT FORMATTING
# ============================================================================

def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_success(message: str):
    """Print a success message."""
    print(f"✓ {message}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"⚠ {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"✗ {message}")


# ============================================================================
# DATASET INSPECTOR
# ============================================================================

class DatasetInspector:
    """
    Inspects dataset structure and validates video files.
    
    Verifies:
    - Directory structure
    - Video file accessibility
    - File formats
    - Basic video properties
    """
    
    def __init__(self, data_root: str):
        """Initialize dataset inspector."""
        self.data_root = Path(data_root)
        self.videos = []
        self.issues = []
    
    def inspect(self) -> Dict:
        """
        Perform comprehensive dataset inspection.
        
        Returns:
            Dictionary with inspection results
        """
        result = {
            'valid': True,
            'issues': [],
            'num_words': 0,
            'total_videos': 0,
            'videos': [],
            'extensions': set(),
            'unreadable_videos': []
        }
        
        # Check if root exists
        if not self.data_root.exists():
            result['valid'] = False
            result['issues'].append(f"Data root does not exist: {self.data_root}")
            return result
        
        if not self.data_root.is_dir():
            result['valid'] = False
            result['issues'].append(f"Data root is not a directory: {self.data_root}")
            return result
        
        # Find word class directories
        word_dirs = [d for d in self.data_root.iterdir() if d.is_dir()]
        
        if len(word_dirs) == 0:
            result['valid'] = False
            result['issues'].append("No word class directories found")
            return result
        
        result['num_words'] = len(word_dirs)
        
        print(f"Found {len(word_dirs)} word class directories")
        
        # Inspect each word directory
        for word_dir in sorted(word_dirs):
            word_class = word_dir.name
            print(f"  Inspecting: {word_class}")
            
            # Look for videos in word directory and subdirectories
            video_files = self._find_videos(word_dir)
            
            if len(video_files) == 0:
                result['issues'].append(f"No videos found in {word_class}")
                continue
            
            print(f"    Found {len(video_files)} videos")
            
            # Test each video
            for video_path in video_files:
                video_info = self._inspect_video(video_path, word_class)
                result['videos'].append(video_info)
                result['extensions'].add(video_path.suffix)
                
                if not video_info['readable']:
                    result['unreadable_videos'].append(str(video_path))
                    result['issues'].append(f"Unreadable video: {video_path}")
        
        result['total_videos'] = len(result['videos'])
        result['extensions'] = list(result['extensions'])
        
        # Check if we have any readable videos
        readable_count = sum(1 for v in result['videos'] if v['readable'])
        
        if readable_count == 0:
            result['valid'] = False
            result['issues'].append("No readable videos found")
        
        if len(result['unreadable_videos']) > 0:
            result['valid'] = False
        
        self.videos = result['videos']
        self.issues = result['issues']
        
        return result
    
    def _find_videos(self, directory: Path) -> List[Path]:
        """Find all video files in directory and subdirectories."""
        video_extensions = ['.mp4', '.mpg', '.avi', '.mov', '.mkv']
        videos = []
        
        for ext in video_extensions:
            videos.extend(directory.rglob(f"*{ext}"))
        
        return videos
    
    def _inspect_video(self, video_path: Path, word_class: str) -> Dict:
        """Inspect a single video file."""
        info = {
            'path': str(video_path),
            'word_class': word_class,
            'readable': False,
            'num_frames': 0,
            'resolution': (0, 0),
            'fps': 0.0,
            'error': None
        }
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                info['error'] = "Failed to open video"
                return info
            
            info['readable'] = True
            info['num_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info['fps'] = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info['resolution'] = (height, width)
            
            cap.release()
            
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    def select_test_videos(self, max_words: int = 3, 
                          max_videos_per_word: int = 3) -> List[Dict]:
        """
        Select a subset of videos for testing.
        
        Args:
            max_words: Maximum number of word classes
            max_videos_per_word: Maximum videos per word class
        
        Returns:
            List of selected video info dictionaries
        """
        # Group videos by word class
        videos_by_word = {}
        for video in self.videos:
            if not video['readable']:
                continue
            
            word = video['word_class']
            if word not in videos_by_word:
                videos_by_word[word] = []
            videos_by_word[word].append(video)
        
        # Select videos
        selected = []
        for word in sorted(videos_by_word.keys())[:max_words]:
            selected.extend(videos_by_word[word][:max_videos_per_word])
        
        return selected
    
    def save_report(self, output_path: Path):
        """Save inspection report to JSON."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_root': str(self.data_root),
            'num_videos': len(self.videos),
            'issues': self.issues,
            'videos': self.videos
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved inspection report to {output_path}")


# ============================================================================
# SINGLE VIDEO TESTER
# ============================================================================

class SingleVideoTester:
    """
    Tests the complete pipeline on a single video with detailed logging.
    """
    
    def __init__(self, config: Dict, debug_dir: Path):
        """Initialize single video tester."""
        self.config = config
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline components
        self.video_reader = VideoReader(
            expected_frames=config['video']['expected_frames']
        )
        
        self.landmark_extractor = FaceLandmarkExtractor(
            confidence_threshold=config['face_detection']['confidence_threshold'],
            model_selection=config['face_detection']['model_selection'],
            lip_indices=config['landmarks']['lip_indices_mediapipe']
        )
        
        self.roi_extractor = MouthROIExtractor(
            padding_factor=config['mouth_roi']['padding_factor'],
            min_size=config['mouth_roi']['min_size'],
            max_size=config['mouth_roi']['max_size'],
            target_size=tuple(config['mouth_roi']['target_size']),
            aspect_ratio=config['mouth_roi']['aspect_ratio']
        )
    
    def test_video(self, video_path: str) -> Dict:
        """
        Test complete pipeline on a single video.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with test results
        """
        result = {
            'success': False,
            'error': None,
            'num_frames': 0,
            'face_detection_rate': 0.0,
            'num_valid_crops': 0
        }
        
        video_path = Path(video_path)
        video_id = video_path.stem
        
        try:
            # Step 1: Load video
            print("  [1/6] Loading video...")
            frames, metadata = self.video_reader.read_video(video_path)
            
            if frames is None:
                result['error'] = f"Failed to load video: {metadata.get('error')}"
                return result
            
            result['num_frames'] = len(frames)
            print(f"        ✓ Loaded {len(frames)} frames")
            print(f"        ✓ Resolution: {frames.shape[1]}x{frames.shape[2]}")
            
            # Save first frame
            self._save_debug_image(frames[0], f"{video_id}_01_original_frame.png")
            
            # Step 2: Extract landmarks
            print("  [2/6] Extracting face landmarks...")
            landmark_results = self.landmark_extractor.process_video_frames(frames)
            
            face_detection_rate = sum(1 for r in landmark_results if r['face_detected']) / len(landmark_results)
            result['face_detection_rate'] = face_detection_rate
            
            print(f"        ✓ Face detection rate: {face_detection_rate:.1%}")
            
            # Interpolate missing landmarks
            landmark_results = interpolate_missing_landmarks(landmark_results)
            
            # Save frame with landmarks
            if landmark_results[0]['landmarks'] is not None:
                frame_with_landmarks = self._draw_landmarks(
                    frames[0], 
                    landmark_results[0]['landmarks']
                )
                self._save_debug_image(
                    frame_with_landmarks, 
                    f"{video_id}_02_landmarks.png"
                )
            
            # Step 3: Compute mouth ROIs
            print("  [3/6] Computing mouth ROIs...")
            roi_results = self.roi_extractor.process_video_frames(frames, landmark_results)
            
            # Fill missing ROI boxes
            roi_results = fill_missing_roi_boxes(roi_results)
            
            num_valid_rois = sum(1 for r in roi_results if r['success'])
            print(f"        ✓ Valid ROIs: {num_valid_rois}/{len(roi_results)}")
            
            # Save frame with ROI box
            if roi_results[0]['roi_box'] is not None:
                frame_with_roi = self._draw_roi_box(
                    frames[0],
                    roi_results[0]['roi_box']
                )
                self._save_debug_image(
                    frame_with_roi,
                    f"{video_id}_03_roi_box.png"
                )
            
            # Step 4: Extract mouth crops
            print("  [4/6] Extracting mouth crops...")
            mouth_crops = []
            for frame, roi_result in zip(frames, roi_results):
                if roi_result['roi_box'] is not None:
                    crop = self.roi_extractor.crop_mouth_region(frame, roi_result['roi_box'])
                    mouth_crops.append(crop)
                else:
                    mouth_crops.append(None)
            
            valid_crops = [c for c in mouth_crops if c is not None]
            result['num_valid_crops'] = len(valid_crops)
            
            print(f"        ✓ Valid crops: {len(valid_crops)}/{len(mouth_crops)}")
            
            # Save first mouth crop
            if valid_crops:
                self._save_debug_image(
                    valid_crops[0],
                    f"{video_id}_04_mouth_crop.png"
                )
            
            # Step 5: Create comparison visualization
            print("  [5/6] Creating visualizations...")
            if valid_crops:
                comparison = self._create_comparison(
                    frames[0],
                    landmark_results[0],
                    roi_results[0],
                    valid_crops[0]
                )
                self._save_debug_image(
                    comparison,
                    f"{video_id}_05_comparison.png"
                )
            
            # Step 6: Create mouth sequence GIF
            print("  [6/6] Creating mouth sequence GIF...")
            if len(valid_crops) >= 10:
                self._create_mouth_gif(
                    valid_crops,
                    self.debug_dir / f"{video_id}_06_mouth_sequence.gif"
                )
            
            result['success'] = True
            print_success("Single video test completed successfully!")
            
        except Exception as e:
            result['error'] = f"Exception: {str(e)}"
            logger.error(f"Error testing video: {e}", exc_info=True)
        
        return result
    
    def _save_debug_image(self, image: np.ndarray, filename: str):
        """Save debug image."""
        output_path = self.debug_dir / filename
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), image_bgr)
    
    def _draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Draw landmarks on frame."""
        annotated = frame.copy()
        for point in landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)
        return annotated
    
    def _draw_roi_box(self, frame: np.ndarray, roi_box: Tuple[int, int, int, int]) -> np.ndarray:
        """Draw ROI box on frame."""
        annotated = frame.copy()
        x, y, w, h = roi_box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return annotated
    
    def _create_comparison(self, frame: np.ndarray, landmark_result: Dict,
                          roi_result: Dict, mouth_crop: np.ndarray) -> np.ndarray:
        """Create side-by-side comparison."""
        # Annotated frame
        annotated = frame.copy()
        if landmark_result['landmarks'] is not None:
            annotated = self._draw_landmarks(annotated, landmark_result['landmarks'])
        if roi_result['roi_box'] is not None:
            annotated = self._draw_roi_box(annotated, roi_result['roi_box'])
        
        # Resize mouth crop to match frame height
        crop_resized = cv2.resize(mouth_crop, (frame.shape[0], frame.shape[0]))
        
        # Concatenate
        comparison = np.hstack([annotated, crop_resized])
        return comparison
    
    def _create_mouth_gif(self, mouth_crops: List[np.ndarray], output_path: Path):
        """Create GIF from mouth crops."""
        try:
            import imageio
            
            # Convert to uint8 if needed
            crops_uint8 = [crop.astype(np.uint8) if crop.dtype != np.uint8 else crop 
                          for crop in mouth_crops if crop is not None]
            
            # Save as GIF
            imageio.mimsave(str(output_path), crops_uint8, duration=0.04)  # 25 fps
            
        except ImportError:
            logger.warning("imageio not installed, skipping GIF creation")
        except Exception as e:
            logger.error(f"Error creating GIF: {e}")


# ============================================================================
# SMALL BATCH TESTER
# ============================================================================

class SmallBatchTester:
    """
    Tests the pipeline on a small batch of videos.
    """
    
    def __init__(self, config: Dict, output_dir: str):
        """Initialize batch tester."""
        self.config = config
        self.output_dir = Path(output_dir)
        
        # Initialize pipeline components
        self.video_reader = VideoReader(
            expected_frames=config['video']['expected_frames']
        )
        
        self.landmark_extractor = FaceLandmarkExtractor(
            confidence_threshold=config['face_detection']['confidence_threshold'],
            model_selection=config['face_detection']['model_selection'],
            lip_indices=config['landmarks']['lip_indices_mediapipe']
        )
        
        self.roi_extractor = MouthROIExtractor(
            padding_factor=config['mouth_roi']['padding_factor'],
            min_size=config['mouth_roi']['min_size'],
            max_size=config['mouth_roi']['max_size'],
            target_size=tuple(config['mouth_roi']['target_size']),
            aspect_ratio=config['mouth_roi']['aspect_ratio']
        )
        
        self.output_saver = OutputSaver(
            output_root=str(self.output_dir),
            frame_format=config['video']['frame_format']
        )
    
    def test_batch(self, videos: List[Dict]) -> List[Dict]:
        """
        Test pipeline on batch of videos.
        
        Args:
            videos: List of video info dictionaries
        
        Returns:
            List of processing results
        """
        results = []
        
        for video_info in tqdm(videos, desc="Processing videos"):
            result = self._process_video(video_info)
            results.append(result)
        
        return results
    
    def _process_video(self, video_info: Dict) -> Dict:
        """Process a single video."""
        result = {
            'video_path': video_info['path'],
            'word_class': video_info['word_class'],
            'success': False,
            'error': None
        }
        
        try:
            video_path = Path(video_info['path'])
            video_id = video_path.stem
            
            # Load video
            frames, metadata = self.video_reader.read_video(video_path)
            
            if frames is None:
                result['error'] = metadata.get('error', 'Failed to read video')
                return result
            
            # Extract landmarks
            landmark_results = self.landmark_extractor.process_video_frames(frames)
            landmark_results = interpolate_missing_landmarks(landmark_results)
            
            # Compute ROIs
            roi_results = self.roi_extractor.process_video_frames(frames, landmark_results)
            roi_results = fill_missing_roi_boxes(roi_results)
            
            # Extract mouth crops
            mouth_crops = []
            for frame, roi_result in zip(frames, roi_results):
                if roi_result['roi_box'] is not None:
                    crop = self.roi_extractor.crop_mouth_region(frame, roi_result['roi_box'])
                    mouth_crops.append(crop)
                else:
                    mouth_crops.append(None)
            
            # Check if we have valid crops
            valid_crops = [c for c in mouth_crops if c is not None]
            if len(valid_crops) == 0:
                result['error'] = "No valid mouth crops extracted"
                return result
            
            # Prepare landmarks array
            lip_landmarks_list = [r['lip_landmarks'] for r in landmark_results]
            lip_landmarks_array = np.array(lip_landmarks_list)
            
            # Create video sample object for saving
            from src.dataset import VideoSample
            video_sample = VideoSample(
                video_path=video_path,
                word_class=video_info['word_class'],
                split='test',  # Use 'test' for smoke test
                video_id=video_id
            )
            
            # Save outputs
            processing_info = {
                'face_detection_rate': sum(1 for r in landmark_results if r['face_detected']) / len(landmark_results),
                'num_valid_crops': len(valid_crops),
                'smoothing_applied': False,
                'roi_boxes': [r['roi_box'] for r in roi_results],
                'crop_size': self.config['mouth_roi']['target_size']
            }
            
            save_success = self.output_saver.save_video_output(
                video_sample,
                mouth_crops,
                lip_landmarks_array,
                processing_info
            )
            
            if not save_success:
                result['error'] = "Failed to save outputs"
                return result
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = f"Exception: {str(e)}"
            logger.error(f"Error processing {video_info['path']}: {e}", exc_info=True)
        
        return result


# ============================================================================
# OUTPUT VERIFIER
# ============================================================================

class OutputVerifier:
    """
    Verifies that saved outputs meet quality standards.
    """
    
    def __init__(self, output_dir: str, config: Dict):
        """Initialize output verifier."""
        self.output_dir = Path(output_dir)
        self.config = config
        self.expected_frames = config['video']['expected_frames']
        self.target_size = tuple(config['mouth_roi']['target_size'])
    
    def verify_all_outputs(self, batch_results: List[Dict]) -> List[Dict]:
        """
        Verify all outputs from batch processing.
        
        Args:
            batch_results: List of batch processing results
        
        Returns:
            List of verification results
        """
        verification_results = []
        
        for result in batch_results:
            if not result['success']:
                continue
            
            video_path = Path(result['video_path'])
            video_id = video_path.stem
            word_class = result['word_class']
            
            verification = self._verify_video_output(word_class, video_id)
            verification_results.append(verification)
        
        return verification_results
    
    def _verify_video_output(self, word_class: str, video_id: str) -> Dict:
        """Verify output for a single video."""
        result = {
            'video_id': video_id,
            'word_class': word_class,
            'valid': True,
            'issues': []
        }
        
        output_dir = self.output_dir / word_class / 'test' / video_id
        
        # Check file existence
        if not output_dir.exists():
            result['valid'] = False
            result['issues'].append("Output directory does not exist")
            return result
        
        frames_dir = output_dir / 'frames'
        landmarks_file = output_dir / 'landmarks.npy'
        metadata_file = output_dir / 'metadata.json'
        
        if not frames_dir.exists():
            result['valid'] = False
            result['issues'].append("Frames directory missing")
        
        if not landmarks_file.exists():
            result['valid'] = False
            result['issues'].append("Landmarks file missing")
        
        if not metadata_file.exists():
            result['valid'] = False
            result['issues'].append("Metadata file missing")
        
        if not result['valid']:
            return result
        
        # Verify frame count
        frame_files = list(frames_dir.glob('*.png'))
        if len(frame_files) == 0:
            result['valid'] = False
            result['issues'].append("No frame files found")
        
        # Verify landmarks shape
        try:
            landmarks = np.load(landmarks_file)
            
            if landmarks.ndim != 3:
                result['valid'] = False
                result['issues'].append(f"Invalid landmarks shape: {landmarks.shape}")
            
            if landmarks.shape[0] != self.expected_frames:
                result['issues'].append(
                    f"Frame count mismatch: expected {self.expected_frames}, got {landmarks.shape[0]}"
                )
            
            # Check for NaNs
            if np.any(np.isnan(landmarks)):
                result['valid'] = False
                result['issues'].append("Landmarks contain NaN values")
            
            # Check for infs
            if np.any(np.isinf(landmarks)):
                result['valid'] = False
                result['issues'].append("Landmarks contain infinite values")
            
        except Exception as e:
            result['valid'] = False
            result['issues'].append(f"Error loading landmarks: {e}")
        
        # Verify frame dimensions
        try:
            first_frame_path = sorted(frame_files)[0]
            frame = cv2.imread(str(first_frame_path))
            
            if frame is None:
                result['valid'] = False
                result['issues'].append("Failed to load frame")
            else:
                h, w = frame.shape[:2]
                expected_h, expected_w = self.target_size
                
                if (h, w) != (expected_h, expected_w):
                    result['valid'] = False
                    result['issues'].append(
                        f"Frame size mismatch: expected {expected_h}x{expected_w}, got {h}x{w}"
                    )
        
        except Exception as e:
            result['valid'] = False
            result['issues'].append(f"Error verifying frames: {e}")
        
        return result


# ============================================================================
# VISUAL CONFIRMATION
# ============================================================================

class VisualConfirmation:
    """
    Creates visual confirmations for human inspection.
    """
    
    def __init__(self, output_dir: str, debug_dir: str):
        """Initialize visual confirmation generator."""
        self.output_dir = Path(output_dir)
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
    
    def create_video_visualization(self, video_path: str, word_class: str, video_id: str):
        """
        Create comprehensive visualization for a processed video.
        
        Args:
            video_path: Path to original video
            word_class: Word class name
            video_id: Video identifier
        """
        # Load original video
        video_reader = VideoReader()
        frames, _ = video_reader.read_video(video_path)
        
        if frames is None:
            logger.warning(f"Could not load original video: {video_path}")
            return
        
        # Load processed outputs
        output_dir = self.output_dir / word_class / 'test' / video_id
        
        if not output_dir.exists():
            logger.warning(f"Output directory not found: {output_dir}")
            return
        
        # Load mouth crops
        frames_dir = output_dir / 'frames'
        frame_files = sorted(frames_dir.glob('*.png'))
        
        mouth_crops = []
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mouth_crops.append(frame_rgb)
        
        if len(mouth_crops) == 0:
            logger.warning(f"No mouth crops found for {video_id}")
            return
        
        # Create grid visualization (first 5 frames)
        num_samples = min(5, len(mouth_crops))
        grid_frames = []
        
        for i in range(num_samples):
            idx = i * len(mouth_crops) // num_samples
            grid_frames.append(mouth_crops[idx])
        
        # Concatenate horizontally
        grid = np.hstack(grid_frames)
        
        # Save grid
        grid_path = self.debug_dir / f"{video_id}_mouth_grid.png"
        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(grid_path), grid_bgr)
        
        # Create GIF if possible
        try:
            import imageio
            gif_path = self.debug_dir / f"{video_id}_mouth_sequence.gif"
            imageio.mimsave(str(gif_path), mouth_crops, duration=0.04)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Could not create GIF: {e}")
        
        # Create side-by-side comparison
        if len(frames) > 0 and len(mouth_crops) > 0:
            # Use middle frame
            mid_idx = len(frames) // 2
            orig_frame = frames[mid_idx]
            mouth_frame = mouth_crops[mid_idx]
            
            # Resize mouth to match original height
            mouth_resized = cv2.resize(
                mouth_frame,
                (orig_frame.shape[0], orig_frame.shape[0])
            )
            
            # Concatenate
            comparison = np.hstack([orig_frame, mouth_resized])
            
            # Save
            comparison_path = self.debug_dir / f"{video_id}_comparison.png"
            comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(comparison_path), comparison_bgr)


# ============================================================================
# FAILURE REPORTER
# ============================================================================

class FailureReporter:
    """
    Tracks and reports processing failures.
    """
    
    def __init__(self, output_path: Path):
        """Initialize failure reporter."""
        self.output_path = output_path
        self.failures = []
    
    def add_failure(self, video_path: str, stage: str, error: str):
        """Add a failure record."""
        self.failures.append({
            'video_path': video_path,
            'stage': stage,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
    
    def has_failures(self) -> bool:
        """Check if any failures were recorded."""
        return len(self.failures) > 0
    
    def save(self):
        """Save failures to CSV file."""
        if not self.failures:
            return
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['video_path', 'stage', 'error', 'timestamp']
            )
            writer.writeheader()
            writer.writerows(self.failures)
        
        logger.info(f"Saved {len(self.failures)} failure records to {self.output_path}")
