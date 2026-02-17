"""
LRW Preprocessing Pipeline - Main Entry Point

This script orchestrates the complete preprocessing pipeline for the LRW dataset,
including video loading, face detection, landmark extraction, ROI computation,
temporal smoothing, and output saving.
"""

import argparse
import logging
import yaml
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np

# Import pipeline modules
from src.dataset import LRWDataset, VideoSample, validate_dataset_structure
from src.video_io import VideoReader
from src.face_landmarks import FaceLandmarkExtractor, interpolate_missing_landmarks
from src.mouth_roi import MouthROIExtractor, fill_missing_roi_boxes
# Temporal smoothing removed as per requirements
from src.save_utils import OutputSaver, save_failed_videos_list, save_processing_summary
from src.visualize_debug import PreprocessingVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for LRW videos.
    
    Orchestrates all preprocessing steps from video loading to output saving.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
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
        
        # Temporal smoothing removed as per requirements
        self.smoother = None
        
        self.output_saver = None  # Initialized when output_dir is known
        
        self.visualizer = None
        if config['debug']['enabled'] and config['debug']['save_debug_images']:
            self.visualizer = PreprocessingVisualizer(
                output_dir=config['debug']['debug_output_dir']
            )
        
        logger.info("Initialized preprocessing pipeline")
    
    def process_video(self, video_sample: VideoSample) -> Dict:
        """
        Process a single video through the complete pipeline.
        
        Args:
            video_sample: VideoSample object
        
        Returns:
            Dictionary with processing results and statistics
        """
        result = {
            'video_id': video_sample.video_id,
            'success': False,
            'error': None,
            'statistics': {}
        }
        
        try:
            # Step 1: Load video frames
            logger.debug(f"Processing {video_sample.video_id}")
            frames, video_metadata = self.video_reader.read_video(video_sample.video_path)
            
            if frames is None:
                result['error'] = video_metadata.get('error', 'Failed to read video')
                return result
            
            result['statistics']['num_frames'] = len(frames)
            
            # Step 2: Extract landmarks
            landmark_results = self.landmark_extractor.process_video_frames(frames)
            
            # Interpolate missing landmarks
            landmark_results = interpolate_missing_landmarks(landmark_results)
            
            face_detection_rate = sum(1 for r in landmark_results if r['face_detected']) / len(landmark_results)
            result['statistics']['face_detection_rate'] = face_detection_rate
            
            if face_detection_rate < self.config['quality']['min_face_detection_rate']:
                logger.warning(
                    f"Low face detection rate for {video_sample.video_id}: "
                    f"{face_detection_rate:.1%}"
                )
            
            # Step 3: Temporal smoothing removed as per requirements
            
            # Step 4: Compute mouth ROIs
            roi_results = self.roi_extractor.process_video_frames(frames, landmark_results)
            
            # Fill missing ROI boxes
            roi_results = fill_missing_roi_boxes(roi_results)
            
            # Step 5: Temporal smoothing removed as per requirements
            
            # Step 6: Extract cropped mouth regions
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
            
            result['statistics']['num_valid_crops'] = len(valid_crops)
            
            # Step 7: Prepare landmarks array
            lip_landmarks_list = [r['lip_landmarks'] for r in landmark_results]
            lip_landmarks_array = np.array(lip_landmarks_list)
            
            # Step 8: Create debug visualizations if enabled
            if self.visualizer:
                self.visualizer.visualize_video_processing(
                    frames, landmark_results, roi_results, video_sample.video_id
                )
            
            # Step 9: Save outputs
            processing_info = {
                'face_detection_rate': face_detection_rate,
                'num_valid_crops': len(valid_crops),
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
            logger.info(f"Successfully processed {video_sample.video_id}")
            
        except Exception as e:
            result['error'] = f"Exception during processing: {str(e)}"
            logger.error(f"Error processing {video_sample.video_id}: {e}", exc_info=True)
        
        return result


def process_video_wrapper(args):
    """
    Wrapper function for multiprocessing.
    
    Args:
        args: Tuple of (video_sample, config, output_dir)
    
    Returns:
        Processing result dictionary
    """
    video_sample, config, output_dir = args
    
    # Create pipeline instance for this worker
    pipeline = PreprocessingPipeline(config)
    pipeline.output_saver = OutputSaver(
        output_root=output_dir,
        frame_format=config['video']['frame_format']
    )
    
    # Check if already processed
    if config['processing']['skip_existing']:
        if pipeline.output_saver.check_output_exists(
            video_sample.word_class,
            video_sample.split,
            video_sample.video_id
        ):
            logger.debug(f"Skipping already processed video: {video_sample.video_id}")
            return {
                'video_id': video_sample.video_id,
                'success': True,
                'skipped': True
            }
    
    # Process video
    return pipeline.process_video(video_sample)


def main():
    """Main entry point for preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description='LRW Silent Speech Recognition Preprocessing Pipeline'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Root directory of LRW dataset'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for preprocessed data'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        default=None,
        help='Process specific split only'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: from config)'
    )
    
    parser.add_argument(
        '--max_videos',
        type=int,
        default=None,
        help='Maximum number of videos to process (for testing)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with visualizations'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line arguments
    if args.num_workers:
        config['processing']['num_workers'] = args.num_workers
    
    if args.max_videos:
        config['processing']['max_videos'] = args.max_videos
    
    if args.debug:
        config['debug']['enabled'] = True
        config['debug']['save_debug_images'] = True
    
    # Setup logging
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("LRW Preprocessing Pipeline")
    logger.info("=" * 80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Workers: {config['processing']['num_workers']}")
    
    # Validate dataset structure
    logger.info("Validating dataset structure...")
    is_valid, issues = validate_dataset_structure(args.input_dir)
    
    if not is_valid:
        logger.error("Dataset validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        sys.exit(1)
    
    logger.info("Dataset structure validated successfully")
    
    # Initialize dataset
    splits = [args.split] if args.split else ['train', 'val', 'test']
    
    # Get dataset config if available
    dataset_config = config.get('dataset', {})
    video_dir = dataset_config.get('video_dir', None)
    video_extension = dataset_config.get('video_extension', '.mp4')
    
    dataset = LRWDataset(
        root_dir=args.input_dir,
        video_dir=video_dir,
        video_extension=video_extension,
        splits=splits
    )
    
    # Get samples to process
    samples = dataset.get_samples(
        split=args.split,
        max_samples=config['processing']['max_videos']
    )
    
    logger.info(f"Processing {len(samples)} videos")
    
    # Prepare arguments for multiprocessing
    process_args = [(sample, config, args.output_dir) for sample in samples]
    
    # Process videos
    num_workers = config['processing']['num_workers']
    
    logger.info(f"Starting processing with {num_workers} workers...")
    
    results = []
    failed_videos = []
    
    if num_workers > 1:
        # Multiprocessing
        with Pool(processes=num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(process_video_wrapper, process_args),
                total=len(samples),
                desc="Processing videos"
            ):
                results.append(result)
                
                if not result['success'] and not result.get('skipped', False):
                    failed_videos.append({
                        'video_id': result['video_id'],
                        'error': result.get('error', 'Unknown error')
                    })
    else:
        # Single process (easier for debugging)
        for args_tuple in tqdm(process_args, desc="Processing videos"):
            result = process_video_wrapper(args_tuple)
            results.append(result)
            
            if not result['success'] and not result.get('skipped', False):
                failed_videos.append({
                    'video_id': result['video_id'],
                    'error': result.get('error', 'Unknown error')
                })
    
    # Compute summary statistics
    num_success = sum(1 for r in results if r['success'])
    num_skipped = sum(1 for r in results if r.get('skipped', False))
    num_failed = len(failed_videos)
    
    summary = {
        'total_videos': len(samples),
        'successful': num_success,
        'skipped': num_skipped,
        'failed': num_failed,
        'success_rate': num_success / len(samples) if samples else 0,
        'timestamp': datetime.now().isoformat(),
        'config': config
    }
    
    # Save results
    output_dir = Path(args.output_dir)
    
    if failed_videos:
        failed_path = output_dir / 'failed_videos.json'
        save_failed_videos_list(failed_videos, str(failed_path))
    
    summary_path = output_dir / 'processing_summary.json'
    save_processing_summary(summary, str(summary_path))
    
    # Print summary
    logger.info("=" * 80)
    logger.info("Processing Complete")
    logger.info("=" * 80)
    logger.info(f"Total videos: {len(samples)}")
    logger.info(f"Successful: {num_success}")
    logger.info(f"Skipped (already processed): {num_skipped}")
    logger.info(f"Failed: {num_failed}")
    logger.info(f"Success rate: {summary['success_rate']:.1%}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Summary: {summary_path}")
    
    if failed_videos:
        logger.warning(f"Failed videos list: {failed_path}")
    
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
