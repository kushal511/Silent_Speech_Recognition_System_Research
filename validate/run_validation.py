"""
Validation Pipeline - Main CLI Entry Point

Orchestrates the complete validation workflow including data loading,
validation, visualization, and reporting.
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

# Import validation components
from validate.logging_config import setup_logging
from validate.dataset_checker import check_all_datasets, print_dataset_instructions
from validate.data_loader import PreprocessedDataLoader
from validate.validate_shapes import ShapeValidator
from validate.validate_detection import DetectionValidator
from validate.validate_temporal import TemporalValidator
from validate.validate_roi import ROIValidator
from validate.visualize_samples import VisualValidator, sample_clips_for_visualization
from validate.generate_gifs import GIFGenerator
from validate.failure_analysis import FailureAnalyzer
from validate.report_generator import ReportGenerator

logger = logging.getLogger('validation')


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point for validation pipeline."""
    parser = argparse.ArgumentParser(
        description='LRW Preprocessing Validation Pipeline'
    )
    
    parser.add_argument(
        '--lrw_dataset',
        type=str,
        required=True,
        help='Path to original LRW dataset'
    )
    
    parser.add_argument(
        '--preprocessed_dir',
        type=str,
        required=True,
        help='Path to preprocessed output directory'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save validation results'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='validate/config.yaml',
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
        '--num_samples',
        type=int,
        default=None,
        help='Number of clips to visualize (default: from config)'
    )
    
    parser.add_argument(
        '--random_seed',
        type=int,
        default=None,
        help='Random seed for sampling (default: from config)'
    )
    
    parser.add_argument(
        '--max_clips',
        type=int,
        default=None,
        help='Maximum number of clips to validate (for testing)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.num_samples:
        config['visualization']['num_samples'] = args.num_samples
    if args.random_seed:
        config['visualization']['random_seed'] = args.random_seed
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = output_dir / config['logging']['log_dir']
    setup_logging(
        level=config['logging']['level'],
        log_dir=str(log_dir),
        save_to_file=config['logging']['save_to_file']
    )
    
    logger.info("=" * 80)
    logger.info("LRW Preprocessing Validation Pipeline")
    logger.info("=" * 80)
    logger.info(f"LRW dataset: {args.lrw_dataset}")
    logger.info(f"Preprocessed dir: {args.preprocessed_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Config: {args.config}")
    
    # Check dataset availability
    logger.info("\nChecking dataset availability...")
    datasets_ok, errors = check_all_datasets(args.lrw_dataset, args.preprocessed_dir)
    
    if not datasets_ok:
        logger.error("Required datasets are missing:")
        for error in errors:
            logger.error(error)
        print_dataset_instructions()
        sys.exit(1)
    
    # Initialize data loader
    logger.info("\nInitializing data loader...")
    data_loader = PreprocessedDataLoader(args.lrw_dataset, args.preprocessed_dir)
    
    # Verify data availability
    availability = data_loader.verify_data_availability()
    
    # Get clips to process
    clip_paths = data_loader.get_clip_paths(split=args.split)
    
    if args.max_clips:
        clip_paths = clip_paths[:args.max_clips]
    
    logger.info(f"\nProcessing {len(clip_paths)} clips")
    
    # Initialize validators
    logger.info("\nInitializing validators...")
    shape_validator = ShapeValidator(
        expected_frames=config['expected_frames'],
        expected_landmarks=config['expected_landmarks'],
        expected_roi_size=tuple(config['expected_roi_size'])
    )
    
    detection_validator = DetectionValidator(
        good_threshold=config['detection_thresholds']['good'],
        ok_threshold=config['detection_thresholds']['ok']
    )
    
    temporal_validator = TemporalValidator(
        bbox_displacement_threshold=config['temporal_stability']['bbox_displacement_std_threshold'],
        bbox_area_variance_threshold=config['temporal_stability']['bbox_area_variance_threshold'],
        landmark_motion_threshold=config['temporal_stability']['landmark_motion_threshold']
    )
    
    roi_validator = ROIValidator(
        expected_roi_size=tuple(config['expected_roi_size']),
        tight_threshold=config['roi_validation']['tight_threshold'],
        loose_threshold=config['roi_validation']['loose_threshold'],
        variance_threshold=config['roi_validation']['area_variance_threshold']
    )
    
    failure_analyzer = FailureAnalyzer()
    
    # Initialize visualization components
    vis_dir = output_dir / 'visualizations'
    visual_validator = VisualValidator(
        output_dir=str(vis_dir),
        landmark_color=tuple(config['visualization']['landmark_color']),
        bbox_color=tuple(config['visualization']['bbox_color']),
        marker_size=config['visualization']['marker_size'],
        line_thickness=config['visualization']['line_thickness']
    )
    
    gif_generator = GIFGenerator(
        fps=config['visualization']['gif_fps']
    )
    
    # Initialize report generator
    report_generator = ReportGenerator(str(output_dir))
    
    # Process clips
    logger.info("\nValidating clips...")
    results_by_clip = defaultdict(list)
    failures = []
    
    start_time = datetime.now()
    
    for clip_path in tqdm(clip_paths, desc="Validating"):
        try:
            # Load clip
            clip_data = data_loader.load_clip(clip_path)
            clip_id = clip_data['clip_id']
            
            # Run validators
            shape_result = shape_validator.validate_clip(clip_data)
            detection_result = detection_validator.validate_clip(clip_data)
            temporal_result = temporal_validator.validate_clip(clip_data)
            roi_result = roi_validator.validate_clip(clip_data)
            
            # Store results
            results_by_clip[clip_id].extend([
                shape_result,
                detection_result,
                temporal_result,
                roi_result
            ])
            
            # Analyze failures
            failure_report = failure_analyzer.analyze_clip(
                clip_data,
                results_by_clip[clip_id]
            )
            
            if len(failure_report.failure_modes) > 0:
                failures.append(failure_report)
            
        except Exception as e:
            logger.error(f"Error processing {clip_path}: {e}")
            continue
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"\nValidation complete in {duration:.1f} seconds")
    logger.info(f"Processing speed: {len(clip_paths) / duration:.2f} clips/second")
    
    # Generate visualizations for sampled clips
    if config['output']['save_visualizations']:
        logger.info("\nGenerating visualizations...")
        sampled_paths = sample_clips_for_visualization(
            clip_paths,
            num_samples=config['visualization']['num_samples'],
            random_seed=config['visualization']['random_seed']
        )
        
        for clip_path in tqdm(sampled_paths, desc="Visualizing"):
            try:
                clip_data = data_loader.load_clip(clip_path)
                visual_validator.visualize_clip(clip_data)
                
                # Generate GIFs if enabled
                if config['output']['save_gifs'] and clip_data.get('mouth_frames') is not None:
                    gif_path = vis_dir / f"{clip_data['clip_id']}_mouth_sequence.gif"
                    gif_generator.create_mouth_sequence_gif(
                        clip_data['mouth_frames'],
                        str(gif_path)
                    )
            except Exception as e:
                logger.error(f"Error visualizing {clip_path}: {e}")
                continue
    
    # Generate reports
    logger.info("\nGenerating reports...")
    
    summary_csv_path = output_dir / 'validation_summary.csv'
    report_generator.generate_summary_csv(results_by_clip, str(summary_csv_path))
    
    if failures:
        failure_csv_path = output_dir / 'failure_report.csv'
        report_generator.generate_failure_report(failures, str(failure_csv_path))
    
    summary_stats = report_generator.generate_summary_statistics(results_by_clip)
    summary_json_path = output_dir / 'summary_statistics.json'
    report_generator.save_summary_statistics(summary_stats, str(summary_json_path))
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total clips validated: {summary_stats['total_clips']}")
    logger.info(f"PASS: {summary_stats['status_counts']['PASS']}")
    logger.info(f"WARN: {summary_stats['status_counts']['WARN']}")
    logger.info(f"FAIL: {summary_stats['status_counts']['FAIL']}")
    logger.info(f"Pass rate: {summary_stats['pass_rate']:.1%}")
    logger.info(f"\nReports saved to: {output_dir}")
    logger.info("=" * 80)
    
    # Determine exit code
    if summary_stats['pass_rate'] >= 0.95:
        logger.info("\n✓ Validation PASSED - Data is ready for training")
        return 0
    else:
        logger.warning("\n⚠ Validation completed with warnings - Review failure report")
        return 1


if __name__ == '__main__':
    sys.exit(main())
