#!/usr/bin/env python3
"""
LRW Preprocessing Pipeline - Smoke Test

This script performs comprehensive smoke testing of the preprocessing pipeline
on a small, manually downloaded subset of the LRW dataset.

Usage:
    python run_smoke_test.py --data_root <path_to_raw_lrw> [options]

Example:
    python run_smoke_test.py --data_root raw_lrw --output_dir smoke_test_output
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2

# Import smoke test utilities
from src.smoke_test_utils import (
    DatasetInspector,
    SingleVideoTester,
    SmallBatchTester,
    OutputVerifier,
    VisualConfirmation,
    FailureReporter,
    print_section_header,
    print_success,
    print_warning,
    print_error
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main smoke test entry point."""
    parser = argparse.ArgumentParser(
        description='Smoke test for LRW preprocessing pipeline'
    )
    
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory of manually downloaded LRW subset'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='smoke_test_output',
        help='Output directory for smoke test results'
    )
    
    parser.add_argument(
        '--debug_dir',
        type=str,
        default='smoke_test_debug',
        help='Directory for debug visualizations'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--max_words',
        type=int,
        default=3,
        help='Maximum number of word classes to test'
    )
    
    parser.add_argument(
        '--max_videos_per_word',
        type=int,
        default=3,
        help='Maximum number of videos per word class'
    )
    
    parser.add_argument(
        '--skip_visuals',
        action='store_true',
        help='Skip visual confirmation step (faster)'
    )
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    debug_dir = Path(args.debug_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup file logging
    log_file = output_dir / f"smoke_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print_error(f"Failed to load config from {args.config}: {e}")
        sys.exit(1)
    
    # Print header
    print_section_header("LRW PREPROCESSING PIPELINE - SMOKE TEST")
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Debug directory: {args.debug_dir}")
    print(f"Max words: {args.max_words}")
    print(f"Max videos per word: {args.max_videos_per_word}")
    print(f"Log file: {log_file}")
    print()
    
    # Initialize failure reporter
    failure_reporter = FailureReporter(output_dir / "smoke_test_failures.csv")
    
    # Track overall results
    all_tests_passed = True
    
    # ========================================================================
    # STEP 1: DATASET SANITY CHECK
    # ========================================================================
    print_section_header("STEP 1: DATASET SANITY CHECK")
    
    inspector = DatasetInspector(args.data_root)
    inspection_result = inspector.inspect()
    
    if not inspection_result['valid']:
        print_error("Dataset inspection failed!")
        for issue in inspection_result['issues']:
            print_error(f"  - {issue}")
        all_tests_passed = False
        
        # Save inspection report
        inspector.save_report(output_dir / "dataset_inspection.json")
        
        print_error("\nSmoke test FAILED at dataset inspection stage.")
        print_error("Please fix the issues above and try again.")
        sys.exit(1)
    
    print_success("Dataset structure validated successfully!")
    print(f"  - Word classes found: {inspection_result['num_words']}")
    print(f"  - Total videos found: {inspection_result['total_videos']}")
    print(f"  - Video extensions: {inspection_result['extensions']}")
    print()
    
    # Save inspection report
    inspector.save_report(output_dir / "dataset_inspection.json")
    
    # ========================================================================
    # STEP 2: SINGLE-VIDEO DRY RUN
    # ========================================================================
    print_section_header("STEP 2: SINGLE-VIDEO DRY RUN")
    
    # Get first video for dry run
    first_video = inspection_result['videos'][0] if inspection_result['videos'] else None
    
    if not first_video:
        print_error("No videos found for dry run!")
        sys.exit(1)
    
    print(f"Testing single video: {first_video['path']}")
    print()
    
    single_tester = SingleVideoTester(config, debug_dir)
    dry_run_result = single_tester.test_video(first_video['path'])
    
    if not dry_run_result['success']:
        print_error(f"Single video test FAILED: {dry_run_result['error']}")
        failure_reporter.add_failure(
            video_path=first_video['path'],
            stage='single_video_dry_run',
            error=dry_run_result['error']
        )
        all_tests_passed = False
        
        print_error("\nSmoke test FAILED at single-video dry run stage.")
        print_error("Check the debug output and fix the pipeline before proceeding.")
        failure_reporter.save()
        sys.exit(1)
    
    print_success("Single video test PASSED!")
    print(f"  - Frames extracted: {dry_run_result['num_frames']}")
    print(f"  - Face detection rate: {dry_run_result['face_detection_rate']:.1%}")
    print(f"  - Valid mouth crops: {dry_run_result['num_valid_crops']}")
    print(f"  - Debug images saved to: {debug_dir}")
    print()
    
    # ========================================================================
    # STEP 3: SMALL-BATCH TEST
    # ========================================================================
    print_section_header("STEP 3: SMALL-BATCH TEST")
    
    # Select videos for batch test
    test_videos = inspector.select_test_videos(
        max_words=args.max_words,
        max_videos_per_word=args.max_videos_per_word
    )
    
    print(f"Testing {len(test_videos)} videos across {args.max_words} word classes")
    print()
    
    batch_tester = SmallBatchTester(config, args.output_dir)
    batch_results = batch_tester.test_batch(test_videos)
    
    # Report batch results
    num_success = sum(1 for r in batch_results if r['success'])
    num_failed = len(batch_results) - num_success
    
    print(f"Batch test completed:")
    print(f"  - Total videos: {len(batch_results)}")
    print(f"  - Successful: {num_success}")
    print(f"  - Failed: {num_failed}")
    
    if num_failed > 0:
        print_warning(f"\n{num_failed} videos failed processing:")
        for result in batch_results:
            if not result['success']:
                print_warning(f"  - {result['video_path']}: {result['error']}")
                failure_reporter.add_failure(
                    video_path=result['video_path'],
                    stage='batch_processing',
                    error=result['error']
                )
        all_tests_passed = False
    else:
        print_success("All videos processed successfully!")
    
    print()
    
    # ========================================================================
    # STEP 4: OUTPUT VERIFICATION
    # ========================================================================
    print_section_header("STEP 4: OUTPUT VERIFICATION")
    
    verifier = OutputVerifier(args.output_dir, config)
    verification_results = verifier.verify_all_outputs(batch_results)
    
    # Report verification results
    num_verified = sum(1 for r in verification_results if r['valid'])
    num_invalid = len(verification_results) - num_verified
    
    print(f"Output verification completed:")
    print(f"  - Total outputs: {len(verification_results)}")
    print(f"  - Valid: {num_verified}")
    print(f"  - Invalid: {num_invalid}")
    
    if num_invalid > 0:
        print_warning(f"\n{num_invalid} outputs failed verification:")
        for result in verification_results:
            if not result['valid']:
                print_warning(f"  - {result['video_id']}")
                for issue in result['issues']:
                    print_warning(f"      {issue}")
                failure_reporter.add_failure(
                    video_path=result['video_id'],
                    stage='output_verification',
                    error='; '.join(result['issues'])
                )
        all_tests_passed = False
    else:
        print_success("All outputs verified successfully!")
    
    print()
    
    # ========================================================================
    # STEP 5: VISUAL CONFIRMATION
    # ========================================================================
    if not args.skip_visuals:
        print_section_header("STEP 5: VISUAL CONFIRMATION")
        
        visualizer = VisualConfirmation(args.output_dir, debug_dir)
        
        # Generate visualizations for successful videos
        successful_videos = [r for r in batch_results if r['success']]
        
        if successful_videos:
            print(f"Generating visual confirmations for {len(successful_videos)} videos...")
            
            for video_result in successful_videos[:3]:  # Visualize first 3
                video_id = Path(video_result['video_path']).stem
                word_class = video_result.get('word_class', 'unknown')
                
                try:
                    visualizer.create_video_visualization(
                        video_result['video_path'],
                        word_class,
                        video_id
                    )
                    print_success(f"  - Created visualization for {video_id}")
                except Exception as e:
                    print_warning(f"  - Failed to create visualization for {video_id}: {e}")
            
            print()
            print_success(f"Visual confirmations saved to: {debug_dir}")
            print("Please review the generated images and GIFs to confirm correctness.")
        else:
            print_warning("No successful videos to visualize")
        
        print()
    else:
        print_warning("STEP 5: VISUAL CONFIRMATION - SKIPPED")
        print()
    
    # ========================================================================
    # STEP 6: GENERATE SUMMARY REPORT
    # ========================================================================
    print_section_header("SUMMARY REPORT")
    
    # Save failure report
    if failure_reporter.has_failures():
        failure_reporter.save()
        print_warning(f"Failure report saved to: {failure_reporter.output_path}")
    
    # Generate summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_root': args.data_root,
        'dataset_inspection': {
            'valid': inspection_result['valid'],
            'num_words': inspection_result['num_words'],
            'total_videos': inspection_result['total_videos']
        },
        'single_video_test': {
            'success': dry_run_result['success'],
            'num_frames': dry_run_result.get('num_frames', 0),
            'face_detection_rate': dry_run_result.get('face_detection_rate', 0)
        },
        'batch_test': {
            'total_videos': len(batch_results),
            'successful': num_success,
            'failed': num_failed,
            'success_rate': num_success / len(batch_results) if batch_results else 0
        },
        'output_verification': {
            'total_outputs': len(verification_results),
            'valid': num_verified,
            'invalid': num_invalid
        },
        'overall_status': 'PASSED' if all_tests_passed else 'FAILED'
    }
    
    # Save summary
    import json
    summary_path = output_dir / "smoke_test_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)
    print(f"Overall Status: {summary['overall_status']}")
    print()
    print("Dataset Inspection:")
    print(f"  ✓ Valid structure: {summary['dataset_inspection']['valid']}")
    print(f"  ✓ Word classes: {summary['dataset_inspection']['num_words']}")
    print(f"  ✓ Total videos: {summary['dataset_inspection']['total_videos']}")
    print()
    print("Single Video Test:")
    print(f"  {'✓' if summary['single_video_test']['success'] else '✗'} Success: {summary['single_video_test']['success']}")
    print(f"  ✓ Frames: {summary['single_video_test']['num_frames']}")
    print(f"  ✓ Face detection: {summary['single_video_test']['face_detection_rate']:.1%}")
    print()
    print("Batch Test:")
    print(f"  ✓ Total: {summary['batch_test']['total_videos']}")
    print(f"  {'✓' if summary['batch_test']['failed'] == 0 else '✗'} Successful: {summary['batch_test']['successful']}")
    print(f"  {'✓' if summary['batch_test']['failed'] == 0 else '✗'} Failed: {summary['batch_test']['failed']}")
    print(f"  ✓ Success rate: {summary['batch_test']['success_rate']:.1%}")
    print()
    print("Output Verification:")
    print(f"  ✓ Total: {summary['output_verification']['total_outputs']}")
    print(f"  {'✓' if summary['output_verification']['invalid'] == 0 else '✗'} Valid: {summary['output_verification']['valid']}")
    print(f"  {'✓' if summary['output_verification']['invalid'] == 0 else '✗'} Invalid: {summary['output_verification']['invalid']}")
    print()
    print("=" * 80)
    print(f"Summary saved to: {summary_path}")
    print(f"Log file: {log_file}")
    
    if not args.skip_visuals:
        print(f"Debug visualizations: {debug_dir}")
    
    print("=" * 80)
    
    # Exit with appropriate code
    if all_tests_passed:
        print_success("\n✓ SMOKE TEST PASSED - Pipeline is ready for full preprocessing!")
        sys.exit(0)
    else:
        print_error("\n✗ SMOKE TEST FAILED - Please review errors and fix issues.")
        sys.exit(1)


if __name__ == '__main__':
    main()
