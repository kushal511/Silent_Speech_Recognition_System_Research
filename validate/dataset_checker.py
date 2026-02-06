"""
Dataset Availability Checker

Checks if required datasets (original LRW and preprocessed outputs) are available
before running validation. Provides helpful error messages and instructions.
"""

import logging
from pathlib import Path
from typing import Tuple, List

logger = logging.getLogger('validation.dataset_checker')


def check_lrw_dataset(lrw_dataset_path: str) -> Tuple[bool, str]:
    """
    Check if LRW dataset exists and is accessible.
    
    Args:
        lrw_dataset_path: Path to LRW dataset root
    
    Returns:
        Tuple of (exists, message)
    """
    lrw_path = Path(lrw_dataset_path)
    
    if not lrw_path.exists():
        message = (
            f"LRW dataset not found at: {lrw_dataset_path}\n\n"
            f"To download the LRW dataset:\n"
            f"  python -m validate.download_lrw --output_dir {lrw_dataset_path}\n\n"
            f"Or download manually from:\n"
            f"  http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html\n"
        )
        return False, message
    
    # Check for word class directories
    word_dirs = [d for d in lrw_path.iterdir() if d.is_dir()]
    
    if len(word_dirs) == 0:
        message = (
            f"LRW dataset directory exists but appears empty: {lrw_dataset_path}\n\n"
            f"Expected structure:\n"
            f"  {lrw_dataset_path}/WORD_CLASS/SPLIT/VIDEO_ID.mp4\n\n"
            f"To download:\n"
            f"  python -m validate.download_lrw --output_dir {lrw_dataset_path}\n"
        )
        return False, message
    
    # Check for video files in at least one word class
    has_videos = False
    for word_dir in word_dirs[:5]:  # Check first 5
        for split in ['train', 'val', 'test']:
            split_dir = word_dir / split
            if split_dir.exists():
                videos = list(split_dir.glob("*.mp4"))
                if len(videos) > 0:
                    has_videos = True
                    break
        if has_videos:
            break
    
    if not has_videos:
        message = (
            f"LRW dataset directory exists but no video files found: {lrw_dataset_path}\n\n"
            f"Expected structure:\n"
            f"  {lrw_dataset_path}/WORD_CLASS/SPLIT/VIDEO_ID.mp4\n\n"
            f"To verify dataset:\n"
            f"  python -m validate.download_lrw --output_dir {lrw_dataset_path} --verify_only\n"
        )
        return False, message
    
    logger.info(f"✓ LRW dataset found: {lrw_dataset_path}")
    logger.info(f"  Found {len(word_dirs)} word class directories")
    
    return True, "LRW dataset is available"


def check_preprocessed_data(preprocessed_path: str) -> Tuple[bool, str]:
    """
    Check if preprocessed data exists and is accessible.
    
    Args:
        preprocessed_path: Path to preprocessed output root
    
    Returns:
        Tuple of (exists, message)
    """
    prep_path = Path(preprocessed_path)
    
    if not prep_path.exists():
        message = (
            f"Preprocessed data not found at: {preprocessed_path}\n\n"
            f"Please run the preprocessing pipeline first:\n"
            f"  python run_preprocess.py \\\n"
            f"    --input_dir /path/to/lrw_dataset \\\n"
            f"    --output_dir {preprocessed_path}\n"
        )
        return False, message
    
    # Check for word class directories
    word_dirs = [d for d in prep_path.iterdir() if d.is_dir()]
    
    if len(word_dirs) == 0:
        message = (
            f"Preprocessed data directory exists but appears empty: {preprocessed_path}\n\n"
            f"Expected structure:\n"
            f"  {preprocessed_path}/WORD_CLASS/SPLIT/VIDEO_ID/\n"
            f"    ├── frames/\n"
            f"    ├── landmarks.npy\n"
            f"    └── metadata.json\n\n"
            f"Please run the preprocessing pipeline first.\n"
        )
        return False, message
    
    # Check for preprocessed clips in at least one word class
    has_clips = False
    for word_dir in word_dirs[:5]:  # Check first 5
        for split in ['train', 'val', 'test']:
            split_dir = word_dir / split
            if split_dir.exists():
                clips = [d for d in split_dir.iterdir() if d.is_dir()]
                for clip_dir in clips[:1]:  # Check first clip
                    if (clip_dir / 'landmarks.npy').exists():
                        has_clips = True
                        break
            if has_clips:
                break
        if has_clips:
            break
    
    if not has_clips:
        message = (
            f"Preprocessed data directory exists but no valid clips found: {preprocessed_path}\n\n"
            f"Expected structure:\n"
            f"  {preprocessed_path}/WORD_CLASS/SPLIT/VIDEO_ID/\n"
            f"    ├── frames/\n"
            f"    ├── landmarks.npy\n"
            f"    └── metadata.json\n\n"
            f"Please run the preprocessing pipeline first.\n"
        )
        return False, message
    
    logger.info(f"✓ Preprocessed data found: {preprocessed_path}")
    logger.info(f"  Found {len(word_dirs)} word class directories")
    
    return True, "Preprocessed data is available"


def check_all_datasets(lrw_dataset_path: str, 
                      preprocessed_path: str) -> Tuple[bool, List[str]]:
    """
    Check availability of all required datasets.
    
    Args:
        lrw_dataset_path: Path to LRW dataset root
        preprocessed_path: Path to preprocessed output root
    
    Returns:
        Tuple of (all_available, list_of_error_messages)
    """
    errors = []
    
    # Check LRW dataset
    lrw_ok, lrw_msg = check_lrw_dataset(lrw_dataset_path)
    if not lrw_ok:
        errors.append(lrw_msg)
    
    # Check preprocessed data
    prep_ok, prep_msg = check_preprocessed_data(preprocessed_path)
    if not prep_ok:
        errors.append(prep_msg)
    
    all_ok = lrw_ok and prep_ok
    
    if all_ok:
        logger.info("✓ All required datasets are available")
    else:
        logger.error("✗ Some required datasets are missing")
    
    return all_ok, errors


def print_dataset_instructions():
    """Print helpful instructions for setting up datasets."""
    print("\n" + "=" * 70)
    print("DATASET SETUP INSTRUCTIONS")
    print("=" * 70)
    print("\nThe validation pipeline requires two datasets:")
    print("\n1. Original LRW Dataset")
    print("   - Download from: http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html")
    print("   - Or use: python -m validate.download_lrw --output_dir /path/to/lrw")
    print("\n2. Preprocessed Outputs")
    print("   - Run: python run_preprocess.py --input_dir /path/to/lrw --output_dir output")
    print("\nOnce both are ready, run validation:")
    print("   python run_validation.py \\")
    print("     --lrw_dataset /path/to/lrw \\")
    print("     --preprocessed_dir output \\")
    print("     --output_dir validation_results")
    print("\n" + "=" * 70 + "\n")
