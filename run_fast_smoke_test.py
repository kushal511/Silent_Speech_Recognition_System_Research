#!/usr/bin/env python3
"""
Fast Smoke Test - Validates entire pipeline in 30-60 seconds

This is a streamlined version that tests all pipeline stages with minimal data:
- 1 word class
- 1 video
- All pipeline stages validated
- Debug images saved for manual inspection
"""

import sys
import subprocess

def main():
    # Get data root from command line or use default
    data_root = sys.argv[1] if len(sys.argv) > 1 else "test_lrw_dataset/data"
    
    print("=" * 80)
    print("FAST SMOKE TEST (30-60 seconds)")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  - 1 word class")
    print("  - 1 video per word")
    print("  - Visual generation: SKIPPED")
    print()
    
    # Run smoke test with minimal configuration
    cmd = [
        "python3", "run_smoke_test.py",
        "--data_root", data_root,
        "--output_dir", "smoke_test_fast",
        "--debug_dir", "smoke_test_fast_debug",
        "--max_words", "1",
        "--max_videos_per_word", "1",
        "--skip_visuals"
    ]
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
