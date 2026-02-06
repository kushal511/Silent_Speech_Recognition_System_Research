#!/usr/bin/env python3
"""
Verify Smoke Test Setup

Checks that all required dependencies are installed and the smoke test
can run successfully.
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python {version.major}.{version.minor} is too old (need 3.8+)")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check required dependencies."""
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'scipy': 'scipy',
        'imageio': 'imageio'
    }
    
    missing = []
    
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    return missing

def check_files():
    """Check required files exist."""
    required_files = [
        'run_smoke_test.py',
        'src/smoke_test_utils.py',
        'config.yaml',
        'requirements.txt'
    ]
    
    missing = []
    
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (missing)")
            missing.append(file)
    
    return missing

def main():
    print("=" * 60)
    print("SMOKE TEST SETUP VERIFICATION")
    print("=" * 60)
    print()
    
    # Check Python version
    print("Python Version:")
    if not check_python_version():
        print("\nPlease upgrade to Python 3.8 or higher.")
        sys.exit(1)
    print()
    
    # Check dependencies
    print("Dependencies:")
    missing_deps = check_dependencies()
    print()
    
    # Check files
    print("Required Files:")
    missing_files = check_files()
    print()
    
    # Summary
    print("=" * 60)
    if missing_deps or missing_files:
        print("✗ SETUP INCOMPLETE")
        print("=" * 60)
        
        if missing_deps:
            print("\nMissing dependencies:")
            for dep in missing_deps:
                print(f"  - {dep}")
            print("\nInstall with:")
            print("  pip3 install -r requirements.txt")
        
        if missing_files:
            print("\nMissing files:")
            for file in missing_files:
                print(f"  - {file}")
        
        sys.exit(1)
    else:
        print("✓ SETUP COMPLETE")
        print("=" * 60)
        print("\nYou can now run the smoke test:")
        print("  ./run_fast_smoke_test.sh test_lrw_dataset/data")
        print("\nOr:")
        print("  python3 run_smoke_test.py --data_root test_lrw_dataset/data")
        sys.exit(0)

if __name__ == '__main__':
    main()
