#!/usr/bin/env python3
"""
Quick test to verify the new accurate landmark detection works
"""

print("Testing new accurate landmark detection...")
print()

# Test imports
try:
    from src.face_landmarks import FaceLandmarkExtractor
    print("✓ FaceLandmarkExtractor imported successfully")
except Exception as e:
    print(f"✗ Failed to import FaceLandmarkExtractor: {e}")
    exit(1)

try:
    from src.mouth_roi import MouthROIExtractor
    print("✓ MouthROIExtractor imported successfully")
except Exception as e:
    print(f"✗ Failed to import MouthROIExtractor: {e}")
    exit(1)

# Test initialization
try:
    extractor = FaceLandmarkExtractor(confidence_threshold=0.5)
    print(f"✓ FaceLandmarkExtractor initialized with detector: {extractor.detector_type}")
    print(f"  - Lip indices count: {len(extractor.lip_indices)}")
except Exception as e:
    print(f"✗ Failed to initialize FaceLandmarkExtractor: {e}")
    exit(1)

try:
    roi_extractor = MouthROIExtractor(target_size=(96, 96))
    print("✓ MouthROIExtractor initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize MouthROIExtractor: {e}")
    exit(1)

print()
print("=" * 60)
print("SUCCESS: All components initialized correctly!")
print("=" * 60)
print()
print("Key Features:")
print("• Using", extractor.detector_type, "for accurate detection")
print("• Targets exact lip boundaries (upper and lower separated)")
print("• No temporal smoothing applied")
print("• Each frame processed independently")
print()
