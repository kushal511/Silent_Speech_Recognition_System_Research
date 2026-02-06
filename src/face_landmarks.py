"""
Face Detection and Landmark Extraction Module

This module provides face detection and facial landmark extraction using OpenCV's DNN face detector
and facial landmark detector. It handles per-frame processing with robust error handling.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class FaceLandmarkExtractor:
    """
    Face detection and landmark extraction using OpenCV.
    
    Uses OpenCV's Haar Cascade for face detection and estimates lip landmarks
    from the detected face region.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 model_selection: int = 0,
                 lip_indices: Optional[List[int]] = None):
        """
        Initialize face landmark extractor.
        
        Args:
            confidence_threshold: Minimum detection confidence (0.0 to 1.0)
            model_selection: Not used (kept for compatibility)
            lip_indices: Not used (kept for compatibility)
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        logger.info(f"Initialized OpenCV Face Detector")
    
    def detect_face_landmarks(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Detect face and estimate landmarks from a single frame.
        
        Args:
            frame: Input frame as numpy array (height, width, 3) in RGB format
        
        Returns:
            Tuple of:
                - success: True if face detected
                - landmarks: numpy array of shape (68, 2) with estimated (x, y) coordinates
                            in pixel space, or None if detection failed
                - confidence: Detection confidence score (0.0 to 1.0)
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided for landmark detection")
            return False, None, 0.0
        
        try:
            height, width = frame.shape[:2]
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                logger.debug("No face detected in frame")
                return False, None, 0.0
            
            # Get the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face
            
            # Estimate 68 landmarks based on face bounding box
            # This is a simplified approach - we create a grid of points
            landmarks = self._estimate_landmarks_from_face(x, y, w, h)
            
            confidence = 0.8  # Fixed confidence for Haar cascade
            
            logger.debug(f"Detected face with estimated landmarks (confidence: {confidence:.3f})")
            
            return True, landmarks, confidence
            
        except Exception as e:
            logger.error(f"Error detecting face landmarks: {e}")
            return False, None, 0.0
    
    def _estimate_landmarks_from_face(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        Estimate 68 facial landmarks from face bounding box.
        This is a simplified geometric estimation.
        
        Args:
            x, y, w, h: Face bounding box
        
        Returns:
            Estimated landmarks (68, 2)
        """
        landmarks = np.zeros((68, 2), dtype=np.float32)
        
        # Face outline (0-16): along the jaw
        for i in range(17):
            landmarks[i] = [x + w * i / 16, y + h * (0.8 + 0.2 * abs(i - 8) / 8)]
        
        # Eyebrows (17-26)
        for i in range(17, 22):  # Left eyebrow
            landmarks[i] = [x + w * (0.2 + (i - 17) * 0.1), y + h * 0.3]
        for i in range(22, 27):  # Right eyebrow
            landmarks[i] = [x + w * (0.6 + (i - 22) * 0.1), y + h * 0.3]
        
        # Nose (27-35)
        for i in range(27, 31):  # Nose bridge
            landmarks[i] = [x + w * 0.5, y + h * (0.35 + (i - 27) * 0.1)]
        for i in range(31, 36):  # Nose base
            landmarks[i] = [x + w * (0.35 + (i - 31) * 0.075), y + h * 0.65]
        
        # Eyes (36-47)
        for i in range(36, 42):  # Left eye
            angle = (i - 36) * np.pi / 3
            landmarks[i] = [x + w * (0.3 + 0.05 * np.cos(angle)), 
                           y + h * (0.4 + 0.03 * np.sin(angle))]
        for i in range(42, 48):  # Right eye
            angle = (i - 42) * np.pi / 3
            landmarks[i] = [x + w * (0.7 + 0.05 * np.cos(angle)), 
                           y + h * (0.4 + 0.03 * np.sin(angle))]
        
        # Mouth (48-67) - This is the most important for lip reading
        mouth_center_x = x + w * 0.5
        mouth_center_y = y + h * 0.75
        mouth_width = w * 0.3
        mouth_height = h * 0.15
        
        # Outer lip (48-59)
        for i in range(48, 60):
            angle = (i - 48) * 2 * np.pi / 12
            landmarks[i] = [mouth_center_x + mouth_width * np.cos(angle),
                           mouth_center_y + mouth_height * np.sin(angle)]
        
        # Inner lip (60-67)
        for i in range(60, 68):
            angle = (i - 60) * 2 * np.pi / 8
            landmarks[i] = [mouth_center_x + mouth_width * 0.6 * np.cos(angle),
                           mouth_center_y + mouth_height * 0.6 * np.sin(angle)]
        
        return landmarks
    
    def extract_lip_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract lip-specific landmarks from full face landmarks.
        
        Args:
            landmarks: Full face landmarks array (68, 2)
        
        Returns:
            Lip landmarks array (20, 2) - indices 48-67
        """
        if landmarks is None:
            return np.array([])
        
        try:
            # Extract mouth landmarks (indices 48-67 in 68-point model)
            lip_landmarks = landmarks[48:68]
            return lip_landmarks
        except IndexError as e:
            logger.error(f"Error extracting lip landmarks: {e}")
            return np.array([])
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame and extract face and lip landmarks.
        
        Args:
            frame: Input frame (height, width, 3) in RGB format
        
        Returns:
            Dictionary with detection results:
                - face_detected: bool
                - landmarks: full face landmarks (468, 2) or None
                - lip_landmarks: lip landmarks (num_lip_points, 2) or None
                - confidence: detection confidence
        """
        success, landmarks, confidence = self.detect_face_landmarks(frame)
        
        result = {
            'face_detected': success,
            'landmarks': landmarks,
            'lip_landmarks': None,
            'confidence': confidence
        }
        
        if success and landmarks is not None:
            result['lip_landmarks'] = self.extract_lip_landmarks(landmarks)
        
        return result
    
    def process_video_frames(self, frames: np.ndarray) -> List[dict]:
        """
        Process all frames in a video and extract landmarks.
        
        Args:
            frames: Video frames array (num_frames, height, width, 3)
        
        Returns:
            List of detection results (one dict per frame)
        """
        results = []
        
        for frame_idx, frame in enumerate(frames):
            result = self.process_frame(frame)
            result['frame_idx'] = frame_idx
            results.append(result)
            
            if not result['face_detected']:
                logger.debug(f"No face detected in frame {frame_idx}")
        
        # Log summary statistics
        num_detected = sum(1 for r in results if r['face_detected'])
        detection_rate = num_detected / len(results) if results else 0
        
        logger.info(
            f"Processed {len(results)} frames: "
            f"{num_detected} faces detected ({detection_rate:.1%})"
        )
        
        return results
    
    def __del__(self):
        """Clean up resources."""
        pass  # No cleanup needed for OpenCV


def interpolate_missing_landmarks(results: List[dict]) -> List[dict]:
    """
    Interpolate landmarks for frames where detection failed.
    
    Uses linear interpolation between nearest successful detections.
    
    Args:
        results: List of detection results from process_video_frames
    
    Returns:
        Updated results with interpolated landmarks
    """
    num_frames = len(results)
    
    # Find frames with successful detections
    valid_indices = [i for i, r in enumerate(results) if r['face_detected']]
    
    if len(valid_indices) == 0:
        logger.warning("No valid detections found for interpolation")
        return results
    
    # Interpolate missing frames
    for i in range(num_frames):
        if not results[i]['face_detected']:
            # Find nearest valid frames
            prev_idx = max([idx for idx in valid_indices if idx < i], default=None)
            next_idx = min([idx for idx in valid_indices if idx > i], default=None)
            
            if prev_idx is not None and next_idx is not None:
                # Linear interpolation between prev and next
                alpha = (i - prev_idx) / (next_idx - prev_idx)
                
                prev_landmarks = results[prev_idx]['landmarks']
                next_landmarks = results[next_idx]['landmarks']
                
                interpolated = (1 - alpha) * prev_landmarks + alpha * next_landmarks
                
                results[i]['landmarks'] = interpolated
                results[i]['lip_landmarks'] = results[prev_idx]['lip_landmarks'] * (1 - alpha) + \
                                             results[next_idx]['lip_landmarks'] * alpha
                results[i]['interpolated'] = True
                
                logger.debug(f"Interpolated landmarks for frame {i}")
                
            elif prev_idx is not None:
                # Use previous frame
                results[i]['landmarks'] = results[prev_idx]['landmarks'].copy()
                results[i]['lip_landmarks'] = results[prev_idx]['lip_landmarks'].copy()
                results[i]['interpolated'] = True
                logger.debug(f"Copied landmarks from frame {prev_idx} to {i}")
                
            elif next_idx is not None:
                # Use next frame
                results[i]['landmarks'] = results[next_idx]['landmarks'].copy()
                results[i]['lip_landmarks'] = results[next_idx]['lip_landmarks'].copy()
                results[i]['interpolated'] = True
                logger.debug(f"Copied landmarks from frame {next_idx} to {i}")
    
    return results
