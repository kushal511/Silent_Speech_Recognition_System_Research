"""
GIF Generator Module

Creates animated sequences (GIFs) from frame sequences for temporal visualization.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional
import imageio

logger = logging.getLogger('validation.gifs')


class GIFGenerator:
    """
    Generates animated GIF sequences from video frames.
    
    Creates GIFs showing:
    - Mouth ROI sequences (29 frames)
    - Annotated original frame sequences
    """
    
    def __init__(self, fps: int = 25, loop: int = 0):
        """
        Initialize GIF generator.
        
        Args:
            fps: Frames per second for GIF playback
            loop: Number of loops (0 = infinite)
        """
        self.fps = fps
        self.loop = loop
        self.duration = 1.0 / fps  # Duration per frame in seconds
        
        logger.info(f"Initialized GIFGenerator: fps={fps}, loop={loop}")
    
    def create_mouth_sequence_gif(self,
                                  mouth_frames: np.ndarray,
                                  output_path: str):
        """
        Create GIF from mouth ROI frames.
        
        Args:
            mouth_frames: Array of mouth frames (29, H, W, C)
            output_path: Path to save GIF
        """
        if mouth_frames is None or len(mouth_frames) == 0:
            logger.warning("No frames provided for GIF generation")
            return
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert frames to list of images
            frames_list = [frame for frame in mouth_frames]
            
            # Save as GIF
            imageio.mimsave(
                output_path,
                frames_list,
                duration=self.duration,
                loop=self.loop
            )
            
            logger.info(f"Created mouth sequence GIF: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating GIF {output_path}: {e}")
    
    def create_annotated_sequence_gif(self,
                                     original_frames: np.ndarray,
                                     landmarks: np.ndarray,
                                     bboxes: list,
                                     output_path: str,
                                     landmark_color: tuple = (0, 255, 0),
                                     bbox_color: tuple = (255, 0, 0)):
        """
        Create GIF with landmarks and bbox overlays.
        
        Args:
            original_frames: Original video frames (29, H, W, C)
            landmarks: Lip landmarks (29, K, 2)
            bboxes: List of bounding boxes
            output_path: Path to save GIF
            landmark_color: RGB color for landmarks
            bbox_color: RGB color for bboxes
        """
        if original_frames is None or len(original_frames) == 0:
            logger.warning("No frames provided for annotated GIF generation")
            return
        
        try:
            import cv2
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            annotated_frames = []
            
            for i, frame in enumerate(original_frames):
                annotated = frame.copy()
                
                # Draw bbox
                if i < len(bboxes) and bboxes[i] is not None:
                    x, y, w, h = bboxes[i]
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), bbox_color, 2)
                
                # Draw landmarks
                if landmarks is not None and i < len(landmarks):
                    frame_landmarks = landmarks[i]
                    for point in frame_landmarks:
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(annotated, (x, y), 3, landmark_color, -1)
                
                annotated_frames.append(annotated)
            
            # Save as GIF
            imageio.mimsave(
                output_path,
                annotated_frames,
                duration=self.duration,
                loop=self.loop
            )
            
            logger.info(f"Created annotated sequence GIF: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating annotated GIF {output_path}: {e}")
