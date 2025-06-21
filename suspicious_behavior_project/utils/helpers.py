"""Utility functions for video processing and visualization.

This module provides helper functions for:
- Video frame processing
- Drawing visualization elements
- Configuration validation
"""

import cv2
import yaml
import numpy as np
from typing import Dict, List, Tuple

def validate_config(config: Dict[str, any]) -> bool:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_keys = [
        'camera',
        'models',
        'loitering_analysis'
    ]
    
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required configuration key: {key}")
            return False
    
    return True

def draw_detection(frame: np.ndarray, bbox: List[int], label: str, color: Tuple[int, int, int] = (0, 255, 0)) -> None:
    """Draw a detection bounding box and label on the frame.
    
    Args:
        frame: Input video frame
        bbox: [x1, y1, x2, y2] bounding box coordinates
        label: Text label to display
        color: RGB color tuple for drawing
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def draw_alert(frame: np.ndarray, alert_msg: str, position: Tuple[int, int] = (50, 50), 
              color: Tuple[int, int, int] = (0, 0, 255)) -> None:
    """Draw an alert message on the frame.
    
    Args:
        frame: Input video frame
        alert_msg: Alert message text
        position: (x, y) coordinates for text
        color: RGB color tuple for text
    """
    cv2.putText(frame, alert_msg, position, 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

def load_camera_source(config: Dict[str, any]) -> cv2.VideoCapture:
    """Load and configure video capture source.
    
    Args:
        config: Configuration dictionary containing camera settings
        
    Returns:
        cv2.VideoCapture: Configured video capture object
    """
    source = config['camera']['source']
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source: {source}")
    
    return cap

def get_frame_size(cap: cv2.VideoCapture) -> Tuple[int, int]:
    """Get frame dimensions from video capture.
    
    Args:
        cap: OpenCV video capture object
        
    Returns:
        Tuple[int, int]: (width, height) of the frame
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height

def normalize_roi_points(roi_points: List[List[int]], frame_size: Tuple[int, int]) -> List[List[int]]:
    """Normalize ROI points to ensure they're within frame boundaries.
    
    Args:
        roi_points: List of [x, y] points defining ROI
        frame_size: (width, height) of the frame
        
    Returns:
        List[List[int]]: Normalized ROI points
    """
    width, height = frame_size
    normalized_points = []
    
    for point in roi_points:
        x = max(0, min(point[0], width - 1))
        y = max(0, min(point[1], height - 1))
        normalized_points.append([x, y])
    
    return normalized_points

def load_config(config_path):
    """Load configuration from YAML file."""
    pass

def preprocess_frame(frame):
    """Preprocess video frame for analysis."""
    pass

def draw_boxes(frame, boxes, labels):
    """Draw bounding boxes on frame with labels."""
    pass
