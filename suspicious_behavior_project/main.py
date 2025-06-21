"""Main application for suspicious behavior detection system.

This module integrates all components of the suspicious behavior detection system:
- Person detection using YOLOv5
- Object tracking using centroid-based tracking
- Behavior analysis for loitering detection
"""

import cv2
import yaml
import time
import numpy as np
from typing import Dict, List, Tuple
from modules.person_detector import PersonDetector
from modules.object_tracker import ObjectTracker
from modules.behavior_analyzer import BehaviorAnalyzer


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {str(e)}")

def convert_detections(detections: List[Tuple[int, int, int, int, float]]) -> Tuple[List[List[float]], List[float]]:
    """Convert detections from (xmin, ymin, xmax, ymax) to (x, y, w, h) format.
    
    Args:
        detections: List of detections in (xmin, ymin, xmax, ymax, confidence) format
        
    Returns:
        Tuple containing:
        - List of detections in (x, y, w, h) format
        - List of confidence scores
    """
    xywh_detections = []
    confidences = []
    for det in detections:
        xmin, ymin, xmax, ymax, conf = det
        width = xmax - xmin
        height = ymax - ymin
        xywh_detections.append([xmin, ymin, width, height])
        confidences.append(conf)
    return xywh_detections, confidences

def draw_detections(frame: np.ndarray,
                   tracked_objects: List[Dict[str, any]],
                   color: Tuple[int, int, int] = (0, 255, 0)) -> None:
    """Draw bounding boxes and IDs on the frame.
    
    Args:
        frame: Input frame
        tracked_objects: List of tracked objects with bbox and track_id
        color: Color for drawing boxes and text
    """
    for obj in tracked_objects:
        bbox = obj['bbox']
        track_id = obj['track_id']
        
        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Calculate center point
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        
        # Draw center point
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
        
        # Draw ID and additional info
        text = f"ID: {track_id}"
        cv2.putText(frame, text, (bbox[0], bbox[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw object size
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        size_text = f"W:{width} H:{height}"
        cv2.putText(frame, size_text, (bbox[0], bbox[1] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw confidence if available
        if 'confidence' in obj:
            conf_text = f"Conf: {obj['confidence']:.2f}"
            cv2.putText(frame, conf_text, (bbox[0], bbox[1] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw tracking history if available
        if 'history' in obj and len(obj['history']) > 1:
            points = obj['history'][-10:]  # Last 10 points
            for i in range(1, len(points)):
                if points[i] is not None and points[i-1] is not None:
                    cv2.line(frame, points[i-1], points[i], (0, 255, 0), 2)

def main():
    """Main function for real-time person tracking and behavior analysis."""
    try:
        # Load configuration
        config = load_config('config/config.yaml')
        
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        
        # Initialize person detector
        detector = PersonDetector(
            model_name=config['model']['weights'],
            confidence_threshold=config['model']['confidence_threshold'],
            device='cpu',
            half_precision=False
        )
        
        # Initialize object tracker
        tracker = ObjectTracker(
            max_disappeared=config['tracking']['max_age'],
            max_distance=config['tracking']['iou_threshold']
        )
        
        # Initialize behavior analyzer
        behavior_analyzer = BehaviorAnalyzer(config['loitering_analysis'])
        
        print("Starting real-time person tracking and behavior analysis...")
        
        # Create OpenCV window with optimized size
        cv2.namedWindow('Person Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Person Detection', 1280, 720)
        
        # Initialize FPS counter
        start_time = time.time()
        frame_count = 0
        frame_rate = 30  # Target frame rate
        
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
                
            # Get detections from person detector
            detections = detector.detect(frame)
            
            # Process detections if any
            if detections:
                # Convert detections to tracker format
                tracking_boxes = []
                for (xmin, ymin, xmax, ymax) in detections:
                    x = xmin
                    y = ymin
                    w = xmax - xmin
                    h = ymax - ymin
                    tracking_boxes.append((x, y, w, h))
                
                # Convert back to display format
                display_boxes = []
                for (x, y, w, h) in tracking_boxes:
                    display_boxes.append((x, y, x + w, y + h))
                
                # Update tracker with detections
                tracked_objects = tracker.update(display_boxes)
                
                # Draw detections (green)
                for i, detection in enumerate(display_boxes):
                    xmin, ymin, xmax, ymax = detection
                    width = xmax - xmin
                    height = ymax - ymin
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    
                    # Draw center point
                    center_x = (xmin + xmax) // 2
                    center_y = (ymin + ymax) // 2
                    cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)
                    
                    # Draw detection info
                    cv2.putText(frame, f"Person {i+1}", (xmin, ymin - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Size: {width}x{height}", (xmin, ymin - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw confidence if available
                    if len(detections) > i and len(detections[i]) == 6:
                        conf_text = f"Conf: {detections[i][4]:.2f}"
                        cv2.putText(frame, conf_text, (xmin, ymin - 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                tracked_objects = []
            
            # Draw tracking results (red)
            for obj in tracked_objects:
                xmin, ymin, xmax, ymax = obj["bbox"]
                width = xmax - xmin
                height = ymax - ymin
                
                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                
                # Draw center point
                center_x = (xmin + xmax) // 2
                center_y = (ymin + ymax) // 2
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                
                # Draw tracking info
                cv2.putText(frame, f"Track {obj['track_id']}", (xmin, ymin - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Size: {width}x{height}", (xmin, ymin - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Draw confidence if available
                if 'confidence' in obj:
                    conf_text = f"Conf: {obj['confidence']:.2f}"
                    cv2.putText(frame, conf_text, (xmin, ymin - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Draw tracking history if available
                if 'history' in obj and len(obj['history']) > 1:
                    points = obj['history'][-10:]  # Last 10 points
                    for i in range(1, len(points)):
                        if points[i] is not None and points[i-1] is not None:
                            cv2.line(frame, points[i-1], points[i], (0, 255, 0), 2)
            
            # Analyze behavior
            alerts = behavior_analyzer.update(tracked_objects)
            
            # Draw ROI regions
            behavior_analyzer.draw_roi(frame)
            
            # Draw detections and alerts
            draw_detections(frame, tracked_objects)
            
            # Draw alerts
            for alert in alerts:
                print(f"Alert: {alert['message']}")
                cv2.putText(frame, alert['message'], (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Calculate and show frame info
            frame_count += 1
            current_time = time.time()
            
            # Calculate FPS
            if frame_count == 10:
                fps = 10.0 / (current_time - start_time)
                start_time = current_time
                frame_count = 0
            else:
                fps = 0 if frame_count == 0 else 10.0 / (current_time - start_time)
            
            # Display information
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Detections: {len(detections)}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Tracked: {len(tracked_objects)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add system info
            cv2.putText(frame, f"Confidence: {detector.model.conf:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"IOU: {detector.model.iou:.2f}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add camera info
            cv2.putText(frame, f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}", 
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS Limit: {frame_rate}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show the frame with optimized size
            cv2.imshow('Person Detection', cv2.resize(frame, (1280, 720)))
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Wait for key press (1ms)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
    finally:
        # Cleanup resources
        print("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        
        # Wait for any remaining frames to be processed
        cv2.waitKey(1)
        print("Program terminated")

if __name__ == "__main__":
    main()
