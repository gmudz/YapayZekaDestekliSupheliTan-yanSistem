"""Module for analyzing behavior patterns in video frames.

This module provides a BehaviorAnalyzer class that implements loitering
detection based on time spent in regions of interest and trajectory analysis.
"""

import time
from collections import defaultdict
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any

class BehaviorAnalyzer:
    """Class for analyzing suspicious behaviors, specifically loitering detection.
    
    Args:
        config (dict): Configuration dictionary containing:
            - time_threshold_seconds: Minimum time in ROI to trigger alert
            - trajectory_efficiency_threshold: Minimum efficiency threshold for loitering
            - regions_of_interest: List of ROI definitions
    """
    
    def __init__(self, config: dict):
        """Initialize BehaviorAnalyzer with configuration."""
        self.rois = config['regions_of_interest']
        self.time_threshold = config['time_threshold_seconds']
        self.efficiency_threshold = config['trajectory_efficiency_threshold']
        
        # Track history and ROI time tracking
        self.track_history = defaultdict(list)
        self.time_in_roi = defaultdict(lambda: {'start_time': None, 'total_time': 0})
        self.alerts = {}
        
    def is_inside_roi(self, point: tuple, roi_name: str) -> bool:
        """Check if a point is inside a specific ROI.
        
        Args:
            point (tuple): (x, y) coordinates
            roi_name (str): Name of the ROI to check
            
        Returns:
            bool: True if point is inside ROI, False otherwise
        """
        for roi in self.rois:
            if roi['name'] == roi_name:
                pts = np.array(roi['points'], np.int32)
                return cv2.pointPolygonTest(pts, point, False) >= 0
        return False

    def update(self, tracks: List[Dict[str, any]]) -> List[str]:
        """Update behavior analysis with new tracked objects.
        
        Args:
            tracks: List of tracked objects with format:
            [{'id': int, 'bbox': [xmin, ymin, xmax, ymax]}]
            
        Returns:
            List[str]: List of generated alerts
        """
        current_alerts = []
        current_time = time.time()
        
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            centroid = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            
            # Update track history
            self.track_history[track_id].append(centroid)
            
            # Check time in ROI
            in_any_roi = False
            for roi in self.rois:
                if self.is_inside_roi(centroid, roi['name']):
                    in_any_roi = True
                    
                    # Start timing if not already started
                    if self.time_in_roi[track_id]['start_time'] is None:
                        self.time_in_roi[track_id]['start_time'] = current_time
                    else:
                        # Calculate elapsed time
                        elapsed = current_time - self.time_in_roi[track_id]['start_time']
                        
                        # Check if time threshold is exceeded
                        if elapsed > self.time_threshold:
                            # Analyze trajectory efficiency
                            path = self.track_history[track_id]
                            if len(path) > 10:  # Minimum path length for analysis
                                start_point = np.array(path[0])
                                end_point = np.array(path[-1])
                                
                                # Calculate net displacement and total distance
                                net_displacement = np.linalg.norm(end_point - start_point)
                                total_distance = sum(
                                    np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
                                    for i in range(1, len(path))
                                )
                                
                                # Calculate efficiency
                                efficiency = net_displacement / (total_distance + 1e-6)
                                
                                # Check if loitering is detected
                                if efficiency < self.efficiency_threshold:
                                    if track_id not in self.alerts:
                                        alert_msg = f"ALERT: Person ID {track_id} is loitering in {roi['name']}"
                                        current_alerts.append(alert_msg)
                                        self.alerts[track_id] = alert_msg
                    break
            
            # Reset tracking if person leaves all ROIs
            if not in_any_roi:
                self.time_in_roi[track_id]['start_time'] = None
                if track_id in self.alerts:
                    del self.alerts[track_id]
        
        return current_alerts

    def draw_roi(self, frame: np.ndarray) -> None:
        """Draw ROIs on the frame.
        
        Args:
            frame: Input frame to draw on
        """
        for roi in self.rois:
            pts = np.array(roi['points'], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(frame, roi['name'], 
                       (pts[0][0][0], pts[0][0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
