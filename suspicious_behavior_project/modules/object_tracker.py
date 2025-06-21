"""Module for tracking objects in video frames.

This module provides an ObjectTracker class that implements a centroid-based
tracking algorithm. It's designed to work with the PersonDetector to maintain
consistent tracking of detected objects across frames.
"""

from scipy.spatial import distance as dist
import numpy as np
from typing import List, Dict, Tuple

class ObjectTracker:
    """Class for tracking objects using centroid-based tracking.
    
    Args:
        max_disappeared (int): Maximum number of frames an object can be missing
                              before it's deregistered
        max_distance (int): Maximum distance between centroids for matching
    """
    def __init__(self, max_disappeared: int = 50, max_distance: int = 50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: Tuple[int, int]):
        """Register a new object with its centroid."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int):
        """Deregister an object by its ID."""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects: List[Tuple[int, int, int, int]]) -> List[Dict[str, any]]:
        """Update the tracker with new detections.
        
        Args:
            rects: List of detections in (x, y, w, h) format
            
        Returns:
            List of tracked objects, each containing:
            - track_id: Unique identifier for the track
            - bbox: Bounding box in (xmin, ymin, xmax, ymax) format
        """
        try:
            if len(rects) == 0:
                for object_id in list(self.disappeared.keys()):
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
                return []

            input_centroids = np.zeros((len(rects), 2), dtype="int")
            
            for (i, (startX, startY, endX, endY)) in enumerate(rects):
                cX = int((startX + endX) / 2.0)
                cY = int((startY + endY) / 2.0)
                input_centroids[i] = (cX, cY)

            if len(self.objects) == 0:
                for i in range(0, len(input_centroids)):
                    self.register(input_centroids[i])
            else:
                object_ids = list(self.objects.keys())
                object_centroids = list(self.objects.values())

                D = dist.cdist(np.array(object_centroids), input_centroids)

                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                used_rows = set()
                used_cols = set()

                for (row, col) in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue

                    if D[row, col] > self.max_distance:
                        continue

                    object_id = object_ids[row]
                    self.objects[object_id] = input_centroids[col]
                    self.disappeared[object_id] = 0

                    used_rows.add(row)
                    used_cols.add(col)

                unused_rows = set(range(0, D.shape[0])).difference(used_rows)
                unused_cols = set(range(0, D.shape[1])).difference(used_cols)

                if D.shape[0] >= D.shape[1]:
                    for row in unused_rows:
                        object_id = object_ids[row]
                        self.disappeared[object_id] += 1
                        if self.disappeared[object_id] > self.max_disappeared:
                            self.deregister(object_id)
                else:
                    for col in unused_cols:
                        self.register(input_centroids[col])

            tracked_objects = []
            for object_id, centroid in self.objects.items():
                startX, startY, endX, endY = rects[0]
                for rect in rects:
                    if dist.euclidean(centroid, (int((rect[0] + rect[2]) / 2.0), int((rect[1] + rect[3]) / 2.0))) < dist.euclidean(centroid, (int((startX + endX) / 2.0), int((startY + endY) / 2.0))):
                        startX, startY, endX, endY = rect
                tracked_objects.append({
                    'track_id': object_id,
                    'bbox': [int(startX), int(startY), int(endX), int(endY)]
                })
            return tracked_objects
        except Exception as e:
            raise RuntimeError(f"Tracking update failed: {str(e)}")

    def __del__(self):
        """Cleanup resources when object is destroyed."""
        pass
