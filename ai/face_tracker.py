"""
EmotiCam Face Tracking Subsystem.
Implements multi-object centroid and spatial tracking across camera frames.
Assigns consistent session tracking IDs (Face #1, Face #2, etc.) to detected faces.
"""

import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

class CentroidTracker:
    """
    Session-based Multi-Face Spatial Tracker.
    Maintains persistent IDs for active faces as they move across consecutive frames.
    """
    def __init__(self, max_disappeared: int = 12, max_distance: float = 85.0):
        # Next unique ID counter for this camera session
        self.next_object_id = 1
        
        # Active tracked objects: {object_id: centroid_tuple (cX, cY)}
        self.objects = OrderedDict()
        
        # Bounding box cache: {object_id: (x, y, w, h)}
        self.bboxes = OrderedDict()
        
        # Disappeared frames counter: {object_id: count}
        self.disappeared = OrderedDict()
        
        # Max frames an object can be missing before being deregistered
        self.max_disappeared = max_disappeared
        
        # Max pixel distance to associate a detection with an existing tracked face
        self.max_distance = max_distance

    def register(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> int:
        """Registers a newly discovered face with a new unique session tracking ID."""
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.bboxes[obj_id] = bbox
        self.disappeared[obj_id] = 0
        self.next_object_id += 1
        return obj_id

    def deregister(self, object_id: int):
        """Deregisters a face that has left the camera field of view."""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.bboxes:
            del self.bboxes[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def reset(self):
        """Resets all tracking states."""
        self.objects.clear()
        self.bboxes.clear()
        self.disappeared.clear()
        self.next_object_id = 1

    def update(self, rects: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Updates tracking state with the latest detected bounding boxes for the current frame.
        Returns: {track_id: (x, y, w, h)}
        """
        # If no faces were detected in this frame
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return {}

        # Compute centroids for all input detection bounding boxes
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(rects):
            cX = int(x + (w / 2.0))
            cY = int(y + (h / 2.0))
            input_centroids[i] = (cX, cY)

        # If we currently have no tracked objects, register all detections
        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(tuple(input_centroids[i]), rects[i])
        else:
            # Grab existing object IDs and centroids
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            # Compute Euclidean distance between every pair of existing centroids and new centroids
            # Shape: (num_existing, num_new)
            distances = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)

            # Sort row/col indices to match closest centroids first
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # Check if the closest distance exceeds spatial threshold
                if distances[row, col] > self.max_distance:
                    continue

                obj_id = object_ids[row]
                self.objects[obj_id] = tuple(input_centroids[col])
                self.bboxes[obj_id] = rects[col]
                self.disappeared[obj_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Check for existing objects that were not matched
            unused_rows = set(range(0, distances.shape[0])).difference(used_rows)
            for row in unused_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            # Check for new detections that were not matched to any existing object
            unused_cols = set(range(0, distances.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(tuple(input_centroids[col]), rects[col])

        # Return only bounding boxes for faces detected/matched in this frame
        return {obj_id: self.bboxes[obj_id] for obj_id in self.bboxes if self.disappeared.get(obj_id, 0) == 0}
