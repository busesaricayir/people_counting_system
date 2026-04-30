"""
centroidtracker.py — Multi-object centroid tracker.

Assigns unique IDs to detected objects and tracks them across frames
using Euclidean distance between centroids.

Developed for KSTS (Kişi Sayma ve Takip Sistemi)
Forte Bilgi İletişim Teknolojileri A.Ş. Internship · July 2023
"""

from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker:
    def __init__(self, max_disappeared=40, max_distance=50):
        """
        Args:
            max_disappeared : Max consecutive frames an object can be
                              missing before its ID is deregistered.
            max_distance    : Max Euclidean distance between centroids
                              to consider them the same object.
        """
        self.next_object_id  = 0
        self.objects         = OrderedDict()   # {object_id: centroid}
        self.disappeared     = OrderedDict()   # {object_id: frame_count}
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance

    def register(self, centroid):
        """Register a new object with the next available ID."""
        self.objects[self.next_object_id]    = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove an object from both tracking dictionaries."""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """
        Update tracker with a list of bounding box rectangles.

        Args:
            rects : List of (xmin, ymin, xmax, ymax) tuples from the detector.

        Returns:
            self.objects : OrderedDict of {object_id: (cx, cy)} for current frame.
        """
        # ── No detections this frame ──────────────────────────────────────────
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        # ── Compute centroids for incoming detections ─────────────────────────
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (xmin, ymin, xmax, ymax) in enumerate(rects):
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            input_centroids[i] = (cx, cy)

        # ── No existing objects — register all ───────────────────────────────
        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects

        # ── Match existing objects to new detections ──────────────────────────
        object_ids       = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # Pairwise Euclidean distances: rows=existing, cols=incoming
        D = dist.cdist(np.array(object_centroids), input_centroids)

        # Sort rows by minimum value, then sort corresponding columns
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            self.objects[obj_id]    = input_centroids[col]
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects
        unused_rows = set(range(D.shape[0])) - used_rows
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        # Register new unmatched centroids
        unused_cols = set(range(D.shape[1])) - used_cols
        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects
