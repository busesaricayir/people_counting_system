"""
object_trackable.py — Trackable object wrapper.

Stores per-object centroid history and counted status.
"""


class TrackableObject:
    def __init__(self, object_id, centroid):
        """
        Args:
            object_id : Unique integer ID assigned by CentroidTracker.
            centroid  : (cx, cy) tuple for the first detection frame.
        """
        self.object_id = object_id
        self.centroids = [centroid]   # history of centroids across frames
        self.counted   = False        # True once this object crosses the line
