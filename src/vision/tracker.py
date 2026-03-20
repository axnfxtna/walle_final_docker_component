"""
ByteTrack implementation for multi-face tracking.

This module implements the ByteTrack algorithm for tracking multiple faces across frames,
maintaining consistent track IDs even through occlusions and appearance changes.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
import time


@dataclass
class Track:
    """Represents a single tracked face."""
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    landmarks: Optional[List[List[float]]] = None
    state: str = "tracked"  # "new", "tracked", "lost"
    frames_since_update: int = 0
    hits: int = 1
    age: int = 1
    
    def update(self, bbox: List[float], confidence: float, landmarks: Optional[List[List[float]]] = None):
        """Update track with new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.landmarks = landmarks
        self.frames_since_update = 0
        self.hits += 1
        self.age += 1
        self.state = "tracked"
    
    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.frames_since_update += 1
        self.age += 1
        if self.frames_since_update > 1:
            self.state = "lost"


class ByteTracker:
    """
    ByteTrack algorithm for multi-object tracking.
    
    Features:
    - Two-stage association (high and low confidence detections)
    - IOU-based matching
    - Track state management (new, tracked, lost)
    - Configurable thresholds and buffer
    """
    
    def __init__(
        self,
        track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_box_area: int = 100
    ):
        """
        Initialize ByteTracker.
        
        Args:
            track_thresh: Confidence threshold for high-confidence detections
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IOU threshold for matching
            min_box_area: Minimum bounding box area
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        
        self.tracks: List[Track] = []
        self.next_track_id = 1
        self.frame_count = 0
        
        logger.info(f"ByteTracker initialized with track_thresh={track_thresh}, match_thresh={match_thresh}")
    
    def update(self, detections: List[Dict]) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detections from detector
        
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Separate high and low confidence detections
        high_conf_dets = []
        low_conf_dets = []
        
        for det in detections:
            bbox = det["bbox"]
            conf = det["confidence"]
            landmarks = det.get("landmarks")
            
            # Filter by minimum box area
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < self.min_box_area:
                continue
            
            if conf >= self.track_thresh:
                high_conf_dets.append(det)
            else:
                low_conf_dets.append(det)
        
        # Get tracked and lost tracks
        # Include both "tracked" and "new" tracks for matching
        tracked_tracks = [t for t in self.tracks if t.state in ["tracked", "new"]]
        lost_tracks = [t for t in self.tracks if t.state == "lost"]
        
        # First association: match high-confidence detections with tracked tracks
        matched_tracks, unmatched_tracks, unmatched_dets = self._associate(
            tracked_tracks, high_conf_dets, self.match_thresh
        )
        
        # Second association: match low-confidence detections with unmatched tracks
        if len(unmatched_tracks) > 0 and len(low_conf_dets) > 0:
            matched_tracks_low, unmatched_tracks_low, _ = self._associate(
                unmatched_tracks, low_conf_dets, self.match_thresh
            )
            matched_tracks.extend(matched_tracks_low)
            unmatched_tracks = unmatched_tracks_low
        
        # Third association: match remaining high-confidence detections with lost tracks
        if len(lost_tracks) > 0 and len(unmatched_dets) > 0:
            matched_tracks_lost, unmatched_lost, unmatched_dets = self._associate(
                lost_tracks, unmatched_dets, self.match_thresh
            )
            matched_tracks.extend(matched_tracks_lost)
        
        # Mark unmatched tracks as missed
        for track in unmatched_tracks:
            track.mark_missed()
        
        # Create new tracks for unmatched detections
        for det in unmatched_dets:
            new_track = Track(
                track_id=self.next_track_id,
                bbox=det["bbox"],
                confidence=det["confidence"],
                landmarks=det.get("landmarks"),
                state="new"
            )
            self.tracks.append(new_track)
            self.next_track_id += 1
        
        # Remove tracks that have been lost for too long
        self.tracks = [
            t for t in self.tracks
            if t.frames_since_update <= self.track_buffer
        ]
        
        # Return active tracks (tracked or new)
        active_tracks = [t for t in self.tracks if t.state in ["tracked", "new"]]
        
        logger.debug(f"Frame {self.frame_count}: {len(active_tracks)} active tracks, {len(self.tracks)} total tracks")
        return active_tracks
    
    def _associate(
        self,
        tracks: List[Track],
        detections: List[Dict],
        threshold: float
    ) -> Tuple[List[Track], List[Track], List[Dict]]:
        """
        Associate tracks with detections using IOU matching.
        
        Args:
            tracks: List of tracks
            detections: List of detections
            threshold: IOU threshold
        
        Returns:
            Tuple of (matched_tracks, unmatched_tracks, unmatched_detections)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], tracks, detections
        
        # Compute IOU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track.bbox, det["bbox"])
        
        # DEBUG: Log IOU values
        if not hasattr(self, '_logged_iou'):
            logger.info(f"IOU Matrix (first detection):")
            logger.info(f"  Threshold: {threshold}")
            if len(tracks) > 0 and len(detections) > 0:
                logger.info(f"  Track bbox: {tracks[0].bbox}")
                logger.info(f"  Detection bbox: {detections[0]['bbox']}")
                logger.info(f"  IOU: {iou_matrix[0, 0]:.4f}")
            self._logged_iou = True
        
        # Greedy matching
        matched_tracks = []
        matched_det_indices = set()
        matched_track_indices = set()
        
        # Sort by IOU (highest first)
        matches = []
        for i in range(len(tracks)):
            for j in range(len(detections)):
                if iou_matrix[i, j] >= threshold:
                    matches.append((i, j, iou_matrix[i, j]))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        
        for track_idx, det_idx, iou in matches:
            if track_idx not in matched_track_indices and det_idx not in matched_det_indices:
                track = tracks[track_idx]
                det = detections[det_idx]
                track.update(det["bbox"], det["confidence"], det.get("landmarks"))
                matched_tracks.append(track)
                matched_track_indices.add(track_idx)
                matched_det_indices.add(det_idx)
        
        # Get unmatched tracks and detections
        unmatched_tracks = [t for i, t in enumerate(tracks) if i not in matched_track_indices]
        unmatched_dets = [d for i, d in enumerate(detections) if i not in matched_det_indices]
        
        return matched_tracks, unmatched_tracks, unmatched_dets
    
    def _iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union (IOU) between two bounding boxes.
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
        
        Returns:
            IOU value
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_track_id = 1
        self.frame_count = 0
        logger.info("ByteTracker reset")
    
    def __repr__(self) -> str:
        return (
            f"ByteTracker(track_thresh={self.track_thresh}, "
            f"match_thresh={self.match_thresh}, "
            f"active_tracks={len([t for t in self.tracks if t.state in ['tracked', 'new']])})"
        )
