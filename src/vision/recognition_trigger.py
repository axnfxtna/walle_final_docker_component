"""
Intelligent recognition triggering logic.

This module decides when to trigger face recognition based on:
- Face confidence
- Head pose (is student looking at robot?)
- Tracking stability
- Cooldown period
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class TriggerDecision:
    """Result of recognition trigger decision."""
    should_trigger: bool
    reason: str
    track_id: int
    confidence: float


class RecognitionTrigger:
    """
    Intelligent recognition triggering to save compute power.
    
    Only triggers face recognition when:
    1. Face confidence is high enough
    2. Student is looking at robot (based on head pose)
    3. Track has been stable for minimum frames
    4. Cooldown period has passed since last recognition
    """
    
    def __init__(
        self,
        min_track_frames: int = 3,
        cooldown_seconds: float = 5.0,
        require_attention: bool = True
    ):
        """
        Initialize recognition trigger.
        
        Args:
            min_track_frames: Minimum frames a track must exist before triggering
            cooldown_seconds: Cooldown period between recognitions for same track
            require_attention: Whether to require student looking at robot
        """
        self.min_track_frames = min_track_frames
        self.cooldown_seconds = cooldown_seconds
        self.require_attention = require_attention
        
        # Track last recognition time for each track_id
        self.last_recognition_time: Dict[int, float] = {}
        
        logger.info(
            f"RecognitionTrigger initialized: min_frames={min_track_frames}, "
            f"cooldown={cooldown_seconds}s, require_attention={require_attention}"
        )
    
    def should_trigger(
        self,
        track_id: int,
        confidence: float,
        track_age: int,
        is_looking: bool
    ) -> TriggerDecision:
        """
        Decide whether to trigger recognition for a track.
        
        Args:
            track_id: Track ID
            confidence: Detection confidence
            track_age: Number of frames track has existed
            is_looking: Whether student is looking at robot
        
        Returns:
            TriggerDecision with should_trigger flag and reason
        """
        current_time = time.time()
        
        # Check if track is stable enough
        if track_age < self.min_track_frames:
            return TriggerDecision(
                should_trigger=False,
                reason=f"Track too new ({track_age}/{self.min_track_frames} frames)",
                track_id=track_id,
                confidence=confidence
            )
        
        # Check if student is looking at robot (if required)
        if self.require_attention and not is_looking:
            return TriggerDecision(
                should_trigger=False,
                reason="Student not looking at robot",
                track_id=track_id,
                confidence=confidence
            )
        
        # Check cooldown period
        if track_id in self.last_recognition_time:
            time_since_last = current_time - self.last_recognition_time[track_id]
            if time_since_last < self.cooldown_seconds:
                return TriggerDecision(
                    should_trigger=False,
                    reason=f"Cooldown active ({time_since_last:.1f}/{self.cooldown_seconds}s)",
                    track_id=track_id,
                    confidence=confidence
                )
        
        # All checks passed - trigger recognition
        self.last_recognition_time[track_id] = current_time
        
        return TriggerDecision(
            should_trigger=True,
            reason="All conditions met",
            track_id=track_id,
            confidence=confidence
        )
    
    def reset_track(self, track_id: int):
        """Reset cooldown for a specific track."""
        if track_id in self.last_recognition_time:
            del self.last_recognition_time[track_id]
    
    def reset_all(self):
        """Reset all cooldowns."""
        self.last_recognition_time.clear()
        logger.info("RecognitionTrigger reset")
    
    def __repr__(self) -> str:
        return (
            f"RecognitionTrigger(min_frames={self.min_track_frames}, "
            f"cooldown={self.cooldown_seconds}s, "
            f"active_tracks={len(self.last_recognition_time)})"
        )
