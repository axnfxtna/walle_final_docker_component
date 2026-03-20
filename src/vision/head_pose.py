"""
Head pose estimation from facial landmarks.

This module estimates head pose (yaw, pitch, roll) from 5-point facial landmarks
to determine if a student is looking at the robot.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from loguru import logger


class HeadPoseEstimator:
    """
    Estimate head pose from facial landmarks.
    
    Uses 5-point landmarks (2 eyes, nose, 2 mouth corners) to estimate
    yaw, pitch, and roll angles.
    """
    
    def __init__(
        self,
        yaw_threshold: float = 30.0,
        pitch_threshold: float = 20.0,
        roll_threshold: float = 45.0
    ):
        """
        Initialize head pose estimator.
        
        Args:
            yaw_threshold: Maximum yaw angle for "looking at robot" (degrees)
            pitch_threshold: Maximum pitch angle for "looking at robot" (degrees)
            roll_threshold: Maximum roll angle for valid detection (degrees)
        """
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold
        
        # 3D model points for canonical face (approximate)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (-30.0, -30.0, -30.0),    # Left eye
            (30.0, -30.0, -30.0),     # Right eye
            (-20.0, 30.0, -20.0),     # Left mouth corner
            (20.0, 30.0, -20.0)       # Right mouth corner
        ], dtype=np.float32)
        
        logger.info(f"HeadPoseEstimator initialized with yaw_thresh={yaw_threshold}, pitch_thresh={pitch_threshold}")
    
    def estimate_pose(
        self,
        landmarks: List[List[float]],
        image_shape: Optional[Tuple[int, int]] = None
    ) -> Dict[str, float]:
        """
        Estimate head pose from facial landmarks.
        
        Args:
            landmarks: 5-point facial landmarks [[x, y], ...]
                Order: [left_eye, right_eye, nose, left_mouth, right_mouth]
            image_shape: Image shape (height, width) for camera matrix
        
        Returns:
            Dictionary with:
                - yaw: Yaw angle in degrees
                - pitch: Pitch angle in degrees
                - roll: Roll angle in degrees
                - is_looking: Whether person is looking at camera
        """
        if len(landmarks) != 5:
            logger.warning(f"Expected 5 landmarks, got {len(landmarks)}")
            return {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "is_looking": False
            }
        
        # Convert landmarks to numpy array
        image_points = np.array(landmarks, dtype=np.float32)
        
        # Use default image shape if not provided
        if image_shape is None:
            image_shape = (640, 640)
        
        # Camera matrix (simplified, assuming no distortion)
        focal_length = image_shape[1]
        center = (image_shape[1] / 2, image_shape[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Distortion coefficients (assuming no distortion)
        dist_coeffs = np.zeros((4, 1))
        
        try:
            # Solve PnP to get rotation and translation vectors
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                logger.warning("PnP solving failed")
                return {
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0,
                    "is_looking": False
                }
            
            # Convert rotation vector to rotation matrix
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            
            # Calculate Euler angles
            yaw, pitch, roll = self._rotation_matrix_to_euler_angles(rotation_mat)
            
            # Check if person is looking at camera
            is_looking = (
                abs(yaw) < self.yaw_threshold and
                abs(pitch) < self.pitch_threshold and
                abs(roll) < self.roll_threshold
            )
            
            return {
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll),
                "is_looking": is_looking
            }
        
        except Exception as e:
            logger.error(f"Error estimating head pose: {e}")
            return {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "is_looking": False
            }
    
    def estimate_simple(self, landmarks: List[List[float]]) -> Dict[str, float]:
        """
        Simple head pose estimation using landmark geometry.
        
        This is a faster, simpler alternative that doesn't require cv2.solvePnP.
        
        Args:
            landmarks: 5-point facial landmarks
        
        Returns:
            Dictionary with yaw, pitch, roll, and is_looking
        """
        if len(landmarks) != 5:
            return {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "is_looking": False
            }
        
        # Extract landmark points
        left_eye = np.array(landmarks[0])
        right_eye = np.array(landmarks[1])
        nose = np.array(landmarks[2])
        left_mouth = np.array(landmarks[3])
        right_mouth = np.array(landmarks[4])
        
        # Calculate eye center
        eye_center = (left_eye + right_eye) / 2
        
        # Calculate mouth center
        mouth_center = (left_mouth + right_mouth) / 2
        
        # Estimate yaw from horizontal face symmetry
        # Positive yaw = face turned right, negative = face turned left
        left_dist = np.linalg.norm(nose - left_eye)
        right_dist = np.linalg.norm(nose - right_eye)
        yaw = (right_dist - left_dist) / (right_dist + left_dist) * 90.0
        
        # Estimate pitch from vertical face position
        # Positive pitch = face looking up, negative = face looking down
        face_height = np.linalg.norm(eye_center - mouth_center)
        nose_to_eyes = np.linalg.norm(nose - eye_center)
        pitch = (nose_to_eyes / face_height - 0.5) * 60.0
        
        # Estimate roll from eye alignment
        # Positive roll = head tilted right, negative = head tilted left
        eye_diff = right_eye - left_eye
        roll = np.degrees(np.arctan2(eye_diff[1], eye_diff[0]))
        
        # Check if person is looking at camera
        is_looking = (
            abs(yaw) < self.yaw_threshold and
            abs(pitch) < self.pitch_threshold and
            abs(roll) < self.roll_threshold
        )
        
        return {
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "is_looking": is_looking
        }
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (yaw, pitch, roll).
        
        Args:
            R: 3x3 rotation matrix
        
        Returns:
            Tuple of (yaw, pitch, roll) in degrees
        """
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            yaw = np.arctan2(R[1, 0], R[0, 0])
            pitch = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[2, 1], R[2, 2])
        else:
            yaw = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            roll = 0
        
        # Convert to degrees
        yaw = np.degrees(yaw)
        pitch = np.degrees(pitch)
        roll = np.degrees(roll)
        
        return yaw, pitch, roll
    
    def __repr__(self) -> str:
        return (
            f"HeadPoseEstimator(yaw_thresh={self.yaw_threshold}, "
            f"pitch_thresh={self.pitch_threshold})"
        )


# Import cv2 only if available (for full PnP-based estimation)
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("OpenCV not available, using simple head pose estimation")
