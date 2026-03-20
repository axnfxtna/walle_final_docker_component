"""
Enrollment Manager

Provides face quality checking and embedding validation for the
multi-angle student enrollment flow in demo_enrollment.py.
"""

import cv2
import numpy as np
from typing import Optional, List
from loguru import logger


class EnrollmentManager:
    """
    Manages face quality assessment during enrollment capture.

    Checks each captured frame for:
    - Face size (must be above min_face_size)
    - Sharpness / blur (Laplacian variance must exceed max_blur_threshold)
    - Basic orientation alignment per angle (via landmark symmetry heuristic)

    Also validates that a final set of embeddings is diverse enough
    (i.e. they are not all identical / degenerate).
    """

    def __init__(
        self,
        min_face_size: int = 112,
        max_blur_threshold: float = 100.0,
        required_angles: Optional[List[str]] = None,
        quality_threshold: float = 0.7,
    ):
        """
        Args:
            min_face_size:       Minimum face bounding-box side length (px).
            max_blur_threshold:  Laplacian variance below this → face is blurry.
            required_angles:     List of angle names expected during enrollment.
            quality_threshold:   Overall quality score [0,1] a face must reach.
        """
        self.min_face_size = min_face_size
        self.max_blur_threshold = max_blur_threshold
        self.required_angles = required_angles or ["straight", "left", "right", "up", "down"]
        self.quality_threshold = quality_threshold

    # ── Public API ─────────────────────────────────────────────────────────────

    def check_face_quality(
        self,
        frame: np.ndarray,
        bbox,
        landmarks=None,
        expected_angle: str = "straight",
    ) -> dict:
        """
        Assess the quality of a detected face in the given frame.

        Returns a dict with keys:
            passed        (bool)  – True if quality meets the threshold
            quality_score (float) – 0.0 … 1.0
            feedback      (str)   – human-readable reason for failure (if any)
        """
        issues = []
        scores = []

        x1, y1, x2, y2 = map(int, bbox)
        face_w = x2 - x1
        face_h = y2 - y1

        # ── 1. Size check ───────────────────────────────────────────────────
        min_dim = min(face_w, face_h)
        if min_dim < self.min_face_size:
            issues.append(f"Face too small ({min_dim}px < {self.min_face_size}px) — move closer")
            size_score = min_dim / self.min_face_size
        else:
            size_score = min(1.0, min_dim / (self.min_face_size * 2))
        scores.append(size_score)

        # ── 2. Blur check ────────────────────────────────────────────────────
        face_roi = frame[max(0, y1):y2, max(0, x1):x2]
        if face_roi.size > 0:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < self.max_blur_threshold:
                issues.append(f"Image blurry (score {laplacian_var:.0f}) — hold still")
                blur_score = laplacian_var / self.max_blur_threshold
            else:
                blur_score = min(1.0, laplacian_var / (self.max_blur_threshold * 5))
        else:
            blur_score = 0.0
            issues.append("Could not crop face region")
        scores.append(blur_score)

        # ── 3. Pose / angle hint (landmark symmetry) ─────────────────────────
        if landmarks and len(landmarks) >= 5:
            pose_score = self._pose_score(landmarks, expected_angle, face_w, face_h)
        else:
            pose_score = 0.8  # No landmarks → neutral score (don't penalise)
        scores.append(pose_score)

        # ── Aggregate ────────────────────────────────────────────────────────
        quality_score = float(np.mean(scores))
        passed = quality_score >= self.quality_threshold

        if issues:
            feedback = issues[0]
        elif not passed:
            feedback = f"Overall quality too low ({quality_score:.2f} < {self.quality_threshold:.2f}) — try better lighting or move closer"
        else:
            feedback = "Good quality"

        return {
            "passed": passed,
            "quality_score": quality_score,
            "feedback": feedback,
            "details": {
                "size_score":  round(size_score, 3),
                "blur_score":  round(blur_score, 3),
                "pose_score":  round(pose_score, 3),
                "laplacian":   round(laplacian_var if face_roi.size > 0 else 0, 1),
            },
        }

    def validate_embeddings(self, embeddings: list) -> bool:
        """
        Validate a list of face embeddings captured at different angles.

        Checks:
        - At least one embedding was captured.
        - No embedding is all-zeros (failed extraction).
        - Embeddings are not all identical (would indicate a bug).
        """
        if not embeddings:
            logger.warning("validate_embeddings: no embeddings provided")
            return False

        arrays = []
        for emb in embeddings:
            arr = np.array(emb, dtype=np.float32)
            if np.allclose(arr, 0.0):
                logger.warning("validate_embeddings: zero-vector embedding detected")
                return False
            arrays.append(arr)

        # Check that not all embeddings are identical
        if len(arrays) > 1:
            first = arrays[0]
            if all(np.allclose(first, a, atol=1e-6) for a in arrays[1:]):
                logger.warning("validate_embeddings: all embeddings are identical")
                return False

        return True

    # ── Private helpers ────────────────────────────────────────────────────────

    def _pose_score(self, landmarks, expected_angle: str, face_w: int, face_h: int) -> float:
        """
        Heuristic pose score based on 5-point landmark geometry.
        Landmarks order: left-eye, right-eye, nose, left-mouth, right-mouth.
        """
        try:
            lm = np.array(landmarks, dtype=np.float32)
            if lm.shape[0] < 5:
                return 0.8

            left_eye   = lm[0]
            right_eye  = lm[1]
            nose       = lm[2]
            left_mouth = lm[3]
            right_mouth = lm[4]

            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2

            # Horizontal offset: nose relative to eye mid-point (−1 left, +1 right)
            h_offset = (nose[0] - eye_center_x) / max(face_w, 1)

            # Vertical offset: nose relative to eye-mouth mid-point
            eye_y = (left_eye[1] + right_eye[1]) / 2
            mouth_y = (left_mouth[1] + right_mouth[1]) / 2
            mid_y = (eye_y + mouth_y) / 2
            v_offset = (nose[1] - mid_y) / max(face_h, 1)

            if expected_angle == "straight":
                score = 1.0 - min(1.0, abs(h_offset) * 4 + abs(v_offset) * 4)
            elif expected_angle == "left":
                score = 1.0 - min(1.0, max(0.0, h_offset + 0.05) * 8)
            elif expected_angle == "right":
                score = 1.0 - min(1.0, max(0.0, -h_offset + 0.05) * 8)
            elif expected_angle == "up":
                score = 1.0 - min(1.0, max(0.0, v_offset + 0.05) * 8)
            elif expected_angle == "down":
                score = 1.0 - min(1.0, max(0.0, -v_offset + 0.05) * 8)
            else:
                score = 0.8

            return float(np.clip(score, 0.0, 1.0))

        except Exception:
            return 0.8
