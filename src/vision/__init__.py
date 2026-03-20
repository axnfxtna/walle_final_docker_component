"""
Vision module for face detection, tracking, and recognition.

This module provides:
- Face detection using SCRFD (InsightFace) - supports both ONNX and PyTorch
- Multi-face tracking using ByteTrack
- Head pose estimation from facial landmarks
- Face recognition using ArcFace
- Student identification with Milvus vector database
"""

from .detector import SCRFDDetector
from .detector_factory import create_scrfd_detector
from .tracker import ByteTracker
from .head_pose import HeadPoseEstimator
from .recognition_trigger import RecognitionTrigger
from .recognizer import FaceRecognizer

try:
    from .detector_pytorch import SCRFDPyTorchDetector
    _pytorch_available = True
except ImportError:
    SCRFDPyTorchDetector = None
    _pytorch_available = False

try:
    from .student_db import StudentDatabase
    _student_db_available = True
except ImportError:
    StudentDatabase = None
    _student_db_available = False

try:
    from .enrollment import EnrollmentManager
    _enrollment_available = True
except ImportError:
    EnrollmentManager = None
    _enrollment_available = False

try:
    from .pipeline import FaceRecognitionPipeline
    _pipeline_available = True
except ImportError:
    FaceRecognitionPipeline = None
    _pipeline_available = False

__all__ = [
    "SCRFDDetector",
    "create_scrfd_detector",
    "ByteTracker",
    "HeadPoseEstimator",
    "RecognitionTrigger",
    "FaceRecognizer",
]
if _pytorch_available:
    __all__.append("SCRFDPyTorchDetector")
if _student_db_available:
    __all__.append("StudentDatabase")
if _enrollment_available:
    __all__.append("EnrollmentManager")
if _pipeline_available:
    __all__.append("FaceRecognitionPipeline")


