"""
Factory function to create SCRFD detector based on model file type.

Automatically selects between ONNX and PyTorch implementations.
"""

from pathlib import Path
from typing import Tuple, Optional

def create_scrfd_detector(
    model_path: str,
    confidence_threshold: float = 0.7,
    nms_threshold: float = 0.4,
    input_size: Tuple[int, int] = (640, 640),
    use_tensorrt: bool = True,
    device: str = "cuda"
):
    """
    Create SCRFD detector based on model file type.
    
    Automatically detects model format:
    - .onnx files: Use ONNX Runtime implementation
    - .pth files: Use PyTorch implementation
    
    Args:
        model_path: Path to model file (.onnx or .pth)
        confidence_threshold: Minimum confidence for detections
        nms_threshold: NMS IoU threshold
        input_size: Model input size (width, height)
        use_tensorrt: Use TensorRT (ONNX only)
        device: Device to run on ("cuda" or "cpu")
    
    Returns:
        SCRFD detector instance
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Check file extension
    if model_path.suffix == '.onnx':
        # Use ONNX Runtime implementation
        from .detector import SCRFDDetector
        return SCRFDDetector(
            model_path=str(model_path),
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size,
            use_tensorrt=use_tensorrt,
            device=device
        )
    
    elif model_path.suffix == '.pth':
        # Use PyTorch implementation
        from .detector_pytorch import SCRFDPyTorchDetector
        return SCRFDPyTorchDetector(
            model_path=str(model_path),
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size,
            device=device
        )
    
    else:
        raise ValueError(
            f"Unsupported model format: {model_path.suffix}. "
            "Supported formats: .onnx, .pth"
        )
