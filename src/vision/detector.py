"""
SCRFD Face Detector with ONNX Runtime and TensorRT support.

This module implements face detection using the SCRFD model from InsightFace,
optimized for NVIDIA Jetson Orin with FP16 precision and TensorRT execution provider.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import onnxruntime as ort
from loguru import logger


class SCRFDDetector:
    """
    SCRFD face detector with ONNX Runtime and TensorRT support.
    
    Features:
    - FP16 precision for Tensor Core optimization
    - TensorRT execution provider for maximum performance
    - Returns bounding boxes, landmarks (5 keypoints), and confidence scores
    - Configurable confidence and NMS thresholds
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        use_tensorrt: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize SCRFD detector.
        
        Args:
            model_path: Path to SCRFD ONNX model
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
            input_size: Model input size (width, height)
            use_tensorrt: Use TensorRT execution provider
            device: Device to run on ("cuda" or "cpu")
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.use_tensorrt = use_tensorrt
        self.device = device
        
        # Initialize ONNX Runtime session
        self.session = self._create_session()
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        logger.info(f"SCRFD detector initialized with model: {model_path}")
        logger.info(f"Input size: {input_size}, Confidence threshold: {confidence_threshold}")
    
    def _create_session(self) -> ort.InferenceSession:
        """Create ONNX Runtime session with tiered fallback: TensorRT → CUDA → CPU."""
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        def _try_create(providers):
            return ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers,
            )

        if self.device == "cuda":
            # Tier 1: TensorRT + CUDA + CPU
            if self.use_tensorrt:
                trt_opts = {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "./models/trt_cache",
                    "trt_max_workspace_size": 512 * 1024 * 1024,  # 512 MB — prevent OOM on Jetson
                }
                try:
                    session = _try_create([("TensorrtExecutionProvider", trt_opts),
                                           "CUDAExecutionProvider", "CPUExecutionProvider"])
                    logger.info(f"ONNX Runtime providers: {session.get_providers()}")
                    return session
                except Exception as e:
                    logger.warning(f"TensorRT init failed ({type(e).__name__}: {e}), retrying with CUDA only...")

            # Tier 2: CUDA + CPU
            try:
                session = _try_create(["CUDAExecutionProvider", "CPUExecutionProvider"])
                logger.info(f"ONNX Runtime providers: {session.get_providers()}")
                return session
            except Exception as e:
                logger.warning(f"CUDA init failed ({type(e).__name__}: {e}), falling back to CPU...")

        # Tier 3: CPU only
        try:
            session = _try_create(["CPUExecutionProvider"])
            logger.warning("Running on CPU — GPU unavailable (check memory / CUDA context).")
            logger.info(f"ONNX Runtime providers: {session.get_providers()}")
            return session
        except Exception as e:
            logger.error(f"Failed to create ONNX Runtime session even on CPU: {e}")
            raise
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Preprocess image for SCRFD model.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Preprocessed image tensor and scale factor
        """
        # Get original image size
        img_h, img_w = image.shape[:2]
        
        # Calculate scale to fit input size while maintaining aspect ratio
        scale = min(self.input_size[0] / img_w, self.input_size[1] / img_h)
        
        # Resize image
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        transposed = normalized.transpose(2, 0, 1)
        batched = np.expand_dims(transposed, axis=0)
        
        return batched, scale
    
    def postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        original_shape: Tuple[int, int]
    ) -> List[Dict]:
        """
        Postprocess model outputs to get detections.
        
        Args:
            outputs: Model outputs
            scale: Scale factor from preprocessing
            original_shape: Original image shape (height, width)
        
        Returns:
            List of detections with bboxes, confidences, and landmarks
        """
        detections = []
        
        # Handle Buffalo_L det_10g model (9 outputs)
        # This model uses 2 anchors per grid cell
        # Output format: [Score8, Score16, Score32, BBox8, BBox16, BBox32, KPS8, KPS16, KPS32]
        # BBox format: [left_dist, top_dist, right_dist, bottom_dist] from anchor center
        # KPS format: [x_offset, y_offset] pairs from anchor center
        if len(outputs) == 9:
            strides = [8, 16, 32]
            num_anchors = 2
            
            all_bboxes = []
            all_scores = []
            all_kps = []
            
            for i, stride in enumerate(strides):
                score_blob = outputs[i]          # (N, 1)
                bbox_blob = outputs[i + 3]       # (N, 4)
                kps_blob = outputs[i + 6]        # (N, 10)
                
                # Calculate grid dimensions
                grid_h = self.input_size[1] // stride
                grid_w = self.input_size[0] // stride
                
                # Reshape to (grid_h, grid_w, num_anchors, ...)
                scores = score_blob.reshape(grid_h, grid_w, num_anchors)
                bboxes = bbox_blob.reshape(grid_h, grid_w, num_anchors, 4)
                kps = kps_blob.reshape(grid_h, grid_w, num_anchors, 10)
                
                # Find high confidence detections
                high_conf_mask = scores > self.confidence_threshold
                high_conf_indices = np.where(high_conf_mask)
                
                # Decode each detection
                for idx in range(len(high_conf_indices[0])):
                    gy = high_conf_indices[0][idx]
                    gx = high_conf_indices[1][idx]
                    anchor_idx = high_conf_indices[2][idx]
                    
                    score = scores[gy, gx, anchor_idx]
                    bbox = bboxes[gy, gx, anchor_idx]
                    kp = kps[gy, gx, anchor_idx]
                    
                    # Calculate anchor center (no 0.5 offset for det_10g model)
                    cx = gx * stride
                    cy = gy * stride
                    
                    # Decode bbox: distances from anchor center
                    x1 = cx - bbox[0] * stride
                    y1 = cy - bbox[1] * stride
                    x2 = cx + bbox[2] * stride
                    y2 = cy + bbox[3] * stride
                    
                    decoded_bbox = np.array([x1, y1, x2, y2])
                    
                    # Decode keypoints: offsets from anchor center
                    decoded_kps = np.zeros(10)
                    for k in range(5):
                        decoded_kps[k*2] = cx + kp[k*2] * stride
                        decoded_kps[k*2+1] = cy + kp[k*2+1] * stride
                    
                    all_bboxes.append(decoded_bbox)
                    all_scores.append(score)
                    all_kps.append(decoded_kps)
            
            if not all_bboxes:
                return []
            
            # Convert to arrays
            bboxes = np.array(all_bboxes)
            scores = np.array(all_scores)
            landmarks = np.array(all_kps)
            
            # Apply NMS
            keep_indices = self._nms(bboxes, scores, self.nms_threshold)
            
            # Scale back to original image size
            for idx in keep_indices:
                det_bbox = bboxes[idx] / scale
                det_kpts = landmarks[idx].reshape(-1, 2) / scale
                
                detection = {
                    "bbox": det_bbox.tolist(),
                    "confidence": float(scores[idx]),
                    "landmarks": det_kpts.tolist()
                }
                detections.append(detection)
                
        # Handle other model structures (fallback)
        elif len(outputs) >= 3:
            # Original logic for standard models
            bboxes = outputs[0]
            scores = outputs[1]
            landmarks = outputs[2]
            
            # Filter by confidence
            valid_indices = scores > self.confidence_threshold
            
            if np.any(valid_indices):
                valid_bboxes = bboxes[valid_indices]
                valid_scores = scores[valid_indices]
                valid_landmarks = landmarks[valid_indices]
                
                # Apply NMS
                keep_indices = self._nms(valid_bboxes, valid_scores, self.nms_threshold)
                
                # Scale back to original image size
                for idx in keep_indices:
                    bbox = valid_bboxes[idx] / scale
                    kpts = valid_landmarks[idx].reshape(-1, 2) / scale
                    
                    detection = {
                        "bbox": bbox.tolist(),
                        "confidence": float(valid_scores[idx]),
                        "landmarks": kpts.tolist()
                    }
                    detections.append(detection)
        
        return detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """
        Non-Maximum Suppression.
        
        Args:
            boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
            scores: Confidence scores [N]
            threshold: IoU threshold
        
        Returns:
            Indices of boxes to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of detections with bboxes, confidences, and landmarks
        """
        # Preprocess
        input_tensor, scale = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # DEBUG: Print first output shape to identify model type
        if not hasattr(self, '_logged_shapes'):
            logger.info("ONNX Model Output Shapes:")
            for i, out in enumerate(outputs):
                logger.info(f"  Output {i}: {out.shape}")
            self._logged_shapes = True
        
        # Postprocess
        detections = self.postprocess(outputs, scale, image.shape[:2])
        
        logger.debug(f"Detected {len(detections)} faces")
        return detections
    
    def __repr__(self) -> str:
        return (
            f"SCRFDDetector(model={self.model_path}, "
            f"input_size={self.input_size}, "
            f"confidence_threshold={self.confidence_threshold})"
        )
