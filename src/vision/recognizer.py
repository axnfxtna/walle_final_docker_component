"""
Face recognition using ArcFace model.

This module extracts face embeddings using the ArcFace model for face recognition.
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple
import onnxruntime as ort
from loguru import logger


class FaceRecognizer:
    """
    ArcFace-based face recognition for extracting face embeddings.
    
    Features:
    - FP16 precision for Tensor Core optimization
    - TensorRT execution provider
    - Face alignment and preprocessing
    - Normalized 512-D embeddings
    - Batch processing support
    """
    
    def __init__(
        self,
        model_path: str,
        embedding_dim: int = 512,
        use_tensorrt: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize face recognizer.
        
        Args:
            model_path: Path to ArcFace ONNX model
            embedding_dim: Embedding dimension (typically 512)
            use_tensorrt: Use TensorRT execution provider
            device: Device to run on ("cuda" or "cpu")
        """
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        self.use_tensorrt = use_tensorrt
        self.device = device
        
        # Initialize ONNX Runtime session
        self.session = self._create_session()
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = (input_shape[2], input_shape[3])  # (height, width)
        
        logger.info(f"FaceRecognizer initialized with model: {model_path}")
        logger.info(f"Input size: {self.input_size}, Embedding dim: {embedding_dim}")
    
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
    
    def align_face(
        self,
        image: np.ndarray,
        bbox: List[float],
        landmarks: Optional[List[List[float]]] = None
    ) -> np.ndarray:
        """
        Align and crop face from image.
        
        Args:
            image: Input image (BGR format)
            bbox: Bounding box [x1, y1, x2, y2]
            landmarks: Optional 5-point landmarks for better alignment
        
        Returns:
            Aligned face image
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add margin
        margin = 0.2
        width = x2 - x1
        height = y2 - y1
        x1 = max(0, int(x1 - width * margin))
        y1 = max(0, int(y1 - height * margin))
        x2 = min(image.shape[1], int(x2 + width * margin))
        y2 = min(image.shape[0], int(y2 + height * margin))
        
        # Crop face
        face = image[y1:y2, x1:x2]
        
        # Resize to input size
        face = cv2.resize(face, self.input_size)
        
        return face
    
    def preprocess(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for ArcFace model.
        
        Args:
            face: Face image (BGR format)
        
        Returns:
            Preprocessed tensor
        """
        # Convert to RGB
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1]
        normalized = (rgb.astype(np.float32) - 127.5) / 127.5
        
        # Transpose to CHW format and add batch dimension
        transposed = normalized.transpose(2, 0, 1)
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def extract_embedding(
        self,
        image: np.ndarray,
        bbox: List[float],
        landmarks: Optional[List[List[float]]] = None
    ) -> np.ndarray:
        """
        Extract face embedding from image.
        
        Args:
            image: Input image (BGR format)
            bbox: Bounding box [x1, y1, x2, y2]
            landmarks: Optional 5-point landmarks
        
        Returns:
            Normalized face embedding (512-D)
        """
        # Align face
        face = self.align_face(image, bbox, landmarks)
        
        # Preprocess
        input_tensor = self.preprocess(face)
        
        # Run inference
        embedding = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        
        # Normalize embedding
        embedding = embedding.flatten()
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def extract_embeddings_batch(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        landmarks_list: Optional[List[List[List[float]]]] = None
    ) -> List[np.ndarray]:
        """
        Extract embeddings for multiple faces in batch.
        
        Args:
            image: Input image (BGR format)
            bboxes: List of bounding boxes
            landmarks_list: Optional list of landmarks for each face
        
        Returns:
            List of normalized face embeddings
        """
        if len(bboxes) == 0:
            return []
        
        # Prepare batch
        faces = []
        for i, bbox in enumerate(bboxes):
            landmarks = landmarks_list[i] if landmarks_list else None
            face = self.align_face(image, bbox, landmarks)
            faces.append(face)
        
        # Preprocess all faces
        batch = np.concatenate([self.preprocess(face) for face in faces], axis=0)
        
        # Run batch inference
        embeddings = self.session.run([self.output_name], {self.input_name: batch})[0]
        
        # Normalize embeddings
        normalized_embeddings = []
        for embedding in embeddings:
            embedding = embedding.flatten()
            embedding = embedding / np.linalg.norm(embedding)
            normalized_embeddings.append(embedding)
        
        logger.debug(f"Extracted {len(normalized_embeddings)} embeddings in batch")
        return normalized_embeddings
    
    def __repr__(self) -> str:
        return (
            f"FaceRecognizer(model={self.model_path}, "
            f"input_size={self.input_size}, "
            f"embedding_dim={self.embedding_dim})"
        )
