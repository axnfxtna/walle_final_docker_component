"""
STT Pipeline — Typhoon ASR
============================
Wraps TyphoonASR with a full record → transcribe loop.
Uses silence-detection-based recording from utils_stt.
"""

import os
import time
from typing import Dict, Optional
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from loguru import logger

from src.utils_stt import record_with_silence_detection, prepare_audio, save_audio


class TyphoonASR:
    """
    Typhoon ASR Speech-to-Text client for Thai language.
    Uses NeMo toolkit under the hood (scb10x/typhoon-asr-realtime).
    """

    def __init__(
        self,
        model_name: str = "scb10x/typhoon-asr-realtime",
        device: str = "auto",
        language: str = "th",
    ):
        self.model_name = model_name
        self.language = language

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading Typhoon ASR model '{model_name}' on {self.device}...")

        try:
            import nemo.collections.asr as nemo_asr

            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_name,
                map_location=torch.device(self.device),
            )
            logger.success("Typhoon ASR model loaded successfully")
        except ImportError:
            logger.error("Failed to import NeMo toolkit. Install: pip install nemo-toolkit[asr]")
            raise
        except Exception as e:
            logger.error(f"Failed to load Typhoon ASR model: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────
    #  Transcription helpers
    # ─────────────────────────────────────────────────────────────────

    def transcribe_file(self, audio_path: str) -> Dict:
        """Transcribe an audio file on disk."""
        try:
            start = time.time()
            processed = prepare_audio(audio_path)

            transcriptions = self.model.transcribe(audio=[processed])
            duration = time.time() - start

            text = self._extract_text(transcriptions)
            confidence = self._estimate_confidence(text, duration)

            # Remove temp file if created
            if processed != audio_path and os.path.exists(processed):
                os.remove(processed)

            logger.info(f"Transcribed in {duration:.2f}s: '{text[:60]}...'")
            return {
                "text": text,
                "confidence": confidence,
                "language": self.language,
                "duration": duration,
            }
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {"text": "", "confidence": 0.0, "language": self.language, "duration": 0.0, "error": str(e)}

    def transcribe_numpy(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict:
        """Transcribe a numpy audio array."""
        temp_path = "temp_audio_typhoon.wav"
        try:
            saved = save_audio(audio_data, sample_rate, temp_path)
            result = self.transcribe_file(saved)
            return result
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # ─────────────────────────────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────────────────────────────

    def _extract_text(self, transcriptions) -> str:
        if not transcriptions:
            return ""
        result = transcriptions[0]
        if hasattr(result, "text"):
            return (result.text or "").strip()
        return str(result).strip()

    def _estimate_confidence(self, text: str, duration: float) -> float:
        if not text:
            return 0.0
        confidence = 0.8
        if len(text) < 5:
            confidence -= 0.2
        elif len(text) > 200:
            confidence -= 0.1
        if duration < 0.5:
            confidence += 0.1
        elif duration > 5.0:
            confidence -= 0.1
        return round(max(0.0, min(1.0, confidence)), 3)


class STTPipeline:
    """
    High-level STT pipeline:
      record (with silence detection) → transcribe → return text
    """

    def __init__(
        self,
        model_name: str = "scb10x/typhoon-asr-realtime",
        device: str = "auto",
        language: str = "th",
        silence_threshold: float = 0.02,
        silence_duration: float = 3.0,
        max_duration: float = 30.0,
        sample_rate: int = 16000,
    ):
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_duration = max_duration
        self.sample_rate = sample_rate

        print("🔄 Loading Typhoon ASR model...")
        self.asr = TyphoonASR(model_name=model_name, device=device, language=language)
        print("✅ STT pipeline ready")

    def listen_and_transcribe(self) -> Optional[str]:
        """
        Record from microphone until silence, then transcribe.
        Returns the transcribed text, or None on failure.
        """
        audio_data, duration = record_with_silence_detection(
            silence_threshold=self.silence_threshold,
            silence_duration=self.silence_duration,
            max_duration=self.max_duration,
            sample_rate=self.sample_rate,
        )

        if audio_data is None or duration < 0.5:
            print("❌ No audio captured")
            return None

        print("🔄 Transcribing...")
        result = self.asr.transcribe_numpy(audio_data, self.sample_rate)
        text = result.get("text", "").strip()
        confidence = result.get("confidence", 0.0)

        if text:
            print(f"📝 Transcribed: {text}  (confidence: {confidence:.1%})")
            return text
        else:
            print("❌ Empty transcription result")
            return None
