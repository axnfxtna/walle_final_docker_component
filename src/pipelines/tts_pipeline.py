"""
TTS Pipeline — KhanomTan TTS v1.0 (YourTTS / Coqui-TTS)
==========================================================
Wraps PyThaiTTS with a clean synthesize → play interface.

Speaker:  Tsyncone  (Thai female voice from TSync-1 corpus)
Library:  pythaitts + TTS (coqui-tts)

Install:
    pip install pythaitts TTS
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger

from src.utils_tts import (
    clean_thai_text,
    save_wav,
    play_audio_file,
    make_temp_path,
    cleanup,
)

# ── PyThaiTTS availability check ──────────────────────────────────────────────
try:
    from pythaitts import TTS as PyThaiTTS
    KHANOMTAN_AVAILABLE = True
    logger.info("PyThaiTTS (KhanomTan) loaded successfully")
except ImportError as e:
    logger.warning(f"PyThaiTTS not available: {e}")
    KHANOMTAN_AVAILABLE = False


# Default speaker / language used for KhanomTan v1.0
DEFAULT_SPEAKER  = os.getenv("TTS_SPEAKER",  "Tsyncone")
DEFAULT_LANGUAGE = os.getenv("TTS_LANGUAGE", "th-th")


class TTSPipeline:
    """
    High-level TTS pipeline backed by KhanomTan TTS v1.0 (via PyThaiTTS).

    Usage::

        pipeline = TTSPipeline()
        pipeline.speak("สวัสดีครับ")

    Environment overrides:
        TTS_SPEAKER   — speaker name  (default: Tsyncone)
        TTS_LANGUAGE  — language code (default: th-th)
    """

    def __init__(
        self,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        speaking_rate: float = 1.0,
        # Legacy params accepted but ignored (kept for API compatibility)
        model_dir: Optional[str] = None,
        default_model: Optional[str] = None,
        auto_play: bool = True,
    ):
        if not KHANOMTAN_AVAILABLE:
            raise RuntimeError(
                "PyThaiTTS is not installed. "
                "Run: pip install pythaitts TTS"
            )

        self.speaker       = speaker  or DEFAULT_SPEAKER
        self.language      = language or DEFAULT_LANGUAGE
        self.speaking_rate = speaking_rate
        self.auto_play     = auto_play

        # Instantiate the model (downloads weights on first run)
        logger.info(
            f"Loading KhanomTan TTS v1.0  "
            f"(speaker={self.speaker}, lang={self.language})…"
        )
        try:
            self._tts = PyThaiTTS(pretrained="khanomtan")
            logger.success(
                f"TTSPipeline ready  "
                f"(KhanomTan / speaker={self.speaker})"
            )
        except Exception as e:
            logger.error(f"KhanomTan load failed: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        output_path: Optional[str] = None,
        # Legacy param (ignored)
        speaking_rate: Optional[float] = None,
        model_name: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        """
        Synthesize Thai text → WAV file and return (audio_path, metadata).

        Args:
            text:        Thai text to synthesize.
            speaker:     Override the default speaker (e.g., ``"Tsynctwo"``).
            language:    Override the language code.
            output_path: Save to this path; auto temp path if None.

        Returns:
            (wav_file_path, metadata_dict)
        """
        spk  = speaker  or self.speaker
        lang = language or self.language

        cleaned = clean_thai_text(text)
        logger.info(f"Synthesizing: '{cleaned[:60]}' (speaker={spk})")

        audio_path = output_path or make_temp_path("tts_khanomtan", "wav")

        # PyThaiTTS writes the WAV and returns the path
        try:
            result_path = self._tts.tts(
                cleaned,
                speaker_idx=spk,
                language_idx=lang,
                filename=audio_path,
            )
            # result_path may differ from audio_path in some versions
            if result_path and result_path != audio_path:
                audio_path = result_path
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise

        if not os.path.exists(audio_path):
            raise RuntimeError(f"TTS produced no output file at {audio_path}")

        metadata = {
            "text":         text,
            "cleaned_text": cleaned,
            "speaker":      spk,
            "language":     lang,
            "audio_file":   audio_path,
        }
        logger.success(f"Audio → {audio_path}")
        return audio_path, metadata

    def speak(
        self,
        text: str,
        speaker: Optional[str] = None,
        cleanup_after: bool = True,
        # Legacy params (ignored)
        model_name: Optional[str] = None,
    ) -> None:
        """
        Synthesize and immediately play audio through the speaker.

        Args:
            text:          Thai text to speak.
            speaker:       Override the default speaker.
            cleanup_after: Delete the temp WAV after playback.
        """
        audio_path, _ = self.synthesize(text, speaker=speaker)
        try:
            play_audio_file(audio_path)
        finally:
            if cleanup_after:
                cleanup(audio_path)

    # Legacy helper — kept so old code calling get_models() doesn't crash
    def get_models(self) -> list:
        """Return list of available KhanomTan speaker names."""
        return ["Tsyncone", "Tsynctwo", "Linda"]


import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger

from src.utils_tts import clean_thai_text, save_wav, play_audio_file, make_temp_path, cleanup

# ── locate VachanaTTS (cloned by Dockerfile into /app/VachanaTTS) ──
VACHANA_PATH = Path(os.getenv("VACHANA_PATH", "/app/VachanaTTS"))
if str(VACHANA_PATH) not in sys.path:
    sys.path.insert(0, str(VACHANA_PATH))

try:
    from inference.tts_with_voiceclone import generate_speech, save_audio, get_model_names
    VACHANA_AVAILABLE = True
    logger.info("VachanaTTS loaded successfully")
except ImportError as e:
    logger.warning(f"VachanaTTS not available: {e}")
    VACHANA_AVAILABLE = False


class TTSPipeline:
    """
    High-level TTS pipeline backed by VachanaTTS (VITS Thai models).

    Usage:
        pipeline = TTSPipeline()
        pipeline.speak("สวัสดีครับ")
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        default_model: Optional[str] = None,
        speaking_rate: float = 1.0,
        auto_play: bool = True,
    ):
        if not VACHANA_AVAILABLE:
            raise RuntimeError(
                "VachanaTTS is not available. "
                "Make sure /app/VachanaTTS is present in the container."
            )

        self.model_dir = model_dir or str(VACHANA_PATH / "models")
        self.speaking_rate = speaking_rate
        self.auto_play = auto_play

        # Discover models
        try:
            self.available_models: list = get_model_names(self.model_dir)
            logger.info(f"Found {len(self.available_models)} TTS model(s): {self.available_models}")
        except Exception as e:
            logger.error(f"Could not list TTS models: {e}")
            self.available_models = []

        # Pick default model (prefer MALEV1 for Wall-E)
        if default_model and default_model in self.available_models:
            self.default_model = default_model
        elif "MMS-TTS-THAI-MALEV1" in self.available_models:
            self.default_model = "MMS-TTS-THAI-MALEV1"
        elif self.available_models:
            self.default_model = self.available_models[0]
        else:
            self.default_model = None
            logger.warning("No TTS models found — place models in {self.model_dir}")

        logger.success(f"TTSPipeline ready (model: {self.default_model})")

    # ─────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        model_name: Optional[str] = None,
        speaking_rate: Optional[float] = None,
        output_path: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        """
        Synthesize Thai text to speech and return (audio_path, metadata).

        Args:
            text:          Thai text to speak.
            model_name:    Override the default model.
            speaking_rate: Override the default speaking rate.
            output_path:   Save to this path; auto-generated temp if None.

        Returns:
            (audio_file_path, metadata_dict)
        """
        if not self.available_models:
            raise RuntimeError("No TTS models available")

        model_name = model_name or self.default_model
        speaking_rate = speaking_rate if speaking_rate is not None else self.speaking_rate

        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not found. Available: {self.available_models}")

        cleaned = clean_thai_text(text)
        logger.info(f"Synthesizing: '{cleaned[:60]}' (model={model_name}, rate={speaking_rate})")

        sampling_rate, audio_data = generate_speech(
            cleaned, self.model_dir, model_name, speaking_rate
        )

        audio_path = output_path or make_temp_path("tts_output", "wav")
        save_wav(audio_data, sampling_rate, audio_path)

        metadata = {
            "text": text,
            "cleaned_text": cleaned,
            "model": model_name,
            "speaking_rate": speaking_rate,
            "sampling_rate": sampling_rate,
            "duration": len(audio_data) / sampling_rate,
            "audio_file": audio_path,
        }

        logger.success(f"Audio → {audio_path} ({metadata['duration']:.2f}s)")
        return audio_path, metadata

    def speak(self, text: str, model_name: Optional[str] = None, cleanup_after: bool = True) -> None:
        """
        Synthesize and immediately play audio through the speaker.

        Args:
            text:          Thai text to speak.
            model_name:    Override default model.
            cleanup_after: Delete the temp WAV after playback.
        """
        audio_path, _ = self.synthesize(text, model_name=model_name)
        try:
            play_audio_file(audio_path)
        finally:
            if cleanup_after:
                cleanup(audio_path)

    def get_models(self) -> list:
        """Return list of available TTS model names."""
        return self.available_models
