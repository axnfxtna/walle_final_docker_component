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
        save_path: Optional[str] = None,
        # Legacy params (ignored)
        model_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Synthesize and immediately play audio through the speaker.

        Args:
            text:          Thai text to speak.
            speaker:       Override the default speaker.
            cleanup_after: Delete the temp WAV after playback (ignored when save_path given).
            save_path:     If provided, save the WAV here permanently instead of a temp file.

        Returns:
            Path to the saved WAV file, or None if no permanent path was requested.
        """
        audio_path, _ = self.synthesize(text, speaker=speaker, output_path=save_path)
        try:
            play_audio_file(audio_path)
        finally:
            if save_path is None and cleanup_after:
                cleanup(audio_path)
        return audio_path if save_path else None

    # Legacy helper — kept so old code calling get_models() doesn't crash
    def get_models(self) -> list:
        """Return list of available KhanomTan speaker names."""
        return ["Tsyncone", "Tsynctwo", "Linda"]
