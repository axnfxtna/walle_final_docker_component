"""
TTS Utility Functions
======================
Covers:
  - Audio playback helpers
  - Thai text pre-processing
  - Temp file management
"""

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import sounddevice as sd
from loguru import logger
from omegaconf import OmegaConf


# =========================
# CONFIG
# =========================
def load_config(path="configs/tts.yaml"):
    """Load YAML configuration."""
    try:
        return OmegaConf.load(path)
    except Exception as e:
        print(f"❌ Error loading config from {path}: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────
#  Audio playback
# ─────────────────────────────────────────────────────────────────────

def play_audio_file(audio_path: str) -> None:
    """
    Play a WAV/audio file through the default sound device and block until done.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    data, samplerate = sf.read(audio_path)
    logger.info(f"Playing audio: {audio_path} ({len(data)/samplerate:.2f}s)")
    sd.play(data, samplerate)
    sd.wait()


def play_audio_array(audio_data: np.ndarray, sample_rate: int = 22050) -> None:
    """
    Play a numpy audio array and block until playback is complete.
    """
    sd.play(audio_data, sample_rate)
    sd.wait()


# ─────────────────────────────────────────────────────────────────────
#  File helpers
# ─────────────────────────────────────────────────────────────────────

def save_wav(audio_data: np.ndarray, sample_rate: int, output_path: str) -> str:
    """Save numpy audio array to a WAV file and return the path."""
    import scipy.io.wavfile as wavfile

    # scipy expects int16 or int32 for WAV; convert from float if needed
    if audio_data.dtype in (np.float32, np.float64):
        audio_int = (audio_data * 32767).astype(np.int16)
    else:
        audio_int = audio_data

    wavfile.write(output_path, sample_rate, audio_int)
    logger.debug(f"Saved WAV: {output_path}")
    return output_path


def make_temp_path(prefix: str = "tts_output", ext: str = "wav") -> str:
    """Generate a unique temporary file path in /tmp."""
    ts = int(time.time() * 1000)
    return f"/tmp/{prefix}_{ts}.{ext}"


def cleanup(path: Optional[str]) -> None:
    """Remove a file if it exists (silently)."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
#  Thai text helpers
# ─────────────────────────────────────────────────────────────────────

def clean_thai_text(text: str) -> str:
    """
    Lightweight Thai text cleaner compatible with VachanaTTS.
    Falls back to pythainlp if available, otherwise strips extra spaces.
    """
    try:
        from pythainlp.util import normalize
        text = normalize(text)
    except ImportError:
        pass

    # Remove duplicate spaces / newlines
    import re
    text = re.sub(r"\s+", " ", text).strip()
    return text
