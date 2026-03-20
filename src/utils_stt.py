"""
STT Utility Functions
======================
Covers:
  - Audio recording with silence detection
  - Audio preprocessing helpers
"""

import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import os
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger
from omegaconf import OmegaConf


# =========================
# CONFIG
# =========================
def load_config(path="configs/stt.yaml"):
    """Load YAML configuration."""
    try:
        return OmegaConf.load(path)
    except Exception as e:
        print(f"❌ Error loading config from {path}: {e}")
        raise


def record_with_silence_detection(
    silence_threshold: float = 0.02,
    silence_duration: float = 3.0,
    max_duration: float = 30.0,
    sample_rate: int = 16000,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Record audio from the microphone until silence is detected.

    Args:
        silence_threshold: RMS amplitude below which is considered silence.
        silence_duration:  Seconds of continuous silence before stopping.
        max_duration:      Hard cap on recording time (seconds).
        sample_rate:       Sample rate in Hz.

    Returns:
        (audio_data, actual_duration_seconds)
        audio_data is None if nothing was captured.
    """
    print(f"\n🎤 Recording (stops after {silence_duration}s of silence, max {max_duration}s)...")

    audio_chunks: list = []
    silence_start = None
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        nonlocal silence_start
        if status:
            logger.warning(f"Audio status: {status}")

        rms = float(np.sqrt(np.mean(indata ** 2)))
        if rms < silence_threshold:
            if silence_start is None:
                silence_start = time.time()
        else:
            silence_start = None

        audio_chunks.append(indata.copy())

    with sd.InputStream(
        callback=callback,
        channels=1,
        samplerate=sample_rate,
        dtype="float32",
    ):
        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_duration:
                print(f"⏱️  Max duration reached ({max_duration}s)")
                break
            if silence_start and (time.time() - silence_start >= silence_duration):
                print(f"🔇 Silence detected for {silence_duration}s — stopping")
                break
            time.sleep(0.1)

    if audio_chunks:
        audio_data = np.concatenate(audio_chunks, axis=0)
        actual_duration = len(audio_data) / sample_rate
        print(f"✅ Recorded {actual_duration:.1f}s of audio")
        return audio_data, actual_duration

    return None, 0.0


def prepare_audio(audio_path: str, target_sr: int = 16000) -> str:
    """
    Resample audio file to target_sr if needed.

    Returns the path to a ready-to-use 16 kHz WAV file.
    Creates a temporary processed file if resampling is required.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = librosa.load(audio_path, sr=None)
    if sr == target_sr:
        return audio_path

    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    y = y / (max(abs(y)) + 1e-8)

    temp_path = f"processed_{Path(audio_path).stem}.wav"
    sf.write(temp_path, y, target_sr)
    return temp_path


def save_audio(audio_data: np.ndarray, sample_rate: int, path: str = "temp_stt_input.wav") -> str:
    """Save a numpy audio array to a WAV file and return the path."""
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    if audio_data.max() > 1.0:
        audio_data = audio_data / 32768.0
    sf.write(path, audio_data, sample_rate)
    return path
