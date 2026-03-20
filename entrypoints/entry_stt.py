"""
Entry point for the STT service (Wall-E Speech-to-Text).
==========================================================
Runs an interactive loop:
  - Listens for speech via microphone
  - Transcribes Thai audio using Typhoon ASR
  - Displays transcribed text

Usage inside Docker:
    python entrypoints/entry_stt.py
"""

import sys
import os

# Ensure project root is on Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils_stt import load_config
from src.pipelines.stt_pipeline import STTPipeline


def main():
    print("=" * 60)
    print("STT SERVICE  (Wall-E Speech-to-Text)")
    print("=" * 60)

    # Load YAML config (env vars override YAML values)
    print("\n📋 Loading configuration...")
    cfg = load_config()
    print("✅ Configuration loaded")

    model_name = os.getenv("STT_MODEL", cfg.model.name)
    device = os.getenv("STT_DEVICE", cfg.model.device)
    language = os.getenv("STT_LANGUAGE", cfg.model.language)

    pipeline = STTPipeline(
        model_name=model_name,
        device=device,
        language=language,
        silence_threshold=float(os.getenv("STT_SILENCE_THRESHOLD", cfg.recording.silence_threshold)),
        silence_duration=float(os.getenv("STT_SILENCE_DURATION", cfg.recording.silence_duration)),
        max_duration=float(os.getenv("STT_MAX_DURATION", cfg.recording.max_duration)),
        sample_rate=int(os.getenv("STT_SAMPLE_RATE", cfg.recording.sample_rate)),
    )

    print("\nReady — press Ctrl+C to quit\n")

    while True:
        try:
            text = pipeline.listen_and_transcribe()
            if text:
                print(f"\n✅ Result: {text}\n")
            else:
                print("(no speech detected, try again)\n")
        except KeyboardInterrupt:
            print("\n\n👋 Stopping STT service")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
