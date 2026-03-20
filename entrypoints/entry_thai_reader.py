"""
Entry point for the Thai Reader MCP.
=====================================
Interactive test loop — reads raw Thai text from stdin,
prints the syllabified form ready for TTS.

Usage:
    cd /home/sarucha3/walle_capstone/final_docker_component
    python entrypoints/entry_thai_reader.py
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipelines.thai_reader_pipeline import ThaiReaderPipeline


def main():
    print("=" * 60)
    print("THAI READER MCP  (syllabifier for TTS)")
    print("=" * 60)
    print("Type Thai text and press Enter to see syllabified output.")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    reader = ThaiReaderPipeline()

    while True:
        try:
            raw = input("📝 Thai text  : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Exiting Thai Reader...")
            break

        if not raw:
            continue

        if raw.lower() in ("quit", "exit", "q", "ออก"):
            print("👋 Goodbye!")
            break

        result = reader.process(raw)
        print(f"🔊 TTS-ready  : {result}\n")


if __name__ == "__main__":
    main()
