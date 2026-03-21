"""
Entry point for the Thai Reader auto-mode service.
====================================================
Polls per-person chat_history/chat_*.json files (written by walle-rag) and
prints syllabified Thai output to the terminal in real-time.

Each person's answers are saved in their own file, e.g.
  chat_Palm__Krittin_Sakharin.json
The reader watches all chat_*.json files and processes new entries as they
appear.

Usage:
    # Standalone (host):
    cd /home/sarucha3/walle_capstone/final_docker_component
    python entrypoints/entry_thaireader.py

    # Docker:
    sudo docker compose --profile interactive up walle-thaireader

Environment variables:
    CHAT_HISTORY_DIR   Directory containing per-person chat_*.json files.
                       (default: /app/database/chat_history)
    RAG_ANSWERS_PATH   Legacy fallback single-file path used when
                       CHAT_HISTORY_DIR does not exist.
                       (default: /app/database/rag_answers.json)
    POLL_INTERVAL      Seconds between polls. (default: 1.0)
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipelines.thaireader_pipeline import ThaiReaderAutoMode


def main() -> None:
    print("=" * 60)
    print("THAI READER SERVICE  (RAG → Syllabify → Terminal)")
    print("=" * 60)

    chat_history_dir = os.environ.get(
        "CHAT_HISTORY_DIR", "/app/database/chat_history"
    )
    json_path = os.environ.get(
        "RAG_ANSWERS_PATH", "/app/database/rag_answers.json"
    )
    poll_interval = float(os.environ.get("POLL_INTERVAL", "1.0"))

    auto = ThaiReaderAutoMode(
        chat_history_dir=chat_history_dir,
        json_path=json_path,
        poll_interval=poll_interval,
    )
    auto.run()


if __name__ == "__main__":
    main()
