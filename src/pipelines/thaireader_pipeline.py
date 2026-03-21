"""
thaireader_pipeline.py
======================
Polls per-person chat_history/chat_*.json files written by walle-rag,
syllabifies each new Thai answer using ThaiReaderPipeline, and prints the
result to the terminal in real-time.

Each person's answers are stored in their own file, e.g.
  chat_Palm__Krittin_Sakharin.json
so the reader scans all chat_*.json files and tracks the latest entry
seen per file.
"""

import os
import sys
import time
import glob
import json
import urllib.request
import urllib.error

from src.pipelines.thai_reader_pipeline import ThaiReaderPipeline
from src.utils_thaireader import load_latest_rag_answer, load_latest_chat_history_entry

TTS_HTTP_URL = os.getenv("TTS_HTTP_URL", "http://localhost:5002")


# ─────────────────────────────────────────────────────────────────────────────

class ThaiReaderAutoMode:
    """
    Watches per-person chat_history JSON files for new RAG answers and
    syllabifies them.

    Args:
        chat_history_dir:  Directory containing chat_*.json files.
                           Falls back to watching a single rag_answers.json
                           if this directory does not exist.
        json_path:         Legacy fallback path to rag_answers.json.
        poll_interval:     Seconds between polls (default 1.0).
    """

    def __init__(
        self,
        chat_history_dir: str = "/app/database/chat_history",
        json_path: str = "/app/database/rag_answers.json",
        poll_interval: float = 1.0,
    ):
        self.chat_history_dir = chat_history_dir
        self.json_path = json_path
        self.poll_interval = poll_interval
        self.tts_url = TTS_HTTP_URL
        self.reader = ThaiReaderPipeline()

        # Per-file tracking: {file_path: last_seen_timestamp}
        self._last_seen: dict[str, str] = {}

    # ------------------------------------------------------------------
    def _send_to_tts(self, text: str) -> None:
        """POST syllabified text to the TTS docker's /speak endpoint."""
        url = f"{self.tts_url}/speak"
        payload = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
                wav  = body.get("wav", "")
                print(f"🎵 TTS saved : {wav}")
        except urllib.error.URLError as e:
            print(f"⚠️  TTS unreachable ({url}): {e}")
        except Exception as e:
            print(f"⚠️  TTS error: {e}")

    # ------------------------------------------------------------------
    def _process_entry(self, entry: dict) -> None:
        """Syllabify a single chat history entry, print it, then speak via TTS."""
        person   = entry.get("person",   "Unknown")
        question = entry.get("question", "")
        answer   = entry.get("answer",   "")

        if not answer.strip():
            print("⚠️  Empty answer — skipping.")
            return

        syllabified = self.reader.process(answer)

        print("\n" + "─" * 60)
        print(f"🙋 Student  : {person}")
        print(f"❓ Question  : {question}")
        print(f"📥 RAG Answer (raw)       : {answer}")
        print(f"🔊 TTS-ready (syllabified): {syllabified}")
        print("─" * 60)
        sys.stdout.flush()

        self._send_to_tts(syllabified)

    # ------------------------------------------------------------------
    def _poll_chat_history_dir(self) -> None:
        """
        Scan every chat_*.json file in chat_history_dir.
        For each file, check whether its latest entry is newer than what
        we last processed.  If so, process the new entry.
        """
        pattern = os.path.join(self.chat_history_dir, "chat_*.json")
        for file_path in glob.glob(pattern):
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, list) or not data:
                    continue
                entry = data[-1]
                ts = entry.get("timestamp", "")
                if ts and ts != self._last_seen.get(file_path):
                    self._last_seen[file_path] = ts
                    self._process_entry(entry)
            except (json.JSONDecodeError, IOError):
                continue

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Blocking loop — poll until Ctrl+C."""
        print("\n" + "=" * 60)
        print("THAI READER  (Real-time RAG → Syllabify)")
        print("=" * 60)

        use_dir = os.path.isdir(self.chat_history_dir)
        if use_dir:
            print(f"📂 Watching dir : {self.chat_history_dir}")
        else:
            print(f"📂 Watching file: {self.json_path}  (fallback)")
        print("🤖 รอรับคำตอบจาก RAG (Ctrl+C เพื่อออก)...\n")
        sys.stdout.flush()

        while True:
            try:
                if use_dir:
                    self._poll_chat_history_dir()
                else:
                    # Legacy single-file fallback
                    entry = load_latest_rag_answer(self.json_path)
                    if entry:
                        current_time = entry.get("received_at")
                        last = self._last_seen.get("__legacy__")
                        if current_time != last:
                            self._last_seen["__legacy__"] = current_time
                            self._process_entry(entry)

                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                print("\n\n👋 Stopping Thai Reader...")
                break
            except Exception as exc:
                print(f"  ❌ Error in poll loop: {exc}")
                import traceback
                traceback.print_exc()
                time.sleep(2.0)
