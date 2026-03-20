"""
test_rag_to_tts.py
==================
End-to-end pipeline: RAG response → Thai Reader MCP → TTS docker HTTP /speak.

Usage
-----
# Standalone: syllabify a fixed text, then POST to TTS
    python test_rag_to_tts.py --text "ปี 4 ของคุณจะเน้นการปฏิบัติงานจริง"

# Live: read latest event from received_events.json, syllabify and speak
    python test_rag_to_tts.py --live

# Live + keep watching
    python test_rag_to_tts.py --live --watch

Options
-------
--text TEXT         Raw Thai text to syllabify and speak
--live              Read latest RAG response from received_events.json
--json PATH         Path to received_events.json
                    (default: /home/sarucha3/walle_capstone/server_package/received_events.json)
--tts-url URL       TTS HTTP base URL (default: http://localhost:5002)
--watch             Keep watching received_events.json for new events
--no-speak          Syllabify only — do NOT send to TTS docker
"""

import argparse
import json
import os
import sys
import time

import requests

# Ensure project src is on the path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipelines.thai_reader_pipeline import ThaiReaderPipeline


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_JSON_PATH = (
    "/home/sarucha3/walle_capstone/server_package/received_events.json"
)
DEFAULT_TTS_URL = "http://localhost:5002"


# ──────────────────────────────────────────────────────────────────────────────
# TTS HTTP client
# ──────────────────────────────────────────────────────────────────────────────

def check_tts_health(tts_url: str) -> bool:
    """Return True if the TTS HTTP server is up and ready."""
    try:
        r = requests.get(f"{tts_url}/health", timeout=3)
        data = r.json()
        return r.status_code == 200 and data.get("status") == "ready"
    except Exception:
        return False


def send_to_tts(syllabified_text: str, tts_url: str) -> bool:
    """
    POST syllabified Thai text to the TTS docker's /speak endpoint.
    Returns True on success.
    """
    endpoint = f"{tts_url}/speak"
    try:
        r = requests.post(
            endpoint,
            json={"text": syllabified_text},
            timeout=60,          # synthesis can take a few seconds
        )
        if r.status_code == 200:
            print(f"✅ TTS speaking done")
            return True
        else:
            print(f"❌ TTS error {r.status_code}: {r.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to TTS at {endpoint}")
        print("   → Is walle-tts running? (docker compose --profile interactive up walle-tts)")
        return False
    except requests.exceptions.Timeout:
        print("❌ TTS request timed out (synthesis may still be running)")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_latest_event(json_path: str) -> dict | None:
    """Return the newest event from received_events.json, or None."""
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            events = json.load(f)
        if not events:
            print("⚠️  No events in JSON file yet.")
            return None
        return events[-1]
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON decode error (file may be mid-write): {e}")
        return None


def extract_rag_answer(event: dict) -> str:
    """
    Pull the Wall-E answer from an event dict.
    Falls back to the STT question text if 'rag_answer' is absent.
    """
    answer = event.get("rag_answer", "").strip()
    if answer:
        return answer
    print("ℹ️  'rag_answer' key not in event — using STT text as fallback.")
    return event.get("stt", {}).get("text", "").strip()


def process_and_speak(
    raw_text: str,
    reader: ThaiReaderPipeline,
    tts_url: str,
    speak: bool = True,
) -> str:
    """Syllabify *raw_text* and optionally POST it to TTS. Returns syllabified text."""
    print(f"\n📥 Raw text   : {raw_text}")
    syllabified = reader.process(raw_text)
    print(f"🔊 TTS-ready  : {syllabified}")

    if speak:
        print(f"📤 Sending to TTS ({tts_url}/speak)...")
        send_to_tts(syllabified, tts_url)

    return syllabified


# ──────────────────────────────────────────────────────────────────────────────
# Modes
# ──────────────────────────────────────────────────────────────────────────────

def run_standalone(text: str, reader: ThaiReaderPipeline, tts_url: str, speak: bool) -> None:
    print("\n" + "─" * 60)
    process_and_speak(text, reader, tts_url, speak=speak)
    print("─" * 60)


def run_live(
    json_path: str,
    reader: ThaiReaderPipeline,
    tts_url: str,
    speak: bool,
    watch: bool,
) -> None:
    last_seen_time: str | None = None

    print(f"\n📂 Watching: {json_path}")
    if watch:
        print("🔄 Watch mode ON — Ctrl+C to stop\n")

    while True:
        event = load_latest_event(json_path)
        if event:
            current_time = event.get("received_at")
            if current_time != last_seen_time:
                last_seen_time = current_time

                student  = event.get("person_id", "Unknown")
                question = event.get("stt", {}).get("text", "")
                answer   = extract_rag_answer(event)

                print(f"\n{'─' * 60}")
                print(f"🙋 Student  : {student}")
                print(f"❓ Question : {question}")
                print(f"🤖 Answer   : {answer}")

                if answer:
                    process_and_speak(answer, reader, tts_url, speak=speak)
                else:
                    print("⚠️  No answer text to process.")

                print("─" * 60)

        if not watch:
            break
        time.sleep(1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAG response → Thai Reader MCP → TTS docker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--text", "-t", help="Raw Thai text to syllabify and speak")
    p.add_argument("--live", "-l", action="store_true",
                   help="Read latest response from received_events.json")
    p.add_argument("--json", default=DEFAULT_JSON_PATH, metavar="PATH",
                   help=f"Path to received_events.json  (default: {DEFAULT_JSON_PATH})")
    p.add_argument("--tts-url", default=DEFAULT_TTS_URL, metavar="URL",
                   help=f"TTS HTTP base URL  (default: {DEFAULT_TTS_URL})")
    p.add_argument("--watch", "-w", action="store_true",
                   help="Keep watching received_events.json for new events")
    p.add_argument("--no-speak", action="store_true",
                   help="Syllabify only — do NOT send to TTS docker")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    speak = not args.no_speak

    print("=" * 60)
    print("RAG → THAI READER → TTS  |  test script")
    print("=" * 60)

    reader = ThaiReaderPipeline()

    # Check TTS health (non-blocking — only warn)
    if speak:
        print(f"\n🔍 Checking TTS ({args.tts_url}/health)...")
        if check_tts_health(args.tts_url):
            print("✅ TTS docker is ready")
        else:
            print("⚠️  TTS docker not responding — will still try to send")

    if args.text:
        run_standalone(args.text, reader, args.tts_url, speak)

    elif args.live:
        try:
            run_live(
                json_path=args.json,
                reader=reader,
                tts_url=args.tts_url,
                speak=speak,
                watch=args.watch,
            )
        except KeyboardInterrupt:
            print("\n\n👋 Stopped.")

    else:
        # Default: built-in demo
        demo = (
            "ปี 4 ของคุณจะเน้นการปฏิบัติงานจริงและออกแบบระบบ"
            "ในด้านหุ่นยนต์และปัญญาประดิษฐ์อย่างลึกซึ้งครับ "
            "มีรายวิชาหลักๆ รวมถึง ROBOTICS AND AI ENGINEERING PROJECTS"
        )
        print("\n💡 No mode specified — running built-in demo:\n")
        run_standalone(demo, reader, args.tts_url, speak)
        print("\nRun with --help to see all options.\n")


if __name__ == "__main__":
    main()
