"""
Entry point for the TTS service (Wall-E Text-to-Speech).
==========================================================
Two modes run concurrently:

1. HTTP server  (port 5002, always-on)
   POST /speak  {"text": "Thai syllabified text"}
   GET  /health → 200 OK

2. Stdin interactive loop
   Type Thai text and press Enter to speak.
   Type 'quit' or press Ctrl+C to stop.

Usage inside Docker:
    python entrypoints/entry_tts.py
"""

import sys
import os
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Ensure project root is on Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils_tts import load_config
from src.pipelines.tts_pipeline import TTSPipeline

# ── Shared pipeline (set after init) ──────────────────────────────────────────
_pipeline: TTSPipeline | None = None
_speak_lock = threading.Lock()   # prevent overlapping playback

TTS_HTTP_PORT = int(os.getenv("TTS_HTTP_PORT", "5002"))


# ──────────────────────────────────────────────────────────────────────────────
# HTTP handler
# ──────────────────────────────────────────────────────────────────────────────

class TTSHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for /speak and /health endpoints."""

    def log_message(self, fmt, *args):  # silence default access log spam
        pass

    def _send(self, code: int, body: str, content_type: str = "application/json"):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            status = "ready" if _pipeline is not None else "loading"
            self._send(200, json.dumps({"status": status}))
        else:
            self._send(404, json.dumps({"error": "Not found"}))

    def do_POST(self):
        if self.path != "/speak":
            self._send(404, json.dumps({"error": "Not found"}))
            return

        # Read body
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
            text = payload.get("text", "").strip()
        except json.JSONDecodeError:
            self._send(400, json.dumps({"error": "Invalid JSON"}))
            return

        if not text:
            self._send(400, json.dumps({"error": "Empty 'text' field"}))
            return

        if _pipeline is None:
            self._send(503, json.dumps({"error": "TTS pipeline not ready yet"}))
            return

        # Speak (serialised so we don't overlap)
        try:
            with _speak_lock:
                print(f"🔊 [HTTP] Speaking: {text[:60]}...")
                _pipeline.speak(text)
                print("✅ [HTTP] Done")
            self._send(200, json.dumps({"ok": True, "text": text}))
        except Exception as e:
            print(f"❌ [HTTP] Error: {e}")
            self._send(500, json.dumps({"error": str(e)}))


def _run_http_server():
    server = HTTPServer(("0.0.0.0", TTS_HTTP_PORT), TTSHandler)
    print(f"🌐 TTS HTTP server listening on port {TTS_HTTP_PORT}")
    server.serve_forever()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    global _pipeline

    print("=" * 60)
    print("TTS SERVICE  (Wall-E Text-to-Speech)")
    print("=" * 60)

    # Load YAML config (env vars override YAML values)
    print("\n📋 Loading configuration...")
    cfg = load_config()
    print("✅ Configuration loaded")

    model_dir     = os.getenv("TTS_MODEL_DIR",      cfg.model.dir)
    default_model = os.getenv("TTS_DEFAULT_MODEL",  cfg.model.default)
    speaking_rate = float(os.getenv("TTS_SPEAKING_RATE", cfg.speaking_rate))

    _pipeline = TTSPipeline(
        model_dir=model_dir,
        default_model=default_model,
        speaking_rate=speaking_rate,
    )

    # Start HTTP server in a daemon thread
    http_thread = threading.Thread(target=_run_http_server, daemon=True)
    http_thread.start()

    print(f"\nReady — type Thai text and press Enter  |  POST to :{TTS_HTTP_PORT}/speak\n")

    # ── Stdin interactive loop ─────────────────────────────────────────────────
    while True:
        try:
            text = input("🗣️  Text: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Stopping TTS service")
            break

        if not text:
            continue

        if text.lower() in ("quit", "exit", "ออก", "จบ"):
            print("👋 Goodbye!")
            break

        try:
            with _speak_lock:
                print("🔊 Speaking...")
                _pipeline.speak(text)
                print("✅ Done\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()
