"""
Chat History → Milvus watcher pipeline.
========================================
Watches database/chat_history/chat_*.json files and upserts each
Q&A turn into the Milvus 'chat_history' collection.

Schema per record:
  id        : SHA-1(person + question)   – dedup key
  person    : speaker name (e.g. "Palm")
  timestamp : ISO-8601 string
  question  : user's question text
  answer    : Wall-E's answer text
  embedding : sentence-transformer vector of f"{question} {answer}"
"""

import os
import sys
import glob
import json
import time
import hashlib
import threading
from typing import Optional

from pymilvus import Collection, utility, CollectionSchema, FieldSchema, DataType
from sentence_transformers import SentenceTransformer

from src.utils_database import connect_milvus

# ─────────────────────────────────────────────────────────────────
_EMB_MODEL = "BAAI/bge-m3"
_COLLECTION = "chat_history"
_EMB_DIM = 1024
_POLL_INTERVAL = 10          # seconds between folder scans
_MAX_QUESTION_LEN = 4096
_MAX_ANSWER_LEN = 8192
_MAX_PERSON_LEN = 100
_MAX_TS_LEN = 30


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _truncate(s: str, n: int) -> str:
    return s[:n] if len(s) > n else s


def _ensure_collection() -> Collection:
    """Create chat_history collection + index if it doesn't exist, then load it."""
    if utility.has_collection(_COLLECTION):
        col = Collection(_COLLECTION)
        col.load()
        return col

    print(f"🆕 Creating Milvus collection '{_COLLECTION}'...")
    fields = [
        FieldSchema(name="id",        dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
        FieldSchema(name="person",    dtype=DataType.VARCHAR, max_length=_MAX_PERSON_LEN),
        FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=_MAX_TS_LEN),
        FieldSchema(name="question",  dtype=DataType.VARCHAR, max_length=_MAX_QUESTION_LEN),
        FieldSchema(name="answer",    dtype=DataType.VARCHAR, max_length=_MAX_ANSWER_LEN),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=_EMB_DIM),
    ]
    schema = CollectionSchema(fields, description="Chat history turns")
    col = Collection(_COLLECTION, schema)
    col.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}},
    )
    col.load()
    print(f"✅ Collection '{_COLLECTION}' ready")
    return col


def _id_exists(col: Collection, record_id: str) -> bool:
    try:
        res = col.query(expr=f'id == "{record_id}"', output_fields=["id"], limit=1)
        return len(res) > 0
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────
#  Main watcher class
# ─────────────────────────────────────────────────────────────────

class ChatHistoryWatcher:
    """
    Polls CHAT_HISTORY_DIR for chat_*.json files and upserts new
    Q/A turns into Milvus every POLL_INTERVAL seconds.
    """

    def __init__(
        self,
        chat_dir: str = "/app/database/chat_history",
        poll_interval: float = _POLL_INTERVAL,
        emb_model: str = _EMB_MODEL,
    ):
        self.chat_dir = chat_dir
        self.poll_interval = poll_interval

        print("🔌 Connecting to Milvus (ChatHistoryWatcher)...")
        connect_milvus()
        self.col = _ensure_collection()

        print(f"🧠 Loading embedder: {emb_model}")
        self.embedder = SentenceTransformer(emb_model)
        print("✅ ChatHistoryWatcher ready")

    # ──────────────────────────────────────────────────────────────
    def _scan_and_sync(self) -> None:
        """Read all chat_*.json files and upsert any unseen turns."""
        pattern = os.path.join(self.chat_dir, "chat_*.json")
        files = glob.glob(pattern)

        if not files:
            return

        new_ids: list = []
        new_persons: list = []
        new_timestamps: list = []
        new_questions: list = []
        new_answers: list = []
        new_embeddings: list = []

        for filepath in files:
            try:
                with open(filepath, "r", encoding="utf-8") as fh:
                    turns = json.load(fh)
                if not isinstance(turns, list):
                    continue
            except (json.JSONDecodeError, IOError):
                continue

            for turn in turns:
                person    = str(turn.get("person", "guest"))
                timestamp = str(turn.get("timestamp", ""))
                question  = str(turn.get("question", "")).strip()
                answer    = str(turn.get("answer",   "")).strip()

                if not question:
                    continue

                record_id = _sha1(person + question)

                # Skip if already in Milvus
                if _id_exists(self.col, record_id):
                    continue

                # Embed combined text
                combined = f"{question} {answer}"
                emb = self.embedder.encode(combined, normalize_embeddings=True).tolist()

                new_ids.append(record_id)
                new_persons.append(_truncate(person, _MAX_PERSON_LEN))
                new_timestamps.append(_truncate(timestamp, _MAX_TS_LEN))
                new_questions.append(_truncate(question, _MAX_QUESTION_LEN))
                new_answers.append(_truncate(answer, _MAX_ANSWER_LEN))
                new_embeddings.append(emb)

        if new_ids:
            print(f"  💾  Upserting {len(new_ids)} new chat turn(s) to Milvus...")
            self.col.insert([
                new_ids,
                new_persons,
                new_timestamps,
                new_questions,
                new_answers,
                new_embeddings,
            ])
            self.col.flush()
            print(f"  ✅  Upserted {len(new_ids)} turn(s)")

    # ──────────────────────────────────────────────────────────────
    def run_forever(self) -> None:
        """Block forever, polling every poll_interval seconds."""
        print(f"\n👀 ChatHistoryWatcher watching: {self.chat_dir}")
        print(f"   Poll interval: {self.poll_interval}s  (Ctrl+C to stop)\n")
        while True:
            try:
                self._scan_and_sync()
            except Exception as exc:
                print(f"  ⚠️  Watcher scan error: {exc}")
            time.sleep(self.poll_interval)

    # ──────────────────────────────────────────────────────────────
    def start_background(self) -> threading.Thread:
        """Start watcher in a daemon thread and return it."""
        t = threading.Thread(target=self.run_forever, daemon=True, name="ChatHistoryWatcher")
        t.start()
        return t


# ─────────────────────────────────────────────────────────────────
#  Standalone entry (for testing)
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    chat_dir = os.environ.get("CHAT_HISTORY_DIR", "/app/database/chat_history")
    watcher = ChatHistoryWatcher(chat_dir=chat_dir)
    watcher.run_forever()
