"""
utils_thaireader.py
===================
Utility helpers for the Thai Reader auto-mode service.

Handles:
  - Loading / parsing the rag_answers.json bridge file written by walle-rag.
  - Loading the latest entry from per-person chat_history/*.json files.
  - Writing a new rag_answer entry into that same file (used by walle-rag).
"""

import json
import os
import glob
import datetime
import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Reading side  (used by walle-thaireader)
# ─────────────────────────────────────────────────────────────────────────────

def load_latest_rag_answer(json_path: str) -> Optional[dict]:
    """
    Return the most-recent entry from *json_path*, or None if the file is
    missing / empty / unreadable.

    Each entry is a dict with keys:
        received_at   – ISO-8601 timestamp string
        person        – student display name
        question      – original STT question
        answer        – raw LLM answer text
    """
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list) and data:
            return data[-1]
        return None
    except (json.JSONDecodeError, IOError):
        return None


def load_latest_chat_history_entry(chat_history_dir: str) -> Optional[dict]:
    """
    Scan all chat_*.json files in *chat_history_dir* and return the single
    most-recent entry (by 'timestamp') across all per-person files, or None
    if nothing is found.

    Each entry is a dict with keys:
        timestamp  – ISO-8601 timestamp string
        person     – student display name
        question   – original STT question
        answer     – raw LLM answer text

    The returned dict also includes a synthetic 'received_at' key set to
    'timestamp' so that ThaiReaderAutoMode can use its existing dedup logic
    unchanged.
    """
    if not os.path.isdir(chat_history_dir):
        return None

    best_entry: Optional[dict] = None
    best_ts: str = ""

    pattern = os.path.join(chat_history_dir, "chat_*.json")
    for file_path in glob.glob(pattern):
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list) or not data:
                continue
            entry = data[-1]
            ts = entry.get("timestamp", "")
            if ts > best_ts:
                best_ts = ts
                best_entry = entry
        except (json.JSONDecodeError, IOError):
            continue

    if best_entry is None:
        return None

    # Normalise: expose 'received_at' so the caller's dedup logic works.
    result = dict(best_entry)
    result.setdefault("received_at", best_entry.get("timestamp", ""))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Writing side  (used by walle-rag  → imported in rag_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def write_rag_answer(
    json_path: str,
    person: str,
    question: str,
    answer: str,
    max_entries: int = 200,
) -> None:
    """
    Atomically append a new {received_at, person, question, answer} record
    to *json_path*.  Keeps at most *max_entries* entries so the file does not
    grow without bound.
    """
    # Load existing entries
    entries: list = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                entries = json.load(fh)
            if not isinstance(entries, list):
                entries = []
        except (json.JSONDecodeError, IOError):
            entries = []

    # Append new record
    entries.append({
        "received_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "person":      person,
        "question":    question,
        "answer":      answer,
    })

    # Trim to max_entries
    if len(entries) > max_entries:
        entries = entries[-max_entries:]

    # Atomic write
    tmp_path = json_path + ".tmp"
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, json_path)
