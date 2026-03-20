"""
RAG Query Pipeline for Wall-E chatbot.
=======================================
Connects OllamaClient <-> MySQL <-> Milvus and exposes:
  - RAGQueryPipeline : initialise connections, run a single query
  - interactive_mode : REPL loop with MySQL chat-history persistence
"""

import os
import sys
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

from pymilvus import Collection, utility
from sentence_transformers import SentenceTransformer

from src.utils_database import connect_milvus, connect_mysql
from src.utils_rag import (
    SYSTEM_PROMPT,
    augment_query_for_english_model,
    clean_cjk,
    route_query,
    build_dynamic_system_prompt,
)
from src.pipelines.mcp_pipeline import ContextBuilder

# OllamaClient lives in the RobotAI sub-project
_ROBOTAI_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "RobotAI",
)
if _ROBOTAI_DIR not in sys.path:
    sys.path.insert(0, _ROBOTAI_DIR)

from src.utils_rag import OllamaClient  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
#  Pipeline
# ═══════════════════════════════════════════════════════════════════════

class RAGQueryPipeline:
    """
    Full RAG pipeline: question -> route -> retrieve -> LLM -> answer.

    Handles:
      - Milvus vector search  (uni_info, time_table, curriculum)
      - MySQL structured query (students, academic years, timetable text)
      - In-memory conversation history  (last 5 turns)
      - CJK artefact removal from LLM output
    """

    def __init__(
        self,
        cfg,
        mcp_cfg=None,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:7b-instruct",
    ):
        self.cfg = cfg
        self.history: List[Tuple[str, str]] = []  # (question, answer) pairs
        self.last_route: Optional[str] = None
        self.default_collection: str = cfg.routing.default_collection

        # ── Context Builder (MCP) ─────────────────────────────────────
        try:
            self.context_builder = ContextBuilder(mcp_cfg) if mcp_cfg else None
            self.session_id = "rag_session_1"
            if self.context_builder:
                self.context_builder.create_session(self.session_id)
                print("✅ Context Builder ready")
        except Exception as e:
            print(f"❌ Context Builder init failed: {e}")
            self.context_builder = None

        # ── OllamaClient ──────────────────────────────────────────────
        try:
            self.llm = OllamaClient(api_url=ollama_url, model=ollama_model)
            print(f"✅ Ollama ready  (model: {ollama_model})")
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            self.llm = None

        # ── Embedding model ───────────────────────────────────────────
        emb_name = cfg.models.text_embedding.name
        self.emb_dim: int = cfg.models.text_embedding.dim
        try:
            self.embedder = SentenceTransformer(emb_name)
            print(f"✅ Embedder ready  ({emb_name}, dim={self.emb_dim})")
        except Exception as e:
            print(f"❌ Embedder load failed: {e}")
            self.embedder = None

        # ── MySQL ─────────────────────────────────────────────────────
        try:
            self.mysql_conn = connect_mysql()
            print("✅ MySQL connected")
        except Exception as e:
            print(f"❌ MySQL connection failed: {e}")
            self.mysql_conn = None

        # ── Milvus ────────────────────────────────────────────────────
        try:
            connect_milvus()
            print("✅ Milvus connected")
        except Exception as e:
            print(f"❌ Milvus connection failed: {e}")

    # ─────────────────────────────────────────────────────────────────
    #  Milvus helpers
    # ─────────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list:
        if self.embedder is None:
            raise RuntimeError("Embedding model not loaded")
        return self.embedder.encode([text], normalize_embeddings=True)[0].tolist()

    def _search_milvus(
        self, question: str, collection_name: str, top_k: int = 5
    ) -> List[Dict]:
        """Vector search against a Milvus collection."""
        if not utility.has_collection(collection_name):
            print(f"⚠️  Collection '{collection_name}' not found")
            return []

        col = Collection(collection_name)
        col.load()

        # Identify vector vs scalar output fields from schema
        vector_fields, output_fields = [], []
        for f in col.schema.fields:
            if f.dtype == 101:      # FLOAT_VECTOR
                vector_fields.append(f.name)
            elif not f.is_primary:
                output_fields.append(f.name)

        if not vector_fields:
            print(f"❌ No vector field in '{collection_name}'")
            return []

        # Prefer text-oriented embedding fields
        if "doc_embedding" in vector_fields:
            vector_field = "doc_embedding"
        elif "embedding" in vector_fields:
            vector_field = "embedding"
        else:
            vector_field = vector_fields[0]

        # Exclude all vector fields from returned output
        output_fields = [f for f in output_fields if f not in vector_fields]

        # Resolve target dimension from schema
        target_dim = self.emb_dim
        for f in col.schema.fields:
            if f.name == vector_field:
                target_dim = f.params.get("dim", self.emb_dim)
                break

        query_vec = self._embed(question)

        # Pad or truncate to match collection dimension
        if len(query_vec) < target_dim:
            query_vec += [0.0] * (target_dim - len(query_vec))
        elif len(query_vec) > target_dim:
            query_vec = query_vec[:target_dim]

        results = col.search(
            data=[query_vec],
            anns_field=vector_field,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=output_fields,
        )

        hits = []
        for hit in results[0]:
            record = {"score": hit.score, "id": hit.id}
            for field in output_fields:
                try:
                    record[field] = hit.entity.get(field)
                except Exception:
                    record[field] = "N/A"
            hits.append(record)

        return hits

    # ─────────────────────────────────────────────────────────────────
    #  MySQL helpers
    # ─────────────────────────────────────────────────────────────────

    def _ensure_mysql(self) -> None:
        """Reconnect if the MySQL connection has dropped."""
        if self.mysql_conn is None or not self.mysql_conn.is_connected():
            self.mysql_conn = connect_mysql()

    def _fetch_student_context(self) -> List[str]:
        """Return student + academic-year rows as formatted strings."""
        self._ensure_mysql()
        cursor = self.mysql_conn.cursor(dictionary=True)
        parts = []

        try:
            cursor.execute("SELECT * FROM Students LIMIT 50")
            students = cursor.fetchall()
            if students:
                parts.append("=== ข้อมูลนักศึกษา ===")
                for s in students:
                    parts.append(
                        f"รหัส: {s.get('student_id')}, "
                        f"ชื่อ: {s.get('first_name')} {s.get('last_name')} "
                        f"(ชื่อเล่น: {s.get('nick_name')}), "
                        f"อีเมล: {s.get('student_email')}"
                    )
        except Exception as e:
            parts.append(f"(Students error: {e})")

        try:
            cursor.execute("SELECT * FROM Academic_Year")
            years = cursor.fetchall()
            if years:
                parts.append("\n=== ปีการศึกษา ===")
                for a in years:
                    parts.append(
                        f"RAI รุ่น {a.get('RAI_Gen')}, "
                        f"KMITL รุ่น {a.get('KMITL_Gen')}, "
                        f"ปี: {a.get('year_start')}-{a.get('year_end')}"
                    )
        except Exception as e:
            parts.append(f"(Academic_Year error: {e})")

        try:
            cursor.execute("SELECT row_text FROM ExcelTimetableData LIMIT 10")
            rows = cursor.fetchall()
            if rows:
                parts.append("\n=== ตารางเรียน (ตัวอย่าง) ===")
                for r in rows:
                    parts.append(f"  {r.get('row_text', '')}")
        except Exception:
            pass

        return parts

    def _fetch_timetable_by_ids(self, hit_ids: List[str]) -> Dict[str, str]:
        """Fetch ExcelTimetableData rows matching Milvus result IDs."""
        self._ensure_mysql()
        cursor = self.mysql_conn.cursor(dictionary=True)
        placeholders = ",".join(["%s"] * len(hit_ids))
        cursor.execute(
            f"SELECT row_id, row_text FROM ExcelTimetableData "
            f"WHERE row_id IN ({placeholders})",
            hit_ids,
        )
        return {str(r["row_id"]): r["row_text"] for r in cursor.fetchall()}

    # ─────────────────────────────────────────────────────────────────
    #  Context builder
    # ─────────────────────────────────────────────────────────────────

    def _build_context(self, question: str, route: str) -> str:
        """Retrieve relevant data and format it as a context string."""
        parts = []

        if route == "chat_history":
            parts.append("=== ประวัติการสนทนา ===")
            if not self.history:
                parts.append("ไม่มีประวัติการสนทนาในรอบนี้")
            else:
                for idx, (q, a) in enumerate(self.history, 1):
                    parts.append(f"ครั้งที่ {idx}:\nคำถาม: {q}\nคำตอบ: {a}")

        elif route == "mysql_students":
            parts.extend(self._fetch_student_context())

        elif route == "curriculum":
            # Dual search: original Thai query + English-augmented query,
            # then merge and deduplicate by score descending.
            thai_hits = self._search_milvus(question, route, top_k=10)
            eng_hits = self._search_milvus(
                augment_query_for_english_model(question), route, top_k=10
            )
            seen: set = set()
            hits = []
            for h in sorted(
                thai_hits + eng_hits, key=lambda x: x["score"], reverse=True
            ):
                if h["id"] not in seen:
                    seen.add(h["id"])
                    hits.append(h)
                if len(hits) >= 10:
                    break

            parts.append(f"=== ผลลัพธ์จาก collection '{route}' ===")
            for i, h in enumerate(hits, 1):
                chunk = [f"[{i}] (คะแนน: {h['score']:.4f})"]
                for k, v in h.items():
                    if k not in ("score", "id"):
                        chunk.append(f"  {k}: {v}")
                parts.append("\n".join(chunk))

        elif route == "time_table":
            hits = self._search_milvus(question, route, top_k=10)
            parts.append(f"=== ผลลัพธ์จาก collection '{route}' ===")

            if hits and self.mysql_conn:
                hit_ids = [str(h["id"]) for h in hits]
                try:
                    rows = self._fetch_timetable_by_ids(hit_ids)
                    print(f"  ℹ  Milvus IDs: {hit_ids[:5]}")
                    print(f"  ℹ  MySQL matched: {len(rows)} rows")
                    for i, h in enumerate(hits, 1):
                        text = rows.get(str(h["id"]), "")
                        if text:
                            parts.append(f"[{i}] {text}")
                except Exception as e:
                    print(f"⚠️  MySQL timetable fetch error: {e}")
                    for i, h in enumerate(hits, 1):
                        parts.append(f"[{i}] (คะแนน: {h['score']:.4f})")
            else:
                for i, h in enumerate(hits, 1):
                    parts.append(f"[{i}] (คะแนน: {h['score']:.4f})")

        else:
            # Generic Milvus search (uni_info or any other collection)
            hits = self._search_milvus(question, route, top_k=5)
            if hits:
                parts.append(f"=== ผลลัพธ์จาก collection '{route}' ===")
                for i, h in enumerate(hits, 1):
                    chunk = [f"[{i}] (คะแนน: {h['score']:.4f})"]
                    for k, v in h.items():
                        if k not in ("score", "id"):
                            chunk.append(f"  {k}: {v}")
                    parts.append("\n".join(chunk))
            else:
                parts.append(f"(ไม่พบผลลัพธ์ใน collection '{route}')")

        return "\n".join(parts)

    # ─────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────

    def ask(self, question: str, student_name: str = "Guest", student_year: int = 1) -> str:
        """
        Full RAG turn: route -> retrieve -> LLM -> clean -> return answer.
        """
        if self.llm is None:
            return "❌ LLM ไม่พร้อมใช้งาน กรุณาเปิด Ollama"

        # 1. Route
        route = route_query(question, self.last_route, self.default_collection)
        print(f"  ℹ  Route: {route}")

        # 2. Retrieve context
        context = self._build_context(question, route)

        self.status_message = "🤖 Thinking..."
        print("\n🤖 Processing with LLM...")

        try:
            if self.context_builder:
                # Update context with student info
                self.context_builder.update_student_identity(self.session_id, "unknown", student_name, student_year=student_year)
                self.context_builder.add_conversation_turn(self.session_id, "user", question)
                
                # Form memory from RAG
                memories = [{"text": context, "score": 1.0, "memory_type": "rag_retrieval"}]
                
                # Build context-aware prompt
                llm_context = self.context_builder.build_llm_context(self.session_id, retrieved_memories=memories)
                context_text = self.context_builder.format_context_as_prompt(llm_context)
                
                system_prompt = build_dynamic_system_prompt(student_name, student_year)
                user_message = f"{context_text}\n\nนักศึกษา: {question}\n\nน้องบอท:"
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            else:
                # Fallback to Original formatting
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for prev_q, prev_a in self.history[-3:]:
                    messages.append({"role": "user",      "content": prev_q})
                    messages.append({"role": "assistant",  "content": prev_a})
                messages.append({
                    "role": "user",
                    "content": f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {question}",
                })

            # LLM call
            response = self.llm.chat(messages, temperature=0.7, max_tokens=512)
            
            if not response:
                response = f"ขอโทษครับ {student_name} ผมไม่เข้าใจคำถามครับ"
            
            answer = clean_cjk(response)
            
            if self.context_builder:
                self.context_builder.add_conversation_turn(self.session_id, "assistant", answer)

            # Update old RAG state history (for REPL history logic)
            self.last_route = route
            self.history.append((question, answer))
            if len(self.history) > 5:
                self.history = self.history[-5:]

        except Exception as e:
            print(f"❌ LLM Error: {e}")
            answer = f"ขอโทษครับ {student_name} ระบบมีปัญหาครับ"

        return answer

    def run_all(self) -> None:
        """Satisfies the pipeline interface; actual work runs via ask()."""
        print("✅ RAGQueryPipeline ready — call ask() or use interactive_mode()")


# ═══════════════════════════════════════════════════════════════════════
#  MySQL chat-history helpers (module-level, used by interactive_mode)
# ═══════════════════════════════════════════════════════════════════════

def _ensure_chat_history_table(conn) -> None:
    """Create ChatHistory table if it does not already exist."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ChatHistory (
            id         INT AUTO_INCREMENT PRIMARY KEY,
            user_name  VARCHAR(100),
            question   TEXT,
            answer     TEXT,
            timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """)
    conn.commit()
    cursor.close()


def _load_chat_history(conn, user_name: str, limit: int = 3) -> List[Tuple]:
    """Return the most-recent chat turns for a user, oldest-first."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT question, answer FROM ChatHistory "
        "WHERE user_name = %s ORDER BY timestamp DESC LIMIT %s",
        (user_name, limit),
    )
    rows = cursor.fetchall()
    cursor.close()
    return list(reversed(rows))     # chronological order


def _save_chat_turn(conn, user_name: str, question: str, answer: str) -> None:
    """Persist a single Q/A turn to MySQL."""
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO ChatHistory (user_name, question, answer) VALUES (%s, %s, %s)",
        (user_name, question, answer),
    )
    conn.commit()
    cursor.close()


# ═══════════════════════════════════════════════════════════════════════
#  Interactive REPL
# ═══════════════════════════════════════════════════════════════════════

def interactive_mode(pipeline: RAGQueryPipeline) -> None:
    """
    REPL loop for Wall-E chatbot.

    - Persists chat history to MySQL (ChatHistory table).
    - Loads the last 3 turns on login so the LLM remembers context.
    - Exits on 'quit', 'exit', 'ออก', or 'จบ'.
    """
    
    def sanitize_input(text: str) -> str:
        """Strip surrogates and invalid characters read from terminal."""
        if not text:
            return text
        return text.encode('utf-8', 'replace').decode('utf-8')

    print("\n" + "=" * 60)
    print("INTERACTIVE MODE  (พิมพ์ 'ออก' หรือ 'quit' เพื่อจบ)")
    print("=" * 60)

    try:
        user_name = sanitize_input(input("\n  กรุณากรอกชื่อของคุณ (หรือพิมพ์ 'quit' เพื่อออก): ").strip())
    except (KeyboardInterrupt, EOFError):
        print("\nลาก่อน! 👋")
        return

    # Allow exiting at the name prompt
    if not user_name or user_name.lower() in ("quit", "exit", "q", "ออก", "จบ"):
        print("\nลาก่อน! 👋")
        return

    user_name = user_name or "Guest"

    try:
        student_year_str = sanitize_input(input("  กรุณากรอกชั้นปี (1-4, ว่างไว้=1): ").strip())
        student_year = int(student_year_str) if student_year_str.isdigit() else 1
    except (KeyboardInterrupt, EOFError):
        print("\nลาก่อน! 👋")
        return

    # Set up MySQL history persistence
    mysql_ok = False
    if pipeline.mysql_conn and pipeline.mysql_conn.is_connected():
        try:
            _ensure_chat_history_table(pipeline.mysql_conn)
            prior = _load_chat_history(pipeline.mysql_conn, user_name)
            if prior:
                print(f"\n  ยินดีต้อนรับกลับมาคุณ {user_name}!")
                print(f"  (โหลดประวัติการสนทนา {len(prior)} รายการล่าสุด)")
                pipeline.history.extend(prior)
            else:
                print(f"\n  ยินดีต้อนรับคุณ {user_name}!")
            mysql_ok = True
        except Exception as e:
            print(f"  ⚠️  ไม่สามารถโหลดประวัติสนทนา: {e}")
            print(f"\n  ยินดีต้อนรับคุณ {user_name}!")
    else:
        print(f"\n  ยินดีต้อนรับคุณ {user_name}!")

    print("  ลองถามคำถามภาษาไทยเกี่ยวกับมหาวิทยาลัย นักศึกษา หลักสูตร หรือตารางสอบ\n")

    while True:
        try:
            question = sanitize_input(input(f"🙋 {user_name}: ").strip())
        except (KeyboardInterrupt, EOFError):
            print("\n\nหยุดการทำงาน")
            break

        if not question or question.lower() in ("quit", "exit", "q", "ออก", "จบ"):
            print(f"\nลาก่อนคุณ {user_name}! 👋")
            break

        try:
            start = time.time()
            answer = pipeline.ask(question, student_name=user_name, student_year=student_year)
            elapsed = time.time() - start

            print(f"\n🤖 Wall-E: {answer}")
            print(f"  ℹ  (ใช้เวลา {elapsed:.1f} วินาที)\n")

            if mysql_ok:
                try:
                    _save_chat_turn(
                        pipeline.mysql_conn, user_name, question, answer
                    )
                except Exception as e:
                    print(f"  ⚠️  บันทึกประวัติสนทนาล้มเหลว: {e}")

        except Exception as e:
            print(f"  ❌ ผิดพลาด: {e}")
            traceback.print_exc()


def auto_stt_mode(pipeline: RAGQueryPipeline, json_path: str = "/app/received_events.json") -> None:
    """
    Poll a JSON file containing STT events and process new events automatically.
    """
    import json
    import time

    print("\n" + "=" * 60)
    print("AUTO STT MODE (Polling JSON for input)")
    print(f"File: {json_path}")
    print("=" * 60)

    last_processed_time = None

    # Set up MySQL history persistence
    mysql_ok = False
    if pipeline.mysql_conn and pipeline.mysql_conn.is_connected():
        try:
            _ensure_chat_history_table(pipeline.mysql_conn)
            mysql_ok = True
        except Exception as e:
            print(f"  ⚠️  ไม่สามารถตั้งค่าฐานข้อมูลประวัติ: {e}")

    print("🤖 รอรับคำถามจากระบบเสียง (Ctrl+C เพื่อออก)...\n")

    while True:
        try:
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        events = json.load(f)

                    if events:
                        # Process the newest event if we haven't seen it yet
                        latest_event = events[-1]
                        current_time = latest_event.get("received_at")

                        if current_time != last_processed_time:
                            # It's a new event!
                            last_processed_time = current_time

                            stt_data = latest_event.get("stt", {})
                            question = stt_data.get("text", "").strip()
                            student_name = latest_event.get("person_id", "Guest")

                            # Only proceed if there is actual text
                            if question:
                                print(f"\n🙋 {student_name} (STT): {question}")

                                start = time.time()
                                answer = pipeline.ask(question, student_name=student_name, student_year=1)
                                elapsed = time.time() - start

                                print(f"\n🤖 Wall-E: {answer}")
                                print(f"  ℹ  (ใช้เวลา {elapsed:.1f} วินาที)\n")

                                if mysql_ok:
                                    try:
                                        _save_chat_turn(
                                            pipeline.mysql_conn, student_name, question, answer
                                        )
                                    except Exception as e:
                                        print(f"  ⚠️  บันทึกประวัติสนทนาล้มเหลว: {e}")

                except json.JSONDecodeError:
                    # File might be mid-write, ignore and try again next loop
                    pass
                except Exception as e:
                    print(f"⚠️ Error reading JSON: {e}")

            time.sleep(1.0)  # poll every 1 second

        except KeyboardInterrupt:
            print("\n\nหยุดการทำงาน")
            break
        except Exception as e:
            print(f"  ❌ ผิดพลาดใน Auto mode: {e}")
            sys.modules['traceback'].print_exc()
            time.sleep(2)