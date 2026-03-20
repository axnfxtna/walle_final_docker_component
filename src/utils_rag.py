"""
Utility functions for the RAG query pipeline.
==============================================
Covers:
  - CJK character cleaning  (qwen2.5 leaks Chinese/Japanese/Korean)
  - Query routing logic
  - Thai <-> English keyword translation for embedding search
"""

import re
from typing import Optional

from omegaconf import OmegaConf


# =========================
# CONFIG
# =========================
def load_config(path="configs/rag.yaml"):
    """Load YAML configuration."""
    try:
        return OmegaConf.load(path)
    except Exception as e:
        print(f"❌ Error loading config from {path}: {e}")
        raise


"""
Typhoon LLM Client
Handles Thai language conversation through Ollama
"""

import requests
import json
from typing import Dict, Optional, List
from loguru import logger


class OllamaClient:
    def __init__(self, api_url: str = "http://localhost:11434", 
                 model: str = "qwen2.5:7b-instruct"):
        """
        Initialize Ollama LLM client
        
        Args:
            api_url: Ollama API endpoint
            model: Model name in Ollama
        """
        self.api_url = api_url
        self.model = model
        self.generate_endpoint = f"{api_url}/api/generate"
        self.chat_endpoint = f"{api_url}/api/chat"
        
        # Verify connection
        self._verify_connection()
        
    def _verify_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.api_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if self.model in model_names:
                    logger.info(f"Connected to Ollama, model '{self.model}' ready")
                else:
                    logger.warning(f"Model '{self.model}' not found. Available: {model_names}")
            else:
                logger.error(f"Failed to connect to Ollama at {self.api_url}")
        except Exception as e:
            logger.error(f"Ollama connection error: {e}")
    
    def generate(self, prompt: str, temperature: float = 0.5, 
                max_tokens: int = 512, stream: bool = False) -> Optional[str]:
        """
        Generate response from prompt
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            
        Returns:
            Generated text or None if failed
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(self.generate_endpoint, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"LLM request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return None
    
    def chat(self, messages: List[Dict[str, str]], 
            temperature: float = 0.7, max_tokens: int = 512) -> Optional[str]:
        """
        Multi-turn chat interface
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Assistant response or None
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(self.chat_endpoint, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "").strip()
            else:
                logger.error(f"Chat request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return None
    
    def generate_structured(self, system_prompt: str, user_message: str,
                          temperature: float = 0.5) -> Optional[Dict]:
        """
        Generate structured output (expects JSON response)
        
        Args:
            system_prompt: System instructions
            user_message: User input
            temperature: Lower for more deterministic output
            
        Returns:
            Parsed JSON dict or None
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = self.chat(messages, temperature=temperature, max_tokens=512)
        
        if response:
            try:
                # Try to parse JSON
                # Remove markdown code blocks if present
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()
                
                return json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from LLM: {e}")
                logger.debug(f"Raw response: {response}")
                return None
        
        return None

# ─────────────────────────────────────────────────────────────────────
#  Text Cleaning
# ─────────────────────────────────────────────────────────────────────

def clean_cjk(text: str) -> str:
    """
    Strip CJK characters and punctuation from LLM output.

    qwen2.5 (Alibaba) frequently leaks Chinese, Japanese, or Korean
    characters into Thai responses. This removes them in one pass.
    """
    text = re.sub(
        r'['
        r'\u3000-\u303f'   # CJK punctuation  (。、「」)
        r'\u3040-\u309f'   # Hiragana
        r'\u30a0-\u30ff'   # Katakana
        r'\u3400-\u4dbf'   # CJK Extension A
        r'\u4e00-\u9fff'   # CJK Unified Ideographs (中文)
        r'\uac00-\ud7af'   # Hangul Syllables (한글)
        r'\u1100-\u11ff'   # Hangul Jamo
        r'\uf900-\ufaff'   # CJK Compatibility Ideographs
        r'\uff01-\uff60'   # Fullwidth forms  (？，（）：！)
        r'\uffe0-\uffef'   # Fullwidth symbols (￥ etc.)
        r']',
        '',
        text,
    )
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\n +\n', '\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────
#  Keyword Tables
# ─────────────────────────────────────────────────────────────────────

HISTORY_KEYWORDS = [
    "ครั้งที่แล้ว", "เมื่อกี้", "ก่อนหน้า", "ถามอะไร",
    "คุยอะไร", "ประวัติ", "history", "previous",
]

STUDENT_KEYWORDS = [
    "student", "name", "email",
    "นักศึกษา", "ชื่อ", "อีเมล", "นศ", "รหัสนักศึกษา",
    "สมาชิก", "ใครบ้าง", "คนไหน", "รุ่น",
]

# timetable must be checked before curriculum —
# "เรียน" (study/attend class) overlaps with curriculum keywords.
TIMETABLE_KEYWORDS = [
    "ตารางเรียน", "ตารางสอบ", "ตาราง",
    "เวลาเรียน", "คาบเรียน", "วันเรียน",
    "วันไหน", "เวลาไหน", "กี่โมง",
    "exam", "schedule", "class", "สอบ",
]

CURRICULUM_KEYWORDS = [
    "วิชา", "หลักสูตร", "รายวิชา", "หน่วยกิต",
    "เรียน", "คอร์ส",
    "course", "credit", "syllabus", "curriculum",
]

# Thai -> English hints appended to queries sent to English-focused
# embedding models (curriculum PDFs are written in English).
THAI_TO_ENG: dict[str, str] = {
    "วิชา":     "courses subjects",
    "หลักสูตร":  "curriculum program",
    "เรียน":    "study learn",
    "หน่วยกิต":  "credits",
    "รายวิชา":   "course list",
    "คอร์ส":    "courses",
    "สอน":      "teaching",
    "เนื้อหา":   "content",
    "ปี":       "year",
}

# System prompt for the Thai university assistant (Wall-E / RAI @ KMITL)
SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยของมหาวิทยาลัย สำหรับหลักสูตร Robotics and AI (RAI) "
    "ที่สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง (KMITL)\n\n"
    "กฎสำคัญที่ต้องปฏิบัติตามอย่างเคร่งครัด:\n"
    "1. ตอบเป็นภาษาไทยเท่านั้น สามารถใช้คำภาษาอังกฤษได้เฉพาะชื่อเฉพาะ เช่น KMITL, RAI, email\n"
    "2. ห้ามใช้ตัวอักษรภาษาจีน (中文), ภาษาเกาหลี (한국어), หรือภาษาญี่ปุ่น (日本語) โดยเด็ดขาด\n"
    "3. ตอบคำถามโดยใช้ข้อมูลที่ให้มาเท่านั้น\n"
    "4. ถ้าข้อมูลไม่เพียงพอ ให้บอกตรงๆ ว่าไม่มีข้อมูล\n"
    "5. ใช้คำว่า 'ห้องปฏิบัติการ' หรือ 'แลป' แทนคำว่า 'lab' เสมอ"
)


def build_dynamic_system_prompt(student_name: str, student_year: int) -> str:
    """Generate a dynamic system prompt based on student identity."""
    year_greeting = {
        1: "น้องปี 1",
        2: "น้องปี 2", 
        3: "น้องปี 3",
        4: "พี่ปี 4"
    }.get(student_year, "คุณ")
    
    return f"""คุณคือหุ่นยนต์ผู้ช่วยของสถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง (KMITL)

กฎสำคัญ:
1. ใช้ชื่อนักศึกษา "{student_name}" ในทุกประโยคตอบ
2. เรียกนักศึกษาว่า "{year_greeting}" (ปี {student_year})
3. ห้ามใช้คำนำหน้าเช่น "น้องบอท:" หรือ "ผม:"
4. ตอบสั้นและกระชับ (1-2 ประโยค)
5. ใช้ "ครับ" ท้ายประโยค"""


# ─────────────────────────────────────────────────────────────────────
#  Query Router
# ─────────────────────────────────────────────────────────────────────

def route_query(
    question: str,
    last_route: Optional[str] = None,
    default_collection: str = "uni_info",
) -> str:
    """
    Map a question string to a data-source label.

    Returns one of:
        'chat_history'   - question is about previous conversation turns
        'mysql_students' - question is about students / roster
        'time_table'     - question is about schedule / exams
        'curriculum'     - question is about courses / credits
        <default>        - fall back to last_route, then default_collection

    Routing order matters:
        history -> students -> timetable -> curriculum -> default
    timetable is checked before curriculum because "เรียน" appears in both.
    """
    q = question.lower()

    if any(kw in q for kw in HISTORY_KEYWORDS):
        return "chat_history"

    if any(kw in q for kw in STUDENT_KEYWORDS):
        return "mysql_students"

    if any(kw in q for kw in TIMETABLE_KEYWORDS):
        return "time_table"

    if any(kw in q for kw in CURRICULUM_KEYWORDS):
        return "curriculum"

    # Follow-up question: reuse previous route for context continuity
    return last_route if last_route else default_collection


# ─────────────────────────────────────────────────────────────────────
#  Embedding Query Augmentation
# ─────────────────────────────────────────────────────────────────────

def augment_query_for_english_model(question: str) -> str:
    """
    Append English keyword hints to a Thai question.

    Curriculum PDFs are written in English and the embedding model is
    English-focused. Appending translated keywords improves cosine
    similarity scores for Thai queries against English documents.
    """
    additions = [eng for th, eng in THAI_TO_ENG.items() if th in question]
    if additions:
        return question + " " + " ".join(additions)
    return question