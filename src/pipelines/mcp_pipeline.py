"""
MCP Context Builder Pipeline
Constructs rich context for LLM from various sources.
Extracted from demo_end_to_end.py
"""

from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger


class ContextBuilder:
    def __init__(self, cfg):
        """Initialize context builder with configuration"""
        self.cfg = cfg
        self.session_contexts = {}  # session_id -> context dict
        self.max_history = self.cfg.context.max_history
        self.default_student_name = self.cfg.context.default_student_name
        self.default_location = self.cfg.context.default_location
        
    def create_session(self, session_id: str) -> None:
        """Create a new session context"""
        self.session_contexts[session_id] = {
            "session_id": session_id,
            "student_id": None,
            "student_name": None,
            "conversation_history": [],
            "current_location": None,
            "detected_faces": [],
            "environment_info": {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        logger.info(f"Created session: {session_id}")
    
    def _calculate_student_year(self, student_id: str) -> int:
        """Calculate student year from ID (format: YYXXXXXX)"""
        try:
            # Extract enrollment year from first 2 digits
            enrollment_year_short = int(student_id[:2])
            enrollment_year = 2500 + enrollment_year_short  # Buddhist year
            
            # Get current date
            now = datetime.now()
            current_year_gregorian = now.year
            current_year_buddhist = current_year_gregorian + 543
            
            # Academic year starts in August
            # If we're before August, we're still in the previous academic year
            if now.month < 8:
                current_academic_year = current_year_buddhist - 1
            else:
                current_academic_year = current_year_buddhist
            
            # Calculate year (1-4)
            # Students enroll in year YYYY and are Year 1 in academic year YYYY
            year = current_academic_year - enrollment_year + 1
            
            # Clamp to 1-4 range
            return max(1, min(4, year))
        except (ValueError, IndexError):
            logger.warning(f"Could not calculate year from ID: {student_id}")
            return 1  # Default to year 1
    
    def update_student_identity(self, session_id: str, 
                                student_id: str, student_name: str,
                                student_year: Optional[int] = None) -> None:
        """Update student identity in session"""
        if session_id not in self.session_contexts:
            self.create_session(session_id)
        
        # Calculate student year from ID if not provided
        if student_year is None:
            student_year = self._calculate_student_year(student_id)
        
        self.session_contexts[session_id]["student_id"] = student_id
        self.session_contexts[session_id]["student_name"] = student_name
        self.session_contexts[session_id]["student_year"] = student_year
        self.session_contexts[session_id]["last_updated"] = datetime.now().isoformat()
        logger.info(f"Updated identity for session {session_id}: {student_name} (Year {student_year})")
    
    def add_conversation_turn(self, session_id: str, 
                            role: str, content: str) -> None:
        """
        Add a conversation turn
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
        """
        if session_id not in self.session_contexts:
            self.create_session(session_id)
        
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.session_contexts[session_id]["conversation_history"].append(turn)
        
        # Keep only last max_history turns to avoid context overflow
        if len(self.session_contexts[session_id]["conversation_history"]) > self.max_history:
            self.session_contexts[session_id]["conversation_history"] = \
                self.session_contexts[session_id]["conversation_history"][-self.max_history:]
        
        self.session_contexts[session_id]["last_updated"] = datetime.now().isoformat()
    
    def update_location(self, session_id: str, location: str) -> None:
        """Update robot's current location"""
        if session_id not in self.session_contexts:
            self.create_session(session_id)
        
        self.session_contexts[session_id]["current_location"] = location
        self.session_contexts[session_id]["last_updated"] = datetime.now().isoformat()
    
    def update_environment(self, session_id: str, env_data: Dict) -> None:
        """Update environment information"""
        if session_id not in self.session_contexts:
            self.create_session(session_id)
        
        self.session_contexts[session_id]["environment_info"].update(env_data)
        self.session_contexts[session_id]["last_updated"] = datetime.now().isoformat()
    
    def build_llm_context(self, session_id: str, 
                         retrieved_memories: Optional[List[Dict]] = None) -> Dict:
        """
        Build comprehensive context for LLM
        
        Args:
            session_id: Session identifier
            retrieved_memories: Optional memories from vector DB
            
        Returns:
            Context dictionary ready for LLM
        """
        if session_id not in self.session_contexts:
            logger.warning(f"Session {session_id} not found, creating new")
            self.create_session(session_id)
        
        ctx = self.session_contexts[session_id]
        
        # Build context structure
        llm_context = {
            "student_info": {
                "id": ctx.get("student_id", "unknown"),
                "name": ctx.get("student_name", self.default_student_name),
                "year": ctx.get("student_year", 1)
            },
            "conversation_history": ctx.get("conversation_history", [])[-5:],  # Last 5 turns
            "current_location": ctx.get("current_location", self.default_location),
            "environment": ctx.get("environment_info", {}),
            "relevant_memories": []
        }
        
        # Add retrieved memories if available
        if retrieved_memories:
            llm_context["relevant_memories"] = [
                {
                    "content": mem["text"],
                    "relevance_score": mem["score"],
                    "type": mem["memory_type"]
                }
                for mem in retrieved_memories[:3]  # Top 3 most relevant
            ]
        
        return llm_context
    
    def format_context_as_prompt(self, llm_context: Dict) -> str:
        """
        Convert context dict to natural language prompt
        
        Args:
            llm_context: Context dictionary
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Student info with year
        student_name = llm_context["student_info"]["name"]
        student_year = llm_context["student_info"].get("year", 1)
        
        if student_name != self.default_student_name:
            prompt_parts.append(f"คุณกำลังพูดคุยกับ {student_name} (ชั้นปีที่ {student_year})")
        
        # Location
        location = llm_context.get("current_location")
        if location and location != self.default_location:
            prompt_parts.append(f"ตำแหน่งปัจจุบัน: {location}")
        
        # Relevant memories
        memories = llm_context.get("relevant_memories", [])
        if memories:
            prompt_parts.append("\nความทรงจำที่เกี่ยวข้อง:")
            for i, mem in enumerate(memories, 1):
                prompt_parts.append(f"{i}. {mem['content']}")
        
        # Conversation history
        history = llm_context.get("conversation_history", [])
        if history:
            prompt_parts.append("\nบทสนทนาล่าสุด:")
            for turn in history[-3:]:  # Last 3 turns
                role_thai = "นักศึกษา" if turn["role"] == "user" else "หุ่นยนต์"
                prompt_parts.append(f"{role_thai}: {turn['content']}")
        
        return "\n".join(prompt_parts)
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session context"""
        return self.session_contexts.get(session_id)
    
    def clear_session(self, session_id: str) -> None:
        """Clear a session"""
        if session_id in self.session_contexts:
            del self.session_contexts[session_id]
            logger.info(f"Cleared session: {session_id}")
