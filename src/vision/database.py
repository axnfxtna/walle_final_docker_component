"""
Student Enrollment Database Handler

Manages student enrollment data with face embeddings.
Supports JSON-based storage for simplicity.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger


class EnrollmentDatabase:
    """Manage student enrollment database."""
    
    def __init__(self, db_path: str = "data/enrollments.json"):
        """
        Initialize enrollment database.
        
        Args:
            db_path: Path to JSON database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing database or create new
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                data = json.load(f)
                # Convert embedding lists back to numpy arrays
                self.enrollments = {}
                for student_id, info in data.get('enrollments', {}).items():
                    self.enrollments[student_id] = {
                        **info,
                        'embeddings': [np.array(emb) for emb in info['embeddings']]
                    }
            logger.info(f"Loaded {len(self.enrollments)} enrollments from {self.db_path}")
        else:
            self.enrollments = {}
            logger.info(f"Created new enrollment database at {self.db_path}")
    
    def enroll_student(
        self,
        student_id: str,
        embeddings: List[np.ndarray],
        metadata: Optional[Dict] = None,
        fullname_thai: str = "",
        fullname_eng: str = "",
        nickname_thai: str = "",
        nickname_eng: str = "",
        name: str = "",
    ) -> bool:
        """
        Enroll a new student or update existing enrollment.
        
        Args:
            student_id: Unique student identifier
            name: Student name
            embeddings: List of face embeddings (from different angles)
            metadata: Optional metadata (grade, class, etc.)
        
        Returns:
            True if enrollment successful
        """
        try:
            if fullname_eng or fullname_thai:
                name_block = {
                    "fullname_thai": fullname_thai,
                    "fullname_eng":  fullname_eng,
                    "nickname_thai": nickname_thai,
                    "nickname_eng":  nickname_eng,
                }
                display_name = fullname_eng or fullname_thai
            else:
                name_block = name
                display_name = name

            self.enrollments[student_id] = {
                'name': name_block,
                'student_id': student_id,
                'embeddings': embeddings,
                'enrolled_date': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            self._save()
            logger.info(f"Enrolled student: {display_name} ({student_id}) with {len(embeddings)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enroll student {student_id}: {e}")
            return False
    
    def get_student(self, student_id: str) -> Optional[Dict]:
        """Get student information by ID."""
        return self.enrollments.get(student_id)
    
    def get_all_students(self) -> Dict[str, Dict]:
        """Get all enrolled students."""
        return self.enrollments
    
    def delete_student(self, student_id: str) -> bool:
        """Delete a student enrollment."""
        if student_id in self.enrollments:
            del self.enrollments[student_id]
            self._save()
            logger.info(f"Deleted student: {student_id}")
            return True
        return False
    
    def recognize(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.6
    ) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Recognize a face by comparing with enrolled embeddings.
        
        Args:
            query_embedding: Face embedding to recognize
            threshold: Similarity threshold for recognition
        
        Returns:
            Tuple of (student_id, confidence, name) or (None, 0.0, None)
        """
        if not self.enrollments:
            return None, 0.0, None
        
        best_match_id = None
        best_match_name = None
        best_score = 0.0
        
        for student_id, info in self.enrollments.items():
            # Compare with all enrolled embeddings for this student
            scores = []
            for enrolled_emb in info['embeddings']:
                # Cosine similarity
                similarity = self._cosine_similarity(query_embedding, enrolled_emb)
                scores.append(similarity)
            
            # Use average similarity
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_match_id = student_id
                best_match_name = info['name']
        
        # Check threshold
        if best_score >= threshold:
            logger.debug(f"Recognized: {best_match_name} ({best_match_id}) with score {best_score:.3f}")
            return best_match_id, best_score, best_match_name
        else:
            logger.debug(f"No match found (best score: {best_score:.3f})")
            return None, 0.0, None
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
    
    def _save(self):
        """Save database to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        data = {
            'enrollments': {
                student_id: {
                    **info,
                    'embeddings': [emb.tolist() for emb in info['embeddings']]
                }
                for student_id, info in self.enrollments.items()
            }
        }
        
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Saved {len(self.enrollments)} enrollments to {self.db_path}")
    
    def __len__(self):
        """Return number of enrolled students."""
        return len(self.enrollments)
    
    def __repr__(self):
        return f"EnrollmentDatabase({len(self)} students)"
