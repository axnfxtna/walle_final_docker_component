"""
Utility functions for Milvus connections, config loading,
and face collection management (idempotent & safe).
"""

import os
import time
from omegaconf import OmegaConf
from dotenv import load_dotenv

load_dotenv()

# Try importing pymilvus — graceful degradation if not installed
try:
    from pymilvus import (
        connections,
        utility,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
    )
    _MILVUS_AVAILABLE = True
except ImportError:
    _MILVUS_AVAILABLE = False


# =========================
# CONFIG
# =========================
def load_config(path="configs/enrollment.yaml"):
    """Load YAML configuration."""
    try:
        return OmegaConf.load(path)
    except Exception as e:
        print(f"❌ Error loading config from {path}: {e}")
        raise


# =========================
# MILVUS
# =========================
def connect_milvus():
    """Connect to Milvus with retry logic using .env / environment variables."""
    if not _MILVUS_AVAILABLE:
        print("⚠️  pymilvus not installed — Milvus storage disabled")
        return

    max_retries = 30
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            alias = os.getenv("MILVUS_ALIAS", "default")
            uri = os.getenv("URI")

            if not connections.has_connection(alias):
                if uri:
                    connections.connect(alias=alias, uri=uri)
                else:
                    connections.connect(
                        alias=alias,
                        host=os.getenv("MILVUS_HOST", "localhost"),
                        port=os.getenv("MILVUS_PORT", "19530"),
                    )
                print(f"✅ Connected to Milvus ({uri or 'localhost:19530'})")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⏳ Milvus connection failed (attempt {attempt+1}/{max_retries}): {e}")
                print(f"   Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"❌ Milvus connection error after {max_retries} attempts: {e}")
                raise


def get_or_create_face_collection(collection_name="student_face_images"):
    """
    Ensure the student_face_images collection exists in Milvus.
    Creates the collection with schema + index if it doesn't exist,
    otherwise loads the existing one.

    Returns:
        Collection object, or None if Milvus is unavailable.
    """
    if not _MILVUS_AVAILABLE:
        print("⚠️  pymilvus not installed — Milvus storage disabled")
        return None

    try:
        connect_milvus()

        if utility.has_collection(collection_name):
            col = Collection(collection_name)
            col.load()
            print(f"✅ Milvus: loaded existing '{collection_name}'")
            return col

        # Create collection (mirrors FaceRecognitionPipeline schema)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
        ]
        schema = CollectionSchema(fields, description="Student face recognition embeddings")
        col = Collection(collection_name, schema)
        col.create_index(
            field_name="embedding",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128},
            },
        )
        col.load()
        print(f"✅ Milvus: created '{collection_name}'")
        return col

    except Exception as exc:
        print(f"⚠️  Milvus unavailable ({exc}) — saving to JSON only")
        return None


def insert_face_embeddings(collection, student_id: str, embeddings: list):
    """
    Insert all angle embeddings for a student into the Milvus collection.
    Each angle becomes one row tagged '<student_id>_angle<i>'.

    Args:
        collection: Milvus Collection object (or None).
        student_id: Student ID string.
        embeddings: List of numpy arrays / lists (one per angle).
    """
    if collection is None:
        return

    try:
        rows = [
            {
                "image_path": f"{student_id}_angle{i}",
                "embedding": emb.tolist() if hasattr(emb, "tolist") else list(emb),
            }
            for i, emb in enumerate(embeddings)
        ]
        result = collection.insert(rows)
        collection.flush()
        print(f"  ✅ Milvus: inserted {len(rows)} vectors  (ids={result.primary_keys})")
    except Exception as exc:
        print(f"  ⚠️  Milvus insert failed: {exc}")
