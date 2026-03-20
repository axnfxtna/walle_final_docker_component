"""
Utility functions for database connections, encoding, hashing,
and Milvus operations (idempotent & safe).
"""

import os
import time
import hashlib
import torch
import numpy as np
import mysql.connector
from omegaconf import OmegaConf
from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)
from transformers import AutoModel
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================
def load_config(path="configs/database.yaml"):
    """Load YAML configuration."""
    try:
        return OmegaConf.load(path)
    except Exception as e:
        print(f"❌ Error loading config from {path}: {e}")
        raise


# =========================
# MYSQL
# =========================
def connect_mysql(db_name="env"):
    """Connect to MySQL using .env variables with retry logic.
    
    Args:
        db_name (str, optional): Database name to connect to.
            - "env": Use MYSQL_DB from environment (default)
            - None: Connect without selecting a database
            - str: Connect to specific database name
    """
    max_retries = 30
    retry_delay = 2
    
    # Determine which database to use
    if db_name == "env":
        target_db = os.getenv("MYSQL_DB")
    else:
        target_db = db_name

    for attempt in range(max_retries):
        try:
            return mysql.connector.connect(
                host=os.getenv("MYSQL_HOST","localhost"),
                port=int(os.getenv("MYSQL_PORT", 3306)),
                user=os.getenv("MYSQL_USER"),
                password=os.getenv("MYSQL_PASSWORD"),
                database=target_db,
            )
        except mysql.connector.Error as e:
            if attempt < max_retries - 1:
                print(f"⏳ MySQL connection failed (attempt {attempt+1}/{max_retries}): {e}")
                print(f"   Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"❌ MySQL connection error after {max_retries} attempts: {e}")
                raise

# =========================
# MILVUS
# =========================
def connect_milvus():
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

                print(f"✅ Connected to Milvus ({uri or 'milvus:19530'})")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⏳ Milvus connection failed (attempt {attempt+1}/{max_retries}): {e}")
                print(f"   Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"❌ Milvus connection error after {max_retries} attempts: {e}")
                raise



def build_collection(cfg, key):
    """
    Create or load Milvus collection (idempotent).
    Handles both auto_id=True and auto_id=False cases.
    Creates indexes for all vector fields.
    """
    try:
        connect_milvus()

        col_cfg = cfg.collections[key]
        name = col_cfg.name

        # If collection exists, load it
        if utility.has_collection(name):
            col = Collection(name)
            col.load()
            print(f"✅ Using existing collection: {name}")
            return col

        # Build schema based on auto_id setting
        auto_id = col_cfg.primary_key.get("auto_id", False)
        
        fields = [
            FieldSchema(
                name=col_cfg.primary_key.name,
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=auto_id,
                max_length=256,
            )
        ]

        # Track vector fields for indexing
        vector_fields = []
        
        # Add other fields
        for f in col_cfg.fields:
            if f.type == "varchar":
                fields.append(
                    FieldSchema(
                        name=f.name,
                        dtype=DataType.VARCHAR,
                        max_length=f.max_length,
                    )
                )
            elif f.type == "vector":
                fields.append(
                    FieldSchema(
                        name=f.name,
                        dtype=DataType.FLOAT_VECTOR,
                        dim=f.dim,
                    )
                )
                vector_fields.append(f.name)

        schema = CollectionSchema(fields, description=f"{key} collection")
        col = Collection(name, schema)

        # Create index for the specified field (primary vector field)
        col.create_index(
            field_name=col_cfg.index.field,
            index_params={
                "index_type": col_cfg.index.type,
                "metric_type": col_cfg.index.metric,
                "params": {"nlist": col_cfg.index.nlist},
            },
        )
        
        # Create indexes for other vector fields if they exist
        for field_name in vector_fields:
            if field_name != col_cfg.index.field:
                # Create index for additional vector fields
                try:
                    col.create_index(
                        field_name=field_name,
                        index_params={
                            "index_type": col_cfg.index.type,
                            "metric_type": col_cfg.index.metric,
                            "params": {"nlist": col_cfg.index.nlist},
                        },
                    )
                    print(f"   Created index for field: {field_name}")
                except Exception as e:
                    print(f"   ⚠️  Could not create index for {field_name}: {e}")

        col.load()
        print(f"🆕 Created collection: {name}")
        return col
    except Exception as e:
        print(f"❌ Error building collection '{key}': {e}")
        raise


# =========================
# MILVUS HELPERS (ANTI-DUPLICATE)
# =========================
def milvus_uid_exists(collection: Collection, record_id: str) -> bool:
    """
    Check if a document/image ID already exists in Milvus.
    Prevents duplicate vectors.
    """
    try:
        res = collection.query(
            expr=f'id == "{record_id}"',
            output_fields=["id"],
            limit=1,
        )
        return len(res) > 0
    except Exception as e:
        print(f"⚠️  Warning: Error checking ID existence: {e}")
        return False


def load_milvus_records(cfg):
    """
    Load all records from face recognition collection.
    Returns empty list if collection doesn't exist.
    """
    try:
        connect_milvus()
        
        collection_name = cfg.milvus.collection_name
        
        # Check if collection exists
        if not utility.has_collection(collection_name):
            print(f"⚠️  Warning: Collection '{collection_name}' does not exist")
            return []
        
        collection = Collection(collection_name)
        collection.load()
        
        # Query all records - use 'id' field instead of 'uid'
        results = collection.query(
            expr='id >= 0',  # Match all records
            output_fields=["id", "image_path"],
            limit=10000,
        )
        
        return results
    except Exception as e:
        print(f"⚠️  Warning: Error loading Milvus records: {e}")
        return []


def map_milvus_id_by_filename(records):
    """
    Map image filename -> Milvus ID
    """
    try:
        return {
            os.path.basename(r["image_path"]): r["id"]  # Use 'id' instead of 'uid'
            for r in records
            if "image_path" in r and "id" in r
        }
    except Exception as e:
        print(f"⚠️  Warning: Error mapping Milvus IDs: {e}")
        return {}


# =========================
# HASHING
# =========================
def sha1(text: str) -> str:
    """Stable hash for deduplication (Excel rows, text chunks)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# =========================
# ENCODER
# =========================
class Encoder:
    """
    Multi-modal encoder (text + image).
    """

    def __init__(self, model_name: str):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🧠 Encoder device: {self.device}")

            print(f"   Loading model: {model_name}...")
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
            ).to(self.device)

            self.model.set_processor(model_name)
            self.model.eval()
            print(f"✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error initializing encoder: {e}")
            raise

    def encode_text(self, text: str) -> list:
        """Encode text to embedding vector."""
        try:
            with torch.no_grad():
                emb = self.model.encode(text=[text]).cpu()
            return emb.numpy().astype(np.float32)[0].tolist()
        except Exception as e:
            print(f"❌ Error encoding text: {e}")
            raise

    def encode_image(self, image_path: str) -> list:
        """Encode image to embedding vector."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            with torch.no_grad():
                emb = self.model.encode(images=[image_path]).cpu()
            return emb.numpy().astype(np.float32)[0].tolist()
        except Exception as e:
            print(f"❌ Error encoding image '{image_path}': {e}")
            raise