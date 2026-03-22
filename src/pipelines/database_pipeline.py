"""
Database preparation pipelines for RAG ingestion and MySQL setup.
"""
import os
import sys
import uuid
import fitz
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, utility, CollectionSchema, FieldSchema, DataType

from src.utils_database import (
    load_config,
    connect_milvus,
    connect_mysql,
    build_collection,
    Encoder,
    load_milvus_records,
    map_milvus_id_by_filename
)

# # Face Recognition Pipeline - from image
# class FaceRecognitionPipeline:
#     """
#     Handles face image ingestion into Milvus.
#     Creates and populates the student_face_images collection.
#     """
    
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.collection_name = cfg.milvus.collection_name
#         self.face_dir = cfg.paths.face_image_dir
#         connect_milvus()
    
#     def create_face_collection(self):
#         """Create the student_face_images collection if it doesn't exist."""
#         try:
#             # Check if collection already exists
#             if utility.has_collection(self.collection_name):
#                 print(f"✅ Collection '{self.collection_name}' already exists")
#                 col = Collection(self.collection_name)
#                 col.load()
#                 return col
            
#             print(f"🆕 Creating collection '{self.collection_name}'...")
            
#             # Define schema
#             fields = [
#                 FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#                 FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
#                 FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
#             ]
            
#             schema = CollectionSchema(fields, description="Student face recognition images")
#             col = Collection(self.collection_name, schema)
            
#             # Create index on the embedding field
#             index_params = {
#                 "index_type": "IVF_FLAT",
#                 "metric_type": "COSINE",
#                 "params": {"nlist": 128}
#             }
#             col.create_index(field_name="embedding", index_params=index_params)
            
#             col.load()
#             print(f"✅ Collection '{self.collection_name}' created successfully")
#             return col
            
#         except Exception as e:
#             print(f"❌ Error creating collection: {e}")
#             raise
    
#     def ingest_face_images(self):
#         """Ingest face images into Milvus."""
#         try:
#             if not os.path.exists(self.face_dir):
#                 print(f"⚠️  Warning: Face image directory not found: {self.face_dir}")
#                 print("   Skipping face recognition ingestion")
#                 return
            
#             # Create/load collection
#             collection = self.create_face_collection()
            
#             # Check if collection already has data
#             existing_count = collection.num_entities
#             if existing_count > 0:
#                 print(f"ℹ️  Collection already contains {existing_count} records")
                
#                 # Check for data consistency
#                 print("   Checking data consistency...")
                
#                 # Get existing image paths from Milvus
#                 results = collection.query(
#                     expr="id >= 0",
#                     output_fields=["image_path"],
#                     limit=16384  # Adjust limit as needed, or use iterator for very large collections
#                 )
#                 existing_files = set(os.path.basename(r["image_path"]) for r in results)
                
#                 # Get local files
#                 local_files = set()
#                 for filename in os.listdir(self.face_dir):
#                     if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
#                         local_files.add(filename)
                
#                 # Compare
#                 if existing_files == local_files:
#                     print("✅ Data matches exactly. Skipping ingestion.")
#                     return
#                 else:
#                     print("⚠️  Data mismatch detected.")
#                     print(f"   Existing in DB: {len(existing_files)}")
#                     print(f"   Local files: {len(local_files)}")
#                     print("   Dropping and recreating collection...")
#                     utility.drop_collection(self.collection_name)
#                     collection = self.create_face_collection()
            
#             # Initialize encoder
#             print("\n🧠 Initializing encoder...")
#             try:
#                 # Try to use vision-text encoder from config
#                 encoder = Encoder(self.cfg.models.vision_text_embedding.name)
#                 print(f"✅ Using encoder: {self.cfg.models.vision_text_embedding.name}")
#             except Exception as e:
#                 print(f"⚠️  Could not initialize vision encoder: {e}")
#                 print("   Trying face_recognition library...")
#                 try:
#                     import face_recognition
#                     encoder = None
#                     use_face_recognition = True
#                     print("✅ Using face_recognition library")
#                 except ImportError:
#                     print("❌ face_recognition not available either")
#                     print("   Using hash-based fallback (testing only)")
#                     encoder = None
#                     use_face_recognition = False
            
#             # Collect all image paths
#             image_list = []
#             for filename in sorted(os.listdir(self.face_dir)):
#                 if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
#                     filepath = os.path.join(self.face_dir, filename)
#                     image_list.append(filepath)
            
#             if not image_list:
#                 print("⚠️  No images found in face directory")
#                 return
            
#             print(f"\n📂 Processing {len(image_list)} images from: {self.face_dir}")
            
#             # Generate embeddings
#             image_dict = {}
            
#             try:
#                 from tqdm import tqdm
#                 use_tqdm = True
#             except ImportError:
#                 use_tqdm = False
            
#             iterator = tqdm(image_list, desc="Generating embeddings") if use_tqdm else image_list
            
#             for image_path in iterator:
#                 filename = os.path.basename(image_path)
#                 if not use_tqdm:
#                     print(f"   Processing: {filename}")
                
#                 try:
#                     if encoder is not None:
#                         # Use vision encoder
#                         embedding = encoder.encode_image(image_path)
#                     elif use_face_recognition:
#                         # Use face_recognition library
#                         import face_recognition
#                         image = face_recognition.load_image_file(image_path)
#                         face_encodings = face_recognition.face_encodings(image)
                        
#                         if face_encodings:
#                             # Pad 128-dim to 512-dim
#                             face_encoding = face_encodings[0]
#                             embedding = list(face_encoding) + [0.0] * (512 - len(face_encoding))
#                         else:
#                             if not use_tqdm:
#                                 print(f"      ⚠️  No face detected, using zero vector")
#                             embedding = [0.0] * 512
#                     else:
#                         # Hash-based fallback
#                         import hashlib
#                         hash_obj = hashlib.sha256(filename.encode())
#                         hash_bytes = hash_obj.digest()
#                         embedding = []
#                         while len(embedding) < 512:
#                             for byte in hash_bytes:
#                                 embedding.append(float(byte) / 255.0)
#                                 if len(embedding) >= 512:
#                                     break
                    
#                     image_dict[image_path] = embedding
                    
#                 except Exception as e:
#                     print(f"   ❌ Failed to generate embedding for {filename}: {e}")
#                     print(f"      Skipped.")
#                     continue
            
#             if not image_dict:
#                 print("❌ No embeddings generated successfully")
#                 return
            
#             # Insert into Milvus
#             print(f"\n💾 Inserting {len(image_dict)} embeddings into Milvus...")
            
#             # Prepare data in correct format
#             data_to_insert = [
#                 {"image_path": k, "embedding": v} 
#                 for k, v in image_dict.items()
#             ]
            
#             # Insert in batches
#             batch_size = 100
#             for i in range(0, len(data_to_insert), batch_size):
#                 batch = data_to_insert[i:i+batch_size]
#                 collection.insert(batch)
#                 if not use_tqdm:
#                     print(f"   Inserted batch {i//batch_size + 1}/{(len(data_to_insert)-1)//batch_size + 1}")
            
#             collection.flush()
            
#             print(f"\n✅ Face recognition ingestion completed ({len(image_dict)} images)")
            
#             # Verify
#             count = collection.num_entities
#             print(f"✅ Collection now contains {count} records")
            
#             # Show sample records
#             print("\n📊 Sample records:")
#             results = collection.query(
#                 expr="id >= 0",
#                 output_fields=["id", "image_path"],
#                 limit=min(10, count)
#             )
#             for r in results:
#                 print(f"   ID {r['id']}: {os.path.basename(r['image_path'])}")
            
#         except Exception as e:
#             print(f"❌ Error in face recognition ingestion: {e}")
#             import traceback
#             traceback.print_exc()
#             raise
    
#     def run_all(self):
#         """Run face recognition ingestion pipeline."""
#         print("🔧 Starting face recognition ingestion pipeline...")
#         self.ingest_face_images()


# RAG Ingestion Pipeline
class RAGIngestionPipeline:
    """
    Handles data ingestion for RAG system:
    - Uni Info (Images)
    - Time Table (Excel)
    - Curriculum (PDF)
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        connect_milvus()
    
    @staticmethod
    def pdf_to_text(path: str) -> str:
        """Extract text from PDF file."""
        doc = fitz.open(path)
        return " ".join(page.get_text() for page in doc)
    
    @staticmethod
    def docx_to_text(path: str) -> str:
        """Extract text from a .docx file."""
        import docx
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def ingest_uni_info(self):
        """Ingest university information (images + docx) into Milvus."""
        try:
            dataset = self.cfg.datasets.uni_info.path
            if not os.path.exists(dataset):
                print(f"⚠️  Warning: Directory {dataset} does not exist")
                return
            
            collection = build_collection(self.cfg, "uni_info")
            
            # Check if collection already has data
            existing_count = collection.num_entities
            if existing_count > 0:
                print(f"ℹ️  Collection 'uni_info' already contains {existing_count} records")
                
                # Data consistency check
                print("   Checking data consistency...")
                results = collection.query(
                    expr='id != ""',
                    output_fields=["file_path"],
                    limit=16384
                )
                existing_files = set(os.path.basename(r["file_path"]) for r in results)
                
                local_files = set()
                for filename in os.listdir(dataset):
                    if filename.lower().endswith((".jpg", ".png", ".jpeg", ".docx")):
                        local_files.add(filename)
                
                if existing_files == local_files:
                    print("✅ Data matches exactly. Skipping ingestion.")
                    return
                else:
                    print(f"⚠️  Data mismatch (DB: {len(existing_files)}, Local: {len(local_files)})")
                    print("   Dropping and recreating collection...")
                    utility.drop_collection("uni_info")
                    collection = build_collection(self.cfg, "uni_info")
            
            encoder = Encoder(self.cfg.models.vision_text_embedding.name)
            text_model = SentenceTransformer(self.cfg.models.text_embedding.name)
            text_emb_dim = self.cfg.models.text_embedding.dim
            doc_emb_dim = self.cfg.models.text_embedding.dim
            
            record_count = 0
            batch_data = []
            
            # ── Process images ──
            for filename in os.listdir(dataset):
                if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    path = os.path.join(dataset, filename)
                    print(f"   Processing image: {filename}")
                    
                    emb = encoder.encode_image(path)
                    
                    data = {
                        "image_embedding": emb,
                        "doc_embedding": [0.0] * doc_emb_dim,
                        "file_path": path,
                        "file_type": "image",
                        "text_content": "",
                    }
                    batch_data.append(data)
                    record_count += 1
            
            # ── Process docx files (with text chunking) ──
            for filename in os.listdir(dataset):
                if filename.lower().endswith(".docx"):
                    path = os.path.join(dataset, filename)
                    print(f"   Processing docx: {filename}")
                    
                    full_text = self.docx_to_text(path)
                    print(f"      Extracted {len(full_text)} chars")
                    
                    # Chunk the text for better retrieval
                    chunks = self.chunk_text(full_text, chunk_size=2000, overlap=200)
                    print(f"      Split into {len(chunks)} chunks")
                    
                    for chunk in chunks:
                        if len(chunk) > 65000:
                            chunk = chunk[:65000]
                        
                        # Encode text and pad to 768-dim for doc_embedding
                        text_vec = text_model.encode(chunk).tolist()
                        if len(text_vec) < doc_emb_dim:
                            text_vec = text_vec + [0.0] * (doc_emb_dim - len(text_vec))
                        
                        data = {
                            "image_embedding": [0.0] * 512,
                            "doc_embedding": text_vec,
                            "file_path": path,
                            "file_type": "docx",
                            "text_content": chunk,
                        }
                        batch_data.append(data)
                        record_count += 1
                        
                        if len(batch_data) >= 50:
                            collection.insert(batch_data)
                            batch_data = []
            
            # Insert remaining records
            if batch_data:
                collection.insert(batch_data)
            
            collection.flush()
            print(f"✅ Uni info ingestion completed ({record_count} records)")
        except Exception as e:
            print(f"❌ Error in uni info ingestion: {e}")
            raise
    
    def ingest_time_table(self):
        """Ingest timetable Excel files into Milvus and MySQL."""
        try:
            dataset = self.cfg.datasets.time_table.path
            if not os.path.exists(dataset):
                print(f"⚠️  Warning: Directory {dataset} does not exist")
                return
            
            collection = build_collection(self.cfg, "time_table")
            
            # Check if collection already has data
            existing_count = collection.num_entities
            if existing_count > 0:
                print(f"ℹ️  Collection 'time_table' already contains {existing_count} records")
                
                # Data consistency check
                print("   Checking data consistency...")
                try:
                    # Check MySQL data vs Excel files
                    db_check = connect_mysql()
                    cursor_check = db_check.cursor()
                    cursor_check.execute("SELECT row_text FROM ExcelTimetableData")
                    existing_rows = set(r[0] for r in cursor_check.fetchall())
                    db_check.close()
                    
                    local_rows = set()
                    for filename in os.listdir(dataset):
                        if filename.endswith(".xlsx"):
                            df = pd.read_excel(os.path.join(dataset, filename))
                            for _, row in df.iterrows():
                                row_values = []
                                for val in row.values:
                                    if pd.isna(val) or val is None: continue
                                    str_val = str(val).strip()
                                    if str_val and str_val.lower() != 'nan':
                                        row_values.append(str_val)
                                if row_values:
                                    text = " ".join(row_values)
                                    if len(text.strip()) >= 3:
                                        local_rows.add(text)
                    
                    if existing_rows == local_rows:
                        print("✅ Data matches exactly. Skipping ingestion.")
                        return
                    else:
                        print(f"⚠️  Data mismatch (DB: {len(existing_rows)}, Local: {len(local_rows)})")
                        print("   Dropping and recreating collection...")
                        utility.drop_collection("time_table")
                        collection = build_collection(self.cfg, "time_table")
                        
                        # Also clear MySQL data
                        print("   Clearing MySQL timetable data...")
                        db = connect_mysql()
                        cursor = db.cursor()
                        cursor.execute("DELETE FROM ExcelTimetableData")
                        db.commit()
                        db.close()
                        
                except Exception as e:
                     print(f"⚠️  Error checking consistency, defaulting to re-ingestion: {e}")
                     print("   Dropping and recreating collection...")
                     utility.drop_collection("time_table")
                     collection = build_collection(self.cfg, "time_table")
                     
                     db = connect_mysql()
                     cursor = db.cursor()
                     cursor.execute("DELETE FROM ExcelTimetableData")
                     db.commit()
                     db.close()
            
            model = SentenceTransformer(self.cfg.models.text_embedding.name)
            
            # Create timetable table if it doesn't exist
            db = connect_mysql()
            cursor = db.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ExcelTimetableData (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    row_text TEXT NOT NULL
                )
            """)
            db.commit()
            
            row_count = 0
            skipped_count = 0
            batch_ids = []
            batch_embeddings = []
            
            for filename in os.listdir(dataset):
                if filename.endswith(".xlsx"):
                    print(f"   Processing Excel file: {filename}")
                    df = pd.read_excel(os.path.join(dataset, filename))
                    
                    for idx, row in df.iterrows():
                        # Convert row to text, handling NaN values
                        row_values = []
                        for val in row.values:
                            # Skip NaN/None values
                            if pd.isna(val) or val is None:
                                continue
                            # Convert to string and strip whitespace
                            str_val = str(val).strip()
                            if str_val and str_val.lower() != 'nan':
                                row_values.append(str_val)
                        
                        # Skip rows that are empty or only contain NaN
                        if not row_values:
                            skipped_count += 1
                            continue
                        
                        text = " ".join(row_values)
                        
                        # Skip if text is too short (likely header or empty)
                        if len(text.strip()) < 3:
                            skipped_count += 1
                            continue
                        
                        try:
                            # Insert into MySQL with INSERT IGNORE to skip duplicates
                            cursor.execute(
                                "INSERT IGNORE INTO ExcelTimetableData (row_text) VALUES (%s)",
                                (text,)
                            )
                            db.commit()
                            
                            row_id = None
                            
                            # Check if insert was successful
                            if cursor.rowcount > 0:
                                row_id = int(cursor.lastrowid)
                            else:
                                # Row exists, fetch the existing ID
                                cursor.execute(
                                    "SELECT id FROM ExcelTimetableData WHERE row_text = %s LIMIT 1",
                                    (text,)
                                )
                                result = cursor.fetchone()
                                if result:
                                    row_id = result[0]
                            
                            if row_id is not None:
                                # Encode text
                                emb = model.encode(text).tolist()
                                
                                # Batch the data
                                batch_ids.append(str(row_id))
                                batch_embeddings.append(emb)
                                row_count += 1
                                
                                # Insert in batches of 100 for efficiency
                                if len(batch_ids) >= 100:
                                    collection.insert([batch_ids, batch_embeddings])
                                    batch_ids = []
                                    batch_embeddings = []
                            else:
                                print(f"   ⚠️  Could not retrieve ID for row {idx}")
                                skipped_count += 1
                        
                        except Exception as row_error:
                            # Log the error but continue processing
                            print(f"   ⚠️  Skipping row {idx}: {str(row_error)[:100]}")
                            skipped_count += 1
                            continue
            
            # Insert remaining records
            if batch_ids:
                collection.insert([batch_ids, batch_embeddings])
            
            collection.flush()
            db.close()
            
            print(f"✅ Time table ingestion completed ({row_count} rows inserted, {skipped_count} rows skipped)")
        except Exception as e:
            print(f"❌ Error in time table ingestion: {e}")
            if 'db' in locals():
                db.close()
            raise
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list:
        """
        Split text into overlapping chunks for better RAG retrieval.
        
        Args:
            text: Full document text
            chunk_size: Target size per chunk (characters)
            overlap: Overlap between consecutive chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at a sentence or paragraph boundary
            if end < len(text):
                # Look for paragraph break first
                break_pos = text.rfind('\n\n', start + chunk_size // 2, end)
                if break_pos == -1:
                    # Look for sentence break
                    break_pos = text.rfind('. ', start + chunk_size // 2, end)
                if break_pos == -1:
                    # Look for any newline
                    break_pos = text.rfind('\n', start + chunk_size // 2, end)
                if break_pos != -1:
                    end = break_pos + 1
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) >= 20:  # Skip very tiny chunks
                chunks.append(chunk)
            
            start = end - overlap  # Overlap for context continuity
        
        return chunks

    def ingest_curriculum(self):
        """Ingest curriculum PDF files into Milvus (with text chunking)."""
        try:
            dataset = self.cfg.datasets.curriculum.path
            if not os.path.exists(dataset):
                print(f"⚠️  Warning: Directory {dataset} does not exist")
                return
            
            collection = build_collection(self.cfg, "curriculum")
            
            # Check if collection already has data
            existing_count = collection.num_entities
            if existing_count > 0:
                print(f"ℹ️  Collection 'curriculum' already contains {existing_count} records")
                
                # Data consistency check
                print("   Checking data consistency...")
                results = collection.query(
                    expr='id != ""',
                    output_fields=["doc_name"],
                    limit=16384
                )
                existing_files = set(r["doc_name"] for r in results)
                
                local_files = set()
                for filename in os.listdir(dataset):
                    if filename.endswith(".pdf"):
                        local_files.add(filename)
                        
                if existing_files == local_files:
                    print("✅ Data matches exactly. Skipping ingestion.")
                    return
                else:
                    print(f"⚠️  Data mismatch (DB: {len(existing_files)}, Local: {len(local_files)})")
                    print("   Dropping and recreating collection...")
                    utility.drop_collection("curriculum")
                    collection = build_collection(self.cfg, "curriculum")
            
            model = SentenceTransformer(self.cfg.models.text_embedding.name)
            
            total_chunks = 0
            batch_names = []
            batch_texts = []
            batch_embeddings = []
            
            for filename in os.listdir(dataset):
                if filename.endswith(".pdf"):
                    print(f"   Processing PDF: {filename}")
                    full_text = self.pdf_to_text(os.path.join(dataset, filename))
                    print(f"      Extracted {len(full_text)} chars")
                    
                    # Chunk the text
                    chunks = self.chunk_text(full_text, chunk_size=2000, overlap=200)
                    print(f"      Split into {len(chunks)} chunks")
                    
                    for chunk in chunks:
                        # Truncate if still over limit (safety net)
                        if len(chunk) > 65000:
                            chunk = chunk[:65000]
                        
                        emb = model.encode(chunk).tolist()
                        
                        batch_names.append(filename)
                        batch_texts.append(chunk)
                        batch_embeddings.append(emb)
                        total_chunks += 1
                        
                        # Insert in batches of 50
                        if len(batch_names) >= 50:
                            collection.insert([batch_names, batch_texts, batch_embeddings])
                            batch_names = []
                            batch_texts = []
                            batch_embeddings = []
            
            # Insert remaining records
            if batch_names:
                collection.insert([batch_names, batch_texts, batch_embeddings])
            
            collection.flush()
            print(f"✅ Curriculum ingestion completed ({total_chunks} chunks from PDFs)")
        except Exception as e:
            print(f"❌ Error in curriculum ingestion: {e}")
            raise
    
    def run_all(self):
        """Run all ingestion pipelines."""
        print("🔧 Starting RAG ingestion pipeline...")
        
        print("\n📊 Ingesting university info...")
        self.ingest_uni_info()
        
        print("\n📊 Ingesting time table...")
        self.ingest_time_table()
        
        print("\n📊 Ingesting curriculum...")
        self.ingest_curriculum()
        
        print("\n✅ RAG ingestion pipeline completed")


# MySQL Pipeline
class MySQLPipeline:
    """Handles MySQL database setup and initial data population."""
    
    def __init__(self):
        self.cfg = load_config()
        self.db_name = self.cfg.mysql.database
        self.face_dir = self.cfg.paths.face_image_dir
        self.initial_data = self._load_initial_data()
    
    def _load_initial_data(self):
        """Load initial data from the dataset directory."""
        # Add the dataset directory to Python path
        dataset_dir = self.cfg.datasets.initial_data.path
        if dataset_dir not in sys.path:
            sys.path.insert(0, dataset_dir)
        
        try:
            # Try different import methods
            try:
                import dataset.initial_data as initial_data
                return initial_data
            except ImportError:
                # Alternative: direct file import
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "initial_data",
                    os.path.join(dataset_dir, "initial_data.py")
                )
                initial_data = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(initial_data)
                return initial_data
        except Exception as e:
            print(f"❌ Error loading initial_data.py: {e}")
            raise
    
    def create_database(self):
        """Create database if it doesn't exist."""
        try:
            # Connect without selecting a database first
            db = connect_mysql(db_name=None)
            cursor = db.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_name}")
            db.close()
            print(f"✅ Database '{self.db_name}' ready")
        except Exception as e:
            print(f"❌ Error creating database: {e}")
            raise
    
    def create_tables(self):
        """Create all required tables."""
        try:
            db = connect_mysql()
            cursor = db.cursor()
            
            # Academic Year table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Academic_Year (
                    RAI_Gen INT PRIMARY KEY,
                    KMITL_Gen INT UNIQUE NOT NULL,
                    year_start INT UNIQUE NOT NULL,
                    year_end INT UNIQUE NOT NULL,
                    F2D_student_id INT UNIQUE NOT NULL
                )
            """)
            
            # Students table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Students (
                    student_id INT PRIMARY KEY,
                    first_name VARCHAR(100) NOT NULL,
                    last_name VARCHAR(100) NOT NULL,
                    nick_name VARCHAR(100),
                    student_email VARCHAR(150) UNIQUE NOT NULL,
                    enrollment_year INT NOT NULL,
                    FOREIGN KEY (enrollment_year) REFERENCES Academic_Year(year_start)
                )
            """)
            
            # Face Recognition Data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Face_Recognition_Data (
                    face_id CHAR(36) PRIMARY KEY,
                    student_id INT NOT NULL,
                    face_image_path VARCHAR(255) UNIQUE NOT NULL,
                    face_encoding VARCHAR(500) UNIQUE NOT NULL,
                    FOREIGN KEY (student_id) REFERENCES Students(student_id)
                )
            """)
            
            # ExcelTimetableData table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ExcelTimetableData (
                    row_id INT PRIMARY KEY AUTO_INCREMENT,
                    row_text TEXT NOT NULL
                )
            """)
            
            db.commit()
            db.close()
            print("✅ Tables created successfully")
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            if 'db' in locals():
                db.close()
            raise
    
    def insert_academic_years(self):
        """Insert academic year records from initial_data.py."""
        try:
            db = connect_mysql()
            cursor = db.cursor()
            
            # Load data from initial_data module
            data = self.initial_data.ACADEMIC_YEARS
            
            # Convert dict to tuple for SQL
            rows = [
                (
                    d["RAI_Gen"],
                    d["KMITL_Gen"],
                    d["year_start"],
                    d["year_end"],
                    d["F2D_student_id"]
                )
                for d in data
            ]
            
            cursor.executemany("""
                INSERT IGNORE INTO Academic_Year 
                (RAI_Gen, KMITL_Gen, year_start, year_end, F2D_student_id)
                VALUES (%s, %s, %s, %s, %s)
            """, rows)
            
            db.commit()
            db.close()
            print(f"✅ Inserted {len(rows)} academic years")
        except Exception as e:
            print(f"❌ Error inserting academic years: {e}")
            if 'db' in locals():
                db.close()
            raise
    
    def insert_students(self):
        """Insert student records from initial_data.py."""
        try:
            db = connect_mysql()
            cursor = db.cursor()
            
            # Load data from initial_data module
            data = self.initial_data.STUDENTS
            
            rows = [
                (
                    d["student_id"],
                    d["first_name"],
                    d["last_name"],
                    d["nick_name"],
                    d["student_email"],
                    d["enrollment_year"]
                )
                for d in data
            ]
            
            cursor.executemany("""
                INSERT IGNORE INTO Students 
                (student_id, first_name, last_name, nick_name, student_email, enrollment_year)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, rows)
            
            db.commit()
            db.close()
            print(f"✅ Inserted {len(rows)} students")
        except Exception as e:
            print(f"❌ Error inserting students: {e}")
            if 'db' in locals():
                db.close()
            raise
                
        except Exception as e:
            print(f"❌ Error inserting face data: {e}")
            if 'db' in locals():
                db.close()
            # Don't raise - this is non-critical
    
    def run_all(self):
        """Run all MySQL setup steps in sequence."""
        print("🔧 Starting MySQL setup pipeline...")
        
        print("\n📊 Creating database...")
        self.create_database()
        
        print("\n📊 Creating tables...")
        self.create_tables()
        
        print("\n📊 Inserting academic years...")
        self.insert_academic_years()
        
        print("\n📊  Inserting students...")
        self.insert_students()
        
        print("\n✅ MySQL pipeline completed successfully")
    
    def close(self):
        """Close any open connections (placeholder for cleanup)."""
        print("🔒 MySQL connections closed")