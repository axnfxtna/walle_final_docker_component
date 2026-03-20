import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(BASE_DIR, "database")

if DATABASE_DIR not in sys.path:
    sys.path.insert(0, DATABASE_DIR)

from src.utils_database import load_config
from src.pipelines.database_pipeline import (
    MySQLPipeline,
    RAGIngestionPipeline,
    # FaceRecognitionPipeline,
)


def main():
    """Main entry point for database preparation."""
    try:
        print("="*60)
        print("DATABASE PREPARATION PIPELINE")
        print("="*60)
        
        # Load configuration
        print("\n📋 Loading configuration...")
        cfg = load_config()
        print("✅ Configuration loaded")

        # # Face Recognition Pipeline
        
        # print("FACE RECOGNITION INGESTION PIPELINE")
        # print("="*60)
        
        # try:
        #     face_pipeline = FaceRecognitionPipeline(cfg)
        #     face_pipeline.run_all()
        #     print("\n✅ Face recognition pipeline completed successfully")
        # except Exception as e:
        #     print(f"\n❌ Face recognition pipeline failed: {e}")

        # MySQL Pipeline
        print("\n" + "="*60)
        print("MYSQL SETUP PIPELINE")
        print("="*60)
        
        try:
            mysql_pipeline = MySQLPipeline()
            mysql_pipeline.run_all()
            mysql_pipeline.close()
            print("\n✅ MySQL pipeline completed successfully")
        except Exception as e:
            print(f"\n❌ MySQL pipeline failed: {e}")

        # RAG / Milvus Pipeline
        print("\n" + "="*60)
        print("RAG INGESTION PIPELINE")
        print("="*60)
        
        try:
            rag_pipeline = RAGIngestionPipeline(cfg)
            rag_pipeline.run_all()
            print("\n✅ RAG pipeline completed successfully")
        except Exception as e:
            print(f"\n❌ RAG pipeline failed: {e}")
            raise


        # Summary
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        print("✅ All pipelines completed successfully")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()