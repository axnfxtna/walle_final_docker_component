"""
Entry point for RAG Query Pipeline (Wall-E chatbot).
=====================================================
Connects OllamaClient to MySQL + Milvus for Thai-language Q&A.

Usage:
    cd /home/sarucha3/walle_capstone/final_docker_component
    python entrypoints/entry_rag.py
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils_database import load_config
from src.utils_rag import load_config as load_rag_config
from src.utils_mcp import load_config as load_mcp_config
from src.pipelines.rag_pipeline import RAGQueryPipeline, interactive_mode, auto_stt_mode


def main():
    """Main entry point for RAG query chatbot."""
    try:
        print("=" * 60)
        print("RAG QUERY PIPELINE  (Wall-E Chatbot)")
        print("=" * 60)

        # Load configuration
        print("\n📋 Loading configuration...")
        cfg = load_config()
        rag_cfg = load_rag_config()
        mcp_cfg = load_mcp_config()
        print("✅ Configuration loaded")

        # Initialise pipeline
        print("\n" + "=" * 60)
        print("INITIALISING RAG QUERY PIPELINE")
        print("=" * 60)

        try:
            pipeline = RAGQueryPipeline(
                cfg,
                mcp_cfg=mcp_cfg,
                ollama_url=rag_cfg.ollama.api_url,
                ollama_model=rag_cfg.ollama.model,
            )
            print("\n✅ RAG query pipeline initialised successfully")
        except Exception as e:
            print(f"\n❌ RAG query pipeline failed: {e}")
            raise

        # Summary
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print("✅ All components ready")

        # Start auto STT mode
        print("\n" + "=" * 60)
        print("AUTO STT MODE")
        print("=" * 60)

        json_path = os.environ.get("STT_JSON_PATH", "/app/received_events.json")
        auto_stt_mode(pipeline, json_path)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()