"""
Entry point for the Student Enrollment Pipeline.

Usage:
    python entrypoints/entry_enrollment.py
"""

import sys
import os

# Ensure project root is on sys.path so `src.*` imports work
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils_enrollment import load_config
from src.pipelines.enrollment_pipeline import EnrollmentPipeline


def main():
    """Main entry point for enrollment."""
    try:
        print("=" * 60)
        print("STUDENT ENROLLMENT PIPELINE")
        print("=" * 60)

        # Load configuration
        print("\n📋 Loading configuration...")
        cfg = load_config()
        print("✅ Configuration loaded")

        # Run enrollment pipeline
        pipeline = EnrollmentPipeline(cfg)
        pipeline.run()

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
