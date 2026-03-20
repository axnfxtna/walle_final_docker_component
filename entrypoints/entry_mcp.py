"""
Entry point for the MCP Context Builder.
=========================================
Standalone interactive test loop to verify session
and context building logic.

Usage:
    python entrypoints/entry_mcp.py
"""

import sys
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils_mcp import load_config
from src.pipelines.mcp_pipeline import ContextBuilder

def main():
    print("=" * 60)
    print("MCP CONTEXT BUILDER SERVICE")
    print("=" * 60)

    # Load configuration
    print("\n📋 Loading configuration...")
    cfg = load_config()
    print("✅ Configuration loaded")

    # Initialize ContextBuilder
    mcp = ContextBuilder(cfg)
    session_id = "test_session_1"
    mcp.create_session(session_id)

    print("\n" + "=" * 60)
    print("INTERACTIVE TEST MODE  (type 'quit' to exit)")
    print("Available Commands:")
    print("  id <id> <name>  : Update student identity (e.g. id 65000001 Sarucha)")
    print("  loc <location>  : Update robot location (e.g. loc Library)")
    print("  say <message>   : Add user message to conversation history")
    print("  bot <message>   : Add bot message to conversation history")
    print("  show            : Print current context structure and LLM prompt")
    print("=" * 60)

    while True:
        try:
            cmd_line = input("\n📝 Command: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting MCP demo...")
            break

        if not cmd_line:
            continue
            
        parts = cmd_line.split(" ", 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            print("👋 Exiting...")
            break
            
        elif cmd == "id":
            if " " not in args:
                print("⚠️ Usage: id <student_id> <name>")
                continue
            student_id, name = args.split(" ", 1)
            mcp.update_student_identity(session_id, student_id, name)
            print(f"✅ Updated identity: {student_id} -> {name}")
            
        elif cmd == "loc":
            mcp.update_location(session_id, args)
            print(f"✅ Updated location: {args}")
            
        elif cmd == "say":
            mcp.add_conversation_turn(session_id, "user", args)
            print("✅ Added user turn")
            
        elif cmd == "bot":
            mcp.add_conversation_turn(session_id, "assistant", args)
            print("✅ Added bot turn")
            
        elif cmd == "show":
            llm_ctx = mcp.build_llm_context(session_id)
            prompt = mcp.format_context_as_prompt(llm_ctx)
            print("\n" + "-" * 40)
            print("1. Context Dictionary structure:")
            print(json.dumps(llm_ctx, indent=2, ensure_ascii=False))
            print("-" * 40)
            print("2. Generated LLM Prompt String:")
            print(prompt)
            print("-" * 40)
            
        else:
            print("❌ Unknown command")

if __name__ == "__main__":
    main()
