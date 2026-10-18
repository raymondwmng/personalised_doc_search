"""Personalised EUR-Lex Legal Search POC Entry Point."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config


def launch_streamlit():
    """Launches the Streamlit web application."""
    app_path = BASE_DIR / "ui" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        "--theme.base=dark",
    ]
    print(f"Launching Streamlit application from {app_path}...")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Personalised EUR-Lex Document Search POC")
    parser.add_argument("--ui", action="store_true", default=True, help="Launch Streamlit Web UI (default)")
    parser.add_argument("--search", type=str, help="Run a quick command-line search query")
    parser.add_argument("--model", type=str, default=config.DEFAULT_ANALYTICAL_MODEL, help="Model to use (Mistral API e.g. mistral-large-4 / mistral-small-2603, or local Ollama)")
    parser.add_argument("--memory", action="store_true", default=True, help="Enable memory prompt")
    args, unknown = parser.parse_known_args()

    if args.search:
        from src.searcher import SearchEngine
        from src.summariser import LegalSummariser
        from src.memory import UserMemoryStore

        print(f"\n[Query]: {args.search}")
        mem = UserMemoryStore()
        searcher = SearchEngine(memory_store=mem)
        results = searcher.search(args.search, top_n_docs=5, enable_memory=args.memory)
        print(f"Retrieved {len(results)} relevant documents.\n")

        for idx, doc in enumerate(results, 1):
            print(f"{idx}. {doc['title']} (CELEX: {doc['celex']}) - Score: {doc['peak_score']}")
            tags = doc.get("tags", {})
            print(f"   Jurisdiction: {[t['category'] for t in tags.get('Jurisdiction', [])]}")
            print(f"   Topics: {[t['category'] for t in tags.get('Regulatory Topics', [])]}")
            print(f"   Sector: {[t['category'] for t in tags.get('Industry / Sector', [])]}")

        summariser = LegalSummariser()
        mem_ctx = mem.get_prompt_context_string() if args.memory else None
        summary_res = summariser.summarise(
            args.search,
            results,
            model_name=args.model,
            memory_context=mem_ctx,
            max_docs=len(results),
        )
        print(f"\n--- Executive Summary (Synthesised across {summary_res.get('docs_analyzed', len(results))} Acts) ---")
        print(summary_res.get("summary"))
        return

    launch_streamlit()


if __name__ == "__main__":
    main()
