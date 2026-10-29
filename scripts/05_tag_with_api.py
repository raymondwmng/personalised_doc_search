"""Stage 2: Script to analyze and enhance documents using Azure AI Document Intelligence (Section 4.3)."""

import argparse
import json
import logging
from pathlib import Path
import sqlite3
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.tagger_api import AzureDocIntelligenceClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("05_tag_with_api")


def main():
    parser = argparse.ArgumentParser(description="Extract structured layout & tables with Azure Document Intelligence.")
    parser.add_argument("--db-path", type=str, default=str(config.DOCUMENTS_DB_PATH), help="Target SQLite DB path")
    parser.add_argument("--limit", type=int, default=10, help="Max documents to inspect/process")
    parser.add_argument("--force-model", type=str, choices=["layout", "read"], default=None, help="Force specific model")
    args = parser.parse_args()

    client = AzureDocIntelligenceClient()
    logger.info("=" * 70)
    logger.info("  Azure Document Intelligence Layout Extraction (Stage 2)")
    logger.info(f"  Configured: {client.is_configured} | Budget: {client.budget_pages} pages / £5.00 envelope")
    logger.info("=" * 70)

    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, celex, title, full_text FROM documents LIMIT ?", (args.limit,))
    docs = cursor.fetchall()
    conn.close()

    if not docs:
        logger.warning("No documents found in database.")
        return

    logger.info(f"Inspecting {len(docs)} documents for structural complexity...")
    complex_count = 0

    for doc in docs:
        celex = doc["celex"]
        text = doc["full_text"] or ""
        complexity = client.assess_complexity(text)

        status_flag = "🔬 COMPLEX (Layout)" if complexity["is_complex"] else "📄 STANDARD (Read)"
        logger.info(
            f"CELEX {celex:<12} | {status_flag} | "
            f"Tables: {complexity['table_count']:<2} | Annexes: {complexity['annex_count']:<2} | Approx Pages: {complexity['approx_pages']:<3}"
        )

        if complexity["is_complex"]:
            complex_count += 1
            analysis = client.analyze_document(text, celex=celex, force_model=args.force_model)
            table_info = analysis.get("tables", [])
            logger.info(f"   -> Extracted {len(table_info)} structured table blocks via {analysis.get('model_used')}")

    logger.info("=" * 70)
    logger.info("Budget & Usage Summary:")
    summary = client.get_budget_summary()
    for k, v in summary.items():
        logger.info(f"  {k:<25}: {v}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

