"""Stage 1: Script to parse and clean HTML documents, chunk them, and store into SQLite."""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.parser import LegalDocParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("02_parse_and_clean")


def main():
    parser = argparse.ArgumentParser(description="Parse raw HTML files and save clean text & chunks to SQLite.")
    parser.add_argument("--db-path", type=str, default=str(config.DOCUMENTS_DB_PATH), help="Target SQLite DB path")
    args = parser.parse_args()

    doc_parser = LegalDocParser(db_path=Path(args.db_path))

    metadata_path = config.PROCESSED_DIR / "metadata.jsonl"
    metadata_map = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        metadata_map[rec.get("celex")] = rec
                    except Exception:
                        pass

    raw_files = list(config.RAW_DIR.glob("*.html"))
    logger.info(f"Found {len(raw_files)} raw HTML files in {config.RAW_DIR}")

    total_docs = 0
    total_chunks = 0

    for file_path in raw_files:
        celex = file_path.stem
        meta = metadata_map.get(celex, {})
        title = meta.get("title", f"Document {celex}")
        doc_date = meta.get("date", "2026-01-01")
        doc_type = meta.get("type", "Regulation")
        eurovoc_raw = meta.get("eurovoc_raw", "")
        source = meta.get("source", "eurlex")

        html_content = file_path.read_text(encoding="utf-8", errors="ignore")
        doc_id = doc_parser.process_and_store_document(
            celex=celex,
            title=title,
            date=doc_date,
            doc_type=doc_type,
            raw_text_or_html=html_content,
            eurovoc_raw=eurovoc_raw,
            source=source,
            is_html=True,
        )

        if doc_id:
            total_docs += 1

    # Verify counts in SQLite
    with doc_parser.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS cnt FROM documents")
        doc_count = cursor.fetchone()["cnt"]
        cursor.execute("SELECT COUNT(*) AS cnt FROM chunks")
        chunk_count = cursor.fetchone()["cnt"]

    logger.info(f"Parsing complete. Database currently has {doc_count} documents and {chunk_count} chunks.")


if __name__ == "__main__":
    main()
