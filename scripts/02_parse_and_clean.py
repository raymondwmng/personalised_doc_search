"""Stage 1: Script to parse and clean documents, chunk them, and store into SQLite documents.db."""

import argparse
import json
import logging
from pathlib import Path
import re
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.parser import LegalDocParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("02_parse_and_clean")


def is_corrigendum(celex: str) -> bool:
    """Checks if a CELEX identifier represents a Corrigendum / Rectificatif."""
    if not celex:
        return True
    return bool(re.search(r"R\(\d+\)|R\d+$", celex))


def main():
    parser = argparse.ArgumentParser(
        description="Parse raw text/HTML files and save clean text & chunks to SQLite."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(config.DOCUMENTS_DB_PATH),
        help="Target SQLite DB path",
    )
    parser.add_argument(
        "--rechunk",
        action="store_true",
        help="Re-chunk existing documents in documents.db using specified regime (e.g. Meso ~2000 tokens)",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="meso",
        choices=["meso", "micro", "macro"],
        help="Chunking regime to apply (default: meso, Rec 3)",
    )
    parser.add_argument(
        "--from-kaggle",
        action="store_true",
        help="Parse directly from Kaggle EurLex CSV instead of data/raw files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=config.DEFAULT_LOAD_LIMIT,
        help="Limit when using --from-kaggle",
    )
    args = parser.parse_args()

    doc_parser = LegalDocParser(db_path=Path(args.db_path))

    if args.rechunk:
        logger.info(f"Re-chunking documents with regime={args.regime} (target_tokens={config.CHUNK_SIZE_TOKENS})...")
        res = doc_parser.rechunk_database(
            regime=args.regime,
            target_tokens=config.CHUNK_SIZE_TOKENS,
            overlap_tokens=config.CHUNK_OVERLAP_TOKENS,
        )
        logger.info(f"Re-chunking finished: {res}")
        return

    if args.from_kaggle:
        logger.info(f"Parsing directly from Kaggle CSV with limit={args.limit}...")
        from src.downloader import EurlexDatasetLoader
        loader = EurlexDatasetLoader()
        loader.load_and_process(limit=args.limit, ingest_to_db=True, save_raw_txt=True)
        return

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

    # Find raw files (.txt and .html)
    raw_files = list(config.RAW_DIR.glob("*.txt")) + list(config.RAW_DIR.glob("*.html"))
    logger.info(f"Found {len(raw_files)} raw files in {config.RAW_DIR}")

    total_docs = 0
    skipped_corrigenda = 0

    for file_path in raw_files:
        celex = file_path.stem

        # Filter out noisy corrigenda
        if is_corrigendum(celex):
            skipped_corrigenda += 1
            continue

        meta = metadata_map.get(celex, {})
        title = meta.get("title", f"Document {celex}")
        doc_date = meta.get("date", "2026-01-01")
        doc_type = meta.get("type", "Regulation")
        eurovoc_raw = meta.get("eurovoc_raw", "")
        source = meta.get("source", "eurlex")

        content = file_path.read_text(encoding="utf-8", errors="ignore")
        if len(content.strip()) < 100:
            continue

        is_html = file_path.suffix.lower() == ".html"
        doc_id = doc_parser.process_and_store_document(
            celex=celex,
            title=title,
            date=doc_date,
            doc_type=doc_type,
            raw_text_or_html=content,
            eurovoc_raw=eurovoc_raw,
            source=source,
            is_html=is_html,
        )

        if doc_id:
            total_docs += 1

    if skipped_corrigenda > 0:
        logger.info(f"Skipped {skipped_corrigenda} noisy corrigenda files.")

    # Verify counts in SQLite
    with doc_parser.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS cnt FROM documents")
        doc_count = cursor.fetchone()["cnt"]
        cursor.execute("SELECT COUNT(*) AS cnt FROM chunks")
        chunk_count = cursor.fetchone()["cnt"]

    logger.info(
        f"Parsing complete. Database currently has {doc_count} documents and {chunk_count} chunks."
    )


if __name__ == "__main__":
    main()
