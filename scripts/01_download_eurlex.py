"""Stage 1: Script to load EUR-Lex documents and metadata from the comprehensive Kaggle dataset.

Replaces the legacy SPARQL/Cellar scraping mechanism, eliminating noise (like *R*.html corrigenda)
and providing rich, clean full texts with official EUROVOC annotations.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.downloader import EurlexDatasetLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("01_download_eurlex")


def clean_noisy_corrigenda_files(raw_dir: Path) -> int:
    """Removes legacy corrigenda files (*R*.html) left behind by previous SPARQL scrapes."""
    removed = 0
    if not raw_dir.exists():
        return removed

    for p in list(raw_dir.glob("*R*.html")):
        try:
            p.unlink()
            removed += 1
        except Exception as e:
            logger.warning(f"Failed to remove {p}: {e}")
    if removed > 0:
        logger.info(f"Cleaned up {removed} noisy corrigenda files from {raw_dir}")
    return removed


def main():
    parser = argparse.ArgumentParser(
        description="Load EUR-Lex documents and metadata from the Kaggle dataset."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=config.DEFAULT_LOAD_LIMIT,
        help=f"Number of documents to load (default: {config.DEFAULT_LOAD_LIMIT}; set to 0 or negative for all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Load all available documents from the dataset (>90,000 with EuroVoc, 142,000 total)",
    )
    parser.add_argument(
        "--source-csv",
        type=str,
        default=str(config.EURLEX_KAGGLE_CSV),
        help=f"Path to Kaggle EurLex CSV file (default: {config.EURLEX_KAGGLE_CSV})",
    )
    parser.add_argument(
        "--allow-missing-eurovoc",
        action="store_true",
        help="Allow documents that do not have EUROVOC annotations (default is EuroVoc-only)",
    )
    parser.add_argument(
        "--act-type",
        type=str,
        default=None,
        help="Filter by act type: Regulation, Directive, Decision, etc. (default: all)",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        default=True,
        help="Save document full text files to data/raw/{celex}.txt (default: True)",
    )
    parser.add_argument(
        "--ingest-to-db",
        action="store_true",
        default=True,
        help="Directly chunk and store into documents.db during load (default: True)",
    )
    parser.add_argument(
        "--include-local",
        action="store_true",
        default=True,
        help="Ingest local benchmark test JSONL files if present",
    )
    parser.add_argument(
        "--clean-corrigenda",
        action="store_true",
        default=True,
        help="Clean up any old noisy *R*.html corrigenda files in data/raw/",
    )
    args = parser.parse_args()

    # Step 1: Clean up old noisy corrigenda files if requested
    if args.clean_corrigenda:
        clean_noisy_corrigenda_files(config.RAW_DIR)

    loader = EurlexDatasetLoader(kaggle_csv=Path(args.source_csv))

    # Step 2: Ingest local benchmark files if present
    if args.include_local:
        for jsonl_filename in ["test3-legal-documents.jsonl", "long-test3-legal-documents.jsonl"]:
            local_path = config.DATA_DIR / jsonl_filename
            if local_path.exists():
                loader.ingest_local_jsonl(local_path)

    # Step 3: Determine limit
    limit = None if args.all or args.limit <= 0 else args.limit
    eurovoc_only = not args.allow_missing_eurovoc

    logger.info(
        f"Starting dataset load from {args.source_csv} (limit={limit}, eurovoc_only={eurovoc_only}, act_type={args.act_type})..."
    )

    docs = loader.load_and_process(
        limit=limit,
        eurovoc_only=eurovoc_only,
        act_type=args.act_type,
        save_metadata=True,
        save_raw_txt=args.save_raw,
        ingest_to_db=args.ingest_to_db,
    )

    logger.info(
        f"Done! Successfully loaded {len(docs)} documents. "
        f"Metadata written to {config.PROCESSED_DIR / 'metadata.jsonl'}."
    )
    if args.ingest_to_db:
        logger.info(f"Database {config.DOCUMENTS_DB_PATH} updated with clean documents and chunks.")


if __name__ == "__main__":
    main()
