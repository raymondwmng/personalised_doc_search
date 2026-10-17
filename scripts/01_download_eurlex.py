"""Stage 1: Script to download EUR-Lex documents and metadata via SPARQL/Cellar."""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.downloader import EurlexDownloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("01_download_eurlex")


def main():
    parser = argparse.ArgumentParser(description="Download EUR-Lex documents and metadata.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of documents to download (default: 50 for quick MVP test, can increase up to 5000)")
    parser.add_argument("--offset", type=int, default=0, help="SPARQL query offset")
    parser.add_argument("--include-local", action="store_true", default=True, help="Ingest local test documents in data/ if available")
    parser.add_argument("--skip-sparql", action="store_true", help="Skip SPARQL and only process local documents")
    args = parser.parse_args()

    downloader = EurlexDownloader()

    # Step 1: Ingest local benchmark files if present
    if args.include_local:
        for jsonl_filename in ["test3-legal-documents.jsonl", "long-test3-legal-documents.jsonl"]:
            local_path = config.DATA_DIR / jsonl_filename
            if local_path.exists():
                logger.info(f"Ingesting local document file: {local_path}")
                downloader.ingest_local_jsonl(local_path)

    if args.skip_sparql:
        logger.info("Skipping SPARQL download as requested.")
        return

    # Step 2: Query SPARQL endpoint
    logger.info(f"Starting SPARQL metadata retrieval for limit={args.limit}, offset={args.offset}...")
    docs = downloader.query_metadata(limit=args.limit, offset=args.offset)
    existing_meta = downloader.load_existing_metadata()

    success_count = 0
    for idx, doc in enumerate(docs, 1):
        celex = doc["celex"]
        logger.info(f"[{idx}/{len(docs)}] Processing CELEX: {celex} - {doc['title'][:60]}...")
        
        # Download HTML
        html_content = downloader.download_document_html(celex, doc.get("html_url"))
        if html_content:
            success_count += 1
            if celex not in existing_meta:
                downloader.save_metadata_record(doc)
                existing_meta[celex] = doc

    logger.info(f"Download complete: {success_count}/{len(docs)} documents successfully saved to {config.RAW_DIR}")


if __name__ == "__main__":
    main()
