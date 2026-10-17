"""Stage 1: Script to tag all documents in documents.db using TF-IDF and EuroVoc mapping."""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.tagger import TaxonomyTagger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("03_tag_documents")


def main():
    parser = argparse.ArgumentParser(description="Tag documents with 3-dimensional taxonomy.")
    parser.add_argument("--db-path", type=str, default=str(config.DOCUMENTS_DB_PATH), help="Target SQLite DB path")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for vectorized processing and DB commits")
    args = parser.parse_args()

    logger.info("initialise tagger")
    tagger = TaxonomyTagger(db_path=Path(args.db_path))
    count = tagger.tag_all_documents(batch_size=args.batch_size)
    logger.info("All documents tagged")

    logger.info("initialise DB")
    with tagger.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT dimension, COUNT(*) as cnt FROM document_tags GROUP BY dimension")
        stats = cursor.fetchall()
    logger.info("All stats fetched")

    logger.info("Tagging summary:")
    for row in stats:
        logger.info(f"  Dimension: {row['dimension']:<20} -> {row['cnt']} tags assigned")


if __name__ == "__main__":
    main()
