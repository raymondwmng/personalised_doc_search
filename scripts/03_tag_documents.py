"""Script to tag all documents in documents.db using TF-IDF, Legal-BERT, or Hybrid methods (Section 3.3 & 4.2)."""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.tagger import TaxonomyTagger
from src.tagger_bert import LegalBertTagger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("03_tag_documents")


def main():
    parser = argparse.ArgumentParser(description="Tag documents with 3-dimensional taxonomy.")
    parser.add_argument("--db-path", type=str, default=str(config.DOCUMENTS_DB_PATH), help="Target SQLite DB path")
    parser.add_argument(
        "--method",
        type=str,
        default="bert",
        choices=["bert", "tfidf", "hybrid"],
        help="Tagging methodology (default: bert)",
    )
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for DB commits")
    parser.add_argument("--limit", type=int, default=None, help="Optional document count limit")
    args = parser.parse_args()

    db_path = Path(args.db_path)

    if args.method in ["tfidf", "hybrid"]:
        logger.info("Running TF-IDF & EuroVoc Taxonomy Tagger...")
        tfidf_tagger = TaxonomyTagger(db_path=db_path)
        count_tfidf = tfidf_tagger.tag_all_documents(batch_size=args.batch_size)
        logger.info(f"TF-IDF tagging complete: {count_tfidf} documents tagged.")

    if args.method in ["bert", "hybrid"]:
        logger.info("Running Legal-BERT Semantic Taxonomy Tagger...")
        bert_tagger = LegalBertTagger(db_path=db_path)
        count_bert = bert_tagger.tag_all_documents(batch_size=args.batch_size, limit=args.limit)
        logger.info(f"Legal-BERT tagging complete: {count_bert} documents tagged.")

    # Fetch and log summary stats
    with tfidf_tagger.get_connection() if args.method in ["tfidf", "hybrid"] else bert_tagger.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT method, dimension, COUNT(*) as cnt FROM document_tags GROUP BY method, dimension")
        stats = cursor.fetchall()

    logger.info("=" * 60)
    logger.info("Tagging Summary Across Methods:")
    for row in stats:
        method = row["method"] if "method" in row.keys() else "unknown"
        logger.info(f"  Method: {method:<12} | Dimension: {row['dimension']:<22} -> {row['cnt']} tags")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
