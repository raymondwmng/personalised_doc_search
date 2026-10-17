"""Stage 1: Script to generate embeddings for all chunks and construct FAISS index."""

import argparse
import logging
from pathlib import Path
import sqlite3
import sys
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.embedder import DocumentEmbedder
from src.indexer import FaissIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("04_embed_and_index")


def main():
    parser = argparse.ArgumentParser(description="Encode chunks with sentence-transformers and build FAISS index.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    conn = sqlite3.connect(str(config.DOCUMENTS_DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
    SELECT 
        c.id AS chunk_id,
        c.doc_id,
        c.chunk_idx,
        c.text AS chunk_text,
        c.token_count,
        d.celex,
        d.title,
        d.date,
        d.type AS doc_type
    FROM chunks c
    JOIN documents d ON c.doc_id = d.id
    ORDER BY c.doc_id, c.chunk_idx
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        logger.warning("No chunks found in database. Run 02_parse_and_clean.py first.")
        return

    logger.info(f"Loaded {len(rows)} chunks from SQLite database.")

    metadata = []
    chunk_texts = []
    for r in rows:
        chunk_texts.append(r["chunk_text"])
        metadata.append({
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "chunk_idx": r["chunk_idx"],
            "text": r["chunk_text"],
            "token_count": r["token_count"],
            "celex": r["celex"],
            "title": r["title"],
            "date": r["date"],
            "doc_type": r["doc_type"],
        })

    # Embed using SentenceTransformer
    embedder = DocumentEmbedder()
    embeddings = embedder.encode(chunk_texts, batch_size=args.batch_size)

    # Save cached embeddings
    emb_cache_path = config.EMBEDDINGS_DIR / "chunk_embeddings.npy"
    np.save(emb_cache_path, embeddings)
    logger.info(f"Cached {embeddings.shape[0]} embeddings to {emb_cache_path}")

    # Build FAISS index
    indexer = FaissIndexer()
    indexer.build_index(embeddings, metadata)
    indexer.save()

    logger.info("Vector index generation and storage complete.")


if __name__ == "__main__":
    main()
