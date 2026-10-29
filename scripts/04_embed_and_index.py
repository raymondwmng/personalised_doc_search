"""Stage 2: Script to encode Meso chunks and build FAISS IndexFlatIP and BM25Okapi indices (Recommendations 1-3)."""

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


def build_for_model(model_key: str, chunk_texts: list, metadata: list, batch_size: int = 16) -> None:
    """Encodes chunk texts with the specified model and saves FAISS + BM25 indices."""
    spec = config.resolve_embedding_spec(model_key)
    logger.info("=" * 70)
    logger.info(f"Building Index for: {spec['display_name']}")
    logger.info(f"Target Index Path: {spec['index_path']}")
    logger.info(f"Dimension: {spec['dim']} | Max Seq Length: {spec['max_seq_length']}")
    logger.info("=" * 70)

    embedder = DocumentEmbedder(model_name_or_key=model_key)
    embeddings = embedder.encode(chunk_texts, batch_size=batch_size, show_progress_bar=True)

    emb_cache_path = config.EMBEDDINGS_DIR / f"chunk_embeddings_{spec['id']}.npy"
    np.save(emb_cache_path, embeddings)
    logger.info(f"Saved cached embeddings to {emb_cache_path}")

    indexer = FaissIndexer(model_name_or_key=model_key)
    indexer.build_index(embeddings, metadata, build_bm25=True)
    indexer.save()
    logger.info(f"Index successfully built and saved for {spec['id']}.")


def main():
    parser = argparse.ArgumentParser(description="Encode Meso chunks with BGE-M3 and/or ModernBERT and build indices.")
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["bge-m3", "modernbert", "both"],
        help="Embedding model backbone to build index for (default: both)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size (default: 16)")
    parser.add_argument("--limit", type=int, default=None, help="Optional chunk limit for testing")
    args = parser.parse_args()

    conn = sqlite3.connect(str(config.DOCUMENTS_DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    from experiments.long_context_embed_study.hierarchical_chunker import BENCHMARK_GT_CELEX, CASE3_DISTRACTOR_CELEX
    priority_celexes = BENCHMARK_GT_CELEX + CASE3_DISTRACTOR_CELEX
    placeholders = ",".join(f"'{c}'" for c in priority_celexes)

    query = f"""
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
    ORDER BY CASE WHEN d.celex IN ({placeholders}) THEN 0 ELSE 1 END, c.doc_id, c.chunk_idx
    """
    if args.limit:
        query += f" LIMIT {args.limit}"

    cursor.execute(query)
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

    models_to_build = ["bge-m3", "modernbert"] if args.model == "both" else [args.model]

    for m_key in models_to_build:
        build_for_model(m_key, chunk_texts, metadata, batch_size=args.batch_size)

    logger.info("All requested indices successfully constructed.")


if __name__ == "__main__":
    main()
