"""Multi-Model Indexer module using FAISS IndexFlatIP and BM25Okapi for hybrid search (Section 3.2.2 & 4.6)."""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenizes text into lowercase terms for BM25 indexing."""
    if not text:
        return []
    cleaned = re.sub(r"[^\w\s\-]", " ", text.lower())
    return [t for t in cleaned.split() if len(t) >= 2]


class FaissIndexer:
    """Manages FAISS inner product index, BM25Okapi index, and associated chunk metadata."""

    def __init__(
        self,
        model_name_or_key: Optional[str] = None,
        index_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
        bm25_path: Optional[Path] = None,
        dimension: Optional[int] = None,
    ):
        self.spec = config.resolve_embedding_spec(model_name_or_key)
        self.model_key = self.spec.get("id", config.DEFAULT_EMBEDDING_MODEL)

        self.index_path = index_path or self.spec.get("index_path", config.FAISS_INDEX_PATH)
        self.meta_path = meta_path or self.spec.get("meta_path", config.CHUNK_METADATA_PATH)
        self.bm25_path = bm25_path or self.spec.get("bm25_path", config.INDEX_DIR / f"bm25_{self.model_key}.pkl")
        self.dimension = dimension or self.spec.get("dim", config.EMBEDDING_DIM)

        self.index: Optional[faiss.Index] = None
        self.bm25: Optional[BM25Okapi] = None
        self.metadata: List[Dict[str, Any]] = []

    def build_index(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        build_bm25: bool = True,
    ) -> None:
        """Constructs an IndexFlatIP index and optionally builds BM25 index."""
        if embeddings.shape[0] != len(metadata):
            raise ValueError(f"Mismatch: {embeddings.shape[0]} embeddings vs {len(metadata)} metadata items.")

        dim = embeddings.shape[1]
        self.dimension = dim
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata
        logger.info(f"Built FAISS IndexFlatIP ({self.model_key}) with {self.index.ntotal} vectors of dimension {dim}.")

        if build_bm25:
            logger.info("Building BM25Okapi index over chunk texts...")
            tokenized_corpus = [tokenize_for_bm25(item.get("text", "")) for item in metadata]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"Built BM25Okapi index with {len(tokenized_corpus)} documents.")

    def save(self) -> None:
        """Saves FAISS index binary, metadata JSONL, and BM25 index."""
        if self.index is None:
            raise ValueError("No index to save.")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        with open(self.meta_path, "w", encoding="utf-8") as f:
            for item in self.metadata:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        if self.bm25 is not None:
            with open(self.bm25_path, "wb") as f:
                pickle.dump(self.bm25, f)
            logger.info(f"Saved BM25 index to {self.bm25_path}.")

        logger.info(f"Saved FAISS index to {self.index_path} and metadata to {self.meta_path}.")

    def load(self) -> bool:
        """Loads FAISS index, metadata, and BM25 index if files exist."""
        # Check model-specific index first, fallback to default legacy index if not found
        actual_index_path = self.index_path
        actual_meta_path = self.meta_path

        if not actual_index_path.exists() or not actual_meta_path.exists():
            legacy_idx = config.INDEX_DIR / "faiss_flat_ip.index"
            legacy_meta = config.INDEX_DIR / "chunk_meta.jsonl"
            if legacy_idx.exists() and legacy_meta.exists():
                logger.info(f"Model-specific index {actual_index_path} not found. Falling back to {legacy_idx}...")
                actual_index_path = legacy_idx
                actual_meta_path = legacy_meta
            else:
                logger.warning(f"Index or metadata file does not exist at {actual_index_path}.")
                return False

        self.index = faiss.read_index(str(actual_index_path))
        self.metadata = []
        with open(actual_meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))

        logger.info(f"Loaded FAISS index with {self.index.ntotal} items from {actual_index_path}.")

        # Load BM25 index if available, otherwise construct in memory
        if self.bm25_path.exists():
            try:
                with open(self.bm25_path, "rb") as f:
                    self.bm25 = pickle.load(f)
                logger.info(f"Loaded pre-computed BM25 index from {self.bm25_path}.")
            except Exception as e:
                logger.warning(f"Could not load BM25 cache ({e}). Rebuilding in-memory...")
                self._rebuild_bm25_in_memory()
        else:
            self._rebuild_bm25_in_memory()

        return True

    def _rebuild_bm25_in_memory(self) -> None:
        """Rebuilds BM25 index in memory from loaded metadata."""
        if self.metadata:
            tokenized_corpus = [tokenize_for_bm25(item.get("text", "")) for item in self.metadata]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"Constructed in-memory BM25Okapi index with {len(tokenized_corpus)} items.")

    def search_dense(self, query_vector: np.ndarray, top_k: int = 20) -> List[Dict[str, Any]]:
        """Searches FAISS dense index for top_k closest chunks."""
        if self.index is None:
            if not self.load():
                return []

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        scores, indices = self.index.search(query_vector.astype(np.float32), min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta_item = dict(self.metadata[idx])
            meta_item["similarity_score"] = float(score)
            meta_item["dense_score"] = float(score)
            results.append(meta_item)

        return results

    def search_bm25(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Searches BM25 index for top_k keyword matching chunks."""
        if self.bm25 is None:
            if not self.load() or self.bm25 is None:
                return []

        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]

        results = []
        max_bm25 = float(np.max(doc_scores)) if len(doc_scores) > 0 and np.max(doc_scores) > 0 else 1.0

        for idx in top_indices:
            score = float(doc_scores[idx])
            if score <= 0.0:
                continue
            meta_item = dict(self.metadata[idx])
            meta_item["bm25_raw_score"] = score
            meta_item["bm25_score"] = score / max_bm25
            results.append(meta_item)

        return results

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> List[Dict[str, Any]]:
        """Dense search wrapper for backward compatibility."""
        return self.search_dense(query_vector=query_vector, top_k=top_k)
