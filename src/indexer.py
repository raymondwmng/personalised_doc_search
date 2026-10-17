"""Indexer module using FAISS IndexFlatIP for vector search."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FaissIndexer:
    """Manages FAISS inner product index and associated chunk metadata."""

    def __init__(
        self,
        index_path: Path = config.FAISS_INDEX_PATH,
        meta_path: Path = config.CHUNK_METADATA_PATH,
        dimension: int = config.EMBEDDING_DIM,
    ):
        self.index_path = index_path
        self.meta_path = meta_path
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []

    def build_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Constructs an IndexFlatIP index and sets metadata."""
        if embeddings.shape[0] != len(metadata):
            raise ValueError(f"Mismatch: {embeddings.shape[0]} embeddings vs {len(metadata)} metadata items.")

        dim = embeddings.shape[1]
        self.dimension = dim
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = metadata
        logger.info(f"Built FAISS IndexFlatIP with {self.index.ntotal} vectors of dimension {dim}.")

    def save(self) -> None:
        """Saves FAISS index binary and metadata JSONL."""
        if self.index is None:
            raise ValueError("No index to save.")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        with open(self.meta_path, "w", encoding="utf-8") as f:
            for item in self.metadata:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Saved FAISS index to {self.index_path} and metadata to {self.meta_path}.")

    def load(self) -> bool:
        """Loads FAISS index and metadata if files exist."""
        if not self.index_path.exists() or not self.meta_path.exists():
            logger.warning("Index or metadata file does not exist on disk.")
            return False

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))

        logger.info(f"Loaded FAISS index with {self.index.ntotal} items from {self.index_path}.")
        return True

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> List[Dict[str, Any]]:
        """Searches index for top_k closest chunks.
        
        Returns:
            List of dicts containing score, chunk_id, doc_id, chunk_idx, text, etc.
        """
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
            results.append(meta_item)

        return results
