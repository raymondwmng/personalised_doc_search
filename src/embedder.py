"""Embedding module using sentence-transformers (GPU-accelerated when available)."""

import logging
from pathlib import Path
from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """Encodes texts into normalised dense vectors for cosine/inner-product retrieval."""

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        device: Union[str, None] = None,
    ):
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model ready. Dimension: {self.embedding_dim}")

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """Encodes a list of strings into L2-normalised float32 numpy vectors."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes a single query string into a 1xD normalized float32 vector."""
        vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec.astype(np.float32)
