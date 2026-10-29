"""Long-context embedding module supporting BAAI/bge-m3 and ModernBERT-base (Section 3.2.2 & Recommendations 1-3)."""

import logging
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch

# ---------------------------------------------------------
# Dynamic Compatibility Shims for Transformers 5.x / PyTorch
# ---------------------------------------------------------
if "transformers.onnx" not in sys.modules:
    onnx_mod = types.ModuleType("transformers.onnx")
    class OnnxConfig:
        pass
    onnx_mod.OnnxConfig = OnnxConfig
    sys.modules["transformers.onnx"] = onnx_mod

try:
    import transformers.pytorch_utils as pu
    if not hasattr(pu, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index
        pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
except Exception:
    pass

try:
    import transformers.configuration_utils
    _orig_getattr = getattr(transformers.configuration_utils.PretrainedConfig, "__getattr__", None)
    def _config_getattr(self, key):
        defaults = {
            "is_decoder": False,
            "add_cross_attention": False,
            "pruned_heads": set(),
            "chunk_size_feed_forward": 0,
        }
        if key in defaults:
            return defaults[key]
        if _orig_getattr:
            return _orig_getattr(self, key)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    transformers.configuration_utils.PretrainedConfig.__getattr__ = _config_getattr
except Exception:
    pass

try:
    import transformers.modeling_utils
    if not hasattr(transformers.modeling_utils.PreTrainedModel, "get_extended_attention_mask"):
        def get_extended_attention_mask(self, attention_mask, input_shape=None, device=None, dtype=None):
            if attention_mask.dim() == 3:
                extended_attention_mask = attention_mask[:, None, :, :]
            elif attention_mask.dim() == 2:
                extended_attention_mask = attention_mask[:, None, None, :]
            else:
                raise ValueError(f"Wrong shape for attention_mask {attention_mask.shape}")
            dt = dtype or getattr(self, "dtype", torch.float32)
            extended_attention_mask = extended_attention_mask.to(dtype=dt)
            return (1.0 - extended_attention_mask) * torch.finfo(dt).min
        transformers.modeling_utils.PreTrainedModel.get_extended_attention_mask = get_extended_attention_mask
except Exception:
    pass

from sentence_transformers import SentenceTransformer
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """Encodes texts into normalised dense vectors for cosine/inner-product retrieval.
    
    Supports:
      1. BAAI/bge-m3 (Primary Backbone, 1024d, 8K context, multilingual EU)
      2. ModernBERT-base (High-Throughput, 768d, 8K context, RoPE + FlashAttn-2)
    """

    def __init__(
        self,
        model_name_or_key: Optional[str] = None,
        device: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        self.spec = config.resolve_embedding_spec(model_name_or_key)
        self.model_key = self.spec.get("id", config.DEFAULT_EMBEDDING_MODEL)
        self.model_name = self.spec.get("name", config.EMBEDDING_MODEL_NAME)
        self.embedding_dim = self.spec.get("dim", config.EMBEDDING_DIM)
        self.max_seq_length = max_seq_length or self.spec.get("max_seq_length", 2048)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            f"Loading embedding model '{self.model_name}' (key: {self.model_key}) "
            f"on device '{self.device}' with max_seq_length={self.max_seq_length}..."
        )

        if self.device == "cuda":
            torch.cuda.empty_cache()

        self.model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = self.max_seq_length

        if hasattr(self.model, "get_embedding_dimension"):
            actual_dim = self.model.get_embedding_dimension()
        else:
            actual_dim = self.model.get_sentence_embedding_dimension()

        self.embedding_dim = actual_dim
        logger.info(f"Embedding model ready. Key: {self.model_key}, Dimension: {self.embedding_dim}")

    def encode(
        self,
        texts: List[str],
        batch_size: int = 16,
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
