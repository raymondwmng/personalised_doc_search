"""Unified Model Adapter for Long-Context Legal Retrieval Study (Section 3.2.2).

Supports 5 candidate backbones:
  1. ModernBERT-base (nomic-ai/modernbert-embed-base) - 8K context, 768 dim
  2. nomic-embed-text-v1.5 (nomic-ai/nomic-embed-text-v1.5) - 8K context, 768 dim / MRL 512
  3. bge-m3 (BAAI/bge-m3) - 8K context, 1024 dim
  4. jina-embeddings-v2-base-en (jinaai/jina-embeddings-v2-base-en) - 8K context, 768 dim, ALiBi
  5. UltraBERT-Embed-150M (dmis-lab/UltraBERT-Embed-150M) - 32K context, 768 dim
"""

import logging
import sys
import time
import types
from typing import Any, Dict, List, Optional, Tuple, Union
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

if not hasattr(transformers.modeling_utils.PreTrainedModel, "get_head_mask"):
    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            if hasattr(self, "_convert_head_mask_to_5d"):
                head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask
    transformers.modeling_utils.PreTrainedModel.get_head_mask = get_head_mask

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

CANDIDATE_MODEL_SPECS = {
    "ModernBERT-base": {
        "model_id": "nomic-ai/modernbert-embed-base",
        "context_window": 8192,
        "params": "149M",
        "dim": 768,
        "query_prefix": "",
        "doc_prefix": "",
        "loader_type": "sentence_transformers",
        "extra_kwargs": {},
    },
    "nomic-embed-text-v1.5": {
        "model_id": "nomic-ai/nomic-embed-text-v1.5",
        "context_window": 8192,
        "params": "137M",
        "dim": 768,
        "query_prefix": "search_query: ",
        "doc_prefix": "search_document: ",
        "loader_type": "sentence_transformers",
        "extra_kwargs": {"trust_remote_code": True},
    },
    "bge-m3": {
        "model_id": "BAAI/bge-m3",
        "context_window": 8192,
        "params": "567M",
        "dim": 1024,
        "query_prefix": "",
        "doc_prefix": "",
        "loader_type": "sentence_transformers",
        "extra_kwargs": {},
    },
    "jina-embeddings-v2-base-en": {
        "model_id": "jinaai/jina-embeddings-v2-base-en",
        "context_window": 8192,
        "params": "137M",
        "dim": 768,
        "query_prefix": "",
        "doc_prefix": "",
        "loader_type": "sentence_transformers",
        "extra_kwargs": {"trust_remote_code": True},
    },
    "UltraBERT-Embed-150M": {
        "model_id": "dmis-lab/UltraBERT-Embed-150M",
        "context_window": 32768,
        "params": "150M",
        "dim": 768,
        "query_prefix": "Instruct: Retrieve relevant EU legal legislation\nQuery: ",
        "doc_prefix": "",
        "loader_type": "ultrabert",
        "extra_kwargs": {},
    },
}


class LongContextEmbeddingAdapter:
    """Unified wrapper around all candidate models providing batch encoding, prefix formatting, and memory tracking."""

    def __init__(
        self,
        model_key: str,
        device: Optional[str] = None,
        matryoshka_dim: Optional[int] = None,
    ):
        if model_key not in CANDIDATE_MODEL_SPECS:
            raise ValueError(f"Unknown model_key '{model_key}'. Expected one of {list(CANDIDATE_MODEL_SPECS.keys())}")

        self.model_key = model_key
        self.spec = CANDIDATE_MODEL_SPECS[model_key]
        self.model_id = self.spec["model_id"]
        self.context_window = self.spec["context_window"]
        self.raw_dim = self.spec["dim"]
        self.matryoshka_dim = matryoshka_dim
        self.embedding_dim = matryoshka_dim if matryoshka_dim else self.raw_dim
        self.query_prefix = self.spec["query_prefix"]
        self.doc_prefix = self.spec["doc_prefix"]
        self.loader_type = self.spec["loader_type"]

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing adapter for '{model_key}' on device: {self.device}")

        self._load_model()

    def _load_model(self) -> None:
        """Loads model weights onto the target device."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        if self.loader_type == "sentence_transformers":
            kwargs = dict(self.spec["extra_kwargs"])
            self.model = SentenceTransformer(self.model_id, device=self.device, **kwargs)
            if hasattr(self.model, "max_seq_length"):
                self.model.max_seq_length = self.context_window
        elif self.loader_type == "ultrabert":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, extra_special_tokens={}, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = self.model.to(self.device).eval()

    def encode_documents(
        self,
        texts: List[str],
        batch_size: int = 16,
        max_length: Optional[int] = None,
        show_progress_bar: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Encodes document passages into L2-normalised float32 numpy vectors.
        
        Returns:
            (embeddings_array, profiling_metrics_dict)
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32), {}

        # Prepend task doc prefix if required
        formatted = [f"{self.doc_prefix}{t}" for t in texts] if self.doc_prefix else texts

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.time()

        if self.loader_type == "sentence_transformers":
            if max_length and hasattr(self.model, "max_seq_length"):
                self.model.max_seq_length = max_length
            raw_embs = self.model.encode(
                formatted,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=(self.matryoshka_dim is None),
                convert_to_numpy=True,
            )
            if self.matryoshka_dim:
                # Truncate and re-normalize for MRL
                raw_embs = raw_embs[:, : self.matryoshka_dim]
                norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
                raw_embs = raw_embs / np.maximum(norms, 1e-12)
            embs = raw_embs.astype(np.float32)

        elif self.loader_type == "ultrabert":
            all_embs = []
            max_len = min(self.context_window, max_length) if max_length else min(self.context_window, 8192)
            for i in range(0, len(formatted), batch_size):
                batch_texts = formatted[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                # Mean pooling over attention mask
                mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask
                normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
                all_embs.append(normalized.cpu().numpy())

            embs = np.vstack(all_embs).astype(np.float32)

        if self.device == "cuda":
            torch.cuda.synchronize()
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_vram_mb = 0.0

        elapsed_sec = max(time.time() - t0, 1e-5)
        throughput = len(texts) / elapsed_sec

        profiling = {
            "elapsed_sec": round(elapsed_sec, 3),
            "throughput_docs_per_sec": round(throughput, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "peak_vram_gb": round(peak_vram_mb / 1024.0, 3),
        }
        return embs, profiling

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encodes search queries into L2-normalised float32 numpy vectors."""
        if not queries:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        formatted = [f"{self.query_prefix}{q}" for q in queries] if self.query_prefix else queries

        if self.loader_type == "sentence_transformers":
            raw_embs = self.model.encode(
                formatted,
                batch_size=len(queries),
                show_progress_bar=False,
                normalize_embeddings=(self.matryoshka_dim is None),
                convert_to_numpy=True,
            )
            if self.matryoshka_dim:
                raw_embs = raw_embs[:, : self.matryoshka_dim]
                norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
                raw_embs = raw_embs / np.maximum(norms, 1e-12)
            return raw_embs.astype(np.float32)

        elif self.loader_type == "ultrabert":
            inputs = self.tokenizer(
                formatted,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            return normalized.cpu().numpy().astype(np.float32)
