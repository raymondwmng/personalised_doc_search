"""Hardware Profiling Suite for RTX A2000 (8GB VRAM) (Section 3.2.2 Point 3).

Measures:
  1. Peak GPU VRAM allocated during batch encoding across batch sizes (4, 8, 16, 32).
  2. Batch inference throughput (docs/sec and tokens/sec).
  3. FAISS IndexFlatIP build time, RAM footprint, and disk size across dimensions (512, 768, 1024).
  4. End-to-end query latency (query embedding + FAISS inner-product retrieval).
"""

import gc
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import faiss
import numpy as np
import torch

from .model_wrapper import CANDIDATE_MODEL_SPECS, LongContextEmbeddingAdapter

logger = logging.getLogger(__name__)


class HardwareProfiler:
    """Profiles GPU memory, inference throughput, FAISS indexing, and latency on RTX A2000."""

    def __init__(self, device: str = "cuda"):
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    def profile_batch_scaling(
        self,
        model_key: str,
        sample_texts: List[str],
        batch_sizes: List[int] = [4, 8, 16, 32],
    ) -> List[Dict[str, Any]]:
        """Profiles GPU VRAM and docs/sec throughput across batch sizes."""
        logger.info(f"Profiling batch scaling for '{model_key}' across batches {batch_sizes}...")
        adapter = LongContextEmbeddingAdapter(model_key, device=self.device)

        # Warm up GPU
        _ = adapter.encode_documents(sample_texts[:min(4, len(sample_texts))], batch_size=4)

        results = []
        for bs in batch_sizes:
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.time()
            try:
                embs, prof = adapter.encode_documents(sample_texts, batch_size=bs)
                elapsed = time.time() - t0
                vram_mb = prof.get("peak_vram_mb", 0.0)
                throughput = len(sample_texts) / max(elapsed, 1e-4)
                status = "OK"
                logger.info(f"  [{model_key}] Batch {bs}: {vram_mb:.1f} MB VRAM, {throughput:.1f} docs/sec")
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                logger.warning(f"  [{model_key}] Batch {bs} exceeded 8GB VRAM (OOM): {e}")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                vram_mb = 8192.0
                throughput = 0.0
                elapsed = 0.0
                status = "OOM (> 8.0 GB)"

            results.append({
                "model_key": model_key,
                "batch_size": bs,
                "status": status,
                "peak_vram_mb": round(vram_mb, 1),
                "peak_vram_gb": round(vram_mb / 1024.0, 3) if status == "OK" else 8.192,
                "throughput_docs_per_sec": round(throughput, 1),
                "elapsed_sec": round(elapsed, 3),
                "sample_count": len(sample_texts),
            })
            if status != "OK":
                logger.info(f"  [{model_key}] Skipping larger batch sizes due to OOM.")
                break

        # Release model memory
        del adapter
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        return results

    def profile_faiss_dimensions(
        self,
        num_vectors: int = 5000,
        dims: List[int] = [512, 768, 1024],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Profiles FAISS IndexFlatIP index build time, RAM footprint, and search latency across dimensions."""
        logger.info(f"Profiling FAISS index across dimensions {dims} with {num_vectors} vectors...")
        results = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            for d in dims:
                # Generate synthetic normalized vectors
                np.random.seed(42)
                vectors = np.random.randn(num_vectors, d).astype(np.float32)
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / np.maximum(norms, 1e-12)

                # 1. Build FAISS IndexFlatIP
                t0 = time.time()
                index = faiss.IndexFlatIP(d)
                index.add(vectors)
                build_time_ms = (time.time() - t0) * 1000.0

                # 2. Measure Disk Footprint
                index_file = tmp_path / f"faiss_flat_ip_{d}.index"
                faiss.write_index(index, str(index_file))
                disk_mb = index_file.stat().st_size / (1024 * 1024)

                # 3. Measure RAM Footprint (approx bytes: num_vectors * d * 4 bytes float32)
                ram_mb = (num_vectors * d * 4) / (1024 * 1024)

                # 4. Measure Query Latency
                query_vec = vectors[:1]
                t_q0 = time.time()
                num_queries = 100
                for _ in range(num_queries):
                    _, _ = index.search(query_vec, top_k)
                avg_search_latency_ms = ((time.time() - t_q0) / num_queries) * 1000.0

                logger.info(
                    f"  [FAISS Dim {d}] Build: {build_time_ms:.1f}ms, Disk: {disk_mb:.2f}MB, "
                    f"Search Latency: {avg_search_latency_ms:.3f}ms"
                )

                results.append({
                    "dimension": d,
                    "num_vectors": num_vectors,
                    "build_time_ms": round(build_time_ms, 2),
                    "disk_mb": round(disk_mb, 2),
                    "ram_mb": round(ram_mb, 2),
                    "avg_search_latency_ms": round(avg_search_latency_ms, 4),
                })

        return results

    def profile_end_to_end_query_latency(
        self,
        model_key: str,
        queries: List[str],
        faiss_index: faiss.Index,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Measures embedding generation time, FAISS search time, and total latency per query."""
        adapter = LongContextEmbeddingAdapter(model_key, device=self.device)

        latencies = []
        embed_times = []
        search_times = []

        for q in queries:
            # Measure query embedding
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            q_vec = adapter.encode_queries([q])
            if self.device == "cuda":
                torch.cuda.synchronize()
            t_embed = time.time() - t0

            # Measure FAISS retrieval
            t1 = time.time()
            _, _ = faiss_index.search(q_vec, top_k)
            t_search = time.time() - t1

            total_ms = (t_embed + t_search) * 1000.0
            latencies.append(total_ms)
            embed_times.append(t_embed * 1000.0)
            search_times.append(t_search * 1000.0)

        del adapter
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return {
            "model_key": model_key,
            "mean_total_latency_ms": round(float(np.mean(latencies)), 2),
            "p95_total_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "mean_embed_latency_ms": round(float(np.mean(embed_times)), 2),
            "mean_search_latency_ms": round(float(np.mean(search_times)), 2),
        }
