"""Master Orchestrator for Long-Context Legal Retrieval & Hierarchical Chunking Study.

Executes Section 3.2.2 end-to-end:
  1. Corpus Slicing & Chunking across Micro, Meso, Macro regimes.
  2. Hardware & Resource Profiling on NVIDIA RTX A2000 8GB Laptop GPU.
  3. Retrieval Evaluation across 5 candidate models and 3 chunking regimes (9 queries, 3 cases).
  4. Downstream LLM Summarisation Fidelity evaluation with mistral-small-2603.
  5. Publication Figures Generation.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from experiments.long_context_embed_study.hardware_profiler import HardwareProfiler
from experiments.long_context_embed_study.hierarchical_chunker import (
    BENCHMARK_GT_CELEX,
    HierarchicalChunker,
    load_study_corpus,
)
from experiments.long_context_embed_study.model_wrapper import (
    CANDIDATE_MODEL_SPECS,
    LongContextEmbeddingAdapter,
)
from experiments.long_context_embed_study.retrieval_benchmark import (
    BENCHMARK_QUERIES,
    RetrievalBenchmarkSuite,
)
from experiments.long_context_embed_study.summarisation_evaluator import (
    SummarisationFidelityEvaluator,
)
from experiments.long_context_embed_study.visualize_study import generate_all_study_figures

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_study")

STUDY_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = STUDY_DIR / "intermediate_outputs"
FIGURES_DIR = STUDY_DIR / "figures"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_full_study(
    db_path: Path = config.DOCUMENTS_DB_PATH,
    num_background_docs: int = 150,
    test_models: Optional[List[str]] = None,
    eval_summarisation: bool = True,
) -> None:
    """Executes the complete empirical study protocol."""
    start_total_time = time.time()
    logger.info("=================================================================")
    logger.info("  STARTING LONG-CONTEXT EMBEDDING & HIERARCHICAL CHUNKING STUDY  ")
    logger.info("=================================================================")
    logger.info(f"Target DB: {db_path}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    models_to_test = test_models or list(CANDIDATE_MODEL_SPECS.keys())
    logger.info(f"Candidate Models ({len(models_to_test)}): {models_to_test}")

    # -------------------------------------------------------------
    # 1. Corpus Loading and Multi-Resolution Slicing
    # -------------------------------------------------------------
    logger.info("\n--- Phase 1: Corpus Slicing & Hierarchical Chunking ---")
    corpus_docs = load_study_corpus(db_path, num_background_docs=num_background_docs)
    logger.info(f"Total evaluation corpus: {len(corpus_docs)} documents.")

    chunker = HierarchicalChunker()
    regimes = ["micro", "meso", "macro"]
    chunked_corpora: Dict[str, List[Dict[str, Any]]] = {}

    for reg in regimes:
        reg_chunks = []
        for doc in corpus_docs:
            c_list = chunker.chunk_document_by_regime(
                text=doc.get("full_text", ""),
                celex=doc["celex"],
                title=doc["title"],
                regime=reg,
            )
            reg_chunks.extend(c_list)
        chunked_corpora[reg] = reg_chunks
        total_tokens = sum(c["token_count"] for c in reg_chunks)
        logger.info(
            f"Regime '{reg.upper()}': {len(reg_chunks)} chunks total | "
            f"Mean tokens/chunk: {total_tokens / max(len(reg_chunks), 1):.1f} | "
            f"Total tokens encoded: {total_tokens:,}"
        )

    # -------------------------------------------------------------
    # 2. Hardware and Resource Profiling
    # -------------------------------------------------------------
    logger.info("\n--- Phase 2: Hardware & Resource Profiling on RTX A2000 ---")
    hw_cache_path = OUTPUTS_DIR / "hardware_profiling.json"
    if hw_cache_path.exists():
        logger.info(f"Existing hardware profiling data found at {hw_cache_path}. Loading cached profile...")
        with open(hw_cache_path, "r") as f:
            cached_hw = json.load(f)
            hardware_scaling_results = cached_hw.get("batch_scaling", [])
            faiss_scaling_results = cached_hw.get("faiss_dimension_scaling", [])
    else:
        profiler = HardwareProfiler(device="cuda" if torch.cuda.is_available() else "cpu")

        # Sample texts for batch scaling (~400 tokens / 1500 chars to test 4, 8, 16, 32 without OOM)
        sample_texts = [c["text"][:1500] for c in chunked_corpora["micro"][:32]]
        if len(sample_texts) < 32:
            sample_texts = (sample_texts * (32 // max(1, len(sample_texts)) + 1))[:32]

        hardware_scaling_results = []
        for model_key in models_to_test:
            scaling_data = profiler.profile_batch_scaling(
                model_key=model_key,
                sample_texts=sample_texts,
                batch_sizes=[4, 8, 16, 32],
            )
            hardware_scaling_results.extend(scaling_data)

        # FAISS dimension scaling profile
        faiss_scaling_results = profiler.profile_faiss_dimensions(
            num_vectors=len(chunked_corpora["micro"]),
            dims=[512, 768, 1024],
            top_k=10,
        )

        with open(hw_cache_path, "w") as f:
            json.dump(
                {
                    "batch_scaling": hardware_scaling_results,
                    "faiss_dimension_scaling": faiss_scaling_results,
                },
                f,
                indent=2,
            )
        logger.info("Saved Phase 2 hardware profiling to hardware_profiling.json")

    # -------------------------------------------------------------
    # 3. Retrieval Evaluation Across Models & Chunking Regimes
    # -------------------------------------------------------------
    logger.info("\n--- Phase 3: Retrieval Evaluation (9 Queries x 3 Archetypes) ---")
    benchmark_suite = RetrievalBenchmarkSuite(null_threshold=0.40)
    all_configuration_metrics = []
    all_query_evaluations = []

    # Safe adaptive configurations per regime for 8GB VRAM
    regime_configs = {
        "micro": {"batch_size": 16, "max_length": 512},
        "meso": {"batch_size": 4, "max_length": 2048},
        "macro": {"batch_size": 2, "max_length": 8192},
    }

    for model_key in models_to_test:
        logger.info(f"\n========================================================")
        logger.info(f"  BENCHMARKING MODEL: {model_key}")
        logger.info(f"========================================================")

        adapter = LongContextEmbeddingAdapter(model_key, device="cuda" if torch.cuda.is_available() else "cpu")

        for reg in regimes:
            chunks = chunked_corpora[reg]
            texts = [c["text"] for c in chunks]
            cfg = regime_configs.get(reg, {"batch_size": 4, "max_length": 2048})
            b_size = cfg["batch_size"]
            m_len = cfg["max_length"]
            logger.info(f"Encoding {len(texts)} chunks for Model '{model_key}', Regime '{reg}' (batch={b_size}, max_len={m_len})...")

            # Encode chunks with adaptive batch size and sequence length
            embs, prof_info = adapter.encode_documents(texts, batch_size=b_size, max_length=m_len)

            # Build FAISS index
            index, metadata = benchmark_suite.build_faiss_index(embs, chunks)

            # Evaluate the 9 queries
            config_result = benchmark_suite.evaluate_configuration(
                model_key=model_key,
                regime=reg,
                adapter=adapter,
                index=index,
                metadata=metadata,
                top_n_docs=5,
            )

            # Attach encoding throughput and VRAM stats
            config_result["encoding_profile"] = prof_info
            all_configuration_metrics.append(config_result)

            logger.info(
                f"[{model_key} - {reg.upper()}] Case 1 P@1: {config_result['case1_singularity']['p_at_1']:.2f} | "
                f"Case 2 P@5: {config_result['case2_multi_match']['p_at_5']:.2f} (GT: {config_result['case2_multi_match']['mean_gt_recovered_in_top5']}/5) | "
                f"Case 3 Zero-Match Peak: {config_result['case3_zero_match']['mean_peak_score']:.3f} | "
                f"Overall MRR: {config_result['overall_mrr']:.3f}"
            )

            for q_eval in config_result["query_details"]:
                q_record = dict(q_eval)
                q_record["model_key"] = model_key
                q_record["regime"] = reg
                all_query_evaluations.append(q_record)

        # Release adapter from GPU
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    with open(OUTPUTS_DIR / "study_metrics.json", "w") as f:
        json.dump(all_configuration_metrics, f, indent=2)

    with open(OUTPUTS_DIR / "retrieval_evaluations.json", "w") as f:
        json.dump(all_query_evaluations, f, indent=2)
    logger.info("Saved Phase 3 retrieval results to study_metrics.json and retrieval_evaluations.json")

    # -------------------------------------------------------------
    # 4. Downstream LLM Summarisation Fidelity with mistral-small-2603
    # -------------------------------------------------------------
    logger.info("\n--- Phase 4: Downstream Summarisation Fidelity (mistral-small-2603) ---")
    summarisation_results = []
    if eval_summarisation and config.MISTRAL_API_KEY:
        evaluator = SummarisationFidelityEvaluator(model_name="mistral-small-2603")
        # Evaluate representative queries: Q1 (Singularity), Q5 (Multi-Match), Q8 (Zero-Match) across best Macro/Meso regime
        test_queries = [
            ("Q1", "singularity_match", "470-790 MHz frequency band"),
            ("Q5", "multi_match", "passenger cars CO2 emissions innovative technology"),
            ("Q8", "zero_match", "California consumer privacy act CCPA"),
        ]

        # Use UltraBERT or BGE-M3 Macro retrieved results
        target_model = "UltraBERT-Embed-150M" if "UltraBERT-Embed-150M" in models_to_test else models_to_test[0]
        for q_id, c_type, q_text in test_queries:
            matching_evals = [
                e for e in all_query_evaluations
                if e["model_key"] == target_model and e["regime"] == "macro" and e["query_id"] == q_id
            ]
            if matching_evals:
                docs = matching_evals[0].get("retrieved_docs", [])
                logger.info(f"Synthesizing {q_id} ({c_type}) with mistral-small-2603 using {len(docs)} acts...")
                s_result = evaluator.evaluate_query_synthesis(
                    query=q_text,
                    case_type=c_type,
                    retrieved_docs=docs,
                    regime="macro",
                    max_tokens=400,
                )
                summarisation_results.append(s_result)
                logger.info(
                    f"  [{q_id}] Status: {s_result.get('status')} | "
                    f"Refusal Correct: {s_result.get('refusal_correct')} | "
                    f"CELEX Citations: {s_result.get('celex_mentions_count')} | "
                    f"Latency: {s_result.get('latency_sec')}s"
                )

        with open(OUTPUTS_DIR / "summarisation_fidelity.json", "w") as f:
            json.dump(summarisation_results, f, indent=2)
        logger.info("Saved Phase 4 summarisation fidelity to summarisation_fidelity.json")
    else:
        logger.warning("Skipping Phase 4 summarisation evaluation (MISTRAL_API_KEY not found or disabled).")

    # -------------------------------------------------------------
    # 5. Visualizations
    # -------------------------------------------------------------
    logger.info("\n--- Phase 5: Generating Publication Figures ---")
    generate_all_study_figures(
        benchmark_metrics=all_configuration_metrics,
        hardware_data=hardware_scaling_results,
        figures_dir=FIGURES_DIR,
    )

    elapsed_total = round(time.time() - start_total_time, 2)
    logger.info("=================================================================")
    logger.info(f"  LONG-CONTEXT EMBEDDING STUDY COMPLETE IN {elapsed_total}s!      ")
    logger.info("=================================================================")


def main():
    parser = argparse.ArgumentParser(description="Run Long-Context Embedding Study (Section 3.2.2)")
    parser.add_argument("--db-path", type=str, default=str(config.DOCUMENTS_DB_PATH))
    parser.add_argument("--num-docs", type=int, default=150, help="Number of background documents")
    parser.add_argument("--skip-mistral", action="store_true", help="Skip live Mistral API calls")
    args = parser.parse_args()

    run_full_study(
        db_path=Path(args.db_path),
        num_background_docs=args.num_docs,
        eval_summarisation=(not args.skip_mistral),
    )


if __name__ == "__main__":
    main()
