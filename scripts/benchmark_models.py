"""Standardised Benchmark Suite for Local Ollama Models on EUR-Lex Legal Queries (Section 4.5)."""

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
import ollama
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_models")

BENCHMARK_LEGAL_QUERIES = [
    {
        "id": "Q1_AI_ACT",
        "title": "AI Act Deployer Obligations & Penalties",
        "query": "What are the core obligations, conformity assessments, and maximum administrative fines imposed on deployers of high-risk AI systems under the EU AI Act?",
    },
    {
        "id": "Q2_GDPR",
        "title": "GDPR Subject Rights & Cross-Border Transfers",
        "query": "Detail the statutory grounds for lawful processing, data subject rights to erasure, and legal mechanisms for third-country data transfers under Regulation (EU) 2016/679.",
    },
    {
        "id": "Q3_SPECTRUM",
        "title": "470-790 MHz Spectrum Coordination",
        "query": "What are the statutory deadlines and member state coordination obligations for the 470-790 MHz frequency band reallocated under Decision (EU) 2017/899?",
    },
    {
        "id": "Q4_CAR_EMISSIONS",
        "title": "Passenger Cars CO2 Eco-Innovations",
        "query": "Explain the regulatory framework for granting eco-innovation super-credits and CO2 emission reductions for passenger cars under Regulation (EC) No 443/2009.",
    },
    {
        "id": "Q5_STATE_AID",
        "title": "Hospital State Aid under Article 107/108 TFEU",
        "query": "Under what statutory conditions is public financing for hospital networks considered compatible State aid under Articles 107 and 108 TFEU?",
    },
]

CANDIDATE_OLLAMA_MODELS = [
    "gemma2:9b",
    "llama3.1:8b",
    "mistral:7b",
    "gemma3:4b",
    "llama3.2:3b",
]


def get_gpu_vram_mb() -> float:
    """Reads current GPU memory used in MB via torch or nvidia-smi."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


def run_benchmark(
    models: List[str] = CANDIDATE_OLLAMA_MODELS,
    context_windows: List[int] = [4096, 6144],
    output_path: Path = config.BENCHMARK_RESULTS_PATH,
) -> Dict[str, Any]:
    """Runs legal reasoning benchmarks across installed Ollama models and logs metrics."""
    client = ollama.Client()

    # Discover locally available Ollama models
    try:
        installed_list = client.list()
        # Handle both dict-based and object-based ollama responses
        if isinstance(installed_list, dict) and "models" in installed_list:
            available_names = [m.get("name", "").split(":")[0] for m in installed_list["models"]]
        elif hasattr(installed_list, "models"):
            available_names = [m.model.split(":")[0] for m in installed_list.models]
        else:
            available_names = []
    except Exception as e:
        logger.warning(f"Could not query Ollama list ({e}). Assuming standard installed models.")
        available_names = [m.split(":")[0] for m in models]

    logger.info(f"Ollama detected models: {available_names}")

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": "NVIDIA RTX A2000 8GB Laptop GPU",
        "models_evaluated": {},
    }

    for model in models:
        model_family = model.split(":")[0]
        logger.info(f"\n{'='*70}\nBenchmarking Model: {model}\n{'='*70}")

        model_results = {}
        for num_ctx in context_windows:
            logger.info(f"--- Context Window: {num_ctx} tokens ---")
            query_metrics = []

            for q in BENCHMARK_LEGAL_QUERIES:
                q_id = q["id"]
                prompt = (
                    "You are a European Union Senior Legal Counsel. "
                    "Analyze the following regulatory query and provide a structured synthesis "
                    "identifying: (1) Governing Legal Basis, (2) Operative Requirements, and (3) Enforcement Provisions.\n\n"
                    f"Legal Query: {q['query']}\n\n"
                    "Response:"
                )

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                t0 = time.perf_counter()
                try:
                    resp = client.generate(
                        model=model,
                        prompt=prompt,
                        options={
                            "num_ctx": num_ctx,
                            "temperature": 0.2,
                        },
                    )
                    latency = time.perf_counter() - t0
                    response_text = resp.get("response", "")
                    tokens_eval = resp.get("eval_count", len(response_text.split()))
                    tps = round(tokens_eval / max(0.01, latency), 2)
                    vram_mb = get_gpu_vram_mb()

                    logger.info(f"[{q_id}] Latency: {latency:.2f}s | Speed: {tps} tok/s | Output chars: {len(response_text)}")

                    query_metrics.append({
                        "query_id": q_id,
                        "latency_sec": round(latency, 3),
                        "tokens_generated": tokens_eval,
                        "tokens_per_sec": tps,
                        "vram_peak_mb": round(vram_mb, 1),
                        "output_char_count": len(response_text),
                        "has_statutory_citation": bool(any(term in response_text for term in ["Article", "Regulation", "Directive", "Decision", "TFEU"])),
                        "response_snippet": response_text[:200] + "...",
                    })
                except Exception as e:
                    logger.error(f"Error evaluating {model} on {q_id}: {e}")
                    query_metrics.append({
                        "query_id": q_id,
                        "error": str(e),
                        "latency_sec": None,
                    })

            valid = [m for m in query_metrics if "latency_sec" in m and m["latency_sec"] is not None]
            avg_latency = sum(m["latency_sec"] for m in valid) / max(1, len(valid))
            avg_tps = sum(m["tokens_per_sec"] for m in valid) / max(1, len(valid))

            model_results[f"ctx_{num_ctx}"] = {
                "num_ctx": num_ctx,
                "avg_latency_sec": round(avg_latency, 2),
                "avg_tokens_per_sec": round(avg_tps, 2),
                "queries": query_metrics,
            }

        results["models_evaluated"][model] = model_results

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nAll benchmark runs complete. Saved report to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark installed Ollama models on EUR-Lex legal queries.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=CANDIDATE_OLLAMA_MODELS,
        help="List of Ollama models to benchmark",
    )
    parser.add_argument(
        "--num-ctx",
        nargs="+",
        type=int,
        default=[4096, 6144],
        help="Context window sizes to test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.BENCHMARK_RESULTS_PATH),
        help="Target output JSON path",
    )
    args = parser.parse_args()

    run_benchmark(models=args.models, context_windows=args.num_ctx, output_path=Path(args.output))


if __name__ == "__main__":
    main()

