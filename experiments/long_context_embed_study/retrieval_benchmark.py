"""Retrieval Benchmark Suite Across 3 Legal Archetypes (Section 3.2.2 Point 2).

Evaluates 5 candidate embedding backbones across 3 hierarchical chunking regimes:
  - Case 1: Singularity Match (Q1–Q3: P@1, MRR, Rank 1 vs Rank 2 margin)
  - Case 2: Multi-Match (Q4–Q6: P@3, P@5, GT count in Top 5)
  - Case 3: Zero-Match Rejection (Q7–Q9: Peak score, margin below 0.40 threshold)
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np

logger = logging.getLogger(__name__)

BENCHMARK_QUERIES = [
    # --- CASE 1: SINGULARITY MATCH ---
    {
        "id": "Q1",
        "case_type": "singularity_match",
        "query": "470-790 MHz frequency band",
        "description": "Decision on frequency band allocation for wireless broadband",
        "ground_truth_celex": ["32017D0899"],
    },
    {
        "id": "Q2",
        "case_type": "singularity_match",
        "query": "Brussels IRIS hospitals State aid",
        "description": "Commission Decision on public deficit financing for Brussels hospitals",
        "ground_truth_celex": ["32016D2327"],
    },
    {
        "id": "Q3",
        "case_type": "singularity_match",
        "query": "Morocco sustainable fisheries agreement",
        "description": "Council Decision on Sustainable Fisheries Partnership Agreement",
        "ground_truth_celex": ["32018D2068"],
    },

    # --- CASE 2: MULTI-MATCH ---
    {
        "id": "Q4",
        "case_type": "multi_match",
        "query": "aviation civil safety agreement United States",
        "description": "Bilateral EU-US aviation safety agreements and amending decisions",
        "ground_truth_celex": [
            "32019D0572",  # 2019 amendment conclusion
            "32018D0061",  # 2017 amendment signing
            "32011D0719",  # 2011 original agreement conclusion
        ],
    },
    {
        "id": "Q5",
        "case_type": "multi_match",
        "query": "passenger cars CO2 emissions innovative technology",
        "description": "Decisions approving innovative technologies for passenger car CO2 reduction under Reg 443/2009",
        "ground_truth_celex": [
            "32019D1119",  # Efficient LED exterior lighting
            "32017D0785",  # Efficient 12V motor-generator
            "32015D1132",  # Porsche / Audi coasting function
            "32013D0128",  # Audi LED lighting
            "32013D0451",  # Valeo efficient alternator
            "32018D2079",  # Engine idle coasting
        ],
    },
    {
        "id": "Q6",
        "case_type": "multi_match",
        "query": "Schengen acquis participation provisions",
        "description": "Council Decisions on participation, requests, and putting into effect of Schengen acquis provisions",
        "ground_truth_celex": [
            "32012D0764",  # Ireland participation request
            "32010D0491",  # Liechtenstein agreement signing
            "32010D0490",  # Liechtenstein agreement conclusion
            "32012D0193",  # Swiss / Liechtenstein border controls
            "32010D0779",  # UK police cooperation request
            "32019D0304",  # UK immigration liaison notification
        ],
    },

    # --- CASE 3: ZERO-MATCH REJECTION ---
    {
        "id": "Q7",
        "case_type": "zero_match",
        "query": "quantum computing export controls",
        "description": "Out-of-corpus emerging technology; 0 EU acts in dataset",
        "ground_truth_celex": [],
    },
    {
        "id": "Q8",
        "case_type": "zero_match",
        "query": "California consumer privacy act CCPA",
        "description": "Non-EU US state law; out-of-jurisdiction",
        "ground_truth_celex": [],
    },
    {
        "id": "Q9",
        "case_type": "zero_match",
        "query": "deepsea asteroid mining licensing",
        "description": "Fictional / non-existent EU regulation",
        "ground_truth_celex": [],
    },
]


class RetrievalBenchmarkSuite:
    """Executes dense vector retrieval benchmarks over FAISS indices."""

    def __init__(self, null_threshold: float = 0.40):
        self.null_threshold = null_threshold

    def build_faiss_index(
        self,
        embeddings: np.ndarray,
        chunk_metadata: List[Dict[str, Any]],
    ) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
        """Builds an exact inner product FAISS index over normalized chunk embeddings."""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index, chunk_metadata

    def search_and_aggregate_documents(
        self,
        query_vec: np.ndarray,
        index: faiss.IndexFlatIP,
        metadata: List[Dict[str, Any]],
        top_k_chunks: int = 40,
        top_n_docs: int = 5,
    ) -> List[Dict[str, Any]]:
        """Searches FAISS and deduplicates chunks to return top substantive acts by parent CELEX."""
        scores, indices = index.search(query_vec, top_k_chunks)
        scores = scores[0]
        indices = indices[0]

        doc_map: Dict[str, Dict[str, Any]] = {}
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(metadata):
                continue
            meta = metadata[idx]
            celex = meta.get("celex", "")
            if not celex:
                continue

            score_flt = float(score)
            if celex not in doc_map or score_flt > doc_map[celex]["peak_score"]:
                doc_map[celex] = {
                    "celex": celex,
                    "title": meta.get("title", ""),
                    "peak_score": round(score_flt, 4),
                    "best_chunk": meta.get("text", ""),
                    "token_count": meta.get("token_count", 0),
                    "regime": meta.get("regime", ""),
                }

        # Sort documents descending by peak score
        sorted_docs = sorted(doc_map.values(), key=lambda d: d["peak_score"], reverse=True)
        return sorted_docs[:top_n_docs]

    def evaluate_configuration(
        self,
        model_key: str,
        regime: str,
        adapter: Any,
        index: faiss.IndexFlatIP,
        metadata: List[Dict[str, Any]],
        top_n_docs: int = 5,
    ) -> Dict[str, Any]:
        """Runs the 9-query benchmark on a specific (model, regime) configuration."""
        query_results = []

        for q in BENCHMARK_QUERIES:
            q_id = q["id"]
            case_type = q["case_type"]
            query_str = q["query"]
            gt = q["ground_truth_celex"]

            # Encode query
            t0 = time.time()
            q_vec = adapter.encode_queries([query_str])
            t_encode = time.time() - t0

            # Retrieve top acts
            t1 = time.time()
            retrieved_docs = self.search_and_aggregate_documents(
                query_vec=q_vec,
                index=index,
                metadata=metadata,
                top_k_chunks=40,
                top_n_docs=top_n_docs,
            )
            t_search = time.time() - t1

            retrieved_celexes = [d["celex"] for d in retrieved_docs]
            retrieved_scores = [d["peak_score"] for d in retrieved_docs]
            peak_score = retrieved_scores[0] if retrieved_scores else 0.0

            # Compute metrics
            gt_found_count = sum(1 for c in retrieved_celexes if c in gt) if gt else 0
            gt_first_rank = None
            if gt:
                for r_idx, c in enumerate(retrieved_celexes, 1):
                    if c in gt:
                        gt_first_rank = r_idx
                        break

            # Metrics
            p_at_1 = 1.0 if (gt and len(retrieved_celexes) > 0 and retrieved_celexes[0] in gt) else (
                1.0 if not gt and peak_score < self.null_threshold else 0.0
            )
            p_at_3 = (sum(1 for c in retrieved_celexes[:3] if c in gt) / 3.0) if gt else 0.0
            p_at_5 = (sum(1 for c in retrieved_celexes[:5] if c in gt) / 5.0) if gt else 0.0
            mrr = (1.0 / gt_first_rank) if gt_first_rank else 0.0

            # Margin metrics
            rank1_margin = (
                round(retrieved_scores[0] - retrieved_scores[1], 4)
                if len(retrieved_scores) >= 2
                else 0.0
            )
            threshold_margin = round(self.null_threshold - peak_score, 4)

            # Rejection correctness for Case 3
            properly_rejected = (peak_score < self.null_threshold) if not gt else None

            query_results.append({
                "query_id": q_id,
                "case_type": case_type,
                "query": query_str,
                "gt_total": len(gt),
                "gt_recovered_top5": gt_found_count,
                "top_celex": retrieved_celexes[0] if retrieved_celexes else None,
                "peak_score": peak_score,
                "scores_top5": retrieved_scores,
                "p_at_1": p_at_1,
                "p_at_3": p_at_3,
                "p_at_5": p_at_5,
                "mrr": mrr,
                "rank1_margin": rank1_margin,
                "threshold_margin": threshold_margin,
                "properly_rejected": properly_rejected,
                "latency_ms": round((t_encode + t_search) * 1000.0, 2),
                "retrieved_docs": retrieved_docs,
            })

        # Aggregation by Case Archetype
        case1_evals = [r for r in query_results if r["case_type"] == "singularity_match"]
        case2_evals = [r for r in query_results if r["case_type"] == "multi_match"]
        case3_evals = [r for r in query_results if r["case_type"] == "zero_match"]

        case1_p1 = float(np.mean([r["p_at_1"] for r in case1_evals]))
        case1_mrr = float(np.mean([r["mrr"] for r in case1_evals]))
        case1_margin = float(np.mean([r["rank1_margin"] for r in case1_evals]))
        case1_mean_score = float(np.mean([r["peak_score"] for r in case1_evals]))

        case2_p1 = float(np.mean([r["p_at_1"] for r in case2_evals]))
        case2_p3 = float(np.mean([r["p_at_3"] for r in case2_evals]))
        case2_p5 = float(np.mean([r["p_at_5"] for r in case2_evals]))
        case2_gt_rec = float(np.mean([r["gt_recovered_top5"] for r in case2_evals]))
        case2_mean_score = float(np.mean([r["peak_score"] for r in case2_evals]))

        case3_fpr = float(np.mean([0.0 if r["properly_rejected"] else 1.0 for r in case3_evals]))
        case3_mean_score = float(np.mean([r["peak_score"] for r in case3_evals]))
        case3_mean_separation = float(np.mean([r["threshold_margin"] for r in case3_evals]))

        overall_p1 = float(np.mean([r["p_at_1"] for r in query_results]))
        overall_mrr = float(np.mean([r["mrr"] for r in query_results if r["case_type"] != "zero_match"]))

        return {
            "model_key": model_key,
            "regime": regime,
            "overall_p1": round(overall_p1, 3),
            "overall_mrr": round(overall_mrr, 3),
            "case1_singularity": {
                "p_at_1": round(case1_p1, 3),
                "mrr": round(case1_mrr, 3),
                "mean_margin_rank1_over_rank2": round(case1_margin, 3),
                "mean_peak_score": round(case1_mean_score, 3),
            },
            "case2_multi_match": {
                "p_at_1": round(case2_p1, 3),
                "p_at_3": round(case2_p3, 3),
                "p_at_5": round(case2_p5, 3),
                "mean_gt_recovered_in_top5": round(case2_gt_rec, 2),
                "mean_peak_score": round(case2_mean_score, 3),
            },
            "case3_zero_match": {
                "false_positive_rate": round(case3_fpr, 3),
                "mean_peak_score": round(case3_mean_score, 3),
                "mean_separation_below_0_40": round(case3_mean_separation, 3),
            },
            "query_details": query_results,
        }
