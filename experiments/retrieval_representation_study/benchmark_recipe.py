#!/usr/bin/env python3
"""Repeatable Research Benchmark Recipe (9 Queries: 3 Singularity, 3 Multi-Match, 3 Zero-Match).

Investigates Representation, Tagging, and Search/Retrieval in Personalised EURLEX Search,
evaluating:
  - Case 1: Singularity Match (Q1-Q3: exactly 1 narrow target document)
  - Case 2: Multi-Match (Q4-Q6: 3 or more ground-truth documents in a thematic/regulatory regime)
  - Case 3: Zero-Match (Q7-Q9: out-of-corpus / non-existent EU regulation)
Evaluated end-to-end with mistral-small-2603.
"""

import json
import logging
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.searcher import SearchEngine
from src.summariser import LegalSummariser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_recipe")

OUTPUT_DIR = Path(__file__).resolve().parent / "intermediate_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 9 Curated Evaluation Queries across the 3 Search Cases
BENCHMARK_QUERIES = [
    # --- CASE 1: SINGULARITY MATCH (Exact narrow act / specific dispute) ---
    {
        "id": "Q1",
        "case_type": "singularity_match",
        "type": "positive",
        "query": "470-790 MHz frequency band",
        "description": "Decision on frequency band allocation for wireless broadband",
        "ground_truth_celex": ["32017D0899"],
        "expected_sector": "Telecommunications",
        "expected_topic": "Digital Services",
    },
    {
        "id": "Q2",
        "case_type": "singularity_match",
        "type": "positive",
        "query": "Brussels IRIS hospitals State aid",
        "description": "Commission Decision on public deficit financing for Brussels hospitals",
        "ground_truth_celex": ["32016D2327"],
        "expected_sector": "Healthcare and Pharma",
        "expected_topic": "Competition and Antitrust",
    },
    {
        "id": "Q3",
        "case_type": "singularity_match",
        "type": "positive",
        "query": "Morocco sustainable fisheries agreement",
        "description": "Council Decision on Sustainable Fisheries Partnership Agreement",
        "ground_truth_celex": ["32018D2068"],
        "expected_sector": "Agriculture and Food",
        "expected_topic": "Trade and Customs",
    },

    # --- CASE 2: MULTI-MATCH (Regulatory regimes / >= 3 ground truth acts) ---
    {
        "id": "Q4",
        "case_type": "multi_match",
        "type": "positive",
        "query": "aviation civil safety agreement United States",
        "description": "Bilateral EU-US aviation safety agreements and amending decisions",
        "ground_truth_celex": [
            "32019D0572",  # 2019 amendment conclusion
            "32018D0061",  # 2017 amendment signing
            "32011D0719",  # 2011 original agreement conclusion
        ],
        "expected_sector": "Transport",
        "expected_topic": "Trade and Customs",
    },
    {
        "id": "Q5",
        "case_type": "multi_match",
        "type": "positive",
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
        "expected_sector": "Transport",
        "expected_topic": "Environmental Law",
    },
    {
        "id": "Q6",
        "case_type": "multi_match",
        "type": "positive",
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
        "expected_sector": "Defence and Security",
        "expected_topic": "Cross-border",
    },

    # --- CASE 3: ZERO-MATCH (Out-of-corpus / non-existent EU regulation) ---
    {
        "id": "Q7",
        "case_type": "zero_match",
        "type": "negative",
        "query": "quantum computing export controls",
        "description": "Out-of-corpus emerging technology; 0 EU acts in 1952-2019 dataset",
        "ground_truth_celex": [],
        "expected_sector": None,
        "expected_topic": None,
    },
    {
        "id": "Q8",
        "case_type": "zero_match",
        "type": "negative",
        "query": "California consumer privacy act CCPA",
        "description": "Non-EU US state law; out-of-jurisdiction",
        "ground_truth_celex": [],
        "expected_sector": None,
        "expected_topic": None,
    },
    {
        "id": "Q9",
        "case_type": "zero_match",
        "type": "negative",
        "query": "deepsea asteroid mining licensing",
        "description": "Fictional / non-existent EU regulation",
        "ground_truth_celex": [],
        "expected_sector": None,
        "expected_topic": None,
    },
]


def run_benchmark(
    model_name: str = "mistral-small-2603",
    max_tokens: int = 500,
    top_k_chunks: int = 20,
    top_n_docs: int = 5,
) -> Dict[str, Any]:
    """Runs the full research study benchmark across all 3 cases (9 queries)."""
    logger.info("Initializing SearchEngine and LegalSummariser...")
    searcher = SearchEngine()
    summariser = LegalSummariser()

    results_raw = []
    evaluations = []

    for q_meta in BENCHMARK_QUERIES:
        q_id = q_meta["id"]
        q_type = q_meta["type"]
        case_type = q_meta["case_type"]
        query_text = q_meta["query"]
        ground_truth = q_meta["ground_truth_celex"]

        logger.info(f"[{q_id} - {case_type.upper()}] Executing query: '{query_text}'")

        # 1. Query Transformation
        transformed = searcher.transform_query(query_text)
        semantic_query = transformed["semantic_query"]
        lexical_query = transformed["lexical_query"]
        lexical_terms = [t for t in lexical_query.split() if len(t) >= 3]

        # 2. FAISS Retrieval
        q_vec = searcher.embedder.encode_query(semantic_query)
        raw_chunks = searcher.indexer.search(q_vec, top_k=top_k_chunks)

        # 3. Lexical Scoring & Hybrid Score Calculation
        chunk_analysis = []
        for rc in raw_chunks:
            chunk_text = rc.get("text", "")
            faiss_score = float(rc.get("similarity_score", 0.0))
            lex_score = float(searcher._calculate_lexical_score(lexical_terms, chunk_text))
            hybrid_score = (0.75 * faiss_score) + (0.25 * lex_score)
            chunk_analysis.append({
                "chunk_id": rc.get("chunk_id"),
                "doc_id": rc.get("doc_id"),
                "celex": rc.get("celex"),
                "title": rc.get("title"),
                "faiss_score": round(faiss_score, 4),
                "lex_score": round(lex_score, 4),
                "hybrid_score": round(hybrid_score, 4),
                "text_snippet": chunk_text[:250],
            })

        # 4. Document Assembly and Top N Retrieval
        retrieved_docs = searcher.search(
            query=query_text,
            top_k_chunks=top_k_chunks,
            top_n_docs=top_n_docs,
            enable_memory=False,
        )

        # 5. Summarisation Excerpt Preparation
        catalog_text, excerpts_text, total_docs_included = summariser.build_catalog_and_excerpts(
            retrieved_docs, max_docs=top_n_docs
        )

        # 6. Mistral-small-2603 Generation
        t0 = time.time()
        sum_result = summariser.summarise(
            query=query_text,
            retrieved_docs=retrieved_docs,
            model_name=model_name,
            max_docs=top_n_docs,
            max_tokens=max_tokens,
        )
        elapsed_sec = round(time.time() - t0, 3)

        # 7. Evaluate Performance
        retrieved_celexes = [d["celex"] for d in retrieved_docs]
        retrieved_scores = [d["peak_score"] for d in retrieved_docs]

        # Ground truth evaluation
        gt_found_count = 0
        gt_first_rank = None
        if ground_truth:
            for rank_idx, c in enumerate(retrieved_celexes, 1):
                if c in ground_truth:
                    gt_found_count += 1
                    if gt_first_rank is None:
                        gt_first_rank = rank_idx

        gt_found = (gt_found_count > 0)
        p_at_1 = 1.0 if (ground_truth and len(retrieved_celexes) > 0 and retrieved_celexes[0] in ground_truth) else (1.0 if not ground_truth and len(retrieved_celexes) == 0 else 0.0)
        p_at_3 = (sum(1 for c in retrieved_celexes[:3] if c in ground_truth) / 3.0) if ground_truth else 0.0
        p_at_5 = (sum(1 for c in retrieved_celexes[:5] if c in ground_truth) / 5.0) if ground_truth else 0.0
        mrr = (1.0 / gt_first_rank) if (gt_found and gt_first_rank) else 0.0
        gt_coverage = (gt_found_count / len(ground_truth)) if ground_truth else 0.0

        # Tag utilization and accuracy evaluation
        top_doc_tags = retrieved_docs[0]["tags"] if retrieved_docs else {}
        expected_sec = q_meta.get("expected_sector")
        expected_top = q_meta.get("expected_topic")
        
        has_expected_sector = False
        has_expected_topic = False
        if top_doc_tags:
            sec_tags = [t["category"] for t in top_doc_tags.get("Industry / Sector", [])]
            top_tags = [t["category"] for t in top_doc_tags.get("Regulatory Topics", [])]
            has_expected_sector = expected_sec in sec_tags if expected_sec else False
            has_expected_topic = expected_top in top_tags if expected_top else False

        # Negative query false positive check
        false_positive_returned = (q_type == "negative" and len(retrieved_docs) > 0)

        eval_item = {
            "query_id": q_id,
            "case_type": case_type,
            "type": q_type,
            "query": query_text,
            "top_celex": retrieved_celexes[0] if retrieved_celexes else None,
            "top_score": retrieved_scores[0] if retrieved_scores else 0.0,
            "ground_truth_celex": ground_truth,
            "ground_truth_total": len(ground_truth),
            "ground_truth_found_in_top5": gt_found_count,
            "ground_truth_coverage": round(gt_coverage, 3),
            "ground_truth_first_rank": gt_first_rank,
            "precision_at_1": p_at_1,
            "precision_at_3": round(p_at_3, 3),
            "precision_at_5": round(p_at_5, 3),
            "reciprocal_rank": round(mrr, 3),
            "false_positive_returned": false_positive_returned,
            "tag_information_effective_in_retrieval": False,  # 0% weight in retrieval pipeline
            "top_doc_matches_expected_sector": has_expected_sector,
            "top_doc_matches_expected_topic": has_expected_topic,
            "top_doc_tags": top_doc_tags,
            "summariser_latency_sec": elapsed_sec,
            "summariser_status": sum_result.get("status"),
            "tokens_consumed": sum_result.get("usage", {}).get("total_tokens", 0),
        }
        evaluations.append(eval_item)

        results_raw.append({
            "meta": q_meta,
            "transformed_query": transformed,
            "raw_chunks_top5": chunk_analysis[:5],
            "retrieved_documents": [
                {
                    "rank": idx + 1,
                    "celex": d["celex"],
                    "title": d["title"],
                    "date": d["date"],
                    "type": d["type"],
                    "peak_score": d["peak_score"],
                    "avg_score": d["avg_score"],
                    "is_ground_truth": d["celex"] in ground_truth,
                    "tags": d["tags"],
                    "excerpt_length": len(d["best_chunk"]),
                    "excerpt_preview": d["best_chunk"][:300],
                }
                for idx, d in enumerate(retrieved_docs)
            ],
            "catalog_passed_to_llm": catalog_text,
            "summariser_result": sum_result,
            "latency_sec": elapsed_sec,
            "evaluation": eval_item,
        })

    # Summary Metrics across the 3 Cases
    singularity_evals = [e for e in evaluations if e["case_type"] == "singularity_match"]
    multi_evals = [e for e in evaluations if e["case_type"] == "multi_match"]
    zero_evals = [e for e in evaluations if e["case_type"] == "zero_match"]
    pos_evals = [e for e in evaluations if e["type"] == "positive"]

    overall_metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_tested": model_name,
        "total_queries": len(evaluations),
        "case_breakdown": {
            "singularity_match_count": len(singularity_evals),
            "multi_match_count": len(multi_evals),
            "zero_match_count": len(zero_evals),
        },
        # Case 1: Singularity Matches
        "case1_singularity_mrr": round(float(np.mean([e["reciprocal_rank"] for e in singularity_evals])), 3),
        "case1_singularity_p_at_1": round(float(np.mean([e["precision_at_1"] for e in singularity_evals])), 3),
        "case1_singularity_mean_score": round(float(np.mean([e["top_score"] for e in singularity_evals])), 3),

        # Case 2: Multi-Matches (>= 3 ground truth acts)
        "case2_multi_mrr": round(float(np.mean([e["reciprocal_rank"] for e in multi_evals])), 3),
        "case2_multi_p_at_1": round(float(np.mean([e["precision_at_1"] for e in multi_evals])), 3),
        "case2_multi_p_at_3": round(float(np.mean([e["precision_at_3"] for e in multi_evals])), 3),
        "case2_multi_p_at_5": round(float(np.mean([e["precision_at_5"] for e in multi_evals])), 3),
        "case2_multi_mean_gt_recovered_in_top5": round(float(np.mean([e["ground_truth_found_in_top5"] for e in multi_evals])), 2),
        "case2_multi_mean_score": round(float(np.mean([e["top_score"] for e in multi_evals])), 3),

        # Case 3: Zero-Matches
        "case3_zero_false_positive_rate": round(float(sum(1 for e in zero_evals if e["false_positive_returned"]) / len(zero_evals)), 3),
        "case3_zero_mean_score": round(float(np.mean([e["top_score"] for e in zero_evals])), 3),

        # Systemic Metrics
        "overall_positive_precision_at_1": round(float(np.mean([e["precision_at_1"] for e in pos_evals])), 3),
        "overall_positive_mrr": round(float(np.mean([e["reciprocal_rank"] for e in pos_evals])), 3),
        "tag_utilization_in_retrieval_percentage": 0.0,
        "total_tokens_consumed": sum(e["tokens_consumed"] for e in evaluations),
    }

    full_output = {
        "overall_metrics": overall_metrics,
        "evaluations": evaluations,
        "raw_results": results_raw,
    }

    # Save outputs
    raw_path = OUTPUT_DIR / "benchmark_results_raw.json"
    eval_path = OUTPUT_DIR / "query_evaluations.json"
    metrics_path = OUTPUT_DIR / "benchmark_metrics.json"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluations, f, indent=2, ensure_ascii=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)

    logger.info(f"Benchmark completed successfully! Saved results to {OUTPUT_DIR}")
    print("\n" + "=" * 65)
    print("           BENCHMARK OVERALL SUMMARY (3 CASES / 9 QUERIES)")
    print("=" * 65)
    for k, v in overall_metrics.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sub_k, sub_v in v.items():
                print(f"    {sub_k:38s}: {sub_v}")
        else:
            print(f"  {k:42s}: {v}")
    print("=" * 65)

    return full_output


if __name__ == "__main__":
    run_benchmark()
