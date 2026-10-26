"""Downstream Summarisation Fidelity Evaluator with mistral-small-2603 (Section 3.2.2 Point 4 & Section 3.5).

Evaluates the Case-Adaptive prompt:
  1. Multi-Match Synthesis (Case 2): Unified thematic summary + comparative distinctions across acts.
  2. Singularity Analysis (Case 1): Deep dive into legal basis, key articles, operative terms.
  3. Graceful Refusal (Case 3): Explicit refusal when documents do not pertain to query.
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

logger = logging.getLogger(__name__)

CASE_ADAPTIVE_PROMPT_TEMPLATE = """You are an expert EU legal document analyst. Summarise the following EU legislation excerpts that are relevant to the user's query.

CRITICAL INSTRUCTIONS:
1. Multi-Match Synthesis (Case 2): When multiple distinct EU acts are provided (e.g. multiple implementing decisions or bilateral agreements), provide:
   (a) A unified thematic summary explaining the overarching legal regime.
   (b) A comparative breakdown highlighting the specific scope, dates, and technical distinctions of each individual act (e.g. distinct technologies approved, or differing country opt-in terms).
2. Singularity Analysis (Case 1): If a single specific instrument is identified, provide a deep dive into its legal basis, key articles, and operative constraints.
3. Graceful Refusal (Case 3): If the retrieved documents do not directly pertain to the user's query, state clearly that no relevant EU legislation was identified in the retrieved set. Do not force an unrelated synthesis.

User query: {query}
Retrieved documents:
{context}
"""


class SummarisationFidelityEvaluator:
    """Evaluates downstream LLM synthesis fidelity across chunking regimes using mistral-small-2603."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "mistral-small-2603",
        timeout: float = 30.0,
    ):
        self.api_key = api_key or config.MISTRAL_API_KEY or os.getenv("MISTRAL_API_KEY", "")
        self.model_name = model_name
        self.timeout = timeout
        self.base_url = "https://api.mistral.ai/v1/chat/completions"

    def format_context(self, retrieved_docs: List[Dict[str, Any]], max_chars: int = 4000) -> str:
        """Budgets document snippets to fit within optimal context constraints."""
        if not retrieved_docs:
            return "[No documents retrieved]"

        blocks = []
        budget_per_doc = max(400, max_chars // len(retrieved_docs))
        for idx, doc in enumerate(retrieved_docs, 1):
            celex = doc.get("celex", "UNKNOWN")
            title = doc.get("title", "")
            raw_text = doc.get("best_chunk", "")
            snippet = raw_text[:budget_per_doc].strip()
            blocks.append(f"Document {idx} (CELEX: {celex})\nTitle: {title}\nExcerpt:\n{snippet}\n")

        return "\n---\n".join(blocks)

    def evaluate_query_synthesis(
        self,
        query: str,
        case_type: str,
        retrieved_docs: List[Dict[str, Any]],
        regime: str,
        max_tokens: int = 450,
    ) -> Dict[str, Any]:
        """Calls mistral-small-2603 with Case-Adaptive prompt and scores fidelity."""
        context_str = self.format_context(retrieved_docs)
        prompt = CASE_ADAPTIVE_PROMPT_TEMPLATE.format(query=query, context=context_str)

        if not self.api_key:
            return {
                "status": "skipped_no_api_key",
                "summary": "Mock summary: MISTRAL_API_KEY not configured.",
                "latency_sec": 0.0,
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,  # Strict legal precision
        }

        t0 = time.time()
        try:
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
            elapsed = time.time() - t0
            if resp.status_code != 200:
                logger.error(f"Mistral API returned error {resp.status_code}: {resp.text}")
                return {
                    "status": "api_error",
                    "status_code": resp.status_code,
                    "error": resp.text,
                    "latency_sec": round(elapsed, 3),
                }

            data = resp.json()
            summary = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            # Evaluation metrics on summary text
            retrieved_celexes = [d["celex"] for d in retrieved_docs if d.get("celex")]
            celex_mentions = [c for c in retrieved_celexes if c in summary]

            # Case 3 Graceful refusal check:
            refusal_patterns = [
                r"no relevant (?:eu\s+)?legislation",
                r"not pertain",
                r"does not contain",
                r"outside (?:the\s+)?scope",
                r"no documents.*identified",
            ]
            contains_refusal = any(re.search(pat, summary, re.IGNORECASE) for pat in refusal_patterns)
            refusal_correct = contains_refusal if case_type == "zero_match" else (not contains_refusal)

            # Case 2 Comparative synthesis check:
            has_comparative_structure = (
                len(celex_mentions) >= 2 or bool(re.search(r"compar|distinction|differ|respectively", summary, re.I))
            )

            return {
                "status": "success",
                "regime": regime,
                "case_type": case_type,
                "query": query,
                "summary": summary,
                "latency_sec": round(elapsed, 3),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "celex_mentions_count": len(celex_mentions),
                "celex_mentions": celex_mentions,
                "contains_refusal": contains_refusal,
                "refusal_correct": refusal_correct,
                "has_comparative_structure": has_comparative_structure,
            }

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"Mistral synthesis failed: {e}")
            return {
                "status": "exception",
                "error": str(e),
                "latency_sec": round(elapsed, 3),
            }
