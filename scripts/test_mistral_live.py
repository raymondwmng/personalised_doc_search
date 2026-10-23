#!/usr/bin/env python3
"""Minimal Token Consumption Live Test for Mistral API Search Summarisation.

This script tests the real Mistral API using your MISTRAL_API_KEY with the
minimum necessary number of tokens (~120 prompt tokens, <= 40 completion tokens),
costing less than £0.00003 GBP ($0.00003 USD) to verify live API connectivity
without depleting your credits.
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.summariser import LegalSummariser


def run_minimal_mistral_test():
    api_key = config.MISTRAL_API_KEY or os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        print("\n❌ Error: MISTRAL_API_KEY is not set in your .env file.")
        print("Please add MISTRAL_API_KEY=your_key to your .env file.")
        sys.exit(1)

    # Mask key for privacy
    masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
    print("=" * 65)
    print("  MISTRAL API LIVE SUMMARY TEST — MINIMAL TOKEN MODE")
    print("=" * 65)
    print(f"• API Key Detected: {masked_key}")
    print(f"• Model: mistral-small-2603 (Most cost-efficient cloud model)")
    print(f"• Max Output Tokens: 40 (Strictly capped to conserve credits)")
    print(f"• Documents Tested: 1 minimal excerpt (~60 words)")
    print("-" * 65)

    # Construct a minimal realistic document excerpt
    mock_docs = [
        {
            "celex": "32024R1689",
            "title": "Artificial Intelligence Act (Regulation (EU) 2024/1689)",
            "date": "2024-07-12",
            "best_chunk": (
                "Article 6 establishes classification rules for high-risk AI systems. "
                "AI systems posing significant risks to health, safety, or fundamental "
                "rights are subject to strict conformity assessments and transparency obligations."
            ),
        }
    ]

    summariser = LegalSummariser()
    query = "What rules apply to high-risk AI systems?"

    start_time = time.time()
    result = summariser.summarise(
        query=query,
        retrieved_docs=mock_docs,
        model_name="mistral-small-2603",
        max_docs=1,
        max_tokens=40,
    )
    elapsed = time.time() - start_time

    status = result.get("status")
    print(f"\n[Response Status]: {status.upper()} (in {elapsed:.2f}s)")
    print(f"[Model Used]: {result.get('model_used')}")
    print("\n--- Generated Legal Summary ---")
    print(result.get("summary"))
    print("-" * 65)

    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    # Pricing for mistral-small-2603: ~$0.10/M input, ~$0.30/M output
    cost_usd = (prompt_tokens * 0.00000010) + (completion_tokens * 0.00000030)
    cost_gbp = cost_usd * 0.79

    print("\n📊 TOKEN CONSUMPTION & COST AUDIT:")
    print(f"  • Prompt (Input) Tokens:      {prompt_tokens}")
    print(f"  • Completion (Output) Tokens: {completion_tokens}")
    print(f"  • Total Tokens Consumed:      {total_tokens}")
    print(f"  • Estimated Cost (USD):       ${cost_usd:.6f}")
    print(f"  • Estimated Cost (GBP):       £{cost_gbp:.6f}")
    print("=" * 65)

    if status == "success":
        print("✅ SUCCESS: Live Mistral API summarisation confirmed with minimal token usage!")
    else:
        print("❌ FAILED: Generation returned an error.")
        sys.exit(1)


if __name__ == "__main__":
    run_minimal_mistral_test()
