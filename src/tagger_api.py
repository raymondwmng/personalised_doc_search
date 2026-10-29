"""Azure AI Document Intelligence integration with Layout model, complexity routing, and budget guardrails (Section 4.3)."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
import requests

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Budget Constants (5 GBP budget envelope)
BUDGET_GBP = 5.0
USD_PER_GBP = 1.26  # ~ $6.30 USD total
TOTAL_BUDGET_USD = BUDGET_GBP * USD_PER_GBP
FREE_TIER_PAGES = 500
COST_PER_1K_PAGES_LAYOUT = 10.00  # Azure S0 Layout: $10 / 1000 pages
COST_PER_1K_PAGES_READ = 1.50     # Azure S0 Read: $1.50 / 1000 pages


class AzureDocIntelligenceClient:
    """Client for Azure AI Document Intelligence (Layout and Read models)
    with budget tracking, complexity classification, and resilient local fallback.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        budget_pages: int = config.AZURE_DOC_INTEL_BUDGET_PAGES,
    ):
        self.endpoint = (endpoint or config.AZURE_DOC_INTEL_ENDPOINT).strip().rstrip("/")
        self.api_key = (api_key or config.AZURE_DOC_INTEL_KEY).strip()
        self.budget_pages = budget_pages

        self.pages_processed = 0
        self.layout_pages = 0
        self.read_pages = 0
        self.estimated_cost_usd = 0.0

        self.is_configured = bool(self.endpoint and self.api_key and "mock" not in self.endpoint.lower())
        if not self.is_configured:
            logger.info("Azure Document Intelligence credentials not configured. Operating in resilient offline/mock mode.")

    def assess_complexity(self, html_or_text: str) -> Dict[str, Any]:
        """Assesses document structural complexity to route between Layout and Read models."""
        table_count = len(re.findall(r"<table\b", html_or_text, re.IGNORECASE))
        row_count = len(re.findall(r"<tr\b", html_or_text, re.IGNORECASE))
        annex_count = len(re.findall(r"(?:Annex|ANNEX)\s+[IVXLCDM\d]+", html_or_text))
        approx_pages = max(1, len(html_or_text) // 3000)

        # Complex documents: multiple tables, dense schedules, or technical annexes
        is_complex = table_count >= 2 or row_count >= 10 or annex_count >= 2
        model_to_use = "layout" if is_complex else "read"

        return {
            "is_complex": is_complex,
            "table_count": table_count,
            "row_count": row_count,
            "annex_count": annex_count,
            "approx_pages": approx_pages,
            "selected_model": model_to_use,
        }

    def _estimate_cost(self, pages: int, model: str) -> float:
        """Calculates marginal cost in USD taking into account 500 free pages."""
        total_prior = self.pages_processed
        free_remaining = max(0, FREE_TIER_PAGES - total_prior)
        billable_pages = max(0, pages - free_remaining)

        rate_per_page = (COST_PER_1K_PAGES_LAYOUT if model == "layout" else COST_PER_1K_PAGES_READ) / 1000.0
        return billable_pages * rate_per_page

    def extract_layout_mock(self, html_or_text: str, celex: str = "") -> Dict[str, Any]:
        """High-fidelity local layout extraction parsing HTML tables, sections, and structured metadata."""
        soup = BeautifulSoup(html_or_text, "lxml" if "<html" in html_or_text.lower() else "html.parser")
        tables_extracted = []

        for idx, table in enumerate(soup.find_all("table")):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
            if rows:
                headers = rows[0] if len(rows) > 1 else []
                tables_extracted.append({
                    "table_idx": idx,
                    "row_count": len(rows),
                    "headers": headers,
                    "sample_rows": rows[1:4],
                })

        approx_pages = max(1, len(html_or_text) // 3000)
        return {
            "celex": celex,
            "model_used": "local_structured_parser",
            "pages": approx_pages,
            "tables": tables_extracted,
            "table_count": len(tables_extracted),
            "status": "success_local",
        }

    def analyze_document(
        self,
        html_or_text: str,
        celex: str = "",
        force_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyzes a legislative document using Azure Document Intelligence or local fallback.
        
        Enforces budget ceiling and tracks cumulative usage.
        """
        complexity = self.assess_complexity(html_or_text)
        model = force_model or complexity["selected_model"]
        pages = complexity["approx_pages"]

        # Budget check
        if self.pages_processed + pages > self.budget_pages:
            logger.warning(
                f"Page budget threshold ({self.budget_pages} pages) reached. "
                f"Routing CELEX {celex} ({pages} pages) to local parser."
            )
            return self.extract_layout_mock(html_or_text, celex=celex)

        marginal_cost = self._estimate_cost(pages, model)
        if (self.estimated_cost_usd + marginal_cost) > TOTAL_BUDGET_USD:
            logger.warning(f"Cost ceiling (${TOTAL_BUDGET_USD:.2f} USD) exceeded. Falling back to local parser.")
            return self.extract_layout_mock(html_or_text, celex=celex)

        if not self.is_configured:
            result = self.extract_layout_mock(html_or_text, celex=celex)
            self.pages_processed += pages
            if model == "layout":
                self.layout_pages += pages
            else:
                self.read_pages += pages
            self.estimated_cost_usd += marginal_cost
            result["budget_stats"] = self.get_budget_summary()
            return result

        # Azure REST API invocation
        api_version = "2024-02-29-preview"
        url = f"{self.endpoint}/documentintelligence/documentModels/prebuilt-{model}:analyze?api-version={api_version}"
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "text/html" if "<html" in html_or_text.lower() else "text/plain",
        }

        try:
            resp = requests.post(url, headers=headers, data=html_or_text.encode("utf-8"), timeout=30)
            if resp.status_code in [200, 202]:
                self.pages_processed += pages
                if model == "layout":
                    self.layout_pages += pages
                else:
                    self.read_pages += pages
                self.estimated_cost_usd += marginal_cost
                return {
                    "celex": celex,
                    "model_used": f"azure_{model}",
                    "pages": pages,
                    "status": "success",
                    "budget_stats": self.get_budget_summary(),
                }
            else:
                logger.warning(f"Azure API returned status {resp.status_code}: {resp.text[:200]}. Falling back to local.")
                return self.extract_layout_mock(html_or_text, celex=celex)
        except Exception as e:
            logger.warning(f"Azure request failed ({e}). Falling back to local parser.")
            return self.extract_layout_mock(html_or_text, celex=celex)

    def get_budget_summary(self) -> Dict[str, Any]:
        """Returns budget tracking metrics."""
        return {
            "total_pages_processed": self.pages_processed,
            "layout_pages": self.layout_pages,
            "read_pages": self.read_pages,
            "free_tier_remaining": max(0, FREE_TIER_PAGES - self.pages_processed),
            "budget_cap_pages": self.budget_pages,
            "estimated_cost_usd": round(self.estimated_cost_usd, 3),
            "budget_limit_usd": round(TOTAL_BUDGET_USD, 2),
            "budget_remaining_usd": round(max(0.0, TOTAL_BUDGET_USD - self.estimated_cost_usd), 2),
        }

