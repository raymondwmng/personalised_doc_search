"""Ollama-based legal summarisation module."""

import json
import logging
from typing import Any, Dict, List, Optional
import ollama

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class LegalSummariser:
    """Generates precise legal syntheses and citations across multiple documents using local Ollama models."""

    def __init__(
        self,
        default_model: str = config.DEFAULT_ANALYTICAL_MODEL,
        fallback_model: str = config.DEFAULT_FALLBACK_MODEL,
        num_ctx: int = config.DEFAULT_NUM_CTX,
        timeout: float = config.DEFAULT_OLLAMA_TIMEOUT,
    ):
        self.default_model = default_model
        self.fallback_model = fallback_model
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.client = ollama.Client()

    def build_catalog_and_excerpts(
        self, retrieved_docs: List[Dict[str, Any]], max_docs: int = 5
    ) -> tuple[str, str, int]:
        """Formats retrieved documents into a catalog index and budgeted excerpts block.
        
        Calculates a dynamic character budget per document based on total documents to prevent
        context window overflow, ensuring all retrieved articles fit without left-truncation.
        
        Returns:
            (catalog_text, excerpts_text, total_docs_included)
        """
        docs_to_include = retrieved_docs[:max_docs]
        total_docs = len(docs_to_include)
        if total_docs == 0:
            return "", "", 0

        # Dynamic character budget: total ~8500 chars (~2100 tokens) across all retrieved docs
        char_budget = max(450, int(8500 / total_docs))

        catalog_lines = []
        excerpts_parts = []

        for i, doc in enumerate(docs_to_include, 1):
            celex = doc.get("celex", "N/A")
            title = doc.get("title", "Untitled")
            date = doc.get("date", "N/A")
            raw_text = doc.get("best_chunk") or doc.get("full_text_excerpt", "")
            clean_excerpt = raw_text.strip()[:char_budget]

            catalog_lines.append(f"{i}. [Doc {i}] (CELEX: {celex}, Date: {date}) - {title}")
            excerpts_parts.append(
                f"### [Doc {i}] CELEX: {celex} | Date: {date}\n"
                f"Title: {title}\n"
                f"Relevant Legal Excerpt:\n{clean_excerpt}\n"
            )

        catalog_text = "\n".join(catalog_lines)
        excerpts_text = "\n".join(excerpts_parts)
        return catalog_text, excerpts_text, total_docs

    def build_context_block(self, retrieved_docs: List[Dict[str, Any]], max_docs: int = 5) -> str:
        """Formats retrieved documents into a structured prompt excerpt (backward compatibility)."""
        _, excerpts_text, _ = self.build_catalog_and_excerpts(retrieved_docs, max_docs=max_docs)
        return excerpts_text

    def summarise(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        memory_context: Optional[str] = None,
        max_docs: int = 5,
    ) -> Dict[str, Any]:
        """Generates a comprehensive cross-document legal synthesis covering all retrieved documents."""
        if not retrieved_docs:
            return {
                "summary": "No relevant documents found to summarise.",
                "model_used": "none",
                "status": "empty",
                "docs_analyzed": 0,
            }

        selected_model = model_name or self.default_model
        catalog_text, excerpts_text, total_docs = self.build_catalog_and_excerpts(
            retrieved_docs, max_docs=max_docs
        )

        system_instruction = (
            "You are an expert EU legal document analyst providing an executive legal briefing. "
            "Your objective is to provide a synthesised, cross-document analysis answering the user's query "
            "by integrating evidence across ALL provided documents. You must cite specific CELEX numbers "
            "and document indices ([Doc 1], [Doc 2], etc.), highlight connections or contrasts between the retrieved acts, "
            "and never restrict your analysis to just one document or the final document."
        )

        user_content_parts = []
        if memory_context and memory_context.strip():
            user_content_parts.append(f"### User Background / Directives:\n{memory_context.strip()}\n")

        user_content_parts.append(
            f"The user has searched EU legislation with the following legal query:\n\"{query.strip()}\"\n"
        )
        user_content_parts.append(
            f"The search engine retrieved {total_docs} relevant legislative documents/decisions:\n{catalog_text}\n"
        )
        user_content_parts.append(
            f"=== RETRIEVED DOCUMENT EXCERPTS ({total_docs} DOCUMENTS) ===\n{excerpts_text}\n=== END OF EXCERPTS ===\n"
        )
        user_content_parts.append(
            f"### MANDATORY SYNTHESIS INSTRUCTIONS:\n"
            f"1. **Cross-Document Executive Summary**: Synthesise insights across ALL {total_docs} retrieved documents. "
            f"Do NOT focus only on the last document or a single document.\n"
            f"2. **Cross-Document Findings & Comparison**: Explicitly address and cite EACH relevant retrieved document "
            f"using both its index and CELEX ([Doc 1, CELEX ...], [Doc 2, CELEX ...], etc.). Group common legal themes "
            f"and explain what each document establishes regarding the user's query.\n"
            f"3. **Industry / Sector Impact**: Detail which business sectors and entities are affected across the retrieved corpus.\n"
            f"4. **Regulatory Trajectory & Takeaways**: Provide an actionable takeaway for the user's query based on these legal instruments.\n"
            f"5. Format the briefing clearly with clean Markdown headings and structured bullet points."
        )

        user_prompt = "\n".join(user_content_parts)

        options = {
            "num_ctx": self.num_ctx,
            "temperature": 0.2,  # Low temperature for precise legal fidelity
        }

        # Try selected model, fall back to fallback_model if needed
        models_to_try = [selected_model]
        if self.fallback_model not in models_to_try:
            models_to_try.append(self.fallback_model)

        last_error = None
        for current_model in models_to_try:
            try:
                logger.info(f"Invoking Ollama model '{current_model}' for {total_docs} documents (timeout={self.timeout}s)...")
                response = self.client.chat(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt},
                    ],
                    options=options,
                )
                summary_text = response.get("message", {}).get("content", "").strip()
                if summary_text:
                    return {
                        "summary": summary_text,
                        "model_used": current_model,
                        "status": "success",
                        "docs_analyzed": total_docs,
                    }
            except Exception as e:
                logger.warning(f"Model '{current_model}' generation failed: {e}")
                last_error = e

        return {
            "summary": f"Summarisation could not be completed: {last_error}",
            "model_used": selected_model,
            "status": "error",
            "docs_analyzed": 0,
        }
