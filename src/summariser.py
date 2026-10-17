"""Ollama-based legal summarisation module."""

import json
import logging
from typing import Any, Dict, List, Optional
import ollama

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class LegalSummariser:
    """Generates precise legal syntheses and citations using local Ollama models."""

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

    def build_context_block(self, retrieved_docs: List[Dict[str, Any]], max_docs: int = 5) -> str:
        """Formats retrieved documents into a structured prompt excerpt."""
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:max_docs], 1):
            celex = doc.get("celex", "N/A")
            title = doc.get("title", "Untitled")
            date = doc.get("date", "N/A")
            excerpt = doc.get("best_chunk") or doc.get("full_text_excerpt", "")
            context_parts.append(
                f"--- [Doc {i}] CELEX: {celex} | Date: {date} | Title: {title} ---\n{excerpt.strip()}\n"
            )
        return "\n".join(context_parts)

    def summarise(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        memory_context: Optional[str] = None,
        max_docs: int = 5,
    ) -> Dict[str, Any]:
        """Generates a legal summary for the query given retrieved documents."""
        if not retrieved_docs:
            return {
                "summary": "No relevant documents found to summarise.",
                "model_used": "none",
                "status": "empty",
            }

        selected_model = model_name or self.default_model
        context_text = self.build_context_block(retrieved_docs, max_docs=max_docs)

        system_instruction = (
            "You are an expert EU legal document analyst. Summarise the retrieved legal excerpts "
            "strictly relevant to the user's query. Be precise, cite specific articles/clauses and CELEX identifiers, "
            "and do not hallucinate facts. Keep the analytical summary clear, structured, and under 300 words."
        )

        user_content_parts = []
        if memory_context and memory_context.strip():
            user_content_parts.append(f"User Background / Directives:\n{memory_context.strip()}\n")

        user_content_parts.append(f"User Legal Query:\n{query}\n")
        user_content_parts.append(f"Retrieved Legal Excerpts:\n{context_text}")

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
                logger.info(f"Invoking Ollama model '{current_model}' (timeout={self.timeout}s)...")
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
                    }
            except Exception as e:
                logger.warning(f"Model '{current_model}' generation failed: {e}")
                last_error = e

        return {
            "summary": f"Summarisation could not be completed: {last_error}",
            "model_used": selected_model,
            "status": "error",
        }
