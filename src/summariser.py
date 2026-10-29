"""Legal summarisation module supporting Mistral API and local Ollama models."""

import json
import logging
import os
from typing import Any, Dict, List, Optional
import ollama
import requests

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class LegalSummariser:
    """Generates precise legal syntheses and citations across multiple documents using Mistral API and local Ollama models."""

    def __init__(
        self,
        default_model: str = config.DEFAULT_ANALYTICAL_MODEL,
        fallback_model: str = config.DEFAULT_FALLBACK_MODEL,
        num_ctx: int = config.DEFAULT_NUM_CTX,
        timeout: float = config.DEFAULT_OLLAMA_TIMEOUT,
        mistral_timeout: float = config.DEFAULT_MISTRAL_TIMEOUT,
    ):
        self.default_model = default_model
        self.fallback_model = fallback_model
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.mistral_timeout = mistral_timeout
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
        max_tokens: Optional[int] = None,
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

        # Check for Case 3 Zero-Match condition (Out-of-domain query below null threshold 0.50)
        is_case_3_zero_match = all(d.get("is_below_null_threshold", False) for d in retrieved_docs[:total_docs])

        if is_case_3_zero_match:
            user_content_parts.append(
                f"### MANDATORY REFUSAL INSTRUCTIONS (Case 3 Zero-Match Rejection):\n"
                f"The similarity scores for all retrieved documents fall below the calibrated threshold (0.50). "
                f"These documents do NOT substantively govern the user's query '{query.strip()}'.\n"
                f"1. **Graceful Refusal**: Begin with an explicit statement: 'Graceful Refusal (Case 3): No Relevant EU Legislation Identified.'\n"
                f"2. Explain clearly that the retrieved European Union legislative acts do not regulate or address this subject.\n"
                f"3. Briefly state why the closest retrieved distractor documents do not apply, preventing any legal hallucination.\n"
            )
        else:
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

        # Route to Mistral Cloud API if model is mistral-large-4, mistral-small-2603, or alias
        if config.is_mistral_model(selected_model):
            mistral_res = self._summarise_with_mistral(
                model_name=selected_model,
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                total_docs=total_docs,
                max_tokens=max_tokens,
            )
            if mistral_res.get("status") == "success":
                return mistral_res
            logger.warning(
                f"Mistral model '{selected_model}' generation failed: {mistral_res.get('summary')}. "
                f"Attempting local fallback to '{self.fallback_model}'..."
            )
            if config.is_mistral_model(self.fallback_model):
                return mistral_res

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

    def _summarise_with_mistral(
        self,
        model_name: str,
        system_instruction: str,
        user_prompt: str,
        total_docs: int,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calls Mistral AI API for legal synthesis using Mistral client or HTTP REST fallback.
        
        Uses the MISTRAL_API_KEY stored in .env without exposing the key in code.
        Supports max_tokens to strictly control token consumption and credit usage.
        """
        api_key = config.MISTRAL_API_KEY or os.getenv("MISTRAL_API_KEY", "")
        if not api_key:
            return {
                "summary": (
                    "⚠️ **Mistral API Key Missing**\n\n"
                    "The selected model requires `MISTRAL_API_KEY` to be set in your `.env` file.\n"
                    "Please check your `.env` file and restart or re-run the query."
                ),
                "model_used": model_name,
                "status": "error",
                "docs_analyzed": 0,
            }

        resolved_model = config.resolve_model_name(model_name)
        model_display = config.ALL_MODELS.get(resolved_model, resolved_model)

        # 1. Primary: Use Mistral SDK Client
        try:
            try:
                from mistralai.client import Mistral
            except ImportError:
                from mistralai import Mistral

            logger.info(f"Invoking Mistral API model '{resolved_model}' via Mistral SDK...")
            sdk_kwargs = {}
            if max_tokens is not None:
                sdk_kwargs["max_tokens"] = max_tokens

            with Mistral(api_key=api_key) as client:
                res = client.chat.complete(
                    model=resolved_model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    **sdk_kwargs,
                )
                summary_text = res.choices[0].message.content
                if summary_text and summary_text.strip():
                    usage_obj = getattr(res, "usage", None)
                    usage = {
                        "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                        "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                        "total_tokens": getattr(usage_obj, "total_tokens", None),
                    } if usage_obj else {}
                    return {
                        "summary": summary_text.strip(),
                        "model_used": model_display,
                        "status": "success",
                        "docs_analyzed": total_docs,
                        "usage": usage,
                    }
        except Exception as sdk_err:
            logger.warning(f"Mistral SDK call encountered error: {sdk_err}. Falling back to Mistral REST API...")

        # 2. Resilient fallback: Direct HTTP REST call to https://api.mistral.ai/v1/chat/completions
        try:
            logger.info(f"Invoking Mistral REST API for '{resolved_model}'...")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": resolved_model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens

            resp = requests.post(
                f"{config.MISTRAL_API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.mistral_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            summary_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            usage = data.get("usage", {})
            if summary_text:
                return {
                    "summary": summary_text,
                    "model_used": model_display,
                    "status": "success",
                    "docs_analyzed": total_docs,
                    "usage": usage,
                }
            raise ValueError("Mistral REST API returned empty choices")
        except Exception as http_err:
            logger.error(f"Mistral API generation error: {http_err}")
            return {
                "summary": f"Mistral API generation failed: {http_err}",
                "model_used": model_name,
                "status": "error",
                "docs_analyzed": 0,
            }

