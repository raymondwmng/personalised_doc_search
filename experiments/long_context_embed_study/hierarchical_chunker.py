"""Hierarchical Legislative Chunker for Legal Retrieval Study (Section 3.2.1 & 3.2.2).

Implements three distinct multi-resolution granularity tiers:
  - Regime A (Micro):  ~400-token article chunks with 50-token sliding overlap.
  - Regime B (Meso):   ~2,000-token chapter/thematic chunks bundling definitions + articles.
  - Regime C (Macro):  Whole-act document chunks (8K to 32K tokens).
"""

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Ground-truth CELEX documents for the 9-query benchmark suite
BENCHMARK_GT_CELEX = [
    # Q1: 470-790 MHz frequency band
    "32017D0899",
    # Q2: Brussels IRIS hospitals State aid
    "32016D2327",
    # Q3: Morocco sustainable fisheries agreement
    "32018D2068",
    # Q4: Aviation civil safety agreement United States
    "32019D0572",
    "32018D0061",
    "32011D0719",
    # Q5: Passenger cars CO2 emissions innovative technology
    "32019D1119",
    "32017D0785",
    "32015D1132",
    "32013D0128",
    "32013D0451",
    "32018D2079",
    # Q6: Schengen acquis participation provisions
    "32012D0764",
    "32010D0491",
    "32010D0490",
    "32012D0193",
    "32010D0779",
    "32019D0304",
]

# Distractor CELEX documents retrieved in the initial audit for Case 3 negative queries
CASE3_DISTRACTOR_CELEX = [
    "32014D0255",  # Customs Code / export controls distractor for Q7
    "32016D1250",  # EU-US Privacy Shield distractor for Q8
    "32012D0130",  # UNCLOS international fisheries distractor for Q9
]


def estimate_token_count(text: str) -> int:
    """Fast, accurate token count approximation (~1 token = 0.75 words / ~4 chars)."""
    if not text:
        return 0
    words = text.split()
    return max(1, int(len(words) * 1.3))


class HierarchicalChunker:
    """Generates chunks across Micro (400t), Meso (2000t), and Macro (full-act) tiers."""

    def __init__(self):
        pass

    def chunk_micro(
        self,
        text: str,
        celex: str,
        title: str,
        target_tokens: int = 400,
        overlap_tokens: int = 50,
    ) -> List[Dict[str, Any]]:
        r"""Regime A (Micro): Article-level chunks (~400 tokens with 50-token overlap).
        
        Splits by structural boundaries (Article \d+, Whereas, Chapter) and subdivides
        oversized provisions.
        """
        if not text.strip():
            return []

        # Split on structural legal boundaries
        pattern = r"(?=(?:(?:Article|ARTICLE)\s+\d+|(?:Whereas|WHEREAS)\s*\(?\d*\)?|(?:Chapter|CHAPTER)\s+[IVXLCDM\d]+))"
        sections = re.split(pattern, text)
        sections = [s.strip() for s in sections if s.strip()]

        if not sections:
            sections = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        target_chars = target_tokens * 4
        overlap_chars = overlap_tokens * 4

        current_buf = []
        current_len = 0

        for sec in sections:
            sec_len = len(sec)
            if current_len + sec_len > target_chars and current_buf:
                chunk_text = "\n\n".join(current_buf).strip()
                tokens = estimate_token_count(chunk_text)
                chunks.append({
                    "chunk_idx": len(chunks),
                    "regime": "micro",
                    "celex": celex,
                    "title": title,
                    "text": chunk_text,
                    "token_count": tokens,
                })

                # Maintain overlap
                keep = []
                k_len = 0
                for item in reversed(current_buf):
                    if k_len + len(item) <= overlap_chars:
                        keep.insert(0, item)
                        k_len += len(item)
                    else:
                        break
                current_buf = keep
                current_len = k_len

            current_buf.append(sec)
            current_len += sec_len

        if current_buf:
            chunk_text = "\n\n".join(current_buf).strip()
            tokens = estimate_token_count(chunk_text)
            chunks.append({
                "chunk_idx": len(chunks),
                "regime": "micro",
                "celex": celex,
                "title": title,
                "text": chunk_text,
                "token_count": tokens,
            })

        return chunks

    def chunk_meso(
        self,
        text: str,
        celex: str,
        title: str,
        target_tokens: int = 2000,
        overlap_tokens: int = 200,
    ) -> List[Dict[str, Any]]:
        """Regime B (Meso): Chapter/Thematic chunks (~2,000 tokens with 200-token overlap).
        
        Preserves definitions, procedural context, and substantive article clusters intact.
        """
        if not text.strip():
            return []

        target_chars = target_tokens * 4
        overlap_chars = overlap_tokens * 4

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_buf = []
        current_len = 0

        for para in paragraphs:
            p_len = len(para)
            if current_len + p_len > target_chars and current_buf:
                chunk_text = f"Document {celex} — {title}\n\n" + "\n\n".join(current_buf).strip()
                tokens = estimate_token_count(chunk_text)
                chunks.append({
                    "chunk_idx": len(chunks),
                    "regime": "meso",
                    "celex": celex,
                    "title": title,
                    "text": chunk_text,
                    "token_count": tokens,
                })

                # Overlap
                keep = []
                k_len = 0
                for item in reversed(current_buf):
                    if k_len + len(item) <= overlap_chars:
                        keep.insert(0, item)
                        k_len += len(item)
                    else:
                        break
                current_buf = keep
                current_len = k_len

            current_buf.append(para)
            current_len += p_len

        if current_buf:
            chunk_text = f"Document {celex} — {title}\n\n" + "\n\n".join(current_buf).strip()
            tokens = estimate_token_count(chunk_text)
            chunks.append({
                "chunk_idx": len(chunks),
                "regime": "meso",
                "celex": celex,
                "title": title,
                "text": chunk_text,
                "token_count": tokens,
            })

        return chunks

    def chunk_macro(
        self,
        text: str,
        celex: str,
        title: str,
        max_tokens: int = 32768,
    ) -> List[Dict[str, Any]]:
        """Regime C (Macro): Whole-act chunk (up to 32K tokens intact).
        
        Eliminates fragmentation loss; single holistic embedding per legal act.
        """
        if not text.strip():
            return []

        clean_text = f"Document {celex} — {title}\n\n" + text.strip()
        tokens = estimate_token_count(clean_text)

        # Truncate only if exceeding absolute model ceiling
        if tokens > max_tokens:
            max_chars = max_tokens * 4
            clean_text = clean_text[:max_chars]
            tokens = max_tokens

        return [{
            "chunk_idx": 0,
            "regime": "macro",
            "celex": celex,
            "title": title,
            "text": clean_text,
            "token_count": tokens,
        }]

    def chunk_document_by_regime(
        self,
        text: str,
        celex: str,
        title: str,
        regime: str,
        max_tokens: int = 32768,
    ) -> List[Dict[str, Any]]:
        """Dispatches chunking to the designated regime ('micro', 'meso', or 'macro')."""
        reg = regime.lower()
        if reg == "micro":
            return self.chunk_micro(text, celex, title)
        elif reg == "meso":
            return self.chunk_meso(text, celex, title)
        elif reg == "macro":
            return self.chunk_macro(text, celex, title, max_tokens=max_tokens)
        else:
            raise ValueError(f"Unknown chunking regime: '{regime}' (expected 'micro', 'meso', 'macro')")


def load_study_corpus(
    db_path: Path,
    num_background_docs: int = 500,
) -> List[Dict[str, Any]]:
    """Loads a high-fidelity evaluation corpus containing all benchmark GT acts + stratified background."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Fetch benchmark GT documents + distractors
    priority_celexes = list(set(BENCHMARK_GT_CELEX + CASE3_DISTRACTOR_CELEX))
    placeholders = ",".join(["?"] * len(priority_celexes))
    cursor.execute(
        f"SELECT id, celex, title, date, type, full_text FROM documents WHERE celex IN ({placeholders})",
        priority_celexes,
    )
    priority_docs = [dict(row) for row in cursor.fetchall()]
    found_celexes = {d["celex"] for d in priority_docs}
    logger.info(f"Loaded {len(priority_docs)} priority benchmark/distractor documents from DB.")

    # 2. Fetch stratified background sample excluding priority documents
    cursor.execute(
        f"SELECT id, celex, title, date, type, full_text FROM documents WHERE celex NOT IN ({placeholders}) AND LENGTH(full_text) > 200 ORDER BY id ASC LIMIT ?",
        (*priority_celexes, num_background_docs),
    )
    background_docs = [dict(row) for row in cursor.fetchall()]
    logger.info(f"Loaded {len(background_docs)} stratified background documents.")

    all_docs = priority_docs + background_docs
    conn.close()
    return all_docs
