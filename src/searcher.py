"""Retrieval and search engine combining FAISS semantic search and keyword re-ranking."""

import logging
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import config
from src.embedder import DocumentEmbedder
from src.indexer import FaissIndexer
from src.memory import UserMemoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SearchEngine:
    """Executes query transformation, FAISS dense retrieval, lexical re-ranking, and document assembly."""

    def __init__(
        self,
        embedder: Optional[DocumentEmbedder] = None,
        indexer: Optional[FaissIndexer] = None,
        db_path: Path = config.DOCUMENTS_DB_PATH,
        memory_store: Optional[UserMemoryStore] = None,
    ):
        self.db_path = db_path
        self.memory_store = memory_store or UserMemoryStore()
        self.embedder = embedder or DocumentEmbedder()
        self.indexer = indexer or FaissIndexer()
        self.indexer.load()

    def _get_db_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def transform_query(self, query: str, enable_memory: bool = False) -> Dict[str, str]:
        """Transforms user query into semantic query and lexical keyword set."""
        cleaned_query = re.sub(r"[^\w\s\-\.]", " ", query).strip()
        semantic_query = cleaned_query

        memory_context = ""
        if enable_memory:
            mem = self.memory_store.load_memory()
            user_context = mem.get("user_context", "")
            topics = " ".join(mem.get("preferred_topics", []))
            sectors = " ".join(mem.get("preferred_sectors", []))
            if user_context or topics or sectors:
                memory_context = f"{user_context} {topics} {sectors}".strip()
                # Lightly bias semantic query with profile context
                semantic_query = f"{cleaned_query} ({memory_context[:100]})"

        return {
            "raw_query": query,
            "semantic_query": semantic_query,
            "lexical_query": cleaned_query,
            "memory_context": memory_context,
        }

    def _calculate_lexical_score(self, query_terms: List[str], text: str) -> float:
        """Simple BM25-lite keyword match density."""
        if not text or not query_terms:
            return 0.0
        text_lower = text.lower()
        score = 0.0
        for term in query_terms:
            if len(term) < 3:
                continue
            count = text_lower.count(term.lower())
            if count > 0:
                score += min(1.0, count * 0.25)
        return score / max(1, len(query_terms))

    def search(
        self,
        query: str,
        top_k_chunks: int = 20,
        top_n_docs: int = 10,
        enable_memory: bool = False,
    ) -> List[Dict[str, Any]]:
        """Executes full search pipeline:
        
        1. Query transformation
        2. Vector similarity search via FAISS
        3. Lexical re-ranking & document aggregation
        4. Enrichment with tags and document metadata
        """
        transformed = self.transform_query(query, enable_memory=enable_memory)
        semantic_q = transformed["semantic_query"]
        lexical_terms = [t for t in transformed["lexical_query"].split() if len(t) >= 3]

        # FAISS search
        q_vec = self.embedder.encode_query(semantic_q)
        raw_chunks = self.indexer.search(q_vec, top_k=top_k_chunks)

        if not raw_chunks:
            logger.warning(f"No matching chunks found for query: {query}")
            return []

        # Group chunks by parent doc_id
        doc_chunk_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for chunk in raw_chunks:
            doc_id = chunk["doc_id"]
            faiss_score = chunk["similarity_score"]
            chunk_text = chunk["text"]

            # Lexical re-ranking component
            lex_score = self._calculate_lexical_score(lexical_terms, chunk_text)
            hybrid_score = (0.75 * faiss_score) + (0.25 * lex_score)
            chunk["combined_score"] = float(hybrid_score)
            doc_chunk_map[doc_id].append(chunk)

        # Retrieve documents and tags from SQLite
        doc_ids = list(doc_chunk_map.keys())
        placeholders = ",".join("?" for _ in doc_ids)

        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
            SELECT id, celex, title, date, type, full_text, eurovoc_raw, source
            FROM documents
            WHERE id IN ({placeholders})
            """, doc_ids)
            doc_records = {r["id"]: dict(r) for r in cursor.fetchall()}

            cursor.execute(f"""
            SELECT doc_id, dimension, category, confidence_score, method
            FROM document_tags
            WHERE doc_id IN ({placeholders})
            ORDER BY confidence_score DESC
            """, doc_ids)
            tag_rows = cursor.fetchall()

        doc_tags_map = defaultdict(lambda: {"Jurisdiction": [], "Regulatory Topics": [], "Industry / Sector": []})
        for tag in tag_rows:
            d_id = tag["doc_id"]
            dim = tag["dimension"]
            cat = tag["category"]
            conf = tag["confidence_score"]
            if dim in doc_tags_map[d_id]:
                doc_tags_map[d_id][dim].append({"category": cat, "confidence": conf, "method": tag["method"]})

        # Rank documents
        aggregated_docs = []
        for doc_id, chunks in doc_chunk_map.items():
            if doc_id not in doc_records:
                continue

            doc_info = doc_records[doc_id]
            # Best chunk score represents the peak relevance of this document
            best_chunk = max(chunks, key=lambda c: c["combined_score"])
            avg_score = float(np.mean([c["combined_score"] for c in chunks]))

            aggregated_docs.append({
                "doc_id": doc_id,
                "celex": doc_info["celex"],
                "title": doc_info["title"],
                "date": doc_info["date"],
                "type": doc_info["type"],
                "source": doc_info["source"],
                "peak_score": round(best_chunk["combined_score"], 4),
                "avg_score": round(avg_score, 4),
                "best_chunk": best_chunk["text"],
                "chunks": chunks,
                "tags": doc_tags_map[doc_id],
                "full_text_excerpt": doc_info["full_text"][:1200] + ("..." if len(doc_info["full_text"]) > 1200 else ""),
            })

        # Sort descending by peak score
        aggregated_docs.sort(key=lambda d: d["peak_score"], reverse=True)
        return aggregated_docs[:top_n_docs]
