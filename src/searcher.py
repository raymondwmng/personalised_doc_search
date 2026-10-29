"""Advanced Search Engine with Query Expansion, Hybrid BM25+Dense FAISS Retrieval, and Cross-Encoder Re-Ranking (Section 3.2.2 & 4.6)."""

import logging
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import config
from src.embedder import DocumentEmbedder
from src.indexer import FaissIndexer
from src.memory import UserMemoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# EuroVoc & Statutory Legal Synonym Expansion Dictionary (Section 4.6)
LEGAL_EXPANSION_RULES: Dict[str, List[str]] = {
    "ai": ["artificial intelligence", "high-risk systems", "conformity assessment", "deployers"],
    "artificial intelligence": ["ai act", "high-risk", "transparency obligations", "penalties", "governance"],
    "privacy": ["personal data", "gdpr", "data subject", "supervisory authority", "controller", "processor"],
    "data protection": ["gdpr", "regulation 2016 679", "personal data", "rights of data subjects", "privacy"],
    "emissions": ["greenhouse gas", "co2", "decarbonisation", "emission reduction targets"],
    "passenger cars": ["co2 emissions", "eco-innovation", "innovative technology", "light commercial vehicles"],
    "frequency": ["frequency band", "spectrum allocation", "wireless broadband", "dtt", "uhf"],
    "spectrum": ["frequency band 470 790 mhz", "decision 2017 899", "radio spectrum", "broadband"],
    "state aid": ["article 107 tfeu", "article 108 tfeu", "internal market", "compatibility", "hospitals"],
    "hospitals": ["state aid", "brussels iris", "public service compensation", "sgeb", "iris-h"],
    "fisheries": ["sustainable fisheries partnership agreement", "exclusive economic zone", "morocco", "catches"],
    "aviation": ["civil aviation safety", "airworthiness", "bilateral agreement", "united states", "faa"],
    "schengen": ["schengen acquis", "border checks", "judicial cooperation", "schengen information system"],
    "customs": ["union customs code", "tariff quotas", "origin of goods", "duties", "export control"],
    "antitrust": ["competition", "dominant position", "merger control", "cartels", "treaty 101 102"],
}


class SearchEngine:
    """Hybrid retrieval and ranking engine combining FAISS dense semantic search,
    BM25Okapi lexical retrieval, EuroVoc query expansion, and cross-encoder re-ranking.
    """

    def __init__(
        self,
        model_name_or_key: Optional[str] = None,
        embedder: Optional[DocumentEmbedder] = None,
        indexer: Optional[FaissIndexer] = None,
        db_path: Path = config.DOCUMENTS_DB_PATH,
        memory_store: Optional[UserMemoryStore] = None,
        enable_hybrid: bool = config.ENABLE_HYBRID_SEARCH,
        enable_cross_encoder: bool = config.ENABLE_CROSS_ENCODER,
        dense_weight: float = config.DENSE_WEIGHT,
        bm25_weight: float = config.BM25_WEIGHT,
        null_threshold: float = config.NULL_REJECTION_THRESHOLD,
    ):
        self.db_path = db_path
        self.memory_store = memory_store or UserMemoryStore()

        self.model_spec = config.resolve_embedding_spec(model_name_or_key)
        self.model_key = self.model_spec.get("id", config.DEFAULT_EMBEDDING_MODEL)

        self.embedder = embedder or DocumentEmbedder(model_name_or_key=self.model_key)
        self.indexer = indexer or FaissIndexer(model_name_or_key=self.model_key)
        self.indexer.load()

        self.enable_hybrid = enable_hybrid
        self.enable_cross_encoder = enable_cross_encoder
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.null_threshold = null_threshold

        self._cross_encoder = None
        self._cross_encoder_loaded = False

    def switch_model(self, model_key: str) -> None:
        """Dynamically switches between supported embedding backbones (e.g. bge-m3 vs modernbert)."""
        logger.info(f"Switching SearchEngine backbone to '{model_key}'...")
        self.model_spec = config.resolve_embedding_spec(model_key)
        self.model_key = self.model_spec.get("id", config.DEFAULT_EMBEDDING_MODEL)
        self.null_threshold = self.model_spec.get("threshold", config.NULL_REJECTION_THRESHOLD)
        self.embedder = DocumentEmbedder(model_name_or_key=self.model_key)
        self.indexer = FaissIndexer(model_name_or_key=self.model_key)
        self.indexer.load()
        logger.info(f"Switched to '{self.model_key}'. Dimension: {self.embedder.embedding_dim}")

    def _get_cross_encoder(self):
        """Lazy loads lightweight cross-encoder for candidate re-ranking."""
        if not self._cross_encoder_loaded and self.enable_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder
                device = "cuda" if self.embedder.device == "cuda" else "cpu"
                logger.info(f"Loading CrossEncoder '{config.CROSS_ENCODER_MODEL_NAME}' on {device}...")
                self._cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL_NAME, device=device)
                self._cross_encoder_loaded = True
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder ({e}). Re-ranking will use hybrid scoring.")
                self._cross_encoder = None
                self._cross_encoder_loaded = True
        return self._cross_encoder

    def _get_db_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def expand_query_terms(self, query: str) -> List[str]:
        """Expands user query using domain-specific EuroVoc concepts and legal synonyms."""
        q_lower = query.lower()
        expanded_tokens = []

        for key, synonyms in LEGAL_EXPANSION_RULES.items():
            pattern = r"\b" + re.escape(key) + r"\b"
            if re.search(pattern, q_lower):
                expanded_tokens.extend(synonyms[:2])

        return expanded_tokens

    def transform_query(self, query: str, enable_memory: bool = False) -> Dict[str, Any]:
        """Transforms user query into semantic query, expanded lexical query, and memory context."""
        cleaned_query = re.sub(r"[^\w\s\-\.]", " ", query).strip()
        expanded_synonyms = self.expand_query_terms(cleaned_query)

        # Build lexical query incorporating expansions
        if expanded_synonyms:
            lexical_query = f"{cleaned_query} {' '.join(expanded_synonyms)}"
        else:
            lexical_query = cleaned_query

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
            "lexical_query": lexical_query,
            "expanded_terms": expanded_synonyms,
            "memory_context": memory_context,
        }

    def search(
        self,
        query: str,
        top_k_chunks: int = 20,
        top_n_docs: int = 10,
        enable_memory: bool = False,
    ) -> List[Dict[str, Any]]:
        """Executes full search pipeline:
        
        1. Query transformation & EuroVoc expansion
        2. Dense vector similarity search via FAISS
        3. Lexical keyword retrieval via BM25Okapi
        4. Hybrid fusion (dense + BM25)
        5. Cross-encoder re-ranking on top candidates
        6. Null rejection threshold evaluation (Rec 5)
        7. Document aggregation and taxonomy tag enrichment
        """
        transformed = self.transform_query(query, enable_memory=enable_memory)
        semantic_q = transformed["semantic_query"]
        lexical_q = transformed["lexical_query"]

        # 1. FAISS Dense Retrieval
        q_vec = self.embedder.encode_query(semantic_q)
        dense_chunks = self.indexer.search_dense(q_vec, top_k=top_k_chunks * 2)

        # 2. BM25 Lexical Retrieval
        bm25_chunks = []
        if self.enable_hybrid:
            bm25_chunks = self.indexer.search_bm25(lexical_q, top_k=top_k_chunks * 2)

        if not dense_chunks and not bm25_chunks:
            logger.warning(f"No matching chunks found for query: {query}")
            return []

        # 3. Hybrid Candidate Pooling & Score Fusion
        pool_by_chunk_id: Dict[int, Dict[str, Any]] = {}

        for c in dense_chunks:
            cid = c.get("chunk_id", hash(c.get("text", "")))
            c_copy = dict(c)
            c_copy["dense_score"] = float(c.get("similarity_score", 0.0))
            c_copy["bm25_score"] = 0.0
            pool_by_chunk_id[cid] = c_copy

        for c in bm25_chunks:
            cid = c.get("chunk_id", hash(c.get("text", "")))
            bm_score = float(c.get("bm25_score", 0.0))
            if cid in pool_by_chunk_id:
                pool_by_chunk_id[cid]["bm25_score"] = bm_score
            else:
                c_copy = dict(c)
                c_copy["dense_score"] = 0.0
                c_copy["bm25_score"] = bm_score
                pool_by_chunk_id[cid] = c_copy

        # Calculate combined hybrid scores
        fused_chunks = list(pool_by_chunk_id.values())
        for chunk in fused_chunks:
            d_score = chunk["dense_score"]
            b_score = chunk["bm25_score"]
            if self.enable_hybrid:
                chunk["combined_score"] = (self.dense_weight * d_score) + (self.bm25_weight * b_score)
            else:
                chunk["combined_score"] = d_score

        # Sort pool descending
        fused_chunks.sort(key=lambda c: c["combined_score"], reverse=True)
        top_candidates = fused_chunks[:top_k_chunks]

        # 4. Cross-Encoder Re-Ranking (if available)
        cross_encoder = self._get_cross_encoder()
        if cross_encoder is not None and top_candidates:
            try:
                pairs = [[query, c.get("text", "")[:1000]] for c in top_candidates]
                ce_scores = cross_encoder.predict(pairs)
                # Normalize CE scores with sigmoid or min-max
                ce_norm = 1.0 / (1.0 + np.exp(-np.array(ce_scores)))
                for c, ce_s in zip(top_candidates, ce_norm):
                    c["cross_encoder_score"] = float(ce_s)
                    # Blend cross-encoder score with initial retrieval score
                    c["combined_score"] = float(0.6 * c["combined_score"] + 0.4 * ce_s)
                top_candidates.sort(key=lambda c: c["combined_score"], reverse=True)
            except Exception as e:
                logger.warning(f"Cross-encoder scoring encountered exception: {e}")

        # Group chunks by doc_id
        doc_chunk_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for chunk in top_candidates:
            doc_id = chunk.get("doc_id")
            if doc_id is not None:
                doc_chunk_map[doc_id].append(chunk)

        # Retrieve document details & tags from SQLite
        doc_ids = list(doc_chunk_map.keys())
        if not doc_ids:
            return []

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
                doc_tags_map[d_id][dim].append({
                    "category": cat,
                    "confidence": conf,
                    "method": tag["method"],
                })

        # Rank documents
        aggregated_docs = []
        for doc_id, chunks in doc_chunk_map.items():
            if doc_id not in doc_records:
                continue

            doc_info = doc_records[doc_id]
            best_chunk = max(chunks, key=lambda c: c["combined_score"])
            avg_score = float(np.mean([c["combined_score"] for c in chunks]))
            peak_dense = max(c.get("dense_score", 0.0) for c in chunks)

            # Null rejection check: is this below the calibrated 0.50 threshold?
            is_below_null = peak_dense < self.null_threshold

            aggregated_docs.append({
                "doc_id": doc_id,
                "celex": doc_info["celex"],
                "title": doc_info["title"],
                "date": doc_info["date"],
                "type": doc_info["type"],
                "source": doc_info["source"],
                "peak_score": round(best_chunk["combined_score"], 4),
                "peak_dense_score": round(peak_dense, 4),
                "avg_score": round(avg_score, 4),
                "is_below_null_threshold": is_below_null,
                "null_threshold": self.null_threshold,
                "best_chunk": best_chunk["text"],
                "chunks": chunks,
                "tags": doc_tags_map[doc_id],
                "full_text_excerpt": doc_info["full_text"][:1200] + ("..." if len(doc_info["full_text"]) > 1200 else ""),
            })

        # Sort descending by peak score
        aggregated_docs.sort(key=lambda d: d["peak_score"], reverse=True)
        return aggregated_docs[:top_n_docs]

    def search_similar_documents(self, celex: str, top_n_docs: int = 5) -> List[Dict[str, Any]]:
        """Finds documents semantically similar to a given CELEX legal act."""
        with self._get_db_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT title, full_text FROM documents WHERE celex = ?", (celex,))
            row = cursor.fetchone()

        if not row:
            logger.warning(f"CELEX '{celex}' not found in database for similarity search.")
            return []

        seed_text = f"{row['title']}. {row['full_text'][:1500]}"
        return self.search(query=seed_text, top_n_docs=top_n_docs + 1)
