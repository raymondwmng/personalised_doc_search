"""Tagger module implementing V1 TF-IDF and EuroVoc mapping into a 3-dimensional taxonomy."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TaxonomyTagger:
    """Classifies legal documents into Jurisdiction, Regulatory Topics, and Industry/Sector dimensions."""

    def __init__(self, db_path: Path = config.DOCUMENTS_DB_PATH):
        self.db_path = db_path
        self.taxonomy = config.TAXONOMY
        self.taxonomy_keywords = config.TAXONOMY_KEYWORDS
        self.init_db()
        self._build_category_prototypes()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initialises the document_tags table."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                dimension TEXT,
                category TEXT,
                confidence_score REAL,
                method TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
            )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_doc_dim ON document_tags(doc_id, dimension)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_category ON document_tags(dimension, category)")
            conn.commit()

    def _build_category_prototypes(self) -> None:
        """Builds prototype text strings for each taxonomy category to compute TF-IDF profiles."""
        self.category_prototypes: Dict[str, Dict[str, str]] = {}
        all_prototype_texts = []

        for dimension, categories in self.taxonomy.items():
            self.category_prototypes[dimension] = {}
            for cat in categories:
                keywords = self.taxonomy_keywords.get(dimension, {}).get(cat, [])
                # Repeat category name and keywords to form a rich representative document
                proto_text = f"{cat} " * 3 + " ".join(keywords * 2)
                self.category_prototypes[dimension][cat] = proto_text
                all_prototype_texts.append(proto_text)

        # Fit TF-IDF on prototype texts with sublinear tf and english stop words
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            min_df=1,
        )
        self.vectorizer.fit(all_prototype_texts)

        # Cache prototype vectors
        self.prototype_vectors: Dict[str, Dict[str, np.ndarray]] = {}
        for dimension, cat_dict in self.category_prototypes.items():
            self.prototype_vectors[dimension] = {}
            for cat, text in cat_dict.items():
                vec = self.vectorizer.transform([text])
                self.prototype_vectors[dimension][cat] = vec

    def map_eurovoc_raw(self, eurovoc_raw: str) -> List[Tuple[str, str, float]]:
        """Extracts taxonomy tags if raw EuroVoc concept labels or URIs match keywords."""
        tags = []
        if not eurovoc_raw:
            return tags

        lower_raw = eurovoc_raw.lower()
        for dimension, categories in self.taxonomy_keywords.items():
            for cat, kw_list in categories.items():
                for kw in kw_list:
                    if kw in lower_raw:
                        tags.append((dimension, cat, 0.95))
                        break
        return tags

    def tag_text(
        self,
        text: str,
        eurovoc_raw: str = "",
        top_k_topics: int = 3,
        top_k_sectors: int = 2,
    ) -> List[Tuple[str, str, float, str]]:
        """Tags a text document across all three dimensions.
        
        Returns:
            List of (dimension, category, confidence_score, method)
        """
        results = []
        eurovoc_matches = self.map_eurovoc_raw(eurovoc_raw)
        eurovoc_found_dims = set()
        for dim, cat, conf in eurovoc_matches:
            results.append((dim, cat, conf, "eurovoc"))
            eurovoc_found_dims.add(dim)

        if not text.strip():
            return results

        # Transform document text
        doc_vec = self.vectorizer.transform([text])

        for dimension, categories in self.taxonomy.items():
            # If EuroVoc gave tags for this dimension, we still check TF-IDF for secondary tags
            scores = []
            for cat in categories:
                proto_vec = self.prototype_vectors[dimension][cat]
                sim = float(cosine_similarity(doc_vec, proto_vec)[0][0])
                scores.append((cat, sim))

            scores.sort(key=lambda x: x[1], reverse=True)

            if dimension == "Jurisdiction":
                # Assign top 1 jurisdiction
                top_cat, top_score = scores[0]
                conf = max(0.40, min(0.99, top_score * 2.0))
                if not any(r[0] == "Jurisdiction" for r in results):
                    results.append((dimension, top_cat, round(conf, 3), "tfidf"))

            elif dimension == "Regulatory Topics":
                k = top_k_topics
                assigned = 0
                for cat, sim in scores:
                    if sim > 0.05 or assigned == 0:
                        conf = max(0.35, min(0.98, sim * 2.5))
                        if not any(r[0] == dimension and r[1] == cat for r in results):
                            results.append((dimension, cat, round(conf, 3), "tfidf"))
                            assigned += 1
                    if assigned >= k:
                        break

            elif dimension == "Industry / Sector":
                k = top_k_sectors
                assigned = 0
                for cat, sim in scores:
                    if sim > 0.04 or assigned == 0:
                        conf = max(0.30, min(0.95, sim * 2.5))
                        if not any(r[0] == dimension and r[1] == cat for r in results):
                            results.append((dimension, cat, round(conf, 3), "tfidf"))
                            assigned += 1
                    if assigned >= k:
                        break

        return results

    def tag_all_documents(self) -> int:
        """Processes all documents in documents.db and writes tags to document_tags."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, celex, title, full_text, eurovoc_raw FROM documents")
            docs = cursor.fetchall()

        total_tagged = 0
        tag_records = []

        for doc in docs:
            doc_id = doc["id"]
            text = f"{doc['title']}\n\n{doc['full_text']}"
            eurovoc_raw = doc["eurovoc_raw"] or ""

            tags = self.tag_text(text, eurovoc_raw)
            for dim, cat, conf, method in tags:
                tag_records.append((doc_id, dim, cat, conf, method))

            total_tagged += 1

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM document_tags")
            cursor.executemany("""
            INSERT INTO document_tags (doc_id, dimension, category, confidence_score, method)
            VALUES (?, ?, ?, ?, ?)
            """, tag_records)
            conn.commit()

        logger.info(f"Tagged {total_tagged} documents with {len(tag_records)} total tag assignments.")
        return total_tagged
