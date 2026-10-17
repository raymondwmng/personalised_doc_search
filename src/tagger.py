"""Tagger module implementing V1 TF-IDF and EuroVoc mapping into a 3-dimensional taxonomy."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

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
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
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

        # Cache prototype vectors and stacked prototype matrices for high-throughput dot-product matching
        self.prototype_vectors: Dict[str, Dict[str, np.ndarray]] = {}
        self.prototype_matrices: Dict[str, Any] = {}

        for dimension, cat_dict in self.category_prototypes.items():
            self.prototype_vectors[dimension] = {}
            cats = self.taxonomy[dimension]
            cat_texts = []
            for cat in cats:
                text = cat_dict[cat]
                vec = self.vectorizer.transform([text])
                self.prototype_vectors[dimension][cat] = vec
                cat_texts.append(text)

            # Stacked prototype matrix of shape (num_categories, vocab_size)
            self.prototype_matrices[dimension] = self.vectorizer.transform(cat_texts)

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
        for dim, cat, conf in eurovoc_matches:
            results.append((dim, cat, conf, "eurovoc"))

        if not text.strip():
            return results

        # Transform document text
        doc_vec = self.vectorizer.transform([text])

        for dimension, categories in self.taxonomy.items():
            # Fast cosine similarity via sparse matrix dot product
            sims = doc_vec.dot(self.prototype_matrices[dimension].T).toarray()[0]
            scores = [(categories[idx], float(sims[idx])) for idx in range(len(categories))]
            scores.sort(key=lambda x: x[1], reverse=True)

            if dimension == "Jurisdiction":
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

    def tag_all_documents(self, batch_size: int = 500) -> int:
        """Processes all documents in documents.db in streaming batches, using vectorized

        matrix operations and committing incrementally to SQLite.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS cnt FROM documents")
            total_docs = cursor.fetchone()["cnt"]

            if total_docs == 0:
                logger.info("No documents found in documents.db to tag.")
                return 0

            # Clear existing tags
            cursor.execute("DELETE FROM document_tags")
            conn.commit()

        logger.info(f"Tagging {total_docs} documents in batches of {batch_size}...")

        total_tagged = 0
        total_tags_saved = 0

        with self.get_connection() as conn:
            cursor = conn.cursor()
            last_id = 0

            with tqdm(total=total_docs, desc="🏷️  Tagging documents", unit="doc") as pbar:
                while True:
                    cursor.execute("""
                        SELECT id, title, full_text, eurovoc_raw
                        FROM documents
                        WHERE id > ?
                        ORDER BY id ASC
                        LIMIT ?
                    """, (last_id, batch_size))
                    batch = cursor.fetchall()
                    if not batch:
                        break

                    last_id = batch[-1]["id"]

                    # Vectorize batch texts at once for high throughput
                    batch_texts = [
                        f"{doc['title'] or ''}\n\n{doc['full_text'] or ''}".strip()
                        for doc in batch
                    ]
                    doc_mat = self.vectorizer.transform(batch_texts)

                    # Compute similarity matrices for each dimension
                    sim_matrices = {}
                    for dim in self.taxonomy:
                        # (batch_size, num_categories_in_dimension)
                        sim_matrices[dim] = doc_mat.dot(self.prototype_matrices[dim].T).toarray()

                    batch_tag_records = []

                    for idx, doc in enumerate(batch):
                        doc_id = doc["id"]
                        eurovoc_raw = doc["eurovoc_raw"] or ""
                        doc_tags = []

                        # 1. EuroVoc direct mapping
                        eurovoc_matches = self.map_eurovoc_raw(eurovoc_raw)
                        for dim, cat, conf in eurovoc_matches:
                            doc_tags.append((doc_id, dim, cat, conf, "eurovoc"))

                        # 2. Vectorized TF-IDF prototype matching
                        for dim, categories in self.taxonomy.items():
                            sims = sim_matrices[dim][idx]
                            scores = [(categories[c_idx], float(sims[c_idx])) for c_idx in range(len(categories))]
                            scores.sort(key=lambda x: x[1], reverse=True)

                            if dim == "Jurisdiction":
                                top_cat, top_score = scores[0]
                                conf = max(0.40, min(0.99, top_score * 2.0))
                                if not any(r[1] == "Jurisdiction" for r in doc_tags):
                                    doc_tags.append((doc_id, dim, top_cat, round(conf, 3), "tfidf"))

                            elif dim == "Regulatory Topics":
                                assigned = 0
                                for cat, sim in scores:
                                    if sim > 0.05 or assigned == 0:
                                        conf = max(0.35, min(0.98, sim * 2.5))
                                        if not any(r[1] == dim and r[2] == cat for r in doc_tags):
                                            doc_tags.append((doc_id, dim, cat, round(conf, 3), "tfidf"))
                                            assigned += 1
                                    if assigned >= 3:
                                        break

                            elif dim == "Industry / Sector":
                                assigned = 0
                                for cat, sim in scores:
                                    if sim > 0.04 or assigned == 0:
                                        conf = max(0.30, min(0.95, sim * 2.5))
                                        if not any(r[1] == dim and r[2] == cat for r in doc_tags):
                                            doc_tags.append((doc_id, dim, cat, round(conf, 3), "tfidf"))
                                            assigned += 1
                                    if assigned >= 2:
                                        break

                        batch_tag_records.extend(doc_tags)

                    # Persist batch to SQLite
                    cursor.executemany("""
                    INSERT INTO document_tags (doc_id, dimension, category, confidence_score, method)
                    VALUES (?, ?, ?, ?, ?)
                    """, batch_tag_records)
                    conn.commit()

                    total_tagged += len(batch)
                    total_tags_saved += len(batch_tag_records)

                    pbar.update(len(batch))
                    pbar.set_postfix({"tags_saved": total_tags_saved, "batch_size": len(batch)})

        logger.info(f"Tagged {total_tagged} documents with {total_tags_saved} total tag assignments.")
        return total_tagged
