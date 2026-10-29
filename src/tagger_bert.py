"""Legal-BERT and zero-shot taxonomy classifier for EUR-Lex legal documents (Section 4.2)."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Detailed statutory descriptions for zero-shot semantic matching
CATEGORY_SEMANTIC_DESCRIPTIONS = {
    "Jurisdiction": {
        "EU-wide": "European Union law applicable across all EU member states, internal market harmonisation, union legislation.",
        "Member State specific": "Provisions directed at specific member state national authorities, national designated bodies or domestic implementation.",
        "International": "Treaties, conventions, and bilateral agreements between the European Union and third countries, WTO, international law.",
        "Cross-border": "Mutual assistance, cross-border police and judicial cooperation, extradition, and transnational legal enforcement.",
    },
    "Regulatory Topics": {
        "Data Protection and Privacy": "General Data Protection Regulation GDPR, personal data processing, data subject rights, supervisory authorities, consent.",
        "Competition and Antitrust": "Articles 101 and 102 TFEU, State aid, anti-competitive agreements, cartels, abuse of dominant position, mergers.",
        "Consumer Protection": "Consumer rights, unfair commercial practices, product liability, guarantees, consumer dispute resolution, withdrawal rights.",
        "Financial Regulation": "Banking union, prudential capital requirements, credit institutions, MiFID, securities, liquidity, solvency framework.",
        "Employment Law": "Worker rights, working time directive, occupational health and safety, dismissals, collective bargaining, equal pay.",
        "Environmental Law": "Greenhouse gas emissions, biodiversity preservation, industrial pollution, waste management, circular economy, water quality.",
        "Trade and Customs": "Union Customs Code, tariffs, import and export regulations, anti-dumping duties, rules of origin, international trade.",
        "Intellectual Property": "EU trademarks, patents, copyright, geographical indications, design rights, enforcement against counterfeiting.",
        "Digital Services": "Digital Services Act DSA, Digital Markets Act DMA, online platforms, gatekeepers, algorithmic transparency, AI regulation.",
        "Health and Safety": "Public health protection, medicinal products, clinical trials, medical devices, epidemic emergency measures, food safety.",
        "Tax": "Value added tax VAT, excise duties, direct corporate taxation, tax avoidance, tax fraud prevention, withholding tax.",
        "Corporate Governance": "Statutory auditing, company law, annual financial statements, shareholder rights, corporate transparency and reporting.",
    },
    "Industry / Sector": {
        "Agriculture and Food": "Common Agricultural Policy CAP, fisheries, food safety standards, livestock, crops, pesticides, veterinary rules.",
        "Energy": "Internal energy market, electricity grid, natural gas, renewable energy sources, nuclear safety, energy efficiency, hydrogen.",
        "Transport": "Aviation civil safety, railways, maritime navigation, road transport freight, passenger rights, transport infrastructure.",
        "Finance and Banking": "Commercial banking, investment firms, insurance undertakings, fintech, payment services, asset management.",
        "Technology and Digital": "Software engineering, cloud computing, cybersecurity, semiconductor manufacturing, artificial intelligence systems, data spaces.",
        "Healthcare and Pharma": "Pharmaceutical manufacturing, hospitals, clinical medicine, biotech therapies, patient care, medical devices.",
        "Manufacturing": "Industrial machinery, chemicals regulation REACH, manufacturing equipment standards, raw materials, heavy industry.",
        "Telecommunications": "Radio frequency spectrum allocation, electronic communications networks, 5G broadband, mobile roaming.",
        "Environment and Climate": "Climate change mitigation, Paris Agreement, nature restoration, carbon neutrality, renewable resources.",
        "Education": "Higher education, vocational qualifications framework, academic mobility, Erasmus, university degree recognition.",
        "Defence and Security": "Defence procurement, border surveillance, counter-terrorism, dual-use goods export control, sanctions.",
    },
}


class LegalBertTagger:
    """Semantic legal taxonomy classifier powered by transformer representations.
    
    Computes cosine similarity between document semantic vectors and canonical
    statutory descriptions across Jurisdiction, Regulatory Topics, and Industry/Sector.
    """

    def __init__(
        self,
        model_name: str = config.LEGAL_BERT_MODEL,
        db_path: Path = config.DOCUMENTS_DB_PATH,
        device: Optional[str] = None,
        confidence_threshold: float = 0.35,
    ):
        self.model_name = model_name
        self.db_path = db_path
        self.confidence_threshold = confidence_threshold

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing LegalBertTagger on {self.device}...")
        # Use sentence transformer encoder for category similarity matching
        # (Fallback gracefully to modernbert or bge-m3 if legal-bert is not cached)
        try:
            self.encoder = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            fallback = "BAAI/bge-m3" if Path("~/.cache/huggingface/hub/models--BAAI--bge-m3").expanduser().exists() else "nomic-ai/modernbert-embed-base"
            logger.warning(f"Could not load '{self.model_name}' ({e}). Falling back to '{fallback}' for semantic tagging.")
            self.encoder = SentenceTransformer(fallback, device=self.device)

        self.init_db()
        self._encode_category_prototypes()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
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

    def _encode_category_prototypes(self) -> None:
        """Encodes canonical category descriptions into normalised vectors."""
        self.category_vectors: Dict[str, Dict[str, np.ndarray]] = {}

        for dimension, categories in CATEGORY_SEMANTIC_DESCRIPTIONS.items():
            self.category_vectors[dimension] = {}
            cat_names = list(categories.keys())
            cat_texts = [f"{name}: {categories[name]}" for name in cat_names]
            vecs = self.encoder.encode(cat_texts, normalize_embeddings=True, show_progress_bar=False)

            for name, vec in zip(cat_names, vecs):
                self.category_vectors[dimension][name] = vec

        logger.info("Category prototype vectors computed across all 3 dimensions.")

    def tag_text(self, text: str, top_k_per_dim: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        """Tags a text excerpt with confidence scores across all 3 dimensions."""
        if not text or not text.strip():
            return {dim: [] for dim in CATEGORY_SEMANTIC_DESCRIPTIONS}

        sample = text[:3000].strip()
        doc_vec = self.encoder.encode([sample], normalize_embeddings=True, show_progress_bar=False)[0]

        tags: Dict[str, List[Tuple[str, float]]] = {}

        for dimension, cat_dict in self.category_vectors.items():
            scores = []
            for cat_name, cat_vec in cat_dict.items():
                cos_sim = float(np.dot(doc_vec, cat_vec))
                if cos_sim >= self.confidence_threshold:
                    scores.append((cat_name, round(cos_sim, 4)))

            scores.sort(key=lambda x: x[1], reverse=True)
            tags[dimension] = scores[:top_k_per_dim]

        return tags

    def tag_all_documents(
        self,
        batch_size: int = 250,
        limit: Optional[int] = None,
        overwrite: bool = True,
    ) -> int:
        """Tags documents in documents.db using Legal-BERT semantic classification."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT id, title, full_text FROM documents"
            if limit:
                query += f" LIMIT {limit}"
            cursor.execute(query)
            docs = cursor.fetchall()

        if not docs:
            logger.warning("No documents found to tag.")
            return 0

        logger.info(f"Tagging {len(docs)} documents with Legal-BERT...")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            if overwrite and (limit is None or limit >= len(docs)):
                cursor.execute("DELETE FROM document_tags WHERE method = 'legal_bert'")
                conn.commit()

            pending_tags = []
            tagged_count = 0

            for i, doc in enumerate(docs, 1):
                doc_id = doc["id"]
                title = doc["title"] or ""
                full_text = doc["full_text"] or ""
                doc_sample = f"{title}. {full_text[:2500]}"

                tags_by_dim = self.tag_text(doc_sample)

                for dim, cat_scores in tags_by_dim.items():
                    for cat, score in cat_scores:
                        pending_tags.append((doc_id, dim, cat, score, "legal_bert"))

                if len(pending_tags) >= batch_size * 3:
                    cursor.executemany("""
                    INSERT INTO document_tags (doc_id, dimension, category, confidence_score, method)
                    VALUES (?, ?, ?, ?, ?)
                    """, pending_tags)
                    conn.commit()
                    pending_tags = []

                tagged_count += 1
                if i % 500 == 0 or i == len(docs):
                    logger.info(f"Tagged {i}/{len(docs)} documents with Legal-BERT...")

            if pending_tags:
                cursor.executemany("""
                INSERT INTO document_tags (doc_id, dimension, category, confidence_score, method)
                VALUES (?, ?, ?, ?, ?)
                """, pending_tags)
                conn.commit()

        logger.info(f"Legal-BERT tagging complete. Tagged {tagged_count} documents.")
        return tagged_count

