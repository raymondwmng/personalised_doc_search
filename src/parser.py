"""Parser and chunker for EUR-Lex and legal documents, storing into SQLite documents.db."""

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from bs4 import BeautifulSoup

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class LegalDocParser:
    """Parses raw HTML documents, extracts clean text/articles, chunks text, and populates SQLite."""

    def __init__(self, db_path: Path = config.DOCUMENTS_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initialises database schema for documents and chunks."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                celex TEXT UNIQUE,
                title TEXT,
                date TEXT,
                type TEXT,
                full_text TEXT,
                eurovoc_raw TEXT,
                source TEXT
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                chunk_idx INTEGER,
                text TEXT,
                token_count INTEGER,
                FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
            )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_celex ON documents(celex)")
            conn.commit()

    def clean_html(self, html_content: str) -> str:
        """Strips HTML markup while preserving paragraph and article structural boundaries."""
        if not html_content or not html_content.strip():
            return ""

        soup = BeautifulSoup(html_content, "lxml")

        # Strip non-content elements
        for element in soup(["script", "style", "nav", "footer", "header", "noscript", "meta"]):
            element.decompose()

        # Normalise line breaks and paragraphs
        text_blocks = []
        for tag in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "li", "tr"]):
            block_text = tag.get_text(" ", strip=True)
            if block_text and len(block_text) > 10:
                text_blocks.append(block_text)

        if not text_blocks:
            # Fallback to entire stripped string
            cleaned = soup.get_text(separator="\n", strip=True)
        else:
            cleaned = "\n\n".join(text_blocks)

        # Normalise whitespace
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
        return cleaned.strip()

    def approximate_token_count(self, text: str) -> int:
        """Fast token count approximation (~1 token = ~0.75 words / 4 chars)."""
        words = text.split()
        return max(1, int(len(words) * 1.3))

    def detect_document_complexity(self, html_or_text: str) -> Dict[str, Any]:
        """Detects layout and structural complexity for Document Intelligence routing."""
        table_count = len(re.findall(r"<table\b", html_or_text, re.IGNORECASE))
        row_count = len(re.findall(r"<tr\b", html_or_text, re.IGNORECASE))
        annex_count = len(re.findall(r"(?:Annex|ANNEX)\s+[IVXLCDM\d]+", html_or_text))
        article_count = len(re.findall(r"(?:Article|ARTICLE)\s+\d+", html_or_text))
        is_complex = table_count >= 2 or row_count >= 10 or annex_count >= 2
        return {
            "is_complex": is_complex,
            "table_count": table_count,
            "row_count": row_count,
            "annex_count": annex_count,
            "article_count": article_count,
            "recommended_extractor": "azure_layout" if is_complex else "html_parser",
        }

    def chunk_text(
        self,
        text: str,
        celex: str = "",
        title: str = "",
        target_tokens: int = config.CHUNK_SIZE_TOKENS,
        overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
        regime: str = "meso",
    ) -> List[Tuple[int, str, int]]:
        """Splits text into chunks under the specified granularity regime.
        
        Args:
            text: Clean document text
            celex: Document CELEX identifier for provenance header
            title: Document title
            target_tokens: Target chunk size (default: 2,000 for Meso, Rec 3)
            overlap_tokens: Overlap size (default: 200 for Meso)
            regime: 'meso' (~2000 tokens, thematic chapter chunks with provenance),
                    'micro' (~400 tokens, article-level with structural boundaries),
                    'macro' (full-act up to 32K tokens)
        
        Returns:
            List of (chunk_index, chunk_text, token_count)
        """
        if not text.strip():
            return []

        if regime == "macro":
            tokens = self.approximate_token_count(text)
            header = f"Document {celex} — {title}\n\n" if (celex or title) else ""
            return [(0, f"{header}{text}".strip(), tokens)]

        if regime == "micro":
            # Article-level splitting on structural boundaries
            pattern = r"(?=(?:(?:Article|ARTICLE)\s+\d+|(?:Whereas|WHEREAS)\s*\(?\d*\)?|(?:Chapter|CHAPTER)\s+[IVXLCDM\d]+))"
            sections = [s.strip() for s in re.split(pattern, text) if s.strip()]
            if not sections:
                sections = [p.strip() for p in text.split("\n\n") if p.strip()]
            target_chars = target_tokens * 4
            overlap_chars = overlap_tokens * 4
            chunks = []
            current_buf = []
            current_len = 0

            for sec in sections:
                sec_len = len(sec)
                if current_len + sec_len > target_chars and current_buf:
                    chunk_str = "\n\n".join(current_buf).strip()
                    tokens = self.approximate_token_count(chunk_str)
                    chunks.append((len(chunks), chunk_str, tokens))
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
                chunk_str = "\n\n".join(current_buf).strip()
                tokens = self.approximate_token_count(chunk_str)
                chunks.append((len(chunks), chunk_str, tokens))
            return chunks

        # Default: Regime B (Meso ~2000 tokens standard chunking, Rec 3)
        target_chars = target_tokens * 4
        overlap_chars = overlap_tokens * 4
        provenance_header = f"Document {celex} — {title}\n\n" if (celex or title) else ""

        # First try splitting by paragraphs
        raw_sections = [p.strip() for p in text.split("\n\n") if p.strip()]

        # If text is monolithic (e.g. collapsed newlines in raw dataset), split on structural legal boundaries
        if len(raw_sections) <= 1:
            pattern = r"(?=(?:(?:Article|ARTICLE)\s+\d+|(?:Whereas|WHEREAS)\s*\(?\d*\)?|(?:Chapter|CHAPTER)\s+[IVXLCDM\d]+|(?:Annex|ANNEX)\s+[IVXLCDM\d]+))"
            raw_sections = [s.strip() for s in re.split(pattern, text) if s.strip()]

        # If still only 1 block and large, split on sentence boundaries
        sections = []
        for sec in raw_sections:
            if len(sec) > target_chars:
                sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", sec)
                sections.extend([s.strip() for s in sentences if s.strip()])
            else:
                sections.append(sec)

        if not sections:
            sections = [text]

        chunks = []
        current_buf = []
        current_len = 0

        for sec in sections:
            sec_len = len(sec)
            if current_len + sec_len > target_chars and current_buf:
                chunk_content = "\n\n".join(current_buf).strip()
                full_chunk_text = f"{provenance_header}{chunk_content}".strip()
                tokens = self.approximate_token_count(full_chunk_text)
                chunks.append((len(chunks), full_chunk_text, tokens))

                # Sliding overlap
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
            chunk_content = "\n\n".join(current_buf).strip()
            full_chunk_text = f"{provenance_header}{chunk_content}".strip()
            tokens = self.approximate_token_count(full_chunk_text)
            chunks.append((len(chunks), full_chunk_text, tokens))

        return chunks

    def rechunk_database(
        self,
        regime: str = "meso",
        target_tokens: int = config.CHUNK_SIZE_TOKENS,
        overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
        limit: Optional[int] = None,
        batch_commit_size: int = 500,
    ) -> Dict[str, int]:
        """Re-chunks existing documents in documents.db under the specified regime (e.g. Meso ~2000t).
        
        Returns:
            Dict containing total documents processed and total chunks created.
        """
        logger.info(f"Re-chunking database using regime='{regime}', target_tokens={target_tokens}, overlap={overlap_tokens}...")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT id, celex, title, full_text FROM documents"
            if limit:
                query += f" LIMIT {limit}"
            cursor.execute(query)
            docs = cursor.fetchall()

        if not docs:
            logger.warning("No documents found in documents table.")
            return {"docs_processed": 0, "chunks_created": 0}

        total_chunks = 0
        doc_count = len(docs)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            # If rechunking all documents, clear existing chunks
            if limit is None or limit >= doc_count:
                cursor.execute("DELETE FROM chunks")
                conn.commit()

            pending_chunk_records = []
            for i, doc in enumerate(docs, 1):
                doc_id = doc["id"]
                celex = doc["celex"] or ""
                title = doc["title"] or ""
                full_text = doc["full_text"] or ""

                if not full_text.strip():
                    continue

                if limit is not None:
                    # Clear this specific doc's chunks
                    cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

                chunks = self.chunk_text(
                    text=full_text,
                    celex=celex,
                    title=title,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                    regime=regime,
                )

                for chunk_idx, chunk_text, tok_count in chunks:
                    pending_chunk_records.append((doc_id, chunk_idx, chunk_text, tok_count))
                    total_chunks += 1

                if len(pending_chunk_records) >= batch_commit_size:
                    cursor.executemany("""
                    INSERT INTO chunks (doc_id, chunk_idx, text, token_count)
                    VALUES (?, ?, ?, ?)
                    """, pending_chunk_records)
                    conn.commit()
                    pending_chunk_records = []

                if i % 1000 == 0 or i == doc_count:
                    logger.info(f"Processed {i}/{doc_count} documents ({total_chunks} chunks generated)...")

            if pending_chunk_records:
                cursor.executemany("""
                INSERT INTO chunks (doc_id, chunk_idx, text, token_count)
                VALUES (?, ?, ?, ?)
                """, pending_chunk_records)
                conn.commit()

        logger.info(f"Re-chunking complete: {doc_count} documents -> {total_chunks} Meso chunks.")
        return {"docs_processed": doc_count, "chunks_created": total_chunks}


    def process_and_store_document(
        self,
        celex: str,
        title: str,
        date: str,
        doc_type: str,
        raw_text_or_html: str,
        eurovoc_raw: str = "",
        source: str = "eurlex",
        is_html: bool = True,
    ) -> Optional[int]:
        """Cleans, chunks, and inserts document and its chunks into SQLite."""
        if is_html:
            clean_text = self.clean_html(raw_text_or_html)
        else:
            clean_text = raw_text_or_html.strip()

        if not clean_text:
            logger.warning(f"Skipping empty document CELEX: {celex}")
            return None

        chunks = self.chunk_text(clean_text, celex=celex, title=title, regime="meso")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                INSERT INTO documents (celex, title, date, type, full_text, eurovoc_raw, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(celex) DO UPDATE SET
                    title=excluded.title,
                    date=excluded.date,
                    type=excluded.type,
                    full_text=excluded.full_text,
                    eurovoc_raw=excluded.eurovoc_raw,
                    source=excluded.source
                """, (celex, title, date, doc_type, clean_text, eurovoc_raw, source))

                cursor.execute("SELECT id FROM documents WHERE celex = ?", (celex,))
                doc_id = cursor.fetchone()["id"]

                # Remove old chunks if updating
                cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

                # Insert chunks
                chunk_records = [
                    (doc_id, idx, chunk_text, tok_count)
                    for idx, chunk_text, tok_count in chunks
                ]
                cursor.executemany("""
                INSERT INTO chunks (doc_id, chunk_idx, text, token_count)
                VALUES (?, ?, ?, ?)
                """, chunk_records)

                conn.commit()
                logger.info(f"Stored CELEX {celex} (doc_id={doc_id}) with {len(chunks)} chunks.")
                return doc_id
            except Exception as e:
                logger.error(f"Failed to store CELEX {celex}: {e}")
                conn.rollback()
                return None
