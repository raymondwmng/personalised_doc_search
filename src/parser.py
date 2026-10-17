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

    def chunk_text(
        self,
        text: str,
        target_tokens: int = config.CHUNK_SIZE_TOKENS,
        overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
    ) -> List[Tuple[int, str, int]]:
        """Splits text into chunks of ~target_tokens with ~overlap_tokens overlap.
        
        Returns:
            List of (chunk_index, chunk_text, token_count)
        """
        if not text.strip():
            return []

        # Target character window calculated from token targets
        target_chars = target_tokens * 4
        overlap_chars = overlap_tokens * 4

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk_paragraphs = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)
            if current_length + para_len > target_chars and current_chunk_paragraphs:
                # Flush current chunk
                chunk_str = "\n\n".join(current_chunk_paragraphs).strip()
                tokens = self.approximate_token_count(chunk_str)
                chunks.append((len(chunks), chunk_str, tokens))

                # Slide window with overlap: keep the tail paragraphs
                keep_paragraphs = []
                keep_length = 0
                for p in reversed(current_chunk_paragraphs):
                    if keep_length + len(p) <= overlap_chars:
                        keep_paragraphs.insert(0, p)
                        keep_length += len(p)
                    else:
                        break
                current_chunk_paragraphs = keep_paragraphs
                current_length = keep_length

            current_chunk_paragraphs.append(para)
            current_length += para_len

        # Flush remainder
        if current_chunk_paragraphs:
            chunk_str = "\n\n".join(current_chunk_paragraphs).strip()
            tokens = self.approximate_token_count(chunk_str)
            chunks.append((len(chunks), chunk_str, tokens))

        return chunks

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

        chunks = self.chunk_text(clean_text)

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
