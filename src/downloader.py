"""Loader and downloader module for the complete EUR-Lex dataset (Kaggle: puskas78/eurlex-dataset).

Replaces legacy SPARQL/Cellar scraping which was slow and produced noisy corrigenda (*R*.html).
Loads clean, full-text EU legislation with official EUROVOC descriptor annotations.
"""

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EurlexDatasetLoader:
    """Manages ingestion and loading of the comprehensive EUR-Lex dataset from Kaggle."""

    def __init__(
        self,
        kaggle_dir: Path = config.EURLEX_KAGGLE_DIR,
        kaggle_csv: Path = config.EURLEX_KAGGLE_CSV,
        raw_dir: Path = config.RAW_DIR,
        processed_dir: Path = config.PROCESSED_DIR,
    ):
        self.kaggle_dir = kaggle_dir
        self.kaggle_csv = kaggle_csv
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.metadata_file = self.processed_dir / "metadata.jsonl"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_corrigendum(celex: str) -> bool:
        """Checks if a CELEX identifier represents a Corrigendum / Rectificatif (e.g. 32014R0668R(02))."""
        if not celex:
            return True
        return bool(re.search(r"R\(\d+\)|R\d+$", celex))

    def check_dataset_exists(self) -> bool:
        """Checks if the Kaggle EurLex CSV dataset is present locally."""
        if self.kaggle_csv.exists() and self.kaggle_csv.stat().st_size > 1000:
            return True
        # Check if zip exists
        zip_path = self.kaggle_dir / "eurlex_dataset_from_kaggle.zip"
        if zip_path.exists():
            logger.info(f"Found {zip_path}, unzipping...")
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(self.kaggle_dir)
            return self.kaggle_csv.exists()
        return False

    def download_from_kaggle(self) -> bool:
        """Attempts to download the dataset using the Kaggle CLI if not already present."""
        if self.check_dataset_exists():
            logger.info(f"Dataset already exists at {self.kaggle_csv}")
            return True

        self.kaggle_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Attempting to download 'puskas78/eurlex-dataset' via Kaggle CLI...")
        try:
            cmd = [
                "kaggle", "datasets", "download",
                "-d", "puskas78/eurlex-dataset",
                "--unzip",
                "-p", str(self.kaggle_dir)
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                logger.info(f"Successfully downloaded dataset to {self.kaggle_dir}")
                return self.check_dataset_exists()
            else:
                logger.error(f"Kaggle CLI failed: {res.stderr}")
                logger.warning(
                    "Please download the dataset manually from: "
                    "https://www.kaggle.com/datasets/puskas78/eurlex-dataset\n"
                    f"and extract it to {self.kaggle_dir}"
                )
                return False
        except FileNotFoundError:
            logger.warning(
                "Kaggle CLI not installed or kaggle.json not configured.\n"
                "Please download the dataset from https://www.kaggle.com/datasets/puskas78/eurlex-dataset\n"
                f"and extract EurLex_all.csv to {self.kaggle_dir}"
            )
            return False

    def stream_documents(
        self,
        csv_path: Optional[Path] = None,
        limit: Optional[int] = None,
        eurovoc_only: bool = True,
        act_type: Optional[str] = None,
        min_text_len: int = 150,
        skip_corrigenda: bool = True,
        chunk_size: int = 10000,
    ) -> Generator[Dict[str, Any], None, None]:
        """Yields clean document dictionaries parsed from the Kaggle EUR-Lex CSV.
        
        Args:
            csv_path: Path to CSV (defaults to config.EURLEX_KAGGLE_CSV)
            limit: Maximum documents to return (None or <= 0 for all)
            eurovoc_only: If True, only yield documents with EuroVoc labels
            act_type: Filter by act type (e.g. 'Regulation', 'Directive', 'Decision')
            min_text_len: Minimum text character count
            skip_corrigenda: If True, skips rectificatifs / corrigenda (*R*.html noise)
            chunk_size: Pandas chunk size for memory-efficient iteration
        """
        path = csv_path or self.kaggle_csv
        if not path.exists():
            raise FileNotFoundError(f"EUR-Lex CSV not found at {path}")

        cols = [
            "CELEX", "Act_name", "Act_type", "Date_document",
            "EUROVOC", "Subject_matter", "Eurlex_link", "act_raw_text"
        ]

        count = 0
        reader = pd.read_csv(path, usecols=cols, chunksize=chunk_size, low_memory=False)

        for chunk in reader:
            for _, row in chunk.iterrows():
                celex = str(row["CELEX"]).strip() if pd.notna(row["CELEX"]) else ""
                if not celex or celex.lower() == "nan":
                    continue

                if skip_corrigenda and self.is_corrigendum(celex):
                    continue

                raw_text = str(row["act_raw_text"]).strip() if pd.notna(row["act_raw_text"]) else ""
                if len(raw_text) < min_text_len or raw_text.lower() == "nan":
                    continue

                eurovoc_raw = str(row["EUROVOC"]).strip() if pd.notna(row["EUROVOC"]) else ""
                if eurovoc_only and (not eurovoc_raw or eurovoc_raw.lower() == "nan"):
                    continue

                curr_type = str(row["Act_type"]).strip() if pd.notna(row["Act_type"]) else "Regulation"
                if act_type and curr_type.lower() != act_type.lower():
                    continue

                title = str(row["Act_name"]).strip() if pd.notna(row["Act_name"]) else f"Document {celex}"
                date = str(row["Date_document"]).strip() if pd.notna(row["Date_document"]) else "2026-01-01"
                subject_matter = str(row["Subject_matter"]).strip() if pd.notna(row["Subject_matter"]) else ""
                url = str(row["Eurlex_link"]).strip() if pd.notna(row["Eurlex_link"]) else f"https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:{celex}"

                doc = {
                    "celex": celex,
                    "title": title,
                    "date": date,
                    "type": curr_type,
                    "eurovoc_raw": eurovoc_raw,
                    "subject_matter": subject_matter,
                    "html_url": url,
                    "text": raw_text,
                    "source": "eurlex_kaggle",
                }

                yield doc
                count += 1
                if limit and limit > 0 and count >= limit:
                    return

    def load_and_process(
        self,
        limit: int = 1000,
        eurovoc_only: bool = True,
        act_type: Optional[str] = None,
        save_metadata: bool = True,
        save_raw_txt: bool = False,
        ingest_to_db: bool = False,
    ) -> List[Dict[str, Any]]:
        """Loads documents from Kaggle CSV and saves to metadata.jsonl, raw files, and/or SQLite."""
        if not self.check_dataset_exists():
            success = self.download_from_kaggle()
            if not success:
                raise RuntimeError(
                    f"Kaggle EUR-Lex dataset not found at {self.kaggle_csv}. Please see instructions."
                )

        logger.info(
            f"Loading EUR-Lex documents from {self.kaggle_csv.name} "
            f"(limit={limit}, eurovoc_only={eurovoc_only}, act_type={act_type})..."
        )

        doc_parser = None
        if ingest_to_db:
            from src.parser import LegalDocParser
            doc_parser = LegalDocParser()

        docs = []
        meta_out = None
        if save_metadata:
            meta_out = open(self.metadata_file, "w", encoding="utf-8")

        try:
            for idx, doc in enumerate(
                self.stream_documents(limit=limit, eurovoc_only=eurovoc_only, act_type=act_type), 1
            ):
                docs.append(doc)

                if meta_out:
                    # Write metadata record (excluding full text to keep metadata.jsonl lightweight)
                    meta_record = {k: v for k, v in doc.items() if k != "text"}
                    meta_out.write(json.dumps(meta_record, ensure_ascii=False) + "\n")

                if save_raw_txt:
                    raw_txt_path = self.raw_dir / f"{doc['celex']}.txt"
                    raw_txt_path.write_text(doc["text"], encoding="utf-8")

                if doc_parser:
                    doc_parser.process_and_store_document(
                        celex=doc["celex"],
                        title=doc["title"],
                        date=doc["date"],
                        doc_type=doc["type"],
                        raw_text_or_html=doc["text"],
                        eurovoc_raw=doc["eurovoc_raw"],
                        source=doc["source"],
                        is_html=False,
                    )

                if idx % 500 == 0:
                    logger.info(f"Loaded and processed {idx} documents...")

        finally:
            if meta_out:
                meta_out.close()

        logger.info(f"Successfully loaded and processed {len(docs)} clean EUR-Lex documents.")
        return docs

    def ingest_local_jsonl(self, jsonl_path: Path) -> List[Dict[str, Any]]:
        """Ingests local test benchmark JSONL files if present."""
        if not jsonl_path.exists():
            logger.warning(f"File {jsonl_path} does not exist.")
            return []

        logger.info(f"Ingesting local benchmark dataset from {jsonl_path}")
        existing_celexes = set()
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            if "celex" in rec:
                                existing_celexes.add(rec["celex"])
                        except Exception:
                            pass

        ingested = []
        with open(jsonl_path, "r", encoding="utf-8") as f, open(self.metadata_file, "a", encoding="utf-8") as meta_out:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                doc_id = item.get("id") or item.get("celex", "DOC_UNKNOWN")
                celex = item.get("celex", doc_id)
                title = item.get("title", f"Legal Document {celex}")
                content = item.get("content", item.get("full_text", ""))
                doc_type = item.get("document_type", item.get("type", "Regulation"))

                raw_txt_path = self.raw_dir / f"{celex}.txt"
                if not raw_txt_path.exists():
                    raw_txt_path.write_text(content, encoding="utf-8")

                record = {
                    "celex": celex,
                    "title": title,
                    "date": item.get("date", "2026-01-01"),
                    "type": doc_type,
                    "eurovoc_raw": item.get("eurovoc", ""),
                    "html_url": f"local://{jsonl_path.name}/{doc_id}",
                    "source": "local_jsonl",
                }

                if celex not in existing_celexes:
                    meta_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    existing_celexes.add(celex)

                ingested.append(record)

        logger.info(f"Ingested {len(ingested)} documents from local dataset.")
        return ingested


# Alias for backwards compatibility
EurlexDownloader = EurlexDatasetLoader
