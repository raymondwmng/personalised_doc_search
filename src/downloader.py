"""Downloader module for EUR-Lex legal documents via SPARQL and Cellar/EUR-Lex HTTP endpoints."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from SPARQLWrapper import JSON, SPARQLWrapper

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

USER_AGENT = "PersonalisedEurlexSearchPOC/1.0 (Research; contact: user@example.com)"


class EurlexDownloader:
    """Manages SPARQL discovery and document retrieval from EUR-Lex / Cellar."""

    def __init__(
        self,
        sparql_endpoint: str = config.SPARQL_ENDPOINT,
        raw_dir: Path = config.RAW_DIR,
        processed_dir: Path = config.PROCESSED_DIR,
        rate_limit: float = config.REQUEST_DELAY_SECONDS,
    ):
        self.sparql_endpoint = sparql_endpoint
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.rate_limit = rate_limit
        self.metadata_file = self.processed_dir / "metadata.jsonl"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def query_metadata(
        self,
        limit: int = 500,
        offset: int = 0,
        years_back: int = 10,
        doc_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Queries EUR-Lex SPARQL endpoint for legislation metadata and manifestation links."""
        if doc_types is None:
            # CDM classes for Regulation, Directive, Decision, Recommendation
            type_clauses = """
                { ?work a cdm:regulation . BIND("Regulation" AS ?type) }
                UNION
                { ?work a cdm:directive . BIND("Directive" AS ?type) }
                UNION
                { ?work a cdm:decision . BIND("Decision" AS ?type) }
                UNION
                { ?work a cdm:recommendation . BIND("Recommendation" AS ?type) }
            """
        else:
            clauses = []
            for dt in doc_types:
                clauses.append(f'{{ ?work a cdm:{dt.lower()} . BIND("{dt}" AS ?type) }}')
            type_clauses = " UNION ".join(clauses)

        sparql_query = f"""
        PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
        SELECT DISTINCT ?work ?celex ?title ?date ?type (GROUP_CONCAT(DISTINCT ?eurovoc; separator="|") AS ?eurovoc_list) WHERE {{
            {type_clauses}
            ?work cdm:resource_legal_id_celex ?celex .
            ?work cdm:work_date_document ?date .
            OPTIONAL {{ ?work cdm:work_is_about_concept_eurovoc ?eurovoc }}
            ?expr cdm:expression_belongs_to_work ?work ;
                  cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/ENG> ;
                  cdm:expression_title ?title .
        }}
        GROUP BY ?work ?celex ?title ?date ?type
        ORDER BY DESC(?date)
        OFFSET {offset}
        LIMIT {limit}
        """

        sparql = SPARQLWrapper(self.sparql_endpoint)
        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        sparql.setTimeout(45)

        logger.info(f"Querying SPARQL (limit={limit}, offset={offset})...")
        try:
            response = sparql.query().convert()
            results = response.get("results", {}).get("bindings", [])
            docs = []
            for item in results:
                celex = item.get("celex", {}).get("value")
                title = item.get("title", {}).get("value")
                doc_date = item.get("date", {}).get("value")
                doc_type = item.get("type", {}).get("value", "Regulation")
                work_uri = item.get("work", {}).get("value")
                eurovoc_raw = item.get("eurovoc_list", {}).get("value", "")

                if celex and title:
                    docs.append({
                        "celex": celex,
                        "title": title,
                        "date": doc_date,
                        "type": doc_type,
                        "work_uri": work_uri,
                        "eurovoc_raw": eurovoc_raw,
                        "html_url": f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}",
                    })
            logger.info(f"Retrieved {len(docs)} documents from SPARQL.")
            return docs
        except Exception as e:
            logger.error(f"Error querying SPARQL endpoint: {e}")
            return []

    def download_document_html(self, celex: str, html_url: Optional[str] = None) -> Optional[str]:
        """Downloads HTML for a single document, caching to data/raw/{celex}.html."""
        target_path = self.raw_dir / f"{celex}.html"
        if target_path.exists() and target_path.stat().st_size > 500:
            logger.debug(f"File {target_path} already exists, skipping download.")
            return target_path.read_text(encoding="utf-8", errors="ignore")

        urls_to_try = []
        if html_url:
            urls_to_try.append(html_url)
        urls_to_try.append(f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}")
        urls_to_try.append(f"http://publications.europa.eu/resource/celex/{celex}")

        for url in urls_to_try:
            try:
                headers = {"Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8"}
                resp = self.session.get(url, headers=headers, timeout=20, allow_redirects=True)
                if resp.status_code == 200 and len(resp.text) > 500:
                    target_path.write_text(resp.text, encoding="utf-8", errors="ignore")
                    logger.info(f"Downloaded CELEX {celex} ({len(resp.text)} chars) from {url}")
                    time.sleep(self.rate_limit)
                    return resp.text
            except Exception as e:
                logger.warning(f"Failed fetching {url} for {celex}: {e}")

        logger.warning(f"Could not retrieve HTML for CELEX {celex}")
        return None

    def save_metadata_record(self, record: Dict[str, Any]) -> None:
        """Appends a metadata record to data/processed/metadata.jsonl."""
        with open(self.metadata_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_existing_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Loads already processed metadata records keyed by CELEX."""
        if not self.metadata_file.exists():
            return {}
        records = {}
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if "celex" in rec:
                            records[rec["celex"]] = rec
                    except Exception:
                        pass
        return records

    def ingest_local_jsonl(self, jsonl_path: Path) -> List[Dict[str, Any]]:
        """Ingests local test JSONL files (e.g. data/test3-legal-documents.jsonl) into the pipeline."""
        if not jsonl_path.exists():
            logger.warning(f"File {jsonl_path} does not exist.")
            return []

        logger.info(f"Ingesting local test dataset from {jsonl_path}")
        existing = self.load_existing_metadata()
        ingested = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
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

                raw_html_path = self.raw_dir / f"{celex}.html"
                if not raw_html_path.exists():
                    html_content = f"<html><body><h1>{title}</h1><div class='document-content'>{content}</div></body></html>"
                    raw_html_path.write_text(html_content, encoding="utf-8")

                record = {
                    "celex": celex,
                    "title": title,
                    "date": item.get("date", "2026-01-01"),
                    "type": doc_type,
                    "eurovoc_raw": item.get("eurovoc", ""),
                    "html_url": f"local://{jsonl_path.name}/{doc_id}",
                    "source": "local_jsonl",
                }

                if celex not in existing:
                    self.save_metadata_record(record)
                    existing[celex] = record

                ingested.append(record)

        logger.info(f"Ingested {len(ingested)} documents from local dataset.")
        return ingested
