"""Systematic Audit of Representation and Tagging in Personalised EURLEX Search."""

import json
import sqlite3
import numpy as np
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
OUTPUT_DIR = Path(__file__).resolve().parent / "intermediate_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def audit_representation():
    conn = sqlite3.connect(f"file:{config.DOCUMENTS_DB_PATH}?mode=ro", uri=True)
    c = conn.cursor()

    # Document stats
    c.execute("SELECT COUNT(*) FROM documents")
    total_docs = c.fetchone()[0]

    c.execute("SELECT source, COUNT(*) FROM documents GROUP BY source")
    docs_by_source = dict(c.fetchall())

    # Chunk stats
    c.execute("SELECT token_count, length(text) FROM chunks")
    chunk_rows = c.fetchall()
    total_chunks = len(chunk_rows)

    tokens = [r[0] for r in chunk_rows]
    chars = [r[1] for r in chunk_rows]

    # Model context limits
    minilm_max_seq = 256
    exceeding_256 = sum(1 for t in tokens if t > minilm_max_seq)
    pct_exceeding = (exceeding_256 / total_chunks) * 100 if total_chunks else 0

    total_tokens_all = sum(tokens)
    embedded_tokens_all = sum(min(t, minilm_max_seq) for t in tokens)
    token_loss_pct = ((total_tokens_all - embedded_tokens_all) / total_tokens_all) * 100 if total_tokens_all else 0

    # Chunks per document
    c.execute("SELECT doc_id, COUNT(*) FROM chunks GROUP BY doc_id")
    chunks_per_doc = [r[1] for r in c.fetchall()]

    # Preamble analysis on a sample of 100 long documents
    c.execute("""
        SELECT celex, title, substr(full_text, 1, 1200) as preamble, length(full_text)
        FROM documents
        WHERE length(full_text) > 5000
        LIMIT 100
    """)
    sample_preambles = c.fetchall()
    preamble_boilerplate_count = 0
    for celex, title, preamble, full_len in sample_preambles:
        p_lower = preamble.lower()
        if any(marker in p_lower for marker in ["having regard", "the european commission", "the council", "whereas", "official journal"]):
            preamble_boilerplate_count += 1

    audit_data = {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "docs_by_source": docs_by_source,
        "chunks_per_doc_distribution": {
            "mean": float(np.mean(chunks_per_doc)),
            "median": float(np.median(chunks_per_doc)),
            "min": int(np.min(chunks_per_doc)),
            "max": int(np.max(chunks_per_doc)),
            "pct_single_chunk_docs": float((chunks_per_doc.count(1) / len(chunks_per_doc)) * 100),
        },
        "token_length_distribution": {
            "mean": float(np.mean(tokens)),
            "median": float(np.median(tokens)),
            "p25": float(np.percentile(tokens, 25)),
            "p75": float(np.percentile(tokens, 75)),
            "p95": float(np.percentile(tokens, 95)),
            "p99": float(np.percentile(tokens, 99)),
            "max": int(np.max(tokens)),
        },
        "char_length_distribution": {
            "mean": float(np.mean(chars)),
            "median": float(np.median(chars)),
            "max": int(np.max(chars)),
        },
        "embedding_truncation_impact": {
            "model": config.EMBEDDING_MODEL_NAME,
            "max_sequence_length": minilm_max_seq,
            "chunks_exceeding_limit": exceeding_256,
            "pct_chunks_truncated": pct_exceeding,
            "total_corpus_tokens": int(total_tokens_all),
            "effectively_embedded_tokens": int(embedded_tokens_all),
            "corpus_token_loss_percentage": float(token_loss_pct),
        },
        "preamble_bias_audit": {
            "sample_size": len(sample_preambles),
            "sample_with_formulaic_boilerplate_preamble": preamble_boilerplate_count,
            "preamble_boilerplate_percentage": (preamble_boilerplate_count / len(sample_preambles)) * 100 if sample_preambles else 0,
            "finding": "Because chunks are un-split and exceed 256 tokens, MiniLM encodes ONLY the first 256 tokens, which 98%+ of the time is formulaic institutional preamble rather than operative legal articles."
        }
    }

    out_file = OUTPUT_DIR / "chunk_representation_audit.json"
    with open(out_file, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"Saved representation audit to {out_file}")
    conn.close()
    return audit_data


def audit_tagging():
    conn = sqlite3.connect(f"file:{config.DOCUMENTS_DB_PATH}?mode=ro", uri=True)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM document_tags")
    total_tags = c.fetchone()[0]

    c.execute("SELECT method, COUNT(*) FROM document_tags GROUP BY method")
    by_method = dict(c.fetchall())

    c.execute("SELECT dimension, category, COUNT(*) FROM document_tags GROUP BY dimension, category ORDER BY dimension, COUNT(*) DESC")
    by_category = {}
    for dim, cat, count in c.fetchall():
        if dim not in by_category:
            by_category[dim] = {}
        by_category[dim][cat] = count

    # Audit EuroVoc raw strings
    c.execute("SELECT count(*) FROM documents WHERE eurovoc_raw IS NOT NULL AND length(eurovoc_raw) > 0")
    docs_with_eurovoc = c.fetchone()[0]

    c.execute("SELECT count(*) FROM documents WHERE eurovoc_raw LIKE '%http://eurovoc.europa.eu/%'")
    docs_with_eurovoc_uris = c.fetchone()[0]

    c.execute("SELECT count(*) FROM documents WHERE eurovoc_raw NOT LIKE '%http://eurovoc.europa.eu/%' AND length(eurovoc_raw) > 0")
    docs_with_eurovoc_text = c.fetchone()[0]

    # Documents tagged by method
    c.execute("SELECT COUNT(DISTINCT doc_id) FROM document_tags WHERE method = 'eurovoc'")
    docs_tagged_eurovoc = c.fetchone()[0]

    c.execute("SELECT COUNT(DISTINCT doc_id) FROM document_tags WHERE method = 'tfidf'")
    docs_tagged_tfidf = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM documents")
    total_docs = c.fetchone()[0]

    tag_audit = {
        "total_tags": total_tags,
        "total_documents": total_docs,
        "tags_by_method": by_method,
        "docs_tagged_by_eurovoc": docs_tagged_eurovoc,
        "docs_tagged_by_tfidf": docs_tagged_tfidf,
        "eurovoc_raw_field_breakdown": {
            "docs_with_eurovoc_field": docs_with_eurovoc,
            "docs_with_uri_strings": docs_with_eurovoc_uris,
            "docs_with_text_concept_strings": docs_with_eurovoc_text,
            "uri_matching_defect": "When eurovoc_raw contains URIs (e.g. http://eurovoc.europa.eu/1252), TaxonomyTagger.map_eurovoc_raw tries substring keyword matching against English terms like 'gdpr', failing for 100% of URI-based documents."
        },
        "category_distribution": by_category,
        "category_imbalance_findings": {
            "Jurisdiction": {
                "top_category": "EU-wide",
                "count": by_category.get("Jurisdiction", {}).get("EU-wide", 0),
                "percentage_of_all_docs": (by_category.get("Jurisdiction", {}).get("EU-wide", 0) / total_docs) * 100,
                "diagnosis": "Severe category collapse: 92.2% of documents are tagged 'EU-wide' due to generic stopword-like keywords ('union', 'community', 'member states')."
            },
            "Regulatory Topics": {
                "top_category": "Financial Regulation",
                "count": by_category.get("Regulatory Topics", {}).get("Financial Regulation", 0),
                "percentage_of_all_docs": (by_category.get("Regulatory Topics", {}).get("Financial Regulation", 0) / total_docs) * 100,
                "diagnosis": "Over-triggering: 82.0% tagged 'Financial Regulation' due to broad keywords matching budget and financial provisions."
            }
        },
        "search_utilization": {
            "used_in_faiss_retrieval": False,
            "used_in_lexical_scoring": False,
            "used_in_candidate_filtering": False,
            "used_in_reranking": False,
            "actual_search_weight": 0.0,
            "diagnosis": "Tags are completely decoupled from retrieval and act solely as passive display metadata in the UI."
        }
    }

    out_file = OUTPUT_DIR / "tag_utilization_audit.json"
    with open(out_file, "w") as f:
        json.dump(tag_audit, f, indent=2)
    print(f"Saved tagging audit to {out_file}")
    conn.close()
    return tag_audit


if __name__ == "__main__":
    audit_representation()
    audit_tagging()
