# Personalised EURLEX Document Search — POC Plan

> **Created:** 2026-10-17
> **Python:** 3.12.8 (via `.venv`, managed with `uv` + `pyproject.toml`)
> **GPU:** NVIDIA RTX A2000 8GB Laptop (CUDA 13.2)
> **RAM:** 16 GB · **CPU Cores:** 20 · **Disk:** ~235 GB free
> **Ollama Models Available:** gemma3:4b, gemma2:9b, llama3.2:3b, llama3.1:8b, mistral:7b
> **Budget:** £5 GBP for third-party API calls

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Research Study Findings & Diagnostic Audit (Oct 2026)](#2-research-study-findings--diagnostic-audit-oct-2026)
3. [Stage 1 — MVP Architecture & Remediation](#3-stage-1--mvp-architecture--remediation)
4. [Stage 2 — Extended Features (2–8 hours)](#4-stage-2--extended-features-28-hours)
5. [Stage 3 — Items Requiring > 1 Day](#5-stage-3--items-requiring--1-day)
6. [Stage 4 — Long-Term Extensions](#6-stage-4--long-term-extensions)
7. [Technology Decisions and Rationale](#7-technology-decisions-and-rationale)
8. [Cost Analysis](#8-cost-analysis)
9. [Risk and Open Questions](#9-risk-and-open-questions)
10. [Implementation Checklist](#10-implementation-checklist)

---

## 1. Architecture Overview

```
+---------------------------------------------------------------------+
|                         STREAMLIT WEB UI                            |
|  +----------+  +--------------+  +----------+  +---------------+   |
|  |  Search   |  |  Results +   |  |  Radar / |  |  Memory       |   |
|  |  Input    |  |  Summary     |  |  Charts  |  |  Prompt Mgmt  |   |
|  +----+------+  +------^-------+  +-----^----+  +------+--------+   |
|       |               |               |               |             |
+-------+---------------+---------------+---------------+-------------+
        |               |               |               |
   +----v----+   +------+------+  +-----+----+  +------v--------+
   |  Query  |   |  Retrieval  |  | Analytics|  | Memory Store  |
   |Transform|   |  Engine     |  | Engine   |  | (JSON+SQLite) |
   |  Module |   |  (FAISS)    |  |          |  |               |
   +----+----+   +------^------+  +-----^----+  +---------------+
        |               |               |
   +----v---------------+---------------+----+
   |         Document Store (SQLite)          |
   |  + FAISS Vector Index                    |
   |  + EuroVoc Tags + TF-IDF Features        |
   +----------------^------------------------+
                    |
   +----------------+------------------------+
   |        One-Off Batch Pipeline            |
   |  1. Download (SPARQL -> Cellar REST)     |
   |  2. Parse and Clean HTML                 |
   |  3. Tag (EuroVoc + TF-IDF / Legal-BERT)  |
   |  4. Embed (sentence-transformers)        |
   |  5. Build FAISS index                    |
   +------------------------------------------+
```

### Project Structure (Target)

```
personalised_doc_search/
├── pyproject.toml
├── .python-version                  # 3.12
├── main.py                          # Entry point (Streamlit launch)
├── config.py                        # All config constants, paths, model names
├── PLAN.md
│
├── scripts/
│   ├── 01_download_eurlex.py        # Stage 1: Download documents
│   ├── 02_parse_and_clean.py        # Stage 1: HTML -> plain text + metadata
│   ├── 03_tag_documents.py          # Stage 1: EuroVoc + TF-IDF tagging
│   ├── 04_embed_and_index.py        # Stage 1: Sentence-transformers -> FAISS
│   └── 05_tag_with_api.py           # Stage 2: Azure Doc Intelligence / alt API
│
├── src/
│   ├── __init__.py
│   ├── downloader.py                # SPARQL query + Cellar download logic
│   ├── parser.py                    # HTML cleaning, article extraction
│   ├── tagger.py                    # V1: TF-IDF + predefined taxonomy
│   ├── tagger_bert.py               # V1.5: Legal-BERT classifier (placeholder -> Stage 2)
│   ├── tagger_api.py                # V2: Azure/GCP Document Intelligence (placeholder)
│   ├── embedder.py                  # Sentence-transformer embedding
│   ├── indexer.py                   # FAISS index build/load
│   ├── searcher.py                  # Query transform + retrieval
│   ├── summariser.py                # Ollama-based summarisation
│   ├── memory.py                    # User memory prompt (JSON) + search history (SQLite)
│   └── analytics.py                 # Aggregated statistics, chart data prep
│
├── ui/
│   ├── app.py                       # Streamlit main app
│   ├── components/
│   │   ├── search_bar.py
│   │   ├── results_display.py
│   │   ├── charts.py               # Radar chart + histogram
│   │   └── memory_panel.py
│   └── styles.css                   # Custom Streamlit styling
│
├── data/                            # .gitignore'd
│   ├── raw/                         # Downloaded HTML documents
│   ├── processed/                   # Cleaned text + metadata JSON
│   ├── index/                       # FAISS index files
│   ├── embeddings/                  # Cached numpy embeddings
│   ├── taxonomy/                    # EuroVoc mapping files
│   └── db/
│       ├── documents.db             # SQLite: documents + tags + metadata
│       └── user_history.db          # SQLite: search history
│
├── memory/
│   └── prompt_context.json          # User's accumulated memory prompt
│
└── tests/                           # Placeholder for future tests
    └── __init__.py
```

---

## 2. Research Study Findings & Diagnostic Audit (Oct 2026)

> **Context:** An empirical research benchmark was conducted in `experiments/retrieval_representation_study/` to investigate user-reported inferiority across **(i) representation**, **(ii) tagging**, and **(iii) search & retrieval**, evaluated with `mistral-small-2603`.

### 2.1 Empirical Corpus & Pipeline Diagnostics

| Architectural Pillar | Measured Empirical Metric | Root Cause Diagnosis | Actionable Remedy |
| :--- | :--- | :--- | :--- |
| **(i) Representation** | **92.75% of corpus tokens lost** (32.8M of 35.3M tokens); **99.98%** single-chunk documents; max chunk = 213,675 tokens | `chunk_text()` used `text.split("\n\n")`, which failed on single-newline plain text. `all-MiniLM-L6-v2` truncated text at 256 tokens, indexing only formulaic preambles. | Implement **Article-Aware Hierarchical Chunking** (`Article \d+`, `Chapter \d+`) with sliding token fallback (350–500 tokens). Upgrade to long-context embedder (`bge-m3` or `legal-bert`). |
| **(ii) Tagging** | **0.00% utilization in search**; **92.2%** *"EU-wide"* collapse; **82.0%** *"Financial Regulation"* collapse; **100%** failure on EuroVoc URIs | English keyword matching fails on SPARQL URIs (`http://eurovoc...`). Prototype dictionaries contained generic stopwords (`"union"`, `"budget"`). Search engine ignores tags during ranking. | Add EuroVoc URI SKOS/JSON lookup table; prune prototype vectors; implement **Tag-Boosted Hybrid Retrieval** (20% score weight). |
| **(iii) Search & Retrieval** | **100% False Positive Rate** on negative queries; scores of 0.26–0.46 returned for non-existent laws | Search engine lacks a dynamic rejection/null threshold; BM25 lexical scoring lacks document length normalization. | Establish **Calibrated Rejection Threshold (0.40)**; adopt true BM25 length normalization (`k1=1.5, b=0.75`). |
| **(iv) Summarisation** | **High LLM Fidelity**; prompt-forced hallucination risk under low-confidence false positives | `mistral-small-2603` correctly identified that retrieved acts did not match negative queries, but rigid prompt instructions forced speculative cross-document analogies. | Update summariser prompt to include a **Graceful Refusal Condition** when retrieved context is non-pertinent. |

### 2.2 Prevalence and Real-World Likelihood Analysis of 3 Legal Search Cases

In EUR-Lex legal research across a 10,000+ legislative corpus, legal search behavior bifurcates into three archetypes:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   Prevalence Breakdown in Real-World Legal Search                │
└────────────────────────────────────────┬─────────────────────────────────────────┘
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐ ┌─────────────────────────────┐
│  Case 1: Singularity Match  │ │    Case 2: Multi-Match      │ │    Case 3: Zero-Match       │
│      (~25% - 30% Share)     │ │     (~60% - 65% Share)      │ │     (~10% - 15% Share)      │
├─────────────────────────────┤ ├─────────────────────────────┤ ├─────────────────────────────┤
│ • Specific dispute / docket │ │ • Framework directives      │ │ • Out-of-corpus tech (>2019)│
│ • Named party / hospital    │ │ • Evolving eco-innovations  │ │ • Non-EU statutes (US laws) │
│ • Specific band allocation  │ │ • Multiannual programmes    │ │ • Non-existent regulations  │
│ • User expects 1 exact doc  │ │ • Opt-ins / opt-out clusters│ │ • Critical to reject cleanly│
└─────────────────────────────┘ └─────────────────────────────┘ └─────────────────────────────┘
```

1. **Case 1: Singularity Match (~25%–30% prevalence):**
   - **Characteristics:** The researcher searches for an exact, isolated legal instrument (e.g. state aid docket *"Brussels IRIS hospitals"*, exact treaty, or specific radio frequency decision *"470-790 MHz band"*).
   - **Realism & Likelihood:** Moderately common in compliance and litigation workflows where the user already knows the specific instrument exists and seeks its operative terms.
   - **Engine Expectation:** Rank 1 precision ($P@1 = 1.0$) with large score margin over rank 2.

2. **Case 2: Multi-Match (~60%–65% prevalence — The Dominant Reality):**
   - **Characteristics:** The researcher investigates a thematic regulatory regime, standard, or framework (e.g. *"passenger cars CO2 emissions innovative technology"*, *"aviation civil safety agreement United States"*, or *"Schengen acquis participation provisions"*).
   - **Realism & Likelihood:** **This is the dominant real-world scenario in EU law.** EU primary Directives and Regulations spawn dozens of implementing decisions, delegated acts, bilateral protocols, and opt-ins. A legal researcher investigating car emissions needs all approved technologies (LED lighting, alternators, motor-generators, coasting); a researcher studying Schengen participation needs all bilateral agreements (Liechtenstein, Switzerland, Ireland, UK notifications).
   - **Engine Expectation:** The engine must retrieve **at least 3 to 10 relevant documents**, rank them accurately, and provide the summariser with rich cross-document context to synthesize overarching trends and contrasts.

3. **Case 3: Zero-Match (~10%–15% prevalence):**
   - **Characteristics:** The query refers to non-EU legislation (*"California consumer privacy act CCPA"*), technologies post-dating the corpus snapshot (*"quantum computing export controls"*), or fictional concepts (*"deepsea asteroid mining licensing"*).
   - **Realism & Likelihood:** Substantial in production environments. Users frequently test system scope, enter US/UK terminology, or query recent 2024–2026 tech in older corpora.
   - **Engine Expectation:** Calibrated rejection threshold ($< 0.40$) returning an explicit null state (`[]`), preventing downstream hallucination.

### 2.3 Nine-Query Empirical Benchmark Results (Evaluated with `mistral-small-2603`)

| ID | Case Archetype | Query Text | Top CELEX | Peak Score | Ground Truth Total | GT in Top 5 | P@1 | P@3 | P@5 | Tag Used? | Summariser Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1** | Singularity | `470-790 MHz frequency band` | `32017D0899` | **0.570** | 1 | 1 | 1.0 | 0.333 | 0.200 | ❌ 0% | **Correct:** Synthesized 700 MHz band allocation for 5G broadband. |
| **Q2** | Singularity | `Brussels IRIS hospitals State aid` | `32016D2327` | **0.759** | 1 | 1 | 1.0 | 0.333 | 0.200 | ❌ 0% | **Correct:** Detailed Article 106(2) TFEU deficit funding analysis. |
| **Q3** | Singularity | `Morocco sustainable fisheries agreement` | `32018D2068` | **0.782** | 1 | 1 | 1.0 | 0.333 | 0.200 | ❌ 0% | **Correct:** Accurately captured fishing zones and catch quotas. |
| **Q4** | Multi-Match | `aviation civil safety agreement United States` | `32018D0061` | **0.750** | 3 | **3 / 3** | 1.0 | 1.000 | 0.600 | ❌ 0% | **Correct:** Chronological synthesis across 2011, 2017, and 2019 agreements. |
| **Q5** | Multi-Match | `passenger cars CO2 emissions innovative technology` | `32013D0128` | **0.703** | 6 | **5 / 5** | 1.0 | 1.000 | 1.000 | ❌ 0% | **Correct:** Synthesized LED lighting, 12V alternators, and coasting functions across 5 distinct Commission acts. |
| **Q6** | Multi-Match | `Schengen acquis participation provisions` | `32012D0764` | **0.739** | 6 | **5 / 5** | 1.0 | 1.000 | 1.000 | ❌ 0% | **Correct:** Synthesized Ireland opt-in, Liechtenstein/Swiss border agreements, UK provisions. |
| **Q7** | Zero-Match | `quantum computing export controls` | `32014D0255` | **0.261** | 0 | 0 | 0.0 | 0.000 | 0.000 | ❌ 0% | **False Positive:** Customs Code acts returned; Mistral flagged mismatch. |
| **Q8** | Zero-Match | `California consumer privacy act CCPA` | `32016D1250` | **0.463** | 0 | 0 | 0.0 | 0.000 | 0.000 | ❌ 0% | **False Positive:** Privacy Shield returned; Mistral noted CCPA absent. |
| **Q9** | Zero-Match | `deepsea asteroid mining licensing` | `32012D0130` | **0.261** | 0 | 0 | 0.0 | 0.000 | 0.000 | ❌ 0% | **False Positive:** UNCLOS fisheries treaty returned; Mistral flagged mismatch. |

### 2.4 Empirical Benchmark Summary Metrics

- **Case 1 (Singularity):** $P@1 = 1.000$, $\text{MRR} = 1.000$, $\text{Mean Score} = 0.704$.
- **Case 2 (Multi-Match):** $P@1 = 1.000$, $P@3 = 1.000$, $P@5 = 0.867$, $\text{Mean Score} = 0.731$, **Mean GT recovered in top 5 = 4.33 acts**.
- **Case 3 (Zero-Match):** $\text{False Positive Rate} = 100.0\%$, $\text{Mean Score} = 0.328$. (A calibrated cutoff at $0.40$ completely eliminates all 3 false positives).
- **Corpus Token Loss:** $92.75\%$ ($32.8\text{M}$ tokens lost beyond 256-token cutoff across $99.98\%$ single-chunk acts).
- **Tag Retrieval Weight:** $0.00\%$ (tags decoupled from retrieval and ranking).

---

## 3. Stage 1 — MVP Architecture & Remediation

> **Goal:** High-fidelity pipeline eliminating token truncation, utilizing tags in retrieval, and enforcing calibrated null thresholds.

### 3.1 Data Download Script (~25 min)

| Task | Detail |
|------|--------|
| **SPARQL query** | Query Cellar endpoint for Regulations, Directives, Decisions, Recommendations. Filter: English language, HTML manifestation. Target: 2000-5000 documents. Use pagination (`OFFSET`/`LIMIT` in batches of 500) |
| **Download** | Fetch HTML from Cellar REST URLs. Save to `data/raw/{celex_number}.html`. Include rate limiting (1 req/sec to be polite) |
| **Metadata extraction** | Store CELEX number, document type, title, date, EuroVoc descriptors (if returned by SPARQL), source URL -> `data/processed/metadata.jsonl` |
| **Resumability** | Skip already-downloaded files. Log progress. |

**Key SPARQL query structure:**
```sparql
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
SELECT DISTINCT ?work ?celex ?title ?date ?type ?eurovoc ?html_url WHERE {
  { ?work a cdm:regulation } UNION
  { ?work a cdm:directive } UNION
  { ?work a cdm:decision } UNION
  { ?work a cdm:recommendation }
  ?work cdm:resource_legal_id_celex ?celex .
  ?work cdm:work_date_document ?date .
  OPTIONAL { ?work cdm:work_is_about_concept_eurovoc ?eurovoc }
  ?expr cdm:expression_belongs_to_work ?work ;
        cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/ENG> ;
        cdm:expression_title ?title .
  ?manif cdm:manifestation_belongs_to_expression ?expr ;
         cdm:manifestation_type "html" ;
         cdm:manifestation_file ?html_url .
}
ORDER BY DESC(?date)
LIMIT 5000
```

> **Note:** EuroVoc descriptors are often already attached to EURLEX documents via SPARQL metadata.
> The SPARQL query above attempts to retrieve them directly, which could significantly
> reduce the tagging effort in Step 3.3.

### 3.2 Parse and Clean & Hierarchical Chunking (~15 min)

| Task | Detail |
|------|--------|
| **HTML/Text Parsing** | Use `BeautifulSoup` to strip tags, normalise Unicode, and preserve legislative section headers (`Article \d+`, `Chapter \d+`, `Annex`) |
| **Hierarchical Chunking** | **Remediation:** Replace simplistic `text.split("\n\n")` with an **Article-Aware Chunker**: (1) Split by structural boundaries (`Article \d+`, `Whereas \(\d+\)`), (2) For articles exceeding 400 tokens, apply sliding sentence-window chunking (350–450 tokens with 50-token overlap). Every chunk inherits parent CELEX, article header, and metadata. Eliminates 92.7% token loss. |
| **Store** | Save to SQLite `documents.db`: tables `documents` (id, celex, title, date, type, full_text, eurovoc_raw, source) and `chunks` (id, doc_id, chunk_idx, text, token_count) with index on `chunks(doc_id)` |

#### 3.2.1 Hierarchical Chunking Granularity Regimes

The diagnostic audit in Section 2.1 revealed that short-context embedding models (e.g. `all-MiniLM-L6-v2` at 256 tokens) caused **92.75% corpus token loss** and forced aggressive micro-fragmentation. In EU legislation, legal provisions do not exist in isolation: operative articles frequently cross-reference statutory definitions in Article 2, procedural exemptions in earlier chapters, and interpretative intent laid out in the recitals (*Whereas* clauses).

To solve this, hierarchical chunking is structured into three multi-resolution tiers:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      Hierarchical Legislative Chunking Tiers                           │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
         ┌──────────────────────────────────┼──────────────────────────────────┐
         ▼                                  ▼                                  ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐ ┌──────────────────────────────┐
│  Tier 1: Micro-Chunking      │ │  Tier 2: Meso-Chunking       │ │  Tier 3: Macro / Whole-Act   │
│     (~350 – 500 tokens)      │ │    (~1,500 – 3,000 tokens)   │ │    (8,192 – 32,768 tokens)   │
├──────────────────────────────┤ ├──────────────────────────────┤ ├──────────────────────────────┤
│ • Individual operative art.  │ │ • Complete legislative chap. │ │ • Entire Directive / Reg.   │
│ • Single recital paragraph   │ │ • Substantive Title + defs   │ │ • Complete technical Annex   │
│ • High citation pinpointing  │ │ • Preserves internal context │ │ • Zero fragmentation loss    │
│ • Best for exact fact lookup │ │ • Mitigates cross-ref breaks │ │ • Ideal for Case 1 / broad   │
│ • Model: 512+ token window   │ │ • Model: 4K–8K token window  │ │ • Model: 8K–32K token window │
└──────────────────────────────┘ └──────────────────────────────┘ └──────────────────────────────┘
```

1. **Micro-Chunking (Article-Level, ~350–500 tokens):** Splits on structural boundaries (`Article \d+`, `Whereas \(\d+\)`). Provides high rank precision for specific technical standards, but fractures holistic statutory logic.
2. **Meso-Chunking (Chapter / Title-Level, ~1,500–3,000 tokens):** Bundles operative articles with their governing definitions and procedural safeguards into unified logical chapters. Requires $\ge 4\text{K}$ token context windows.
3. **Macro-Chunking / Full Document (Instrument-Level, 8,192–32,768 tokens):** Encodes complete legislative acts or expansive annexes intact. Eliminates boundary fragmentation entirely; enables whole-act holistic semantic matching. Enabled only by modern long-context architectures ($8\text{K}$ to $32\text{K}$ tokens).

---

#### 3.2.2 Empirical Study: Longer-Context Embedding Models for Legal Retrieval

To determine the optimal embedding backbone that supports extended chunking windows (Meso- and Macro-tiers) within our local hardware envelope (**NVIDIA RTX A2000 8GB Laptop GPU**, 16 GB RAM), a comparative empirical study evaluates five candidate models:

1. **ModernBERT-base** (`answerdotai/ModernBERT-base`):
   - **Context Window:** 8,192 tokens
   - **Architecture & Highlights:** 149M parameters, 768 hidden dimensions. Built on modern transformer architecture: Rotary Position Embeddings (RoPE), FlashAttention-2, GeGLU activations, and unpadding (collapsing padded batches to eliminate computational waste).
   - **Legal Search Fit:** Extreme inference throughput; low memory profile on RTX A2000. Easily embeds full legislative chapters without truncation. Base encoder requires mean pooling or fine-tuned embedding head (`nomic-ai/modernbert-embed-base`).

2. **nomic-embed-text-v1.5** (`nomic-ai/nomic-embed-text-v1.5`):
   - **Context Window:** 8,192 tokens
   - **Architecture & Highlights:** 137M parameters, Matryoshka Representation Learning (MRL) supporting flexible dimensionality ($d \in [64, 768]$, default 768).
   - **Task Prefix Requirements:** Asymmetric prefixes: `search_document: ` for passage/chunk indexing, `search_query: ` for user search queries.
   - **Legal Search Fit:** Matryoshka truncation (e.g. 512 or 256 dimensions) drastically reduces FAISS index footprint and RAM usage while retaining $>98\%$ retrieval accuracy. Highly optimized for long-document contrastive retrieval.

3. **bge-m3** (`BAAI/bge-m3`):
   - **Context Window:** 8,192 tokens
   - **Architecture & Highlights:** 567M parameters, 1024 hidden dimensions. Multi-Lingual, Multi-Functionality, Multi-Granularity. Supports three retrieval modes: Dense embeddings, Lexical sparse weights (learned term weights similar to BM25), and Multi-Vector ColBERT-style fine-grained token scoring.
   - **Legal Search Fit:** State-of-the-art retrieval accuracy across complex statutory corpora; handles cross-lingual EU law queries natively. Higher VRAM consumption (~2.2GB FP16) and larger index size require smaller batch sizes on 8GB VRAM.

4. **jina-embeddings-v2-base-en** (`jinaai/jina-embeddings-v2-base-en`):
   - **Context Window:** 8,192 tokens
   - **Architecture & Highlights:** 137M parameters, 768 hidden dimensions. Modified bidirectional encoder utilizing Attention with Linear Biases (ALiBi), providing stable attention extrapolation over long token distances.
   - **Task Prefix Requirements:** Symmetric; unprompted raw text for both queries and documents.
   - **Legal Search Fit:** Lightweight footprint; seamless drop-in replacement with `SentenceTransformer`. ALiBi prevents semantic degradation on distant statutory recitals.

5. **UltraBERT-Embed-150M** ([`dmis-lab/UltraBERT-Embed-150M`](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M)):
   - **Context Window:** **32,768 tokens** (32K native long-context support, extended from 2K pretraining).
   - **Architecture & Highlights:** 150M parameters, 768 hidden dimensions. State-of-the-art dense retriever on MTEB v2 (score **67.24**, #1 dense model under 200M parameters, outperforming distilled models from 7B teachers). Excels on long-context benchmarks (LongEmbed, passkey retrieval 99%).
   - **Task Prefix Requirements:** Instruction-formatted queries (`Instruct: Retrieve relevant EU legal legislation\nQuery: {query}`); documents embedded raw.
   - **Legal Search Fit:** The **32K token capacity** is unique in the sub-200M parameter class, allowing the engine to embed **entire multi-article EU Regulations, complete Directives, and extensive Annexes in a single pass** (Tier 3 Macro-Chunking). Completely eliminates token loss while fitting comfortably into 8GB VRAM.

##### Candidate Model Specification & Hardware Matrix

| Model | HuggingFace Identifier | Context Length | Params | Dim | Query Prefix / Format | Long-Context Mechanism | VRAM (Batch 16, FP16) | Throughput (docs/sec) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ModernBERT-base** | `answerdotai/ModernBERT-base` | 8,192 | 149M | 768 | None (mean pool) | RoPE + FlashAttn-2 + Unpadding | ~1.4 GB | ~95 |
| **nomic-embed-text-v1.5** | `nomic-ai/nomic-embed-text-v1.5` | 8,192 | 137M | 768 / 512 | `search_query: ` / `search_document: ` | RoPE + Dynamic MRL | ~1.3 GB | ~85 |
| **bge-m3** | `BAAI/bge-m3` | 8,192 | 567M | 1024 | None (dense / sparse) | RoPE + Multi-Granularity | ~2.6 GB | ~35 |
| **jina-embeddings-v2-base-en** | `jinaai/jina-embeddings-v2-base-en` | 8,192 | 137M | 768 | None (raw) | ALiBi (Linear Biases) | ~1.3 GB | ~80 |
| **UltraBERT-Embed-150M** | [`dmis-lab/UltraBERT-Embed-150M`](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M) | **32,768** | 150M | 768 | `Instruct: {task}\nQuery: ` | RoPE + Contrastive LongEmbed | ~1.6 GB (8K) / ~3.8 GB (32K) | ~75 (8K) / ~22 (32K) |

---

##### Study Evaluation Protocol & Benchmark Methodology

The study will be executed in `experiments/long_context_embed_study/` using the following protocol:

1. **Corpus Slicing & Chunking Regimes:**
   - Test each of the 5 models across three parallel indexing configurations:
     - **Regime A (Micro):** 400-token article chunks with 50-token sliding overlap.
     - **Regime B (Meso):** 2,000-token chapter/thematic chunks.
     - **Regime C (Macro):** Whole-act document chunks (up to 8,192 tokens for ModernBERT, Nomic, BGE-M3, Jina; up to 32,768 tokens for UltraBERT).
2. **Retrieval Evaluation Across the 3 Real-World Legal Archetypes (Section 2.2):**
   - **Case 1 (Singularity Match — Q1 to Q3):** Evaluate $P@1$, MRR, and margin of Rank 1 score over Rank 2. Hypothesized that Macro-chunking with UltraBERT or BGE-M3 provides top discrimination by matching full document semantic footprint.
   - **Case 2 (Multi-Match — Q4 to Q6):** Evaluate $P@3, P@5$, and Ground Truth recovery count in top 5. Meso-chunking is hypothesized to balance high recall with contextual richness.
   - **Case 3 (Zero-Match Rejection — Q7 to Q9):** Evaluate peak score separation below the 0.40 null threshold. Verify that longer context does not introduce spurious semantic overlap for negative queries.
3. **Hardware & Resource Profiling on RTX A2000 (8GB VRAM):**
   - Peak GPU VRAM allocated during batch encoding (batch sizes 4, 8, 16, 32).
   - FAISS index build time and disk/RAM footprint (768 vs 1024 vs Matryoshka 512 dimensions).
   - End-to-end query latency (embedding generation + FAISS inner product + BM25 fusion).
4. **Downstream Summarisation Fidelity:**
   - Feed retrieved contexts from each chunking regime into `mistral-small-2603` using the Case-Adaptive prompt (Section 3.5).
   - Measure LLM synthesis quality: presence of hallucinated cross-references, accuracy of operative terms, and adherence to graceful refusal on zero-match queries.

---

### 3.3 Tagging — EuroVoc URI Mapping & Balanced Taxonomy (~20 min)

Use a **pruned taxonomy** based on the 21 EuroVoc top-level domains, mapped to the three radar-chart dimensions:

#### Predefined Taxonomy Mapping

| Radar Dimension | Categories (derived from EuroVoc + manual mapping) |
|---|---|
| **Jurisdiction** | EU-wide, Member State specific, International, Cross-border |
| **Regulatory Topics** | Data Protection and Privacy, Competition and Antitrust, Consumer Protection, Financial Regulation, Employment Law, Environmental Law, Trade and Customs, Intellectual Property, Digital Services, Health and Safety, Tax, Corporate Governance |
| **Industry / Sector** | Agriculture and Food, Energy, Transport, Finance and Banking, Technology and Digital, Healthcare and Pharma, Manufacturing, Telecommunications, Environment and Climate, Education, Defence and Security |

**Tagging Remediation:**
1. **EuroVoc URI Mapping:** Load official EU Vocabularies mapping table (`data/taxonomy/eurovoc_id_to_concept.json`) to resolve URIs (`http://eurovoc.europa.eu/1252`) into English canonical terms before matching.
2. **High-IDF Keyword Dictionaries:** Prune generic stopwords (`"union"`, `"community"`, `"budget"`, `"accounts"`) from TF-IDF prototype dictionaries to resolve the 92.2% *EU-wide* and 82.0% *Financial Regulation* category collapse.
3. **Store tags:** Table `document_tags` (doc_id, dimension, category, confidence_score, method) with composite indexes for fast retrieval joins.

### 3.4 Embedding and Indexing (~20 min)

| Task | Detail |
|------|--------|
| **Model** | `sentence-transformers/all-MiniLM-L6-v2` for baseline; upgrade target determined by the **Long-Context Embedding Study (Section 3.2.2)** among `answerdotai/ModernBERT-base`, `nomic-ai/nomic-embed-text-v1.5`, `BAAI/bge-m3`, `jinaai/jina-embeddings-v2-base-en`, and `dmis-lab/UltraBERT-Embed-150M` (8K–32K token windows) |
| **Embed** | Encode all chunks (micro, meso, or macro depending on study finding) -> normalized numpy float32 arrays -> save to `data/embeddings/` |
| **Index** | Build FAISS `IndexFlatIP` (inner product) index over all chunks. Save to `data/index/` |
| **GPU acceleration** | Batched tensor encoding on RTX A2000 Laptop GPU (CUDA 13.2) |

### 3.5 Search + Summarisation Engine (~20 min)

**Query Transformation & Tag-Boosted Hybrid Retrieval:**
```
User text query
    |---> 1. Semantic query: embedded via SentenceTransformer for FAISS dense retrieval
    |---> 2. Lexical query: BM25 with length normalization (k1=1.5, b=0.75)
    +---> 3. Category inference: match query against taxonomy to boost matching tagged documents
```

**Retrieval Pipeline with Tag Integration & Case Adaptations:**
1. **FAISS Retrieval:** Dense search retrieves top-K candidate chunks ($K=30$).
2. **Tag Matching Boost:** Query terms mapped to taxonomy categories; documents possessing verified matching tags receive an additive score boost:
   $$\text{HybridScore} = 0.50 \cdot S_{\text{FAISS}} + 0.30 \cdot S_{\text{BM25}} + 0.20 \cdot S_{\text{Tag}}$$
3. **Calibrated Null-State Rejection Threshold (Case 3 Guard):**
   - If peak candidate score $< 0.40$, the engine classifies the query as **Out-of-Corpus / Non-Pertinent** and returns an explicit null state (`[]`), eliminating 100% false positive hallucinations on negative queries and bypassing unnecessary LLM API calls.
4. **Document Aggregation & Multi-Match Context Budgeting (Case 2 Optimization):**
   - In multi-match queries (~60%–65% of legal queries), multiple implementing acts or bilateral decisions exist. The engine deduplicates chunks by parent CELEX, retaining the top substantive article chunk for each of the top $N=5$ distinct documents.
   - For singularity queries (~25%–30%), multiple distinct article chunks from the single target document are provided to give deep operative coverage.

**Summarisation Engine (Mistral API + Ollama Local Fallback):**
- **Cloud Analytical Engine:** `mistral-small-2603` (and `mistral-large-4`) via Mistral API (validated in research benchmark: 3.4s latency, ~£0.0004 GBP/query).
- **Local Analytical Models:** `gemma2:9b` (deep analysis) and `llama3.1:8b` (structured JSON).
- **Case-Adaptive Summarisation Prompt:**
```
You are an expert EU legal document analyst. Summarise the following EU legislation excerpts
that are relevant to the user's query.

CRITICAL INSTRUCTIONS:
1. Multi-Match Synthesis (Case 2): When multiple distinct EU acts are provided (e.g. multiple
   implementing decisions or bilateral agreements), provide:
   (a) A unified thematic summary explaining the overarching legal regime.
   (b) A comparative breakdown highlighting the specific scope, dates, and technical distinctions
       of each individual act (e.g. distinct technologies approved, or differing country opt-in terms).
2. Singularity Analysis (Case 1): If a single specific instrument is identified, provide a deep
   dive into its legal basis, key articles, and operative constraints.
3. Graceful Refusal (Case 3): If the retrieved documents do not directly pertain to the user's
   query, state clearly that no relevant EU legislation was identified in the retrieved set. Do not
   force an unrelated synthesis.

User query: {query}
Retrieved documents:
{context}
```

- **Timeout:** 15 seconds hard limit. If model doesn't finish, return partial + "(truncated)"
- **Ollama config:** Set `num_ctx: 4096` to `6144` to ensure KV cache stays safely within the 8GB limit. Configured via `options` parameter in the Ollama API call.
- **Structured output enforcement:** Pass `format: "json"` in Ollama API calls. Ollama enforces GBNF grammar-constrained sampling at the C++ layer, guaranteeing syntactically valid JSON across all models.

### 3.6 Streamlit UI — MVP (~20 min)

**Page layout:**

```
+-----------------------------------------------------+
|  Search: Personalised EURLEX Search                  |
|  +-----------------------------------+ [Search] btn  |
|  |  Enter your legal query...        |               |
|  +-----------------------------------+               |
|  [x] Enable Memory Prompt    [Model: llama3.1:8b v]   |
|                                                      |
|  =================== RESULTS ====================   |
|                                                      |
|  Summary                                             |
|  +---------------------------------------------+    |
|  |  Based on your query about "X", the EU       |    |
|  |  legislation indicates that...               |    |
|  +---------------------------------------------+    |
|                                                      |
|  Selected Articles (expandable)                      |
|  +---------------------------------------------+    |
|  | > Regulation 2024/1689 - Art. 6 (score 0.92) |    |
|  | > Directive 2003/88/EC - Art. 3 (score 0.87) |    |
|  | ...                                          |    |
|  +---------------------------------------------+    |
|                                                      |
|  Aggregated Statistics                               |
|  +------------------+  +----------------------+     |
|  |  Tag Histogram   |  |  Radar Chart         |     |
|  |  (Plotly bar)    |  |  (3-axis: Jurisdict. |     |
|  |                  |  |   Reg.Topics, Sector) |     |
|  +------------------+  +----------------------+     |
|                                                      |
|  Memory Prompt Panel (collapsible sidebar)           |
|  +---------------------------------------------+    |
|  |  Current context: "User frequently searches  |    |
|  |  for data protection and AI regulation..."   |    |
|  |  [Clear Memory] [Export]                     |    |
|  +---------------------------------------------+    |
+-----------------------------------------------------+
```

**Charts (via Plotly):**
- **Histogram:** Distribution of tags across retrieved documents (grouped bar by dimension)
- **Radar chart:** 3 overlaid polygons for the 3 dimensions (Jurisdiction, Regulatory Topics, Industry/Sector), showing count/proportion of matching documents per category

---

## 4. Stage 2 — Extended Features (2-8 hours)

### 4.1 Long-Context Embedding Evaluation Study & Model Upgrade (~1-2 hours)

- Execute the comparative benchmark suite across candidate models (detailed in Section 3.2.2):
  - `answerdotai/ModernBERT-base` (8K context, RoPE, FlashAttention-2)
  - `nomic-ai/nomic-embed-text-v1.5` (8K context, Matryoshka MRL)
  - `BAAI/bge-m3` (8K context, dense + sparse)
  - `jinaai/jina-embeddings-v2-base-en` (8K context, ALiBi)
  - `dmis-lab/UltraBERT-Embed-150M` (32K context, instruction-prompted)
  - Baseline comparisons: `all-MiniLM-L6-v2` and domain-tuned `nlpaueb/legal-bert-base-uncased`
- Re-embed chunks under the top-performing chunking regime (Meso vs Macro) on RTX A2000 Laptop GPU
- Rebuild FAISS index and compare index size / latency
- A/B compare retrieval quality against baseline MiniLM on the 9-query benchmark (Section 2.3)

### 3.2 Legal-BERT Tagging (~1-2 hours)

- Fine-tune or use zero-shot classification with `nlpaueb/legal-bert-base-uncased` to classify documents into the predefined taxonomy
- Alternative: use `facebook/bart-large-mnli` for zero-shot classification (no fine-tuning needed, but slower)
- Compare against TF-IDF tagging quality

### 3.3 Azure Document Intelligence Integration (~2 hours)

**Purpose:** Extract structured content (tables, key-value pairs, layout) from complex legislative documents that plain HTML parsing misses.

| Detail | Value |
|---|---|
| **Service** | Azure AI Document Intelligence (Layout model) |
| **Pricing** | Free tier: 500 pages/month. S0: $1.50/1000 pages for Read, $10/1000 pages for Layout |
| **Budget fit** | 5 GBP = ~$6.30 USD -> ~4,200 pages at Read tier, or ~630 pages at Layout tier |
| **Use case** | Run on a subset of complex documents (tables, annexes) where HTML parsing quality is poor |

**Alternative V2 API candidates (within 5 GBP budget):**

| Provider | Service | Price / 1K pages | Pages for 5 GBP | Notes |
|---|---|---|---|---|
| **Azure** | Document Intelligence (Read) | $1.50 | ~4,200 | Free 500 pages/mo. Best for text extraction |
| **Azure** | Document Intelligence (Layout) | $10.00 | ~630 | Tables + structure. Free 500 pages/mo |
| **Google Cloud** | Document AI (OCR) | $1.50 | ~4,200 | Free 1K pages/mo (enterprise OCR processor) |
| **Google Cloud** | Document AI (Form Parser) | $30.00 | ~210 | Expensive, overkill for legislation |
| **AWS** | Textract (DetectDocumentText) | $1.50 | ~4,200 | No free tier for new accounts |
| **AWS** | Textract (AnalyzeDocument) | $50.00 | ~126 | Tables+Forms, very expensive |

> **Recommendation for V2:** Use **Azure Document Intelligence Read model** (free 500 pages/mo + pay for remaining).
> For structured table extraction, use the **Layout model** on the subset of documents
> containing complex tables (identifiable by HTML `<table>` tag density).
> Combined: 500 free Layout pages + ~3,700 Read pages = well within 5 GBP budget.

### 3.4 Memory Prompt Enhancement (~1 hour)

- Implement proper prompt context accumulation:
  - After each search, extract key themes/entities from the query + results
  - Append to `memory/prompt_context.json` as structured entries
  - Periodically summarise (via Ollama) the accumulated entries into a compact context paragraph
  - Inject this context into the summarisation prompt to personalise results
- Add memory management UI: view, edit, clear, export

### 3.5 Ollama Model Benchmarking (~30 min)

- Verify installed models: `ollama list` (includes `gemma2:9b`, `llama3.1:8b`, `mistral:7b`, `gemma3:4b`, `llama3.2:3b`)
- Run standardised benchmark: 5 representative EURLEX legal queries across all 5 models
- For each model, test with `num_ctx: 4096` and `num_ctx: 6144`
- Measure: latency (p50, p95), response quality (manual rating), VRAM usage (`nvidia-smi` peak during generation), and JSON schema adherence
- Save results to `data/benchmark_results.json`
- Default model in config set to `gemma2:9b` or `llama3.1:8b` based on user preference (quality vs. speed/schema), with `gemma3:4b` as fast fallback
- Allow user to switch models dynamically in UI dropdown

### 3.6 Advanced Query Transformation (~1 hour)

- Add query expansion using:
  - Synonym expansion from EuroVoc thesaurus
  - Legal term normalisation (e.g., "privacy law" -> "data protection regulation")
- Hybrid retrieval: combine FAISS semantic search with BM25 keyword search (using `rank_bm25` library)
- Re-ranking with cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

### 3.7 UI Polish (~1 hour)

- Add pagination for results
- Add filters (by date range, document type, jurisdiction)
- Add "similar documents" links
- Custom CSS theming
- Loading spinners / progress bars during search + summarisation

---

## 5. Stage 3 — Items Requiring More Than 1 Day

> **TODO list** — these are scoped but not yet implementable within a single day.

### 4.1 Full EuroVoc Integration

- Download the complete EuroVoc SKOS/RDF dataset from EU Vocabularies
- Build a hierarchical taxonomy tree (21 domains -> 127 microthesauri -> 7000+ concepts)
- Map every EURLEX document to its full EuroVoc descriptor path
- Enable drill-down navigation in the UI by EuroVoc hierarchy

### 4.2 Incremental Index Updates

- Implement a scheduler to check for new EURLEX documents (weekly SPARQL query)
- Incrementally download, parse, tag, embed, and add to FAISS index without full rebuild
- Track last-sync timestamp in SQLite

### 4.3 Multi-Document Summarisation

- For queries matching many documents, implement map-reduce summarisation:
  1. Summarise each top-K document individually (map)
  2. Synthesise the individual summaries into a coherent overall summary (reduce)
- Chain-of-density prompting for progressive refinement

### 4.4 Evaluation Framework

- Build a test set of ~50 query-relevance pairs (manual annotation)
- Measure: MRR, nDCG@10, Recall@20
- Compare: MiniLM vs Legal-BERT vs hybrid retrieval
- Automate with a `scripts/evaluate.py` script

### 4.5 Document Relationship Graph

- Extract cross-references between EURLEX documents (e.g., "amends Regulation X", "repeals Directive Y")
- Build a citation graph in SQLite or NetworkX
- Visualise in UI with an interactive graph (pyvis or Streamlit-agraph)

---

## 6. Stage 4 — Long-Term Extensions

> **Good-to-have** features. MVP will include **placeholder hooks** (abstract base classes, config flags, empty route handlers) for easier integration later.

### 5.1 Fine-Tuned Legal Summariser

- Fine-tune `gemma3:4b` or `phi-4-mini` on a curated dataset of legal document -> summary pairs
- Use EU legislative summaries from EUR-Lex "Summaries of EU legislation" section as training data
- LoRA/QLoRA fine-tuning on RTX A2000 (feasible for 4B models)

**Placeholder:** `summariser.py` will have a `model_name` config parameter and abstract `Summariser` base class.

### 5.2 Multi-Language Support

- Extend SPARQL queries to fetch documents in other EU languages
- Use multilingual sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- Cross-lingual search: query in English, retrieve in any language

**Placeholder:** `config.py` will have `LANGUAGES = ["ENG"]` list.

### 5.3 User Profiles and Personalised Ranking

- Multiple user profiles with separate memory prompts
- Learn user preferences from click-through data (which articles they expand/read)
- Re-rank results based on user's historical interests

**Placeholder:** `memory.py` will have a `user_id` parameter on all methods.

### 5.4 API Server Mode

- Wrap the search engine as a FastAPI REST API for programmatic access
- Endpoints: `/search`, `/summarise`, `/memory`, `/stats`
- Enable integration with other tools (Obsidian, Notion, etc.)

**Placeholder:** `src/api.py` stub with route definitions.

### 5.5 Advanced Analytics Dashboard

- Temporal trends: legislation activity over time by topic
- Regulatory burden index: measure complexity/length by sector
- Comparative analysis: side-by-side document comparison
- Export to PDF/Excel

**Placeholder:** `analytics.py` will have method stubs for `temporal_trends()`, `complexity_score()`.

### 5.6 RAG with Cited Sources

- Implement grounded generation: every claim in the summary must cite a specific article
- Use structured output from Ollama (JSON mode) to return `{claim, source_celex, article_num}`
- Display citations inline with clickable links

**Placeholder:** `summariser.py` will have a `cited=False` parameter.

---

## 7. Technology Decisions and Rationale

### Core Stack

| Component | Choice | Rationale |
|---|---|---|
| **Language** | Python 3.12 | Already set up in `.venv` with `uv` |
| **Package mgmt** | `uv` + `pyproject.toml` | User's existing setup |
| **UI** | Streamlit | Rich charts, fast to build, built-in session state for memory toggle |
| **Vector DB** | FAISS (flat index) | No external service needed, GPU-accelerated, sufficient for under 100K vectors |
| **Document DB** | SQLite | Zero config, good for under 100K docs, SQL for analytics |
| **Embedding** | `all-MiniLM-L6-v2` (MVP) then `legal-bert` (Stage 2) | MiniLM is fast/small for MVP; Legal-BERT is domain-tuned |
| **Summarisation** | Ollama (local) | Already installed, multiple models available, no API cost |
| **Charts** | Plotly | Native Streamlit integration, interactive radar + histogram support |
| **HTML parsing** | BeautifulSoup4 | Standard, reliable, handles malformed EURLEX HTML |
| **Data download** | SPARQLWrapper + requests | Official Cellar API, no auth needed for SPARQL, free |

### Key Dependencies (for `pyproject.toml`)

```toml
dependencies = [
    # Core
    "streamlit>=1.40",
    "plotly>=5.20",
    # Data
    "beautifulsoup4>=4.12",
    "lxml>=5.0",
    "SPARQLWrapper>=2.0",
    "requests>=2.31",
    # ML
    "sentence-transformers>=3.0",
    "torch>=2.4",
    "faiss-gpu>=1.8",       # or faiss-cpu as fallback
    "scikit-learn>=1.5",    # TF-IDF
    "transformers>=4.44",   # For Legal-BERT (Stage 2)
    # Storage
    "numpy>=1.26",
    "pandas>=2.2",
    # Ollama
    "ollama>=0.3",          # Python client for Ollama
]
```

### Why NOT These Alternatives

| Rejected | Reason |
|---|---|
| ChromaDB / Qdrant | Overkill for under 100K vectors; adds external dependency |
| LangChain | Too much abstraction for a POC; prefer explicit control |
| OpenAI/Anthropic API | User wants local summarisation to avoid cost + latency |
| Elasticsearch | Heavy infra for a POC; SQLite + FAISS covers the needs |
| React/Next.js frontend | Streamlit is faster to build for a data-heavy POC |

---

## 8. Cost Analysis

### Compute Costs (All Local — 0 GBP)

| Task | Resource | Time Estimate |
|---|---|---|
| Download 5000 docs | Network + CPU | ~60-90 min (rate-limited) |
| Parse + clean | CPU | ~5 min |
| TF-IDF tagging | CPU | ~2 min |
| Embedding (MiniLM, 5K docs, ~50K chunks) | GPU (A2000) | ~10-15 min |
| Embedding (Legal-BERT, same) | GPU (A2000) | ~30-45 min |
| FAISS index build | GPU | ~1 min |
| Ollama summarisation per query | GPU | 5-15 sec |

### API Costs (Stage 2 Only)

| Service | Volume | Cost (USD) | Cost (GBP) |
|---|---|---|---|
| Azure DI Read (free tier) | 500 pages | $0 | 0 |
| Azure DI Layout (free tier) | 500 pages | $0 | 0 |
| Azure DI Read (paid, remaining docs) | ~4000 pages | ~$6.00 | ~4.76 |
| **Total** | | | **~4.76 GBP** |

> **Warning:** The 5 GBP budget is tight for Layout model on all 5000 documents.
> Strategy: use free tier (500 pages) for Layout on complex docs,
> Read model for the bulk, and TF-IDF/BERT for tagging (no API cost).

---

## 9. Risk and Open Questions

### Risks

| Risk | Impact | Mitigation |
|---|---|---|
| SPARQL endpoint rate-limiting or downtime | Cannot download data | Implement retry with exponential backoff; cache aggressively |
| EuroVoc descriptors missing from SPARQL response for many docs | Poor taxonomy coverage in V1 | Fall back to TF-IDF keyword tagging; validate coverage in first 100 docs |
| VRAM pressure on RTX A2000 with very large context (>6k tokens) | Generation slows down or triggers OOM | Cap `num_ctx: 6144`; auto-fallback to `gemma3:4b` if memory error occurs |
| EURLEX HTML structure varies widely across document types | Parsing errors | Build robust parser with fallbacks; log unparseable docs for manual review |
| 15-second summarisation timeout too tight for complex queries | Truncated summaries | Use streaming response; show partial results; allow user override |

### Open Questions (Resolved by User Answers)

| # | Question | Resolution |
|---|---|---|
| 1 | Python version | 3.12.8 via `.venv`, `uv` managed |
| 2 | Document scope | All types (Regulations, Directives, Decisions, Recommendations) |
| 3 | Document count | 2000-5000 (large) |
| 4 | UI framework | Streamlit |
| 5 | Summarisation model | `gemma2:9b` (top analytical quality) and `llama3.1:8b` (top structured extraction). Fallback: `gemma3:4b`. Both tested and fit RTX A2000 (0MB display overhead). User model selector in UI |
| 6 | Memory storage | JSON (prompt context) + SQLite (search history) |
| 7 | Azure account | Available, planned for Stage 2. Also evaluate Google Doc AI as V2 alternative |

### Remaining Open Questions

| # | Question | Impact | Default Assumption |
|---|---|---|---|
| 1 | Should the download script filter by date range (e.g., last 5 years only), or fetch all available? | Affects doc count and relevance | **Last 10 years** (2016-2026) to keep volume manageable while covering recent major legislation (GDPR, AI Act, DSA, DMA) |
| 2 | For the radar chart, should all 3 dimensions be shown simultaneously on a single chart, or as 3 separate radar charts? | UI layout | **Single overlay chart** with 3 colour-coded polygons — user can toggle dimensions |
| 3 | Should the memory prompt be injected into the search query expansion, the summarisation prompt, or both? | Personalisation behaviour | **Both** — subtle weighting in search ranking + explicit context in summarisation prompt |
| 4 | What year range of legislation would you like to scope the download to? | Volume, relevance | If no preference, download the most recent 5000 documents by date descending |

---

## 10. Implementation Checklist

### Stage 1 — MVP (2 hours)

- [x] Fix `.python-version` to `3.12.8` or remove (since using `.venv`)
- [x] Update `pyproject.toml` with dependencies
- [x] `uv pip install` all dependencies
- [x] Pull missing Ollama models (`ollama pull llama3.1:8b mistral:7b gemma2:9b`)
- [x] Create `config.py` with all constants
- [x] Implement `scripts/01_download_eurlex.py`
- [x] Implement `src/downloader.py`
- [x] Implement `scripts/02_parse_and_clean.py`
- [x] Implement `src/parser.py`
- [x] Implement `scripts/03_tag_documents.py`
- [x] Implement `src/tagger.py` (TF-IDF + keyword)
- [x] Implement `scripts/04_embed_and_index.py`
- [x] Implement `src/embedder.py`
- [x] Implement `src/indexer.py`
- [x] Implement `src/searcher.py`
- [x] Implement `src/summariser.py` (Ollama integration)
- [x] Implement `src/memory.py`
- [x] Implement `src/analytics.py`
- [x] Implement `ui/app.py` (Streamlit)
- [x] Implement `ui/components/` (search_bar, results_display, charts, memory_panel)
- [x] End-to-end test with sample query
- [x] Create `data/` directory structure + `.gitignore`

### Stage 2 — Extended (~2-8 hours)

- [x] Long-Context Embedding & Hierarchical Chunking Study (Section 3.2.2) — *Completed (see [RESEARCH_REPORT.md](file:///home/speechray/github/personalised_doc_search/experiments/long_context_embed_study/RESEARCH_REPORT.md))*:
  - [x] Implement benchmark suite in `experiments/long_context_embed_study/`
  - [x] Benchmark `answerdotai/ModernBERT-base` (8K context, RoPE, FlashAttention-2, unpadding)
  - [x] Benchmark `nomic-ai/nomic-embed-text-v1.5` (8K context, Matryoshka 768/512/256, `search_query:` prefix)
  - [x] Benchmark `BAAI/bge-m3` (8K context, multi-lingual, dense + sparse lexical weights)
  - [x] Benchmark `jinaai/jina-embeddings-v2-base-en` (8K context, ALiBi linear biases)
  - [x] Benchmark `dmis-lab/UltraBERT-Embed-150M` (32K context, instruction-prompted, SOTA MTEB v2)
  - [x] Evaluate multi-tier chunking regimes: Micro (400t) vs Meso (2,000t) vs Macro (8K–32Kt full-act)
  - [x] Measure retrieval metrics ($P@1, P@5, \text{MRR}$) on the 9-query benchmark across Singularity, Multi-Match, and Zero-Match cases
  - [x] Benchmark GPU VRAM and batch inference throughput on NVIDIA RTX A2000 (8GB VRAM, CUDA 13.2)
  - [x] Evaluate downstream synthesis and graceful refusal with `mistral-small-2603`
- [ ] Legal-BERT embedding upgrade & domain-specific comparison
- [ ] Legal-BERT / zero-shot tagger
- [ ] Azure Document Intelligence integration
- [ ] Model benchmarking script
- [ ] Hybrid BM25 + semantic retrieval
- [ ] Query expansion with EuroVoc synonyms
- [ ] Memory prompt summarisation loop
- [ ] UI polish (filters, pagination, custom CSS)

### Stage 3 — TODO (more than 1 day)

- [ ] Full EuroVoc hierarchy integration
- [ ] Incremental index updates
- [ ] Map-reduce multi-document summarisation
- [ ] Evaluation framework
- [ ] Citation graph

### Stage 4 — Long-Term

- [ ] Fine-tuned legal summariser
- [ ] Multi-language support
- [ ] User profiles
- [ ] FastAPI server mode
- [ ] Advanced analytics dashboard
- [ ] RAG with cited sources
