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
2. [Stage 1 — MVP (2 hours)](#2-stage-1--mvp-2-hours)
3. [Stage 2 — Extended Features (2–8 hours)](#3-stage-2--extended-features-28-hours)
4. [Stage 3 — Items Requiring > 1 Day](#4-stage-3--items-requiring--1-day)
5. [Stage 4 — Long-Term Extensions](#5-stage-4--long-term-extensions)
6. [Technology Decisions and Rationale](#6-technology-decisions-and-rationale)
7. [Cost Analysis](#7-cost-analysis)
8. [Risk and Open Questions](#8-risk-and-open-questions)

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

## 2. Stage 1 — MVP (2 hours)

> **Goal:** End-to-end working pipeline from download -> search -> summarise, with a functional Streamlit UI.

### 2.1 Data Download Script (~25 min)

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
> reduce the tagging effort in Step 2.3.

### 2.2 Parse and Clean (~15 min)

| Task | Detail |
|------|--------|
| **HTML to Text** | Use `BeautifulSoup` to strip tags, extract article text. Preserve article/section structure where possible |
| **Chunking** | Split each document into chunks of ~400 tokens with 50-token overlap. Each chunk keeps parent doc metadata |
| **Store** | Save to SQLite `documents.db`: tables `documents` (id, celex, title, date, type, full_text, eurovoc_raw) and `chunks` (id, doc_id, chunk_idx, text, token_count) |

### 2.3 Tagging — V1 TF-IDF + Predefined Taxonomy (~20 min)

Use a **predefined taxonomy** based on the 21 EuroVoc top-level domains, mapped to the three radar-chart dimensions:

#### Predefined Taxonomy Mapping

| Radar Dimension | Categories (derived from EuroVoc + manual mapping) |
|---|---|
| **Jurisdiction** | EU-wide, Member State specific, International, Cross-border |
| **Regulatory Topics** | Data Protection and Privacy, Competition and Antitrust, Consumer Protection, Financial Regulation, Employment Law, Environmental Law, Trade and Customs, Intellectual Property, Digital Services, Health and Safety, Tax, Corporate Governance |
| **Industry / Sector** | Agriculture and Food, Energy, Transport, Finance and Banking, Technology and Digital, Healthcare and Pharma, Manufacturing, Telecommunications, Environment and Climate, Education, Defence and Security |

**V1 Tagging approach:**
1. If EuroVoc descriptors were retrieved from SPARQL -> map them to the taxonomy above using a lookup table
2. For documents missing EuroVoc tags -> use TF-IDF with keyword matching:
   - Build a keyword dictionary per taxonomy category (e.g., "GDPR", "personal data", "consent" -> Data Protection)
   - Compute TF-IDF vectors, match against category keyword vectors
   - Assign top-3 categories per dimension per document
3. Store tags in SQLite: table `document_tags` (doc_id, dimension, category, confidence_score, method)

### 2.4 Embedding and Indexing (~20 min)

| Task | Detail |
|------|--------|
| **Model** | `sentence-transformers/all-MiniLM-L6-v2` (fast, 80MB, good baseline) for MVP. Placeholder for `nlpaueb/legal-bert-base-uncased` in Stage 2 |
| **Embed** | Encode all chunks -> numpy arrays -> save to `data/embeddings/` |
| **Index** | Build FAISS `IndexFlatIP` (inner product) index. Save to `data/index/` |
| **GPU acceleration** | Use GPU for batch encoding (RTX A2000 handles this easily) |

### 2.5 Search + Summarisation Engine (~20 min)

**Query Transformation:**
```
User text query
    |---> 1. Target text: cleaned, expanded query for keyword/BM25 matching
    +---> 2. Semantic query: embedded via same sentence-transformer for FAISS search
```

**Retrieval pipeline:**
1. Embed user query -> FAISS top-K retrieval (K=20 chunks)
2. Re-rank by combining FAISS cosine score + keyword overlap (BM25-lite via TF-IDF)
3. Deduplicate by parent document, keep top-N documents (N=10)
4. Fetch full context for top-5 documents

**Summarisation:**
- Call Ollama Python client (`ollama.generate()` or `ollama.chat()`)
- **Primary analytical model:** `gemma2:9b` — highest legal comprehension and synthesis quality (~7.2 GB peak VRAM on headless RTX A2000)
- **Primary structured/tool model:** `llama3.1:8b` — natively fine-tuned for function calling and JSON schemas
- **Fast/safe fallback:** `gemma3:4b` — low VRAM footprint (~3.8-4.5 GB) when memory headroom or speed is required
- System prompt:

```
You are a legal document analyst. Summarise the following EU legislation excerpts
that are relevant to the user's query. Be precise, cite article numbers, and
do not hallucinate. Keep the summary under 300 words.

User query: {query}
Retrieved documents:
{context}
```

- **Timeout:** 15 seconds hard limit. If model doesn't finish, return partial + "(truncated)"
- **Ollama config:** Set `num_ctx: 4096` to `6144` to ensure KV cache stays safely within the 8GB limit. Configured via `options` parameter in the Ollama API call.
- **Structured output enforcement:** Pass `format: "json"` in Ollama API calls. Ollama enforces GBNF grammar-constrained sampling at the C++ layer, guaranteeing syntactically valid JSON across all models.

**Model benchmark plan** (run once during setup):

| Model | Weights (Q4) | VRAM w/ 3-6K ctx | Fits 8GB? | Analytical Quality | Structured Output (Validated) | Recommendation |
|---|---|---|---|---|---|---|
| `gemma2:9b` | ~5.4 GB | ~7.0-7.4 GB | **Yes** (tight, fits w/ 0MB display overhead) | **Very High** | **Good-High** (100% valid JSON with `format: "json"`; excels at domain classification) | **Top Choice: Deep Legal Analysis** |
| `llama3.1:8b` | ~4.9 GB | ~6.5-7.2 GB | **Yes** (fits comfortably) | High | **Excellent** (trained explicitly on tool calling / JSON; extracts detailed field constraints) | **Top Choice: Structured JSON & Extraction** |
| `gemma3:4b` | ~2.6 GB | ~3.8-4.5 GB | **Yes** (plenty headroom) | Good-High | **High** (compact, responsive, low memory) | **Speed & Safety Fallback** |
| `mistral:7b` | ~4.4 GB | ~5.8-6.5 GB | **Yes** (comfortable) | Moderate-High | **Good** (solid instruction following) | Viable alternative |
| `llama3.2:3b` | ~2.0 GB | ~2.8-3.4 GB | **Yes** (fastest) | Moderate | **Good** (simple schemas only) | Quick summaries / low latency |

> **VRAM Verification Note:**
> On systems where the GPU drives a display, 1.0–1.5 GB is reserved by X11/Wayland, which would cause `gemma2:9b` to spill.
> However, on your RTX A2000, `nvidia-smi` confirms `Disp.A: Off` (0 MB display baseline).
> Testing confirms `gemma2:9b` runs a ~3,000-word document at ~7.2 GB peak VRAM without OOM or CPU offloading.
> Set `num_ctx: 4096-6144` as an upper ceiling to keep a safety buffer for activations.

### 2.6 Streamlit UI — MVP (~20 min)

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

## 3. Stage 2 — Extended Features (2-8 hours)

### 3.1 Legal-BERT Embedding Upgrade (~1 hour)

- Replace `all-MiniLM-L6-v2` with `nlpaueb/legal-bert-base-uncased` wrapped via `sentence-transformers` with mean pooling
- Re-embed all chunks (GPU-accelerated, ~30 min for 5K docs)
- Rebuild FAISS index
- A/B compare retrieval quality against MiniLM

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

## 4. Stage 3 — Items Requiring More Than 1 Day

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

## 5. Stage 4 — Long-Term Extensions

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

## 6. Technology Decisions and Rationale

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

## 7. Cost Analysis

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

## 8. Risk and Open Questions

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

## Implementation Checklist

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

- [ ] Legal-BERT embedding upgrade
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
