# Personalised EUR-Lex Legal Search (POC)

An intelligent, GPU-accelerated legal search and retrieval engine over European Union legislation and court decisions. The system combines dense vector retrieval (FAISS + Sentence-Transformers), multi-dimensional legal taxonomy classification (Jurisdiction, Regulatory Topics, Industry / Sector), adaptive user memory profiling, and cross-document legal synthesis powered by **Mistral AI** cloud models and local **Ollama** models.

---

## 📑 Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [How to Start the System](#how-to-start-the-system)
  - [1. Launching the Streamlit Web Application](#1-launching-the-streamlit-web-application)
  - [2. Running CLI Search Queries](#2-running-cli-search-queries)
- [Testing Mistral API Summarisation (Token & Credit Conservation)](#testing-mistral-api-summarisation-token--credit-conservation)
  - [Minimal Token Live Test Script](#minimal-token-live-test-script)
  - [CLI Search with Token Capping](#cli-search-with-token-capping)
- [Supported LLM Models](#supported-llm-models)
- [Data Pipeline Scripts](#data-pipeline-scripts)
- [Running Unit Tests](#running-unit-tests)
- [Project Directory Layout](#project-directory-layout)

---

## 🚀 Key Features

1. **Dense Semantic Retrieval**:
   - Chunks documents with overlap (~400 tokens / 1600 characters).
   - Generates 384-dimensional dense embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
   - Performs fast vector similarity search using FAISS (`IndexFlatIP`).
2. **Three-Dimensional Legal Taxonomy Profiling**:
   - Automatically maps EU legislation into:
     - **Jurisdiction** (EU-wide, Member State specific, International, Cross-border)
     - **Regulatory Topics** (Data Protection, Competition, Consumer Protection, Financial, etc.)
     - **Industry / Sector** (Technology, Finance, Energy, Healthcare, Transport, etc.)
3. **Cross-Document Legal Synthesis**:
   - Synthesises comparative legal briefings across multiple retrieved acts with explicit CELEX citations.
   - Dual-engine support: **Mistral AI Cloud API** (`mistral-small-2603`, `mistral-large-4`) and **Local Ollama** (`gemma2:9b`, `llama3.1:8b`, etc.).
   - Built-in resilient HTTP REST fallback if SDK imports or connections fail.
4. **Adaptive User Memory Store**:
   - Maintains user background, preferred topics, preferred sectors, and custom directives in `memory/prompt_context.json`.
   - Dynamically injects memory context into synthesis prompts and hybrid search ranking.
5. **Interactive Analytics Dashboard**:
   - Built with Streamlit and Plotly.
   - Features spider/radar charts of topic distributions and relevance score histograms.

---

## 🏛️ System Architecture

```
+-------------------------------------------------------------------------+
|                           STREAMLIT WEB UI                              |
|   +---------------+   +------------------+   +----------------------+   |
|   |  Search Bar   |   | Results & Summary|   | Radar & Histograms   |   |
|   +-------+-------+   +--------^---------+   +----------^-----------+   |
|           |                    |                        |               |
+-----------+--------------------+------------------------+---------------+
            |                    |                        |
       +----v-----+       +------+-------+        +-------+------+
       |  Search  |       | Legal        |        | User Memory  |
       |  Engine  | <---> | Summariser   | <----> | Store        |
       |  (FAISS) |       | (Mistral/LLM)|        | (SQLite/JSON)|
       +----+-----+       +--------------+        +--------------+
            |
     +------v--------------------+
     |      Storage Layer        |
     | • SQLite: documents.db    |
     | • FAISS: faiss_flat_ip    |
     | • Chunks: chunk_meta.jsonl|
     +---------------------------+
```

---

## 📋 Prerequisites

- **Operating System**: Linux / macOS / Windows WSL
- **Python**: 3.12 (managed with `uv` or standard Python virtual environment)
- **Mistral API Key** (optional, required only for Mistral cloud models):
  - Obtain a key from [console.mistral.ai](https://console.mistral.ai/)
- **Ollama** (optional, required only for local models):
  - Install Ollama from [ollama.ai](https://ollama.ai/) if using local models such as `gemma2:9b` or `llama3.1:8b`.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/speechray/personalised_doc_search.git
cd personalised_doc_search
```

### 2. Set Up Virtual Environment

You can use `uv` (recommended) or standard `venv`:

**Using `uv`:**
```bash
# uv will automatically configure the environment according to pyproject.toml
uv sync
```

**Using standard Python `venv`:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Configure Environment Variables (`.env`)

Create your `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Open `.env` and configure your Mistral API key:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

---

## 🏁 How to Start the System

### 1. Launching the Streamlit Web Application

Run any of the following commands from the project root:

**Using `uv`:**
```bash
uv run streamlit run ui/app.py
```

**Or using `.venv` directly:**
```bash
.venv/bin/streamlit run ui/app.py
```

**Or using the `main.py` entrypoint:**
```bash
.venv/bin/python main.py
```

Open your browser at **`http://localhost:8501`**.

---

### 2. Running CLI Search Queries

You can execute searches and legal syntheses directly from your terminal:

```bash
# Basic search with default model (mistral-large-4)
.venv/bin/python main.py --search "AI liability and high risk systems"

# Search using the cost-efficient mistral-small-2603 model
.venv/bin/python main.py --search "GDPR cross-border transfer" --model mistral-small-2603

# Search using local Ollama model
.venv/bin/python main.py --search "Maritime emissions" --model gemma2:9b
```

---

## 💡 Testing Mistral API Summarisation (Token & Credit Conservation)

When developing or verifying your Mistral API connection, you want to avoid depleting your API credits. We have implemented strict token budgeting and dedicated tools to keep token consumption as close to zero as possible.

### Minimal Token Live Test Script

A lightweight test script (`scripts/test_mistral_live.py`) runs a real live test against the Mistral API using:
- **Model**: `mistral-small-2603` (20x cheaper than `mistral-large-4`)
- **Input Tokens**: ~500 tokens (1 minimal document excerpt)
- **Output Tokens (`max_tokens`)**: Capped strictly at **40 tokens**
- **Cost**: **~£0.000049 GBP** (~$0.000063 USD) per run (less than 0.005 pence!)

Run the test:
```bash
.venv/bin/python scripts/test_mistral_live.py
```

Expected output:
```text
=================================================================
  MISTRAL API LIVE SUMMARY TEST — MINIMAL TOKEN MODE
=================================================================
• API Key Detected: mstrl_...
• Model: mistral-small-2603 (Most cost-efficient cloud model)
• Max Output Tokens: 40 (Strictly capped to conserve credits)
• Documents Tested: 1 minimal excerpt (~60 words)
-----------------------------------------------------------------
[Response Status]: SUCCESS (in 0.88s)
[Model Used]: mistral-small-2603 (gpt-5-mini alternative)

--- Generated Legal Summary ---
# **Executive Legal Briefing: Rules Applicable to High-Risk AI Systems in the EU**
...
-----------------------------------------------------------------
📊 TOKEN CONSUMPTION & COST AUDIT:
  • Prompt (Input) Tokens:      506
  • Completion (Output) Tokens: 40
  • Total Tokens Consumed:      546
  • Estimated Cost (USD):       $0.000063
  • Estimated Cost (GBP):       £0.000049
=================================================================
✅ SUCCESS: Live Mistral API summarisation confirmed with minimal token usage!
```

---

### CLI Search with Token Capping

When running searches via `main.py`, you can use `--top-docs` and `--max-tokens` to strictly limit token consumption:

```bash
# Retrieve only 1 document and cap output generation to 35 tokens
.venv/bin/python main.py --search "AI systems" --top-docs 1 --max-tokens 35 --model mistral-small-2603
```

This displays the exact token usage breakdown:
```text
[Token Usage]: Prompt: 808 | Completion: 35 | Total: 843
```

---

## 🤖 Supported LLM Models

| Model Name | Type | Notes | Recommended Use Case |
|---|---|---|---|
| `mistral-small-2603` | Mistral Cloud API | $0.10/M input, $0.30/M output | Fast, ultra-cheap testing & daily search |
| `mistral-large-4` | Mistral Cloud API | $2.00/M input, $6.00/M output | In-depth legal synthesis & complex multi-act comparison |
| `gemma2:9b` | Local (Ollama) | Free, requires GPU/VRAM | High local legal comprehension |
| `llama3.1:8b` | Local (Ollama) | Free, requires GPU/VRAM | Structured summaries & tooling |
| `gemma3:4b` | Local (Ollama) | Free, lightweight | Fast local fallback |

---

## 🛠️ Data Pipeline Scripts

If you need to rebuild or re-index the dataset, run the pipeline scripts in order:

1. **Download Raw Data**:
   ```bash
   .venv/bin/python scripts/01_download_eurlex.py
   ```
   Queries the EU Cellar SPARQL endpoint and downloads HTML legal manifestations to `data/raw/`.

2. **Parse and Clean**:
   ```bash
   .venv/bin/python scripts/02_parse_and_clean.py
   ```
   Parses raw HTML, removes legal boilerplate, extracts CELEX IDs, and outputs clean documents to `data/processed/`.

3. **Taxonomy Tagging**:
   ```bash
   .venv/bin/python scripts/03_tag_documents.py
   ```
   Tags documents across Jurisdiction, Regulatory Topics, and Industry/Sector, storing records in `data/db/documents.db`.

4. **Embedding & Indexing**:
   ```bash
   .venv/bin/python scripts/04_embed_and_index.py
   ```
   Chunks text, computes dense embeddings, and generates the FAISS index (`data/index/faiss_flat_ip.index`).

---

## 🧪 Running Unit Tests

Unit tests verify configuration integrity, model resolution, error handling for missing API keys, and mocked API responses:

```bash
.venv/bin/python -m unittest discover tests
```

---

## 📂 Project Directory Layout

```
personalised_doc_search/
├── config.py                 # Core paths, models, and taxonomy configurations
├── main.py                   # System entrypoint (Streamlit launch & CLI search)
├── pyproject.toml            # Project dependencies and packaging
├── README.md                 # Project documentation
├── .env.example              # Sample environment variables file
│
├── data/                     # Data storage (git-ignored)
│   ├── db/
│   │   ├── documents.db      # SQLite database storing documents & multi-label tags
│   │   └── user_history.db   # Search query history
│   └── index/
│       ├── faiss_flat_ip.index # Vector index for fast cosine retrieval
│       └── chunk_meta.jsonl    # Metadata mapping vector IDs to document chunks
│
├── memory/
│   └── prompt_context.json   # User memory profile (topics, sectors, background)
│
├── scripts/
│   ├── 01_download_eurlex.py # Cellar SPARQL/REST download pipeline
│   ├── 02_parse_and_clean.py # HTML parser and text normalizer
│   ├── 03_tag_documents.py   # Multi-label taxonomy tagger
│   ├── 04_embed_and_index.py # Embedding generator and FAISS indexer
│   └── test_mistral_live.py  # Minimal token test for Mistral API
│
├── src/
│   ├── embedder.py           # Sentence-transformers embedding wrapper
│   ├── memory.py             # User profile & search history manager
│   ├── searcher.py           # Hybrid retrieval engine with FAISS
│   └── summariser.py         # Cross-document legal synthesiser (Mistral/Ollama)
│
├── tests/
│   └── test_mistral_models.py # Unit tests for Mistral model routing
│
└── ui/
    ├── app.py                # Main Streamlit web application
    ├── styles.css            # Custom CSS styling
    └── components/           # UI components (search bar, charts, memory panel)
```
