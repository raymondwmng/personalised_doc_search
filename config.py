import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
TAXONOMY_DIR = DATA_DIR / "taxonomy"
DB_DIR = DATA_DIR / "db"
DOCUMENTS_DB_PATH = DB_DIR / "documents.db"
USER_HISTORY_DB_PATH = DB_DIR / "user_history.db"

MEMORY_DIR = BASE_DIR / "memory"
PROMPT_CONTEXT_PATH = MEMORY_DIR / "prompt_context.json"

# Ensure runtime directories exist
for path in [RAW_DIR, PROCESSED_DIR, INDEX_DIR, EMBEDDINGS_DIR, TAXONOMY_DIR, DB_DIR, MEMORY_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# EUR-Lex Dataset Paths (Kaggle: puskas78/eurlex-dataset)
EURLEX_KAGGLE_DIR = DATA_DIR / "eurlex_kaggle"
EURLEX_KAGGLE_CSV = EURLEX_KAGGLE_DIR / "EurLex_all.csv"
DEFAULT_LOAD_LIMIT = 1000

# Chunking Configuration (Recommendation 3: Meso ~2000 tokens standard chunking)
CHUNK_SIZE_TOKENS = 2000
CHUNK_OVERLAP_TOKENS = 200
CHUNK_APPROX_CHARS = 8000
CHUNK_OVERLAP_CHARS = 800

# Supported Embedding Models (Recommendation 1 & 2)
SUPPORTED_EMBEDDING_MODELS = {
    "bge-m3": {
        "id": "bge-m3",
        "name": "BAAI/bge-m3",
        "display_name": "BAAI/bge-m3 (Multilingual EU, 567M, 1024d)",
        "dim": 1024,
        "context_window": 8192,
        "max_seq_length": 2048,  # Cap sequence length for RTX A2000 VRAM stability
        "index_path": INDEX_DIR / "faiss_bge_m3.index",
        "meta_path": INDEX_DIR / "chunk_meta_bge_m3.jsonl",
        "bm25_path": INDEX_DIR / "bm25_bge_m3.pkl",
        "threshold": 0.50,  # Calibrated null cutoff (Rec 5)
    },
    "modernbert": {
        "id": "modernbert",
        "name": "nomic-ai/modernbert-embed-base",
        "display_name": "ModernBERT-base (High Throughput, 149M, 768d)",
        "dim": 768,
        "context_window": 8192,
        "max_seq_length": 2048,
        "index_path": INDEX_DIR / "faiss_modernbert.index",
        "meta_path": INDEX_DIR / "chunk_meta_modernbert.jsonl",
        "bm25_path": INDEX_DIR / "bm25_modernbert.pkl",
        "threshold": 0.50,
    },
}

DEFAULT_EMBEDDING_MODEL = "bge-m3"

# Active Model Aliases (backwards compatibility)
EMBEDDING_MODEL_NAME = SUPPORTED_EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]["name"]
EMBEDDING_DIM = SUPPORTED_EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]["dim"]
FAISS_INDEX_PATH = SUPPORTED_EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]["index_path"]
CHUNK_METADATA_PATH = SUPPORTED_EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]["meta_path"]
NULL_REJECTION_THRESHOLD = 0.50

# Hybrid Retrieval & Re-ranking Configuration (Section 4.6)
ENABLE_HYBRID_SEARCH = True
ENABLE_CROSS_ENCODER = True
BM25_WEIGHT = 0.3
DENSE_WEIGHT = 0.7
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Azure Document Intelligence Configuration (Section 4.3)
AZURE_DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOC_INTEL_ENDPOINT", "")
AZURE_DOC_INTEL_KEY = os.getenv("AZURE_DOC_INTEL_KEY", "")
AZURE_DOC_INTEL_BUDGET_PAGES = 500

# Legal-BERT Tagging Configuration (Section 4.2)
LEGAL_BERT_MODEL = "nlpaueb/legal-bert-base-uncased"

# Benchmark Results Path (Section 4.5)
BENCHMARK_RESULTS_PATH = DATA_DIR / "benchmark_results.json"

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# Ollama Models & Inference Settings (Local)
OLLAMA_MODELS = {
    "gemma2:9b": "gemma2:9b (High Legal Comprehension)",
    "llama3.1:8b": "llama3.1:8b (Structured / Tool Tuning)",
    "gemma3:4b": "gemma3:4b (Fast / Low Footprint)",
    "mistral:7b": "mistral:7b (Balanced)",
    "llama3.2:3b": "llama3.2:3b (Ultra Fast)",
}

# Mistral API Models (Cloud)
MISTRAL_MODELS = {
    "mistral-large-4": "mistral-large-4",
    "mistral-small-2603": "mistral-small-2603",
}

# Display annotations for Mistral models
MISTRAL_MODEL_ANNOTATIONS = {
    "mistral-large-4": "gpt-5 alternative",
    "mistral-small-2603": "gpt-5-mini alternative",
}

# Combined Models for UI & Summarisation (Annotated display format)
ALL_MODELS = {
    "mistral-large-4": f"mistral-large-4 ({MISTRAL_MODEL_ANNOTATIONS['mistral-large-4']})",
    "mistral-small-2603": f"mistral-small-2603 ({MISTRAL_MODEL_ANNOTATIONS['mistral-small-2603']})",
    **OLLAMA_MODELS,
}

DEFAULT_ANALYTICAL_MODEL = "mistral-large-4"
DEFAULT_STRUCTURED_MODEL = "llama3.1:8b"
DEFAULT_FALLBACK_MODEL = "gemma3:4b"
DEFAULT_OLLAMA_TIMEOUT = 25.0
DEFAULT_MISTRAL_TIMEOUT = 45.0
DEFAULT_NUM_CTX = 4096
MISTRAL_API_BASE_URL = "https://api.mistral.ai/v1"


def is_mistral_model(model_name: str) -> bool:
    """Checks whether the requested model is a Mistral API cloud model."""
    if not model_name:
        return False
    return model_name.strip().lower() in MISTRAL_MODELS


def resolve_model_name(model_name: str) -> str:
    """Resolves model name to canonical ID."""
    clean = model_name.strip().lower()
    return MISTRAL_MODELS.get(clean, model_name)


def resolve_embedding_spec(model_key_or_name: Optional[str] = None) -> dict:
    """Resolves embedding model key or name to its specification dict."""
    if not model_key_or_name:
        return SUPPORTED_EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]
    clean = model_key_or_name.strip().lower()
    if clean in SUPPORTED_EMBEDDING_MODELS:
        return SUPPORTED_EMBEDDING_MODELS[clean]
    for key, spec in SUPPORTED_EMBEDDING_MODELS.items():
        if clean == spec["name"].lower() or clean in spec["display_name"].lower():
            return spec
    return SUPPORTED_EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]




# Predefined Taxonomy Mapping
TAXONOMY = {
    "Jurisdiction": [
        "EU-wide",
        "Member State specific",
        "International",
        "Cross-border",
    ],
    "Regulatory Topics": [
        "Data Protection and Privacy",
        "Competition and Antitrust",
        "Consumer Protection",
        "Financial Regulation",
        "Employment Law",
        "Environmental Law",
        "Trade and Customs",
        "Intellectual Property",
        "Digital Services",
        "Health and Safety",
        "Tax",
        "Corporate Governance",
    ],
    "Industry / Sector": [
        "Agriculture and Food",
        "Energy",
        "Transport",
        "Finance and Banking",
        "Technology and Digital",
        "Healthcare and Pharma",
        "Manufacturing",
        "Telecommunications",
        "Environment and Climate",
        "Education",
        "Defence and Security",
    ],
}

# Category Keywords for TF-IDF Taxonomy Matching
TAXONOMY_KEYWORDS = {
    "Jurisdiction": {
        "EU-wide": ["union", "european union", "internal market", "harmonisation", "member states", "community"],
        "Member State specific": ["national law", "competent authority", "designated authority", "national provisions"],
        "International": ["third countries", "international agreement", "treaty", "wto", "convention", "global"],
        "Cross-border": ["cross-border", "mutual assistance", "cooperation mechanism", "transnational", "extradition"],
    },
    "Regulatory Topics": {
        "Data Protection and Privacy": ["gdpr", "personal data", "data subject", "controller", "processor", "privacy", "processing of personal data", "supervisory authority", "consent"],
        "Competition and Antitrust": ["competition", "state aid", "cartel", "merger", "dominant position", "anti-competitive", "antitrust", "monopoly"],
        "Consumer Protection": ["consumer", "unfair commercial practices", "consumer rights", "withdrawal", "defective products", "guarantee"],
        "Financial Regulation": ["credit institutions", "solvency", "capital requirements", "banking union", "securities", "mifid", "liquidity", "prudential"],
        "Employment Law": ["working hours", "worker", "employment", "occupational health", "dismissal", "collective agreement", "maternity", "equal treatment"],
        "Environmental Law": ["emissions", "pollution", "biodiversity", "waste", "carbon", "greenhouse gas", "circular economy", "water quality"],
        "Trade and Customs": ["customs", "tariffs", "import", "export", "anti-dumping", "origin of goods", "duties", "free trade"],
        "Intellectual Property": ["trademark", "patent", "copyright", "geographical indications", "intellectual property", "counterfeiting"],
        "Digital Services": ["digital services", "online platform", "gatekeeper", "algorithm", "artificial intelligence", "intermediary services", "very large online platform", "content moderation"],
        "Health and Safety": ["public health", "medicinal products", "clinical trials", "food safety", "medical devices", "epidemic", "hazard"],
        "Tax": ["value added tax", "vat", "excise duty", "direct taxation", "tax avoidance", "tax evasion", "withholding tax"],
        "Corporate Governance": ["annual accounts", "board of directors", "shareholder rights", "transparency", "auditing", "statutory audit", "reporting obligations"],
    },
    "Industry / Sector": {
        "Agriculture and Food": ["agriculture", "fisheries", "food safety", "livestock", "crops", "common agricultural policy", "cap", "pesticides", "veterinary"],
        "Energy": ["electricity", "natural gas", "renewable energy", "grid", "internal energy market", "nuclear", "energy efficiency", "hydrogen"],
        "Transport": ["rail", "aviation", "maritime", "road transport", "vehicles", "air carrier", "passenger rights", "infrastructure"],
        "Finance and Banking": ["banks", "investment firms", "insurance", "payment services", "central bank", "fintech", "asset management"],
        "Technology and Digital": ["software", "cloud computing", "cybersecurity", "hardware", "artificial intelligence", "data spaces", "semiconductors"],
        "Healthcare and Pharma": ["pharmaceutical", "hospitals", "patient care", "biotechnology", "medicines", "medical devices"],
        "Manufacturing": ["industrial machinery", "chemicals", "reach", "raw materials", "factory", "equipment standards", "heavy industry"],
        "Telecommunications": ["telecom", "5g", "spectrum", "electronic communications networks", "broadband", "roaming"],
        "Environment and Climate": ["climate change", "paris agreement", "renewable resources", "nature restoration", "carbon neutrality"],
        "Education": ["higher education", "vocational training", "erasmus", "academic recognition", "qualifications framework"],
        "Defence and Security": ["defence procurement", "security", "border control", "counter-terrorism", "dual-use goods", "sanctions"],
    },
}
