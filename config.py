import os
from pathlib import Path

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

# SPARQL and EURLEX Cellar Endpoints
SPARQL_ENDPOINT = "http://publications.europa.eu/webapi/rdf/sparql"
CELLAR_REST_BASE = "http://publications.europa.eu/resource/cellar"
REQUEST_DELAY_SECONDS = 1.0  # Polite rate limiting
DEFAULT_SPARQL_LIMIT = 5000

# Chunking Configuration
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
CHUNK_APPROX_CHARS = 1600
CHUNK_OVERLAP_CHARS = 200

# Embedding Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
FAISS_INDEX_PATH = INDEX_DIR / "faiss_flat_ip.index"
CHUNK_METADATA_PATH = INDEX_DIR / "chunk_meta.jsonl"

# Ollama Models & Inference Settings
OLLAMA_MODELS = {
    "gemma2:9b": "gemma2:9b (High Legal Comprehension)",
    "llama3.1:8b": "llama3.1:8b (Structured / Tool Tuning)",
    "gemma3:4b": "gemma3:4b (Fast / Low Footprint)",
    "mistral:7b": "mistral:7b (Balanced)",
    "llama3.2:3b": "llama3.2:3b (Ultra Fast)",
}
DEFAULT_ANALYTICAL_MODEL = "gemma2:9b"
DEFAULT_STRUCTURED_MODEL = "llama3.1:8b"
DEFAULT_FALLBACK_MODEL = "gemma3:4b"
DEFAULT_OLLAMA_TIMEOUT = 25.0
DEFAULT_NUM_CTX = 4096

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
