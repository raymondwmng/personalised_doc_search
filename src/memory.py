"""User memory management, interaction accumulation, and search history tracking (Section 4.4)."""

from collections import Counter
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import ollama

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MEMORY = {
    "user_context": "Focus on high-impact AI safety regulations, GDPR data subject rights, and cross-border tech compliance.",
    "preferred_topics": ["Data Protection and Privacy", "Digital Services"],
    "preferred_sectors": ["Technology and Digital", "Finance and Banking"],
    "notes": "Interested in direct accountability for deployers and liability caps.",
    "interactions": [],
}


class UserMemoryStore:
    """Manages persistent user memory (JSON), structured interaction accumulation, and search history (SQLite)."""

    def __init__(
        self,
        json_path: Path = config.PROMPT_CONTEXT_PATH,
        history_db_path: Path = config.USER_HISTORY_DB_PATH,
    ):
        self.json_path = json_path
        self.history_db_path = history_db_path
        self.history_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_history_db()
        self._init_memory_json()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.history_db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_history_db(self) -> None:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                top_celex TEXT,
                summary TEXT,
                tags_json TEXT,
                model_used TEXT
            )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_searches_time ON searches(timestamp)")
            conn.commit()

    def _init_memory_json(self) -> None:
        if not self.json_path.exists():
            self.reset_to_defaults()

    def load_memory(self) -> Dict[str, Any]:
        """Loads user memory dictionary from JSON."""
        try:
            if self.json_path.exists():
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        if "interactions" not in data:
                            data["interactions"] = []
                        return data
        except Exception as e:
            logger.error(f"Error reading {self.json_path}: {e}")
        return {
            "user_context": "",
            "preferred_topics": [],
            "preferred_sectors": [],
            "notes": "",
            "interactions": [],
            "updated_at": datetime.now().isoformat(),
        }

    def save_memory(self, memory_dict: Dict[str, Any]) -> None:
        """Saves user memory dictionary to JSON."""
        memory_dict["updated_at"] = datetime.now().isoformat()
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(memory_dict, f, indent=2, ensure_ascii=False)

    def accumulate_interaction(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        summary: str = "",
        model_used: str = "",
    ) -> None:
        """Extracts key themes from query and results and appends to structured interactions log."""
        mem = self.load_memory()
        interactions = mem.get("interactions", [])

        # Extract top categories from retrieved docs
        all_topics = []
        all_sectors = []
        top_celex = []

        for doc in retrieved_docs[:5]:
            top_celex.append(doc.get("celex", ""))
            tags = doc.get("tags", {})
            for item in tags.get("Regulatory Topics", []):
                all_topics.append(item.get("category", ""))
            for item in tags.get("Industry / Sector", []):
                all_sectors.append(item.get("category", ""))

        topic_counts = Counter([t for t in all_topics if t]).most_common(3)
        sector_counts = Counter([s for s in all_sectors if s]).most_common(3)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "top_celex": [c for c in top_celex if c],
            "key_topics": [t[0] for t in topic_counts],
            "key_sectors": [s[0] for s in sector_counts],
            "model_used": model_used,
        }

        # Keep rolling window of last 25 interactions
        interactions.append(entry)
        mem["interactions"] = interactions[-25:]

        # Also auto-update preferred topics/sectors if interactions show strong patterns
        if len(interactions) >= 3:
            recent_topics = [t for it in interactions[-10:] for t in it.get("key_topics", [])]
            top_t = [t[0] for t in Counter(recent_topics).most_common(4)]
            if top_t:
                # Merge into existing preferred topics without duplicates
                current_p = mem.get("preferred_topics", [])
                merged = list(dict.fromkeys(current_p + top_t))
                mem["preferred_topics"] = merged[:6]

        self.save_memory(mem)

    def synthesise_memory_profile(self, model_name: Optional[str] = None) -> str:
        """Summarises accumulated interactions into a compact 2-3 sentence persona paragraph."""
        mem = self.load_memory()
        interactions = mem.get("interactions", [])

        if not interactions:
            return mem.get("user_context", "")

        # Collect interaction history
        queries = [it["query"] for it in interactions[-10:]]
        topics = [t for it in interactions[-10:] for t in it.get("key_topics", [])]
        most_common_topics = [t[0] for t in Counter(topics).most_common(4)]

        prompt = (
            "Based on the following recent EUR-Lex legal search queries and identified regulatory themes, "
            "write a concise 2-sentence user persona describing their legal research focus and priority interests.\n\n"
            f"Recent Search Queries: {', '.join(queries[-5:])}\n"
            f"Key Regulatory Topics: {', '.join(most_common_topics)}\n\n"
            "Provide only the 2-sentence user persona statement without preamble."
        )

        synthesised_text = ""
        model = model_name or config.DEFAULT_FALLBACK_MODEL
        try:
            client = ollama.Client()
            resp = client.generate(model=model, prompt=prompt, options={"temperature": 0.3})
            synthesised_text = resp.get("response", "").strip()
        except Exception as e:
            logger.warning(f"Ollama synthesis failed ({e}). Generating template persona.")
            topics_str = ", ".join(most_common_topics) if most_common_topics else "EU regulatory compliance"
            synthesised_text = f"Legal researcher focused on {topics_str}. Specialises in European Union statutory obligations and market impacts."

        if synthesised_text:
            mem["user_context"] = synthesised_text
            self.save_memory(mem)

        return synthesised_text

    def get_prompt_context_string(self) -> str:
        """Constructs an analytical context block to inject into prompts."""
        mem = self.load_memory()
        context_parts = []
        if mem.get("user_context"):
            context_parts.append(f"User Profile & Interests: {mem['user_context']}")
        if mem.get("preferred_topics"):
            topics = mem["preferred_topics"]
            topics_str = ", ".join(topics) if isinstance(topics, list) else str(topics)
            context_parts.append(f"Priority Regulatory Topics: {topics_str}")
        if mem.get("preferred_sectors"):
            sectors = mem["preferred_sectors"]
            sectors_str = ", ".join(sectors) if isinstance(sectors, list) else str(sectors)
            context_parts.append(f"Priority Industry Sectors: {sectors_str}")
        if mem.get("notes"):
            context_parts.append(f"Specific Directives: {mem['notes']}")
        return "\n".join(context_parts)

    def export_memory_json(self) -> str:
        """Exports user memory and accumulated interactions as formatted JSON."""
        return json.dumps(self.load_memory(), indent=2, ensure_ascii=False)

    def export_memory_markdown(self) -> str:
        """Exports user memory as human-readable Markdown report."""
        mem = self.load_memory()
        topics = ", ".join(mem.get("preferred_topics", [])) or "None specified"
        sectors = ", ".join(mem.get("preferred_sectors", [])) or "None specified"
        interactions = mem.get("interactions", [])

        md = [
            "# Personalised EUR-Lex Legal Search Profile",
            f"**Last Updated:** {mem.get('updated_at', 'N/A')}\n",
            "## User Persona & Background",
            f"{mem.get('user_context', 'No context set.')}\n",
            "## Regulatory Focus",
            f"- **Priority Topics:** {topics}",
            f"- **Industry Sectors:** {sectors}",
            f"- **Directives:** {mem.get('notes', 'None')}\n",
            f"## Accumulated Search Interactions ({len(interactions)} recorded)",
        ]

        for i, it in enumerate(reversed(interactions[-10:]), 1):
            md.append(f"{i}. **{it['query']}** ({it['timestamp'][:16]}) — *Topics: {', '.join(it.get('key_topics', []))}*")

        return "\n".join(md)

    def reset_to_defaults(self) -> Dict[str, Any]:
        """Resets memory to default template state."""
        state = dict(DEFAULT_MEMORY)
        self.save_memory(state)
        return state

    def clear_memory(self) -> None:
        """Resets memory to clean state."""
        self.save_memory({
            "user_context": "",
            "preferred_topics": [],
            "preferred_sectors": [],
            "notes": "",
            "interactions": [],
            "updated_at": datetime.now().isoformat(),
        })

    def log_search(
        self,
        query: str,
        top_celex_list: List[str],
        summary: str,
        tags: Optional[Dict[str, Any]] = None,
        model_used: str = "ollama",
    ) -> int:
        """Records a search interaction into SQLite history."""
        timestamp = datetime.now().isoformat()
        tags_str = json.dumps(tags or {})
        top_celex_str = ",".join(top_celex_list)

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO searches (query, timestamp, top_celex, summary, tags_json, model_used)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (query, timestamp, top_celex_str, summary, tags_str, model_used))
            conn.commit()
            return cursor.lastrowid

    def get_recent_searches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetches the most recent searches."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT id, query, timestamp, top_celex, summary, tags_json, model_used
            FROM searches
            ORDER BY id DESC
            LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
