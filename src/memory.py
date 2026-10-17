"""User memory management and search history tracking."""

import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_MEMORY = {
    "user_context": "Focus on high-impact AI safety regulations, GDPR data subject rights, and cross-border tech compliance.",
    "preferred_topics": ["Data Protection and Privacy", "Digital Services"],
    "preferred_sectors": ["Technology and Digital", "Finance and Banking"],
    "notes": "Interested in direct accountability for deployers and liability caps.",
}


class UserMemoryStore:
    """Manages persistent user memory (JSON) and search history (SQLite)."""

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
                        return data
        except Exception as e:
            logger.error(f"Error reading {self.json_path}: {e}")
        return {
            "user_context": "",
            "preferred_topics": [],
            "preferred_sectors": [],
            "notes": "",
            "updated_at": datetime.now().isoformat(),
        }

    def save_memory(self, memory_dict: Dict[str, Any]) -> None:
        """Saves user memory dictionary to JSON."""
        memory_dict["updated_at"] = datetime.now().isoformat()
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(memory_dict, f, indent=2, ensure_ascii=False)

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
