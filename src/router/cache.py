"""SQLite response cache keyed by (model_id, prompt).

The single most important cost control in this project. Your benchmark will
be re-run many times while you debug the scoring code; without this, every
re-run is a fresh API bill. With it, only genuinely new (model, prompt) pairs
cost money.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .config import CACHE_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def _key(model_id: str, prompt: str) -> str:
    return hashlib.sha256(f"{model_id}::{prompt}".encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: Path = CACHE_PATH, enabled: bool = True):
        self.enabled = enabled
        self.path = path
        # FastAPI runs sync endpoints on a thread pool, so one connection is
        # shared across threads; sqlite3 leaves serialising that to the caller.
        self._lock = threading.Lock()
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(path, check_same_thread=False)
            self.conn.execute(_SCHEMA)
            self.conn.commit()

    def get(self, model_id: str, prompt: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self._lock:
            row = self.conn.execute(
                "SELECT payload FROM cache WHERE key = ?", (_key(model_id, prompt),)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, model_id: str, prompt: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO cache (key, model_id, payload) VALUES (?, ?, ?)",
                (_key(model_id, prompt), model_id, json.dumps(payload)),
            )
            self.conn.commit()

    def stats(self) -> dict[str, int]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT model_id, COUNT(*) FROM cache GROUP BY model_id"
            ).fetchall()
        return {model_id: count for model_id, count in rows}
