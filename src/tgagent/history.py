"""SQLite-backed chat history with token-budget context building."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from .splitter import est_tokens

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
    content TEXT NOT NULL,
    est_tokens INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
"""


class History:
    """Stores user/assistant turns per chat; builds a token-budgeted context."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    def append(self, chat_id: int, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages(chat_id, role, content, est_tokens) VALUES (?, ?, ?, ?)",
                (chat_id, role, content, est_tokens(content)),
            )
            self._conn.commit()

    def reset(self, chat_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            self._conn.commit()

    def import_messages(
        self, chat_id: int, messages: list[tuple[str, str]], reset: bool = True
    ) -> None:
        """Insert a batch of (role, content) turns, optionally replacing the chat's history."""
        with self._lock:
            if reset:
                self._conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            self._conn.executemany(
                "INSERT INTO messages(chat_id, role, content, est_tokens) VALUES (?, ?, ?, ?)",
                [(chat_id, role, content, est_tokens(content)) for role, content in messages],
            )
            self._conn.commit()

    def all_messages(self, chat_id: int) -> list[tuple[str, str]]:
        """Return the full conversation for a chat, oldest first (for /export)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id",
                (chat_id,),
            ).fetchall()
        return [(row["role"], row["content"]) for row in rows]

    def build_context(
        self, chat_id: int, system_prompt: str, budget_tokens: int
    ) -> list[dict[str, Any]]:
        """Return [system] + newest turns that fit into the budget (oldest-first)."""
        system_cost = est_tokens(system_prompt)
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, est_tokens FROM messages WHERE chat_id = ? ORDER BY id DESC",
                (chat_id,),
            ).fetchall()
        picked: list[sqlite3.Row] = []
        used = system_cost
        for row in rows:  # newest first; always keep at least the newest turn
            if used + row["est_tokens"] > budget_tokens and picked:
                break
            picked.append(row)
            used += row["est_tokens"]
        picked.reverse()
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend({"role": row["role"], "content": row["content"]} for row in picked)
        return messages

    def stats(self, chat_id: int) -> dict[str, int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(est_tokens), 0) AS tokens"
                " FROM messages WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
        return {"messages": row["n"], "est_tokens": row["tokens"]}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
