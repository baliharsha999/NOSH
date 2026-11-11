# -*- coding: utf-8 -*-
"""
backend/src/session_store.py
----------------------------
Persistent SQLite-Backed Conversation & Thread State Store for NOSH.

Solves the multi-turn memory persistence requirement:
  - Tracks unique conversation / thread IDs (`thread_id`).
  - Persists accumulated dietary restrictions across application restarts.
  - Retains full conversation turn history with query intents, LLM chef responses,
    and structured card payloads.
  - Storage location: `data_recipes/nosh_sessions.db` (zero external service dependency).
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Set, Any, Optional
from datetime import datetime

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data_recipes" / "nosh_sessions.db"


class SQLiteSessionStore:
    """
    Manages persistent SQLite storage for multi-turn conversational threads.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """Creates tables for threads and turn history if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_threads (
                    thread_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    active_search_query TEXT DEFAULT '',
                    accumulated_restrictions TEXT DEFAULT '[]'
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    user_query TEXT NOT NULL,
                    intent_detected TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    structured_data TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(thread_id) REFERENCES conversation_threads(thread_id) ON DELETE CASCADE
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_turns_thread ON conversation_turns(thread_id);")
            conn.commit()

    def get_or_create_thread(self, thread_id: str) -> Dict[str, Any]:
        """
        Retrieves an existing thread state or creates a new one in SQLite.
        """
        thread_id = thread_id.strip() or "default"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM conversation_threads WHERE thread_id = ?", (thread_id,))
            row = cursor.fetchone()
            if row:
                try:
                    restrictions = json.loads(row["accumulated_restrictions"])
                except Exception:
                    restrictions = []
                return {
                    "thread_id": row["thread_id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "active_search_query": row["active_search_query"] or "",
                    "accumulated_restrictions": set(restrictions)
                }

            # Create new thread row
            now_iso = datetime.utcnow().isoformat()
            cursor.execute("""
                INSERT INTO conversation_threads (thread_id, created_at, updated_at, active_search_query, accumulated_restrictions)
                VALUES (?, ?, ?, '', '[]')
            """, (thread_id, now_iso, now_iso))
            conn.commit()
            return {
                "thread_id": thread_id,
                "created_at": now_iso,
                "updated_at": now_iso,
                "active_search_query": "",
                "accumulated_restrictions": set()
            }

    def load_turn_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Loads all prior conversational turns for a specific thread."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT turn_index, user_query, intent_detected, response_text, structured_data, created_at
                FROM conversation_turns
                WHERE thread_id = ?
                ORDER BY turn_index ASC
            """, (thread_id,))
            rows = cursor.fetchall()
            history = []
            for r in rows:
                try:
                    structured = json.loads(r["structured_data"]) if r["structured_data"] else None
                except Exception:
                    structured = None
                history.append({
                    "turn_index": r["turn_index"],
                    "user_query": r["user_query"],
                    "intent_detected": r["intent_detected"],
                    "response_text": r["response_text"],
                    "structured_data": structured,
                    "created_at": r["created_at"]
                })
            return history

    def save_turn(
        self,
        thread_id: str,
        user_query: str,
        intent_detected: str,
        response_text: str,
        structured_data: Optional[Dict[str, Any]],
        active_search_query: str,
        accumulated_restrictions: Set[str]
    ):
        """
        Atomically saves a new conversation turn and updates the thread's memory state.
        """
        now_iso = datetime.utcnow().isoformat()
        restrictions_json = json.dumps(list(accumulated_restrictions))
        structured_json = json.dumps(structured_data or {})

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Update thread metadata
            cursor.execute("""
                INSERT INTO conversation_threads (thread_id, created_at, updated_at, active_search_query, accumulated_restrictions)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    active_search_query = excluded.active_search_query,
                    accumulated_restrictions = excluded.accumulated_restrictions
            """, (thread_id, now_iso, now_iso, active_search_query, restrictions_json))

            # 2. Get next turn index
            cursor.execute("SELECT COALESCE(MAX(turn_index), 0) + 1 FROM conversation_turns WHERE thread_id = ?", (thread_id,))
            next_turn_idx = cursor.fetchone()[0]

            # 3. Insert turn record
            cursor.execute("""
                INSERT INTO conversation_turns (thread_id, turn_index, user_query, intent_detected, response_text, structured_data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (thread_id, next_turn_idx, user_query, intent_detected, response_text, structured_json, now_iso))

            conn.commit()

    def update_restrictions(self, thread_id: str, accumulated_restrictions: Set[str]):
        """Updates dietary restrictions for a thread in SQLite."""
        restrictions_json = json.dumps(list(accumulated_restrictions))
        now_iso = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE conversation_threads
                SET accumulated_restrictions = ?, updated_at = ?
                WHERE thread_id = ?
            """, (restrictions_json, now_iso, thread_id))
            conn.commit()

    def clear_restrictions(self, thread_id: str):
        """Clears all accumulated dietary restrictions for a thread."""
        self.update_restrictions(thread_id, set())

    def list_threads(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists recent conversation threads with turn counts and last updated times."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT t.thread_id, t.created_at, t.updated_at, t.active_search_query, t.accumulated_restrictions,
                       COUNT(c.id) as total_turns
                FROM conversation_threads t
                LEFT JOIN conversation_turns c ON t.thread_id = c.thread_id
                GROUP BY t.thread_id
                ORDER BY t.updated_at DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            threads = []
            for r in rows:
                try:
                    restrs = json.loads(r["accumulated_restrictions"])
                except Exception:
                    restrs = []
                threads.append({
                    "thread_id": r["thread_id"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "active_search_query": r["active_search_query"],
                    "accumulated_restrictions": restrs,
                    "total_turns": r["total_turns"]
                })
            return threads

    def delete_thread(self, thread_id: str):
        """Deletes a thread and all associated turns."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversation_turns WHERE thread_id = ?", (thread_id,))
            cursor.execute("DELETE FROM conversation_threads WHERE thread_id = ?", (thread_id,))
            conn.commit()
