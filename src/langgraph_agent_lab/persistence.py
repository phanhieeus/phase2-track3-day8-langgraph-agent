"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    - "none"   → no persistence
    - "memory" → in-process MemorySaver (default)
    - "sqlite" → durable SqliteSaver (survives process restarts → crash-resume)
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        # database_url like "sqlite:///outputs/checkpoints.db" or a plain path.
        db_path = (database_url or "outputs/checkpoints.db").replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # langgraph-checkpoint-sqlite 3.x: construct from a sqlite3 connection.
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return SqliteSaver(conn=conn)
    if kind == "postgres":
        raise NotImplementedError(
            "TODO(student): implement Postgres checkpointer (optional extension)"
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
