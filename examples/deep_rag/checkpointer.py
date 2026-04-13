"""Persistent checkpointer for the Deep RAG agent.

Loaded by langgraph.json via ``"checkpointer": {"path": "./checkpointer.py:create_checkpointer"}``.
LangGraph calls this async context manager once at server startup and injects
the yielded saver into every graph invocation, providing cross-restart
conversation history without requiring LangGraph Cloud.

Backend selection (checked in order):
  1. ``POSTGRES_URI`` is set → ``AsyncPostgresSaver`` (requires ``langgraph-checkpoint-postgres``).
  2. Otherwise            → ``AsyncSqliteSaver``    (built-in transitive dep, zero config).

SQLite database is written to ``AGENT_DATA_DIR/checkpoints.sqlite`` (defaults to
``./agent_data/checkpoints.sqlite`` relative to the project root).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path


@asynccontextmanager
async def create_checkpointer():
    """Yield a checkpoint saver backed by Postgres or SQLite.

    LangGraph Platform calls this at startup. The yielded saver is used for
    all thread checkpoints, giving persistent conversation history across
    server restarts.
    """
    postgres_uri = os.getenv("POSTGRES_URI")
    if postgres_uri:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as exc:
            raise ImportError(
                "POSTGRES_URI is set but langgraph-checkpoint-postgres is not installed.\n"
                "Run: uv pip install 'langgraph-checkpoint-postgres>=2.0.0' 'psycopg[binary]>=3.1.0'"
            ) from exc

        async with AsyncPostgresSaver.from_conn_string(postgres_uri) as saver:
            await saver.setup()  # idempotent — creates tables if absent
            yield saver
    else:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        agent_data_dir = Path(os.getenv("AGENT_DATA_DIR", "./agent_data")).resolve()
        agent_data_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(agent_data_dir / "checkpoints.sqlite")

        async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
            yield saver
