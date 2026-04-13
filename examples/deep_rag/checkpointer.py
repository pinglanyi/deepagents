"""Persistent SQLite checkpointer for the Deep RAG agent.

Referenced from langgraph.json so `langgraph dev` uses disk storage
instead of the default in-memory SQLite that comes with [inmem].

Database location: AGENT_DATA_DIR/checkpoints.db
  (default: ./agent_data/checkpoints.db)

To move the database, set AGENT_DATA_DIR in .env:
  AGENT_DATA_DIR=/path/to/your/storage
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path


@asynccontextmanager
async def create_checkpointer():
    """Yield an AsyncSqliteSaver backed by a file on disk.

    Called by LangGraph Platform at server startup. The database path
    is read from the AGENT_DATA_DIR env var so it can be changed without
    touching code.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    agent_data_dir = Path(os.getenv("AGENT_DATA_DIR", "./agent_data")).resolve()
    agent_data_dir.mkdir(parents=True, exist_ok=True)

    db_path = agent_data_dir / "checkpoints.db"

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        yield saver
