"""Deep RAG Agent — LangGraph entry point.

Fast-retrieval mode:
  1. Extract model number from question (no tool)
  2. ragflow_list_datasets(name_filter=<model>) → real dataset IDs
  3. ragflow_retrieve(dataset_ids=<matched>) → specific dataset first
  4. Fallback: ragflow_retrieve(dataset_ids=[]) if specific yields nothing
  5. get_next_chunks() at most once if more detail needed
  6. Answer inline — no planning, no file writing

Persistence:
  - FilesystemBackend  → write_file / edit_file go to AGENT_DATA_DIR on disk
  - memory=["/AGENTS.md"] → cross-session long-term memory, loaded on every request
  - SqliteSaver checkpointer → conversation history in AGENT_DATA_DIR/checkpoints.db
    Baked into the compiled graph so langgraph dev uses it automatically.
    Location configurable via AGENT_DATA_DIR env var.

Usage:
  uv run langgraph dev --port 8122   # LangGraph Studio
  uv run agent.py                    # interactive terminal
"""

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from rag_agent.prompts import (
    DEEP_RAG_ANSWER_FORMAT,
    DEEP_RAG_WORKFLOW_INSTRUCTIONS,
)
from rag_agent.tools import get_next_chunks, ragflow_list_datasets, ragflow_retrieve

load_dotenv()

# ---------------------------------------------------------------------------
# Persistent storage directory
# ---------------------------------------------------------------------------
# All file operations and checkpoints live under AGENT_DATA_DIR.
# Change the location by setting AGENT_DATA_DIR in .env.

AGENT_DATA_DIR = Path(os.getenv("AGENT_DATA_DIR", "./agent_data")).resolve()
AGENT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# FilesystemBackend — write_file / edit_file go to disk, not graph state
# ---------------------------------------------------------------------------

_agents_md = AGENT_DATA_DIR / "AGENTS.md"
if not _agents_md.exists():
    _agents_md.write_text(
        "# Agent Long-term Memory\n\n"
        "This file is loaded automatically at the start of every conversation.\n"
        "Update it with: read_file('/AGENTS.md') then edit_file('/AGENTS.md', ...)\n\n"
        "## User Preferences\n\n(none recorded yet)\n\n"
        "## Domain Knowledge\n\n(none recorded yet)\n",
        encoding="utf-8",
    )

backend = FilesystemBackend(root_dir=AGENT_DATA_DIR, virtual_mode=True)

# ---------------------------------------------------------------------------
# SqliteSaver checkpointer — conversation history on disk
# ---------------------------------------------------------------------------
# Baked into the compiled graph. langgraph dev detects this and uses it
# instead of its own in-memory checkpointer.
# DB location: AGENT_DATA_DIR/checkpoints.db

_conn = sqlite3.connect(
    str(AGENT_DATA_DIR / "checkpoints.db"),
    check_same_thread=False,  # langgraph calls from multiple threads
)
checkpointer = SqliteSaver(_conn)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = DEEP_RAG_WORKFLOW_INSTRUCTIONS + "\n\n" + "=" * 72 + "\n\n" + DEEP_RAG_ANSWER_FORMAT

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_model_name = os.getenv("DEEP_RAG_MODEL", "deepseek-chat")
_api_key = os.getenv("DEEP_RAG_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
_base_url = os.getenv("DEEP_RAG_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "")

model_kwargs: dict = {"model": _model_name, "temperature": 0.0}
if _api_key:
    model_kwargs["api_key"] = _api_key
if _base_url:
    model_kwargs["base_url"] = _base_url

model = ChatOpenAI(**model_kwargs)

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = create_deep_agent(
    model=model,
    tools=[ragflow_list_datasets, ragflow_retrieve, get_next_chunks],
    system_prompt=SYSTEM_PROMPT,
    backend=backend,
    memory=["/AGENTS.md"],
    checkpointer=checkpointer,
)

