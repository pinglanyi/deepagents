"""Deep RAG Agent — LangGraph entry point.

Fast-retrieval mode:
  1. Extract model number from question (no tool)
  2. ragflow_list_datasets(name_filter=<model>) → real dataset IDs
  3. ragflow_retrieve(dataset_ids=<matched>) → specific dataset first
  4. Fallback: ragflow_retrieve(dataset_ids=[]) if specific yields nothing
  5. get_next_chunks() at most once if more detail needed
  6. Answer inline — no planning, no file writing

Persistence:
  - FilesystemBackend → all write_file / edit_file calls go to AGENT_DATA_DIR on disk
  - memory=["/AGENTS.md"] → loaded into system prompt on every request (cross-session memory)
  - Conversation history: managed by LangGraph Platform when using `langgraph dev`
    (each thread_id keeps its own turn-by-turn history in Platform's SQLite)

Usage:
  langgraph dev --port 8122   # LangGraph Studio with auto checkpointing
  uv run agent.py             # interactive terminal (no checkpointing)
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from rag_agent.prompts import (
    DEEP_RAG_ANSWER_FORMAT,
    DEEP_RAG_WORKFLOW_INSTRUCTIONS,
)
from rag_agent.tools import get_next_chunks, ragflow_list_datasets, ragflow_retrieve

load_dotenv()

# ---------------------------------------------------------------------------
# Persistent file storage
# ---------------------------------------------------------------------------
# All write_file / edit_file / read_file calls go to this directory on disk.
# - StateBackend (default) keeps files in graph state → lost on new thread
# - FilesystemBackend writes to real disk → persists across sessions/threads

AGENT_DATA_DIR = Path(os.getenv("AGENT_DATA_DIR", "./agent_data")).resolve()
AGENT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Pre-create AGENTS.md so the agent can edit it without a write-first step.
_agents_md = AGENT_DATA_DIR / "AGENTS.md"
if not _agents_md.exists():
    _agents_md.write_text(
        "# Agent Long-term Memory\n\n"
        "This file is loaded automatically at the start of every conversation.\n"
        "Use edit_file('/AGENTS.md', old_string='...', new_string='...') to update it.\n\n"
        "## User Preferences\n\n(none recorded yet)\n\n"
        "## Domain Knowledge\n\n(none recorded yet)\n",
        encoding="utf-8",
    )

backend = FilesystemBackend(root_dir=AGENT_DATA_DIR, virtual_mode=True)

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
    # /AGENTS.md → AGENT_DATA_DIR/AGENTS.md on disk.
    # Loaded into system prompt on every request → cross-session long-term memory.
    memory=["/AGENTS.md"],
)
