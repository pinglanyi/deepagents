"""Agent singleton — initialised once at server startup, shared across requests.

Each user gets isolated conversation threads via the PostgreSQL checkpointer.
Memory is loaded from two sources:
  - ``/AGENTS.md`` — global system-wide knowledge (shared across all users)
  - ``/users/{user_id}/AGENTS.md`` — per-user memory injected at request time

Skills are loaded from the ``./skills/`` directory at startup and exposed via
progressive disclosure in the system prompt.

Context management:
  - Automatic summarization via SummarizationMiddleware (built into create_deep_agent)
  - Manual compaction via compact_conversation tool (SummarizationToolMiddleware)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from core.config import settings
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.summarization import (
    SummarizationToolMiddleware,
    create_summarization_middleware,
)
from rag_agent.prompts import DEEP_RAG_ANSWER_FORMAT, DEEP_RAG_WORKFLOW_INSTRUCTIONS
from rag_agent.tools import (
    complete_model_number,
    get_kb_datasets_by_type,
    get_next_chunks,
    ragflow_list_datasets,
    ragflow_retrieve,
)

_state: dict[str, Any] = {}


def get_agent() -> Any:
    """Return the initialised agent.  Raises RuntimeError if not yet ready."""
    agent = _state.get("agent")
    if agent is None:
        raise RuntimeError("Agent not initialised — server is still starting up")
    return agent


def get_backend() -> Any:
    """Return the FilesystemBackend used by the agent."""
    backend = _state.get("backend")
    if backend is None:
        raise RuntimeError("Agent not initialised — server is still starting up")
    return backend


def _build_model() -> ChatOpenAI:
    kwargs: dict[str, Any] = {
        "model": settings.deep_rag_model,
        "temperature": 0.0,
    }
    if settings.deep_rag_api_key:
        kwargs["api_key"] = settings.deep_rag_api_key
    if settings.deep_rag_base_url:
        kwargs["base_url"] = settings.deep_rag_base_url
    return ChatOpenAI(**kwargs)


def init_agent(checkpointer: AsyncPostgresSaver) -> None:
    """Build the agent with all middleware and store it as a singleton.

    This is called once at server startup (see ``main.py:lifespan``).
    """
    data_dir = settings.agent_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── Global memory ────────────────────────────────────────────────────
    agents_md = data_dir / "AGENTS.md"
    if not agents_md.exists():
        agents_md.write_text(
            "# Global Agent Memory\n\n"
            "This file is loaded automatically for every conversation.\n"
            "It contains system-wide knowledge shared across all users.\n"
            "Update it via read_file('/AGENTS.md') then edit_file('/AGENTS.md', ...).\n\n"
            "## System Knowledge\n\n(none recorded yet)\n\n"
            "## Domain Knowledge\n\n(none recorded yet)\n",
            encoding="utf-8",
        )

    backend = FilesystemBackend(root_dir=data_dir, virtual_mode=True)

    # ── Skills ───────────────────────────────────────────────────────────
    _skills_dir = (Path(__file__).parent.parent / "skills").resolve()

    # ── System prompt ────────────────────────────────────────────────────
    system_prompt = (
        DEEP_RAG_WORKFLOW_INSTRUCTIONS
        + "\n\n"
        + "=" * 72
        + "\n\n"
        + DEEP_RAG_ANSWER_FORMAT
    )

    # ── Context management ───────────────────────────────────────────────
    model = _build_model()
    _summ_mw = create_summarization_middleware(model, backend)
    _compact_mw = SummarizationToolMiddleware(_summ_mw)

    # ── Agent ────────────────────────────────────────────────────────────
    agent = create_deep_agent(
        model=model,
        tools=[
            get_kb_datasets_by_type,
            complete_model_number,
            ragflow_list_datasets,
            ragflow_retrieve,
            get_next_chunks,
        ],
        system_prompt=system_prompt,
        backend=backend,
        skills=[str(_skills_dir)],
        memory=["/AGENTS.md"],
        middleware=[_compact_mw],
        checkpointer=checkpointer,
    )

    _state["agent"] = agent
    _state["backend"] = backend
    _state["model"] = model
