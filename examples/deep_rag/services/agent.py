"""Agent singleton — initialised once at server startup, shared across requests."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from core.config import settings
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
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


def init_agent(checkpointer: Any) -> None:
    """Build the agent with the given LangGraph checkpointer and store it."""
    data_dir = settings.agent_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    agents_md = data_dir / "AGENTS.md"
    if not agents_md.exists():
        agents_md.write_text(
            "# Agent Long-term Memory\n\n"
            "This file is loaded automatically at the start of every conversation.\n"
            "Update it via read_file('/AGENTS.md') then edit_file('/AGENTS.md', ...).\n\n"
            "## User Preferences\n\n(none recorded yet)\n\n"
            "## Domain Knowledge\n\n(none recorded yet)\n",
            encoding="utf-8",
        )

    backend = FilesystemBackend(root_dir=data_dir, virtual_mode=True)
    system_prompt = (
        DEEP_RAG_WORKFLOW_INSTRUCTIONS
        + "\n\n"
        + "=" * 72
        + "\n\n"
        + DEEP_RAG_ANSWER_FORMAT
    )
    _state["agent"] = create_deep_agent(
        model=_build_model(),
        tools=[
            get_kb_datasets_by_type,
            complete_model_number,
            ragflow_list_datasets,
            ragflow_retrieve,
            get_next_chunks,
        ],
        system_prompt=system_prompt,
        backend=backend,
        memory=["/AGENTS.md"],
        checkpointer=checkpointer,
    )
