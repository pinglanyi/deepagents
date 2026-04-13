"""Production FastAPI server for the Deep RAG agent.

Runs the agent directly in-process — no langgraph dev, no closed-source platform.
Uses the same SQLite/Postgres checkpointer logic as checkpointer.py but wires it
directly into the agent rather than delegating to langgraph.json.

Start:
    uv run uvicorn server:app --host 0.0.0.0 --port 8123
    uv run uvicorn server:app --host 0.0.0.0 --port 8123 --workers 1  # prod (single worker for SQLite)
    uv run uvicorn server:app --host 0.0.0.0 --port 8123 --workers 4  # prod (Postgres only)

With gunicorn (async workers):
    uv run gunicorn server:app -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:8123

API:
    POST /chat                         non-streaming chat
    POST /chat/stream                  SSE streaming chat
    GET  /threads                      list all threads
    GET  /threads/{id}                 get a single thread's message history
    DELETE /threads/{id}               delete a thread
    GET  /health                       liveness probe

Thread ID:
    Pass ``thread_id`` in the request body to continue an existing conversation.
    Omit it to start a new conversation (a UUID is generated automatically).
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from rag_agent.prompts import DEEP_RAG_ANSWER_FORMAT, DEEP_RAG_WORKFLOW_INSTRUCTIONS
from rag_agent.tools import get_next_chunks, ragflow_list_datasets, ragflow_retrieve

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AGENT_DATA_DIR = Path(os.getenv("AGENT_DATA_DIR", "./agent_data")).resolve()
AGENT_DATA_DIR.mkdir(parents=True, exist_ok=True)

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


def _build_model() -> ChatOpenAI:
    model_name = os.getenv("DEEP_RAG_MODEL", "deepseek-chat")
    api_key = os.getenv("DEEP_RAG_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEP_RAG_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "")
    kwargs: dict[str, Any] = {"model": model_name, "temperature": 0.0}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Lifespan — owns the checkpointer for the full server lifetime
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialise the checkpointer and agent, then clean up on shutdown."""
    postgres_uri = os.getenv("POSTGRES_URI")
    if postgres_uri:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as exc:
            raise ImportError(
                "POSTGRES_URI is set but langgraph-checkpoint-postgres is not installed.\n"
                "Run: uv pip install '.[postgres]'"
            ) from exc

        async with AsyncPostgresSaver.from_conn_string(postgres_uri) as checkpointer:
            await checkpointer.setup()
            _state["agent"] = _make_agent(checkpointer)
            yield
    else:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = str(AGENT_DATA_DIR / "checkpoints.sqlite")
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            _state["agent"] = _make_agent(checkpointer)
            yield

    _state.clear()


def _make_agent(checkpointer: Any):
    backend = FilesystemBackend(root_dir=AGENT_DATA_DIR, virtual_mode=True)
    system_prompt = (
        DEEP_RAG_WORKFLOW_INSTRUCTIONS + "\n\n" + "=" * 72 + "\n\n" + DEEP_RAG_ANSWER_FORMAT
    )
    return create_deep_agent(
        model=_build_model(),
        tools=[ragflow_list_datasets, ragflow_retrieve, get_next_chunks],
        system_prompt=system_prompt,
        backend=backend,
        memory=["/AGENTS.md"],
        checkpointer=checkpointer,
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Deep RAG Agent", version="0.1.0", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Chat request body."""

    thread_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    """Non-streaming chat response."""

    thread_id: str
    response: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent():
    a = _state.get("agent")
    if a is None:
        raise RuntimeError("Agent not initialised")
    return a


def _extract_text(content: Any) -> str:
    """Normalise an LLM message content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content)


# ---------------------------------------------------------------------------
# Routes — chat
# ---------------------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Send a message and receive the agent's full response."""
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = await _agent().ainvoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config=config,
    )
    last = result["messages"][-1]
    return ChatResponse(
        thread_id=thread_id,
        response=_extract_text(getattr(last, "content", str(last))),
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream the agent's response as Server-Sent Events.

    Each event is ``data: <json>\\n\\n`` where the JSON is one of:
      - ``{"thread_id": "..."}``          — emitted first, before any text
      - ``{"text": "..."}``               — incremental text token
      - ``{"done": true}``                — final event, stream is complete
    """
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def generate():
        yield f"data: {json.dumps({'thread_id': thread_id})}\n\n"
        try:
            async for event in _agent().astream_events(
                {"messages": [{"role": "user", "content": req.message}]},
                config=config,
                version="v2",
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    text = _extract_text(getattr(chunk, "content", ""))
                    if text:
                        yield f"data: {json.dumps({'text': text})}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
            "X-Thread-Id": thread_id,
        },
    )


# ---------------------------------------------------------------------------
# Routes — thread management
# ---------------------------------------------------------------------------


@app.get("/threads")
async def list_threads(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """List conversation threads, newest first."""
    checkpointer = _agent().checkpointer
    threads: list[dict[str, Any]] = []
    async for config, metadata, _, _, _ in checkpointer.alist(None, limit=limit + offset):
        tid = (config.get("configurable") or {}).get("thread_id")
        if tid:
            threads.append({"thread_id": tid, "metadata": metadata})
    return threads[offset : offset + limit]


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str) -> dict[str, Any]:
    """Return the message history for a single thread."""
    config = {"configurable": {"thread_id": thread_id}}
    state = await _agent().aget_state(config)
    if state is None or not state.values:
        raise HTTPException(status_code=404, detail="Thread not found")
    messages = []
    for m in state.values.get("messages", []):
        role = getattr(m, "type", None) or getattr(m, "role", "unknown")
        content = _extract_text(getattr(m, "content", ""))
        messages.append({"role": role, "content": content})
    return {"thread_id": thread_id, "messages": messages}


@app.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(thread_id: str) -> None:
    """Delete a thread and all its checkpoints."""
    checkpointer = _agent().checkpointer
    config = {"configurable": {"thread_id": thread_id}}
    # AsyncSqliteSaver / AsyncPostgresSaver both expose adelete
    if hasattr(checkpointer, "adelete"):
        await checkpointer.adelete(config)
    elif hasattr(checkpointer, "delete"):
        checkpointer.delete(config)
    else:
        raise HTTPException(status_code=501, detail="Checkpointer does not support deletion")


# ---------------------------------------------------------------------------
# Routes — ops
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
