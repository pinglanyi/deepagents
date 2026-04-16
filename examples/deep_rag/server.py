"""Production FastAPI server for the Deep RAG agent.

Runs the agent directly in-process — no langgraph dev, no closed-source platform.
Uses the same SQLite/Postgres checkpointer logic as checkpointer.py but wires it
directly into the agent rather than delegating to langgraph.json.

Start:
    uv run uvicorn server:app --host 0.0.0.0 --port 8123
    uv run uvicorn server:app --host 0.0.0.0 --port 8123 --workers 4  # Postgres only

With gunicorn (async workers, production):
    uv run gunicorn server:app -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8123
    Note: use --workers 1 for SQLite (single-process only)

API:
    POST /chat                         non-streaming chat
    POST /chat/stream                  SSE streaming chat
    GET  /threads                      list threads (newest first, with preview)
    GET  /threads/{id}                 full conversation history
    GET  /threads/{id}/export          export as plain-text transcript
    DELETE /threads/{id}               delete a thread
    DELETE /threads                    delete ALL threads
    GET  /health                       liveness probe

RAGFlow knowledge-base management (prefix: /ragflow):
    POST   /ragflow/datasets                                      create dataset
    GET    /ragflow/datasets                                      list datasets
    PUT    /ragflow/datasets/{dataset_id}                         update dataset
    DELETE /ragflow/datasets                                      delete datasets

    POST   /ragflow/datasets/{id}/documents/upload                upload single doc
    POST   /ragflow/datasets/{id}/documents/upload/batch          upload multiple docs
    GET    /ragflow/datasets/{id}/documents                       list documents
    PUT    /ragflow/datasets/{id}/documents/{doc_id}              update document
    DELETE /ragflow/datasets/{id}/documents                       delete documents

    POST   /ragflow/datasets/{id}/documents/parse                 start parsing
    DELETE /ragflow/datasets/{id}/documents/parse                 stop parsing

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
from fastapi.responses import PlainTextResponse, StreamingResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from rag_agent.prompts import DEEP_RAG_ANSWER_FORMAT, DEEP_RAG_WORKFLOW_INSTRUCTIONS
from rag_agent.ragflow_routes import router as ragflow_router
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
    kwargs: dict[str, Any] = {
        "model": os.getenv("DEEP_RAG_MODEL", "deepseek-chat"),
        "temperature": 0.0,
    }
    api_key = os.getenv("DEEP_RAG_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEP_RAG_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "")
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

        async with AsyncPostgresSaver.from_conn_string(postgres_uri) as cp:
            await cp.setup()
            _state["agent"] = _make_agent(cp)
            _state["checkpointer"] = cp
            _state["backend"] = "postgres"
            yield
    else:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = str(AGENT_DATA_DIR / "checkpoints.sqlite")
        async with AsyncSqliteSaver.from_conn_string(db_path) as cp:
            _state["agent"] = _make_agent(cp)
            _state["checkpointer"] = cp
            _state["backend"] = "sqlite"
            yield

    _state.clear()


def _make_agent(checkpointer: Any) -> Any:
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


def _agent() -> Any:
    a = _state.get("agent")
    if a is None:
        raise RuntimeError("Agent not initialised — server is still starting up")
    return a


def _checkpointer() -> Any:
    return _state["checkpointer"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_text(content: Any) -> str:
    """Normalise LLM message content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _messages_from_checkpoint(cp_tuple: Any) -> list[Any]:
    """Extract the messages list from a CheckpointTuple safely.

    Reads from ``checkpoint["channel_values"]["messages"]`` which contains
    LangChain message objects (HumanMessage, AIMessage, …).  This avoids
    ``metadata["writes"]`` which can contain non-serialisable LangGraph
    internals such as ``langgraph.types.Send``.
    """
    try:
        checkpoint = cp_tuple.checkpoint
        if not isinstance(checkpoint, dict):
            return []
        return checkpoint.get("channel_values", {}).get("messages", []) or []
    except Exception:  # noqa: BLE001
        return []


def _serialize_messages(messages: list[Any]) -> list[dict[str, str]]:
    """Convert LangChain message objects to JSON-safe dicts."""
    result = []
    for m in messages:
        mtype = getattr(m, "type", None) or getattr(m, "role", "unknown")
        # Normalise role names
        role = {"human": "user", "ai": "assistant"}.get(mtype, mtype)
        content = _extract_text(getattr(m, "content", ""))
        if content:  # skip empty tool-call scaffolding messages
            result.append({"role": role, "content": content})
    return result


# ---------------------------------------------------------------------------
# Thread deletion helpers
# ---------------------------------------------------------------------------


async def _delete_thread_sqlite(thread_id: str) -> None:
    """Delete all checkpoints for *thread_id* from SQLite."""
    conn = _checkpointer().conn  # aiosqlite.Connection
    await conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
    await conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
    await conn.commit()


async def _delete_thread_postgres(thread_id: str) -> None:
    """Delete all checkpoints for *thread_id* from Postgres."""
    cp = _checkpointer()
    # Newer langgraph-checkpoint-postgres exposes adelete_thread directly.
    if hasattr(cp, "adelete_thread"):
        await cp.adelete_thread(thread_id)
        return
    # Fallback: raw SQL via the underlying psycopg pool.
    if hasattr(cp, "conn"):
        async with await cp.conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,)
            )
            await cur.execute(
                "DELETE FROM writes WHERE thread_id = %s", (thread_id,)
            )
        await cp.conn.commit()
        return
    raise HTTPException(status_code=501, detail="Postgres delete not supported by this version")


async def _delete_thread(thread_id: str) -> None:
    if _state["backend"] == "sqlite":
        await _delete_thread_sqlite(thread_id)
    else:
        await _delete_thread_postgres(thread_id)


async def _list_all_thread_ids() -> list[str]:
    """Return deduplicated thread IDs from the checkpointer, newest first."""
    seen: dict[str, str] = {}  # thread_id → latest ts
    async for cp in _checkpointer().alist(None):
        cfg = cp.config.get("configurable") or {}
        tid = cfg.get("thread_id")
        if not tid:
            continue
        ts = ""
        if isinstance(cp.checkpoint, dict):
            ts = str(cp.checkpoint.get("ts") or "")
        if tid not in seen or ts > seen[tid]:
            seen[tid] = ts
    return [tid for tid, _ in sorted(seen.items(), key=lambda x: x[1], reverse=True)]


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


class MessageOut(BaseModel):
    """A single message in a conversation."""

    role: str
    content: str


class ThreadSummary(BaseModel):
    """Brief thread metadata for list views."""

    thread_id: str
    updated_at: str
    message_count: int
    last_message: str  # truncated last user message


class ThreadDetail(BaseModel):
    """Full thread conversation."""

    thread_id: str
    messages: list[MessageOut]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Deep RAG Agent", version="0.1.0", lifespan=_lifespan)
app.include_router(ragflow_router)


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

    Event format — each line is ``data: <json>\\n\\n``:

    * ``{"thread_id": "..."}`` — emitted first, carry this for follow-up turns
    * ``{"text": "..."}``      — incremental text token
    * ``{"done": true}``       — stream finished
    * ``{"error": "..."}``     — unrecoverable error
    """
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def _generate():
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
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "X-Thread-Id": thread_id,
        },
    )


# ---------------------------------------------------------------------------
# Routes — thread management
# ---------------------------------------------------------------------------


@app.get("/threads", response_model=list[ThreadSummary])
async def list_threads(limit: int = 50, offset: int = 0) -> list[ThreadSummary]:
    """List threads newest-first with a preview of the last user message.

    Deduplicates across checkpoints (LangGraph writes one checkpoint per agent
    step, not per conversation turn).  Only ``checkpoint["channel_values"]``
    is read — ``metadata["writes"]`` is intentionally skipped to avoid
    ``langgraph.types.Send`` serialisation errors.
    """
    # One pass: collect latest checkpoint per thread_id
    best: dict[str, Any] = {}  # thread_id → CheckpointTuple
    async for cp in _checkpointer().alist(None):
        cfg = cp.config.get("configurable") or {}
        tid = cfg.get("thread_id")
        if not tid:
            continue
        ts = str(cp.checkpoint.get("ts") or "") if isinstance(cp.checkpoint, dict) else ""
        if tid not in best or ts > str(best[tid].checkpoint.get("ts") or ""):
            best[tid] = cp

    # Build summary rows
    rows: list[ThreadSummary] = []
    for tid, cp in sorted(
        best.items(),
        key=lambda kv: str(kv[1].checkpoint.get("ts") or ""),
        reverse=True,
    ):
        ts = str(cp.checkpoint.get("ts") or "") if isinstance(cp.checkpoint, dict) else ""
        msgs = _messages_from_checkpoint(cp)
        last_user = ""
        for m in reversed(msgs):
            if getattr(m, "type", None) == "human":
                last_user = _extract_text(getattr(m, "content", ""))[:120]
                break
        rows.append(
            ThreadSummary(
                thread_id=tid,
                updated_at=ts,
                message_count=len([m for m in msgs if getattr(m, "type", None) in ("human", "ai")]),
                last_message=last_user,
            )
        )

    return rows[offset : offset + limit]


@app.get("/threads/{thread_id}", response_model=ThreadDetail)
async def get_thread(thread_id: str) -> ThreadDetail:
    """Return the full conversation history for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    state = await _agent().aget_state(config)
    if state is None or not state.values:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
    messages = _serialize_messages(state.values.get("messages", []))
    return ThreadDetail(thread_id=thread_id, messages=[MessageOut(**m) for m in messages])


@app.get("/threads/{thread_id}/export", response_class=PlainTextResponse)
async def export_thread(thread_id: str) -> str:
    """Export the conversation as a plain-text transcript."""
    config = {"configurable": {"thread_id": thread_id}}
    state = await _agent().aget_state(config)
    if state is None or not state.values:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")

    lines = [f"# Conversation — thread {thread_id}", ""]
    for m in _serialize_messages(state.values.get("messages", [])):
        speaker = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"## {speaker}")
        lines.append(m["content"])
        lines.append("")
    return "\n".join(lines)


@app.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(thread_id: str) -> None:
    """Delete a single thread and all its checkpoints."""
    await _delete_thread(thread_id)


@app.delete("/threads", status_code=204)
async def delete_all_threads() -> None:
    """Delete every thread (irreversible)."""
    thread_ids = await _list_all_thread_ids()
    for tid in thread_ids:
        await _delete_thread(tid)


# ---------------------------------------------------------------------------
# Routes — ops
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "backend": _state.get("backend", "unknown")}
