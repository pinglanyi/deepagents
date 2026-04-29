"""Chat router — invoke the agent with per-user memory injection.

Each chat request injects the user's personal memory (from
``/users/{user_id}/AGENTS.md``) into the agent invocation alongside the
global memory loaded at startup.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.database import get_db
from core.deps import get_current_user
from memory_manager import MemoryManager
from models.thread import Thread
from models.user import User
from schemas.thread import ChatRequest, ChatResponse
from services.agent import get_agent

router = APIRouter(prefix="/chat", tags=["Chat"])

# Per-user memory manager — shares the same data_dir as the agent backend
_memory_manager = MemoryManager(settings.agent_data_dir)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


def _build_memory_augmented_messages(
    user: User,
    original_message: str,
) -> list[dict[str, Any]]:
    """Inject per-user memory context into the first user message.

    The memory context is prepended so the agent sees it as persistent
    context from previous conversations.
    """
    user_id = str(user.id)
    _memory_manager.ensure_user_memory(user_id, user.email)

    memory_ctx = _memory_manager.build_memory_context(user_id)
    memory_path = _memory_manager.get_user_memory_path(user_id)

    augmented_content = (
        f"{memory_ctx}\n\n"
        f"---\n\n"
        f"Your memory file is at `{memory_path}`. "
        f"Use read_file('{memory_path}') to read it and "
        f"edit_file('{memory_path}', ...) to update it.\n\n"
        f"User message:\n{original_message}"
    )

    return [{"role": "user", "content": augmented_content}]


async def _get_or_create_thread(
    thread_id: uuid.UUID | None,
    first_message: str,
    user: User,
    db: AsyncSession,
) -> Thread:
    """Return an existing thread (verifying ownership) or create a new one."""
    if thread_id is not None:
        result = await db.execute(
            select(Thread).where(
                Thread.id == thread_id,
                Thread.user_id == user.id,
            )
        )
        thread = result.scalar_one_or_none()
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        return thread

    # Auto-title from first 60 chars of the opening message
    title = first_message.strip()[:60] or "New Chat"
    thread = Thread(user_id=user.id, title=title)
    db.add(thread)
    await db.commit()
    await db.refresh(thread)
    return thread


async def _touch_thread(thread: Thread, user_message: str, db: AsyncSession) -> None:
    """Update denormalised preview fields after a successful exchange."""
    thread.message_count += 2  # user turn + assistant turn
    thread.last_message_preview = user_message[:200]
    thread.updated_at = datetime.now(timezone.utc)
    await db.commit()


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a message and receive the full response",
)
async def chat(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    thread = await _get_or_create_thread(
        req.thread_id, req.message, current_user, db
    )
    config = {
        "configurable": {"thread_id": str(thread.id)},
        "recursion_limit": settings.agent_recursion_limit,
    }

    # Inject per-user memory into the message
    messages = _build_memory_augmented_messages(current_user, req.message)

    result = await get_agent().ainvoke(
        {"messages": messages},
        config=config,
    )

    await _touch_thread(thread, req.message, db)

    last = result["messages"][-1]
    return ChatResponse(
        thread_id=thread.id,
        thread_title=thread.title,
        response=_extract_text(getattr(last, "content", str(last))),
    )


@router.post(
    "/stream",
    summary="Send a message and receive the response as Server-Sent Events",
)
async def chat_stream(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """SSE event format (each line: ``data: <json>\\n\\n``):

    * ``{"thread_id": "...", "thread_title": "..."}`` — first event
    * ``{"text": "..."}``  — incremental token
    * ``{"done": true}``   — stream complete
    * ``{"error": "..."}`` — unrecoverable error
    """
    thread = await _get_or_create_thread(
        req.thread_id, req.message, current_user, db
    )
    config = {
        "configurable": {"thread_id": str(thread.id)},
        "recursion_limit": settings.agent_recursion_limit,
    }

    # Inject per-user memory into the message
    messages = _build_memory_augmented_messages(current_user, req.message)

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def _generate():
        yield sse({'thread_id': str(thread.id), 'thread_title': thread.title})

        try:
            async for event in get_agent().astream_events(
                {"messages": messages},
                config=config,
                version="v2",
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    text = _extract_text(getattr(chunk, "content", ""))
                    if text:
                        yield sse({'text': text})
        except Exception as exc:  # noqa: BLE001
            yield sse({'error': str(exc)})
        finally:
            # Update thread stats even for streams (best-effort)
            try:
                await _touch_thread(thread, req.message, db)
            except Exception:  # noqa: BLE001
                pass
        yield sse({'done': True})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Thread-Id": str(thread.id),
        },
    )
