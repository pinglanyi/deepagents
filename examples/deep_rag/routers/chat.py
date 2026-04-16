"""Chat router — invoke the agent and manage per-user thread state."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.deps import get_current_user
from models.thread import Thread
from models.user import User
from schemas.thread import ChatRequest, ChatResponse
from services.agent import get_agent

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


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
    config = {"configurable": {"thread_id": str(thread.id)}}

    result = await get_agent().ainvoke(
        {"messages": [{"role": "user", "content": req.message}]},
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
    config = {"configurable": {"thread_id": str(thread.id)}}

    async def _generate():
        yield (
            f"data: {json.dumps({'thread_id': str(thread.id), 'thread_title': thread.title})}\n\n"
        )
        try:
            async for event in get_agent().astream_events(
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
        finally:
            # Update thread stats even for streams (best-effort)
            try:
                await _touch_thread(thread, req.message, db)
            except Exception:  # noqa: BLE001
                pass
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Thread-Id": str(thread.id),
        },
    )
