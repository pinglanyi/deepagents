"""Threads router — per-user conversation history management."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.deps import get_current_user
from models.thread import Thread
from models.user import User
from schemas.thread import MessageOut, ThreadDetail, ThreadOut, ThreadRename
from services.agent import get_agent

router = APIRouter(prefix="/threads", tags=["Threads"])


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


def _serialize_messages(messages: list[Any]) -> list[MessageOut]:
    result = []
    for m in messages:
        mtype = getattr(m, "type", None) or getattr(m, "role", "unknown")
        role = {"human": "user", "ai": "assistant"}.get(mtype, mtype)
        content = _extract_text(getattr(m, "content", ""))
        if content:
            result.append(MessageOut(role=role, content=content))
    return result


async def _require_thread(
    thread_id: uuid.UUID, user: User, db: AsyncSession
) -> Thread:
    """Return the thread if owned by the user, else raise 404."""
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


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get(
    "",
    response_model=list[ThreadOut],
    summary="List all threads for the current user (newest first)",
)
async def list_threads(
    page: int = 1,
    page_size: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Thread]:
    result = await db.execute(
        select(Thread)
        .where(Thread.user_id == current_user.id)
        .order_by(Thread.updated_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(result.scalars().all())


@router.get(
    "/{thread_id}",
    response_model=ThreadDetail,
    summary="Get full message history for a thread",
)
async def get_thread(
    thread_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ThreadDetail:
    thread = await _require_thread(thread_id, current_user, db)

    config = {"configurable": {"thread_id": str(thread_id)}}
    state = await get_agent().aget_state(config)
    messages: list[MessageOut] = []
    if state and state.values:
        messages = _serialize_messages(state.values.get("messages", []))

    return ThreadDetail(
        id=thread.id,
        title=thread.title,
        messages=messages,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
    )


@router.patch(
    "/{thread_id}",
    response_model=ThreadOut,
    summary="Rename a thread",
)
async def rename_thread(
    thread_id: uuid.UUID,
    req: ThreadRename,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Thread:
    thread = await _require_thread(thread_id, current_user, db)
    thread.title = req.title
    await db.commit()
    await db.refresh(thread)
    return thread


@router.delete(
    "/{thread_id}",
    status_code=204,
    summary="Delete a single thread and its checkpoints",
)
async def delete_thread(
    thread_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    thread = await _require_thread(thread_id, current_user, db)
    await db.delete(thread)
    await db.commit()


@router.delete(
    "",
    status_code=204,
    summary="Delete ALL threads for the current user",
)
async def delete_all_threads(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    result = await db.execute(
        select(Thread).where(Thread.user_id == current_user.id)
    )
    for thread in result.scalars().all():
        await db.delete(thread)
    await db.commit()


@router.get(
    "/{thread_id}/export",
    response_class=PlainTextResponse,
    summary="Export a conversation as plain text",
)
async def export_thread(
    thread_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> str:
    thread = await _require_thread(thread_id, current_user, db)

    config = {"configurable": {"thread_id": str(thread_id)}}
    state = await get_agent().aget_state(config)

    if not state or not state.values:
        return f"# {thread.title}\n\n(empty conversation)"

    lines = [f"# {thread.title}", f"Thread: {thread_id}", ""]
    for m in _serialize_messages(state.values.get("messages", [])):
        speaker = "User" if m.role == "user" else "Assistant"
        lines += [f"## {speaker}", m.content, ""]
    return "\n".join(lines)
