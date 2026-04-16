"""Pydantic schemas for threads and chat."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class MessageOut(BaseModel):
    role: str
    content: str


class ThreadOut(BaseModel):
    """Thread summary shown in list views."""

    id: uuid.UUID
    title: str
    message_count: int
    last_message_preview: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ThreadDetail(BaseModel):
    """Full thread with message history."""

    id: uuid.UUID
    title: str
    messages: list[MessageOut]
    created_at: datetime
    updated_at: datetime


class ThreadRename(BaseModel):
    title: str = Field(min_length=1, max_length=500)


class ChatRequest(BaseModel):
    thread_id: uuid.UUID | None = Field(
        None,
        description="Continue an existing thread. Omit to start a new conversation.",
    )
    message: str = Field(min_length=1, max_length=32_000)


class ChatResponse(BaseModel):
    thread_id: uuid.UUID
    thread_title: str
    response: str
