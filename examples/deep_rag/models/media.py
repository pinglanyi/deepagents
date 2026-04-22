"""MediaIndex — tracks uploaded file/image/video/product documents across all KBs."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MediaIndex(Base):
    __tablename__ = "media_index"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # Original filename of the uploaded document
    filename: Mapped[str] = mapped_column(String(500), index=True)
    # Canonical download URL — globally unique (based on RAGFlow doc_id)
    url: Mapped[str] = mapped_column(String(2000), unique=True, index=True)
    # RAGFlow document ID in the primary KB — unique per upload
    doc_id: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    # RAGFlow dataset ID of the primary KB
    dataset_id: Mapped[str] = mapped_column(String(200), index=True)
    # KB type: "file" | "image" | "video" | "product"
    kb_type: Mapped[str] = mapped_column(String(20), index=True)
    # RAGFlow chunk ID of the auto-generated {name, url} index chunk
    chunk_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # For product KB entries: the cross-reference doc in the file KB
    linked_file_doc_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    linked_file_dataset_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
