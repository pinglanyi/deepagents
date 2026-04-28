"""Shared helpers for RAGFlow media upload, index-chunk creation, and cross-referencing.

These helpers are used by both routers/ragflow.py (existing upload endpoints)
and routers/media.py (type-specific unified upload endpoints).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings


# ── URL construction ──────────────────────────────────────────────────────────


def construct_doc_url(base_url: str, doc_id: str) -> str:
    """Build the canonical URL for a RAGFlow document."""
    return f"{base_url.rstrip('/')}/api/v1/document/preview?doc_id={doc_id}"


# ── KB type detection ─────────────────────────────────────────────────────────


def detect_kb_type(dataset_name: str) -> str | None:
    """Infer KB type from dataset name using configured name substrings."""
    name_lower = dataset_name.lower()
    if settings.product_kb_name.lower() in name_lower:
        return "product"
    if settings.image_kb_name.lower() in name_lower:
        return "image"
    if settings.video_kb_name.lower() in name_lower:
        return "video"
    if settings.file_kb_name.lower() in name_lower:
        return "file"
    if settings.general_kb_name.lower() in name_lower:
        return "general"
    if settings.program_kb_name.lower() in name_lower:
        return "program"
    if settings.experience_kb_name.lower() in name_lower:
        return "experience"
    return None


def kb_name_for_type(kb_type: str) -> str:
    """Return the configured dataset name substring for a KB type."""
    return {
        "product": settings.product_kb_name,
        "image": settings.image_kb_name,
        "file": settings.file_kb_name,
        "video": settings.video_kb_name,
        "general": settings.general_kb_name,
        "program": settings.program_kb_name,
        "experience": settings.experience_kb_name,
    }.get(kb_type, kb_type)


# ── RAGFlow dataset lookup ────────────────────────────────────────────────────


def _ragflow_headers(content_type: str | None = None) -> tuple[str, dict[str, str]]:
    base_url = settings.ragflow_base_url.rstrip("/")
    hdrs: dict[str, str] = {"Authorization": f"Bearer {settings.ragflow_api_key}"}
    if content_type:
        hdrs["Content-Type"] = content_type
    return base_url, hdrs


async def fetch_dataset_name(base_url: str, hdrs: dict, dataset_id: str) -> str:
    """Return the dataset name for a given dataset_id, or '' on failure."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{base_url}/api/v1/datasets",
                headers=hdrs,
                params={"page": 1, "page_size": 100},
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                for ds in data.get("data", []):
                    if ds.get("id") == dataset_id:
                        return ds.get("name", "")
    except Exception:
        pass
    return ""


async def find_kb_dataset_id(base_url: str, hdrs: dict, kb_type: str) -> str | None:
    """Return the first RAGFlow dataset ID whose name matches the given KB type."""
    name_filter = kb_name_for_type(kb_type)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{base_url}/api/v1/datasets",
                headers=hdrs,
                params={"page": 1, "page_size": 100},
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                for ds in data.get("data", []):
                    if name_filter.lower() in ds.get("name", "").lower():
                        return ds.get("id")
    except Exception:
        pass
    return None


async def list_kb_datasets(base_url: str, hdrs: dict, kb_type: str) -> list[dict]:
    """Return all RAGFlow datasets matching the given KB type."""
    name_filter = kb_name_for_type(kb_type)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{base_url}/api/v1/datasets",
                headers=hdrs,
                params={"page": 1, "page_size": 100},
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return [
                    ds for ds in data.get("data", [])
                    if name_filter.lower() in ds.get("name", "").lower()
                ]
    except Exception:
        pass
    return []


# ── Index chunk operations ────────────────────────────────────────────────────


def _build_index_content(filename: str, url: str, kb_type: str) -> str:
    """Build JSON string for the name+url index chunk."""
    key = f"{kb_type}_name"
    return json.dumps({key: filename, "url": url}, ensure_ascii=False)


async def add_index_chunk(
    base_url: str,
    dataset_id: str,
    doc_id: str,
    filename: str,
    url: str,
    kb_type: str,
) -> str | None:
    """Add a {name, url} index chunk to a RAGFlow document. Returns chunk_id or None."""
    _, hdrs = _ragflow_headers("application/json")
    content = _build_index_content(filename, url, kb_type)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/api/v1/datasets/{dataset_id}/documents/{doc_id}/chunks",
                headers=hdrs,
                json={"content": content, "important_keywords": []},
                timeout=30.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == 0:
                    chunk = data.get("data", {}).get("chunk", {})
                    return chunk.get("id")
    except Exception:
        pass
    return None


async def delete_index_chunk(
    base_url: str,
    dataset_id: str,
    doc_id: str,
    chunk_id: str,
) -> bool:
    """Delete a specific chunk from a RAGFlow document. Returns True on success."""
    _, hdrs = _ragflow_headers("application/json")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"{base_url}/api/v1/datasets/{dataset_id}/documents/{doc_id}/chunks",
                headers=hdrs,
                json={"chunk_ids": [chunk_id]},
                timeout=15.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("code") == 0
    except Exception:
        pass
    return False


# ── Product KB → File KB cross-reference ─────────────────────────────────────


async def cross_ref_to_file_kb(
    base_url: str,
    hdrs: dict,
    filename: str,
    url: str,
    product_doc_id: str,
) -> dict[str, Any]:
    """Create a metadata-only reference document in the file KB for a product upload.

    Returns a dict with keys: file_kb_dataset_id, file_kb_doc_id, chunk_id (optional),
    or a 'warning' key if the operation failed non-fatally.
    """
    file_dataset_id = await find_kb_dataset_id(base_url, hdrs, "file")
    if not file_dataset_id:
        return {"warning": "No file KB dataset found; cross-reference skipped"}

    content = json.dumps(
        {"file_name": filename, "url": url, "source_kb": "product", "product_doc_id": product_doc_id},
        ensure_ascii=False,
    )
    ref_filename = f"_ref_{product_doc_id[:12]}_{filename}"
    text_bytes = content.encode("utf-8")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/api/v1/datasets/{file_dataset_id}/documents",
                headers=hdrs,
                files={"file": (ref_filename, text_bytes, "text/plain")},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return {"warning": f"Cross-reference upload failed: {exc}"}

    if data.get("code") != 0:
        return {"warning": data.get("message", "Cross-reference upload failed")}

    docs = data.get("data", [])
    file_doc = (docs[0] if isinstance(docs, list) and docs else docs) or {}
    file_doc_id: str | None = file_doc.get("id")

    # Add index chunk to the reference document
    chunk_id: str | None = None
    if file_doc_id:
        chunk_id = await add_index_chunk(
            base_url, file_dataset_id, file_doc_id, filename, url, "file"
        )

    return {
        "file_kb_dataset_id": file_dataset_id,
        "file_kb_doc_id": file_doc_id,
        "chunk_id": chunk_id,
    }


# ── MediaIndex DB helpers ─────────────────────────────────────────────────────


async def check_duplicate_filename(
    db: AsyncSession,
    filename: str,
    dataset_id: str,
) -> Any:
    """Return existing MediaIndex record if filename already exists in this dataset."""
    from models.media import MediaIndex  # avoid circular import at module level

    result = await db.execute(
        select(MediaIndex).where(
            MediaIndex.filename == filename,
            MediaIndex.dataset_id == dataset_id,
        )
    )
    return result.scalar_one_or_none()


async def check_duplicate_url(db: AsyncSession, url: str) -> Any:
    """Return existing MediaIndex record if URL is already registered."""
    from models.media import MediaIndex

    result = await db.execute(
        select(MediaIndex).where(MediaIndex.url == url)
    )
    return result.scalar_one_or_none()


async def save_media_record(
    db: AsyncSession,
    *,
    filename: str,
    url: str,
    doc_id: str,
    dataset_id: str,
    kb_type: str,
    chunk_id: str | None = None,
    linked_file_doc_id: str | None = None,
    linked_file_dataset_id: str | None = None,
) -> Any:
    """Insert a new MediaIndex record and return it."""
    from models.media import MediaIndex

    record = MediaIndex(
        filename=filename,
        url=url,
        doc_id=doc_id,
        dataset_id=dataset_id,
        kb_type=kb_type,
        chunk_id=chunk_id,
        linked_file_doc_id=linked_file_doc_id,
        linked_file_dataset_id=linked_file_dataset_id,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


async def get_media_record_by_doc_id(db: AsyncSession, doc_id: str) -> Any:
    """Return MediaIndex record for a RAGFlow document ID, or None."""
    from models.media import MediaIndex

    result = await db.execute(
        select(MediaIndex).where(MediaIndex.doc_id == doc_id)
    )
    return result.scalar_one_or_none()


async def delete_media_record(db: AsyncSession, doc_id: str) -> bool:
    """Delete MediaIndex record by doc_id. Returns True if a record was deleted."""
    from models.media import MediaIndex

    result = await db.execute(
        select(MediaIndex).where(MediaIndex.doc_id == doc_id)
    )
    record = result.scalar_one_or_none()
    if record:
        await db.delete(record)
        await db.commit()
        return True
    return False
