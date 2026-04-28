"""Unified media upload and management router.

Provides type-specific upload endpoints for image / video / file KBs that:
  1. Auto-resolve the target dataset by KB type from settings (no dataset_id needed)
  2. Upload the file to RAGFlow
  3. Add a {<type>_name, url} index chunk to the uploaded document
  4. Track every upload in the local MediaIndex table for deduplication

Endpoints:
    POST   /media/upload/image                  Upload one image to the image KB
    POST   /media/upload/image/batch            Upload multiple images to the image KB
    POST   /media/upload/video                  Upload one video to the video KB
    POST   /media/upload/video/batch            Upload multiple videos to the video KB
    POST   /media/upload/file                   Upload one file to the file KB
    POST   /media/upload/file/batch             Upload multiple files to the file KB

    GET    /media                               List MediaIndex records (filterable)
    GET    /media/stats                         Counts by KB type
    GET    /media/{doc_id}                      Get a single MediaIndex record
    DELETE /media/{doc_id}                      Delete record + RAGFlow document
    POST   /media/reindex                       Rebuild MediaIndex from RAGFlow (admin)
    POST   /media/check-duplicate               Check if a filename already exists
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.deps import get_current_admin, get_current_user
from core.media_helpers import (
    _ragflow_headers,
    add_index_chunk,
    check_duplicate_filename,
    construct_doc_url,
    cross_ref_to_file_kb,
    delete_media_record,
    detect_kb_type,
    find_kb_dataset_id,
    get_media_record_by_doc_id,
    list_kb_datasets,
    save_media_record,
)

router = APIRouter(prefix="/media", tags=["Media"])

KBType = Literal["image", "video", "file"]


# ── Internal helpers ──────────────────────────────────────────────────────────


async def _upload_to_kb(
    kb_type: KBType,
    files_data: list[tuple[str, bytes, str]],
    db: AsyncSession,
    *,
    dataset_id: Optional[str] = None,
    skip_duplicate_check: bool = False,
) -> dict[str, Any]:
    """Core upload logic shared by single and batch endpoints.

    ``files_data`` is a list of (filename, content, content_type) tuples.
    Returns a result dict suitable for returning directly from the endpoint.
    """
    base_url, hdrs = _ragflow_headers()

    # Resolve target dataset
    if dataset_id:
        target_dataset_id = dataset_id
    else:
        target_dataset_id = await find_kb_dataset_id(base_url, hdrs, kb_type)
        if not target_dataset_id:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No dataset found for KB type '{kb_type}'. "
                    "Create a dataset whose name contains the configured "
                    f"'{kb_type}_kb_name' substring, or pass dataset_id explicitly."
                ),
            )

    warnings: list[str] = []

    # Deduplication filter
    accepted: list[tuple[str, bytes, str]] = []
    skipped_duplicates: list[str] = []
    for fname, content, ct in files_data:
        if not skip_duplicate_check:
            existing = await check_duplicate_filename(db, fname, target_dataset_id)
            if existing:
                skipped_duplicates.append(fname)
                continue
        accepted.append((fname, content, ct))

    if skipped_duplicates:
        warnings.extend(
            f"Skipped duplicate '{f}'" for f in skipped_duplicates
        )

    if not accepted:
        raise HTTPException(
            status_code=409,
            detail=(
                "All provided files are already in this dataset. "
                "Pass skip_duplicate_check=true to force re-upload."
            ),
        )

    # Upload to RAGFlow
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/api/v1/datasets/{target_dataset_id}/documents",
                headers=hdrs,
                files=[("file", (name, data, ct)) for name, data, ct in accepted],
                timeout=300.0,
            )
            resp.raise_for_status()
            upload_data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if upload_data.get("code") != 0:
        raise HTTPException(
            status_code=400,
            detail=upload_data.get("message", "RAGFlow upload error"),
        )

    docs: list[dict] = upload_data.get("data", [])
    if not isinstance(docs, list):
        docs = [docs] if docs else []

    # Add index chunks and persist MediaIndex records
    indexed: list[dict[str, Any]] = []
    for doc in docs:
        doc_id: str | None = doc.get("id")
        doc_filename: str = doc.get("name", "")
        if not doc_id:
            continue

        doc_url = construct_doc_url(
            base_url, doc_id, dataset_id=target_dataset_id, location=doc.get("location"),
        )
        chunk_id = await add_index_chunk(
            base_url, target_dataset_id, doc_id, doc_filename, doc_url, kb_type
        )

        try:
            await save_media_record(
                db,
                filename=doc_filename,
                url=doc_url,
                doc_id=doc_id,
                dataset_id=target_dataset_id,
                kb_type=kb_type,
                chunk_id=chunk_id,
            )
        except Exception as exc:
            warnings.append(f"MediaIndex save failed for '{doc_filename}': {exc}")

        indexed.append(
            {"doc_id": doc_id, "filename": doc_filename, "url": doc_url, "chunk_id": chunk_id}
        )

    result: dict[str, Any] = {
        "message": f"{len(docs)} file(s) uploaded to {kb_type} KB",
        "kb_type": kb_type,
        "dataset_id": target_dataset_id,
        "documents": docs,
        "media_index": indexed,
    }
    if warnings:
        result["warnings"] = warnings
    return result


# ── Image upload endpoints ────────────────────────────────────────────────────


@router.post("/upload/image", summary="[Admin] Upload a single image to the image KB")
async def upload_image(
    file: UploadFile = File(...),
    dataset_id: Optional[str] = Form(None, description="Override auto-resolved image KB dataset"),
    skip_duplicate_check: bool = Form(False),
    _=Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Upload a single image and register it in the image KB with a {image_name, url} chunk."""
    fname = file.filename or "image"
    content = await file.read()
    ct = file.content_type or "image/octet-stream"
    return await _upload_to_kb(
        "image", [(fname, content, ct)], db,
        dataset_id=dataset_id, skip_duplicate_check=skip_duplicate_check,
    )


@router.post("/upload/image/batch", summary="[Admin] Batch upload images to the image KB")
async def upload_images_batch(
    files: list[UploadFile] = File(...),
    dataset_id: Optional[str] = Form(None),
    skip_duplicate_check: bool = Form(False),
    _=Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Upload multiple images in one request. Each gets its own {image_name, url} index chunk."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    files_data = [(f.filename or "image", await f.read(), f.content_type or "image/octet-stream") for f in files]
    return await _upload_to_kb(
        "image", files_data, db,
        dataset_id=dataset_id, skip_duplicate_check=skip_duplicate_check,
    )


# ── Video upload endpoints ────────────────────────────────────────────────────


@router.post("/upload/video", summary="[Admin] Upload a single video to the video KB")
async def upload_video(
    file: UploadFile = File(...),
    dataset_id: Optional[str] = Form(None, description="Override auto-resolved video KB dataset"),
    skip_duplicate_check: bool = Form(False),
    _=Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Upload a single video and register it in the video KB with a {video_name, url} chunk."""
    fname = file.filename or "video"
    content = await file.read()
    ct = file.content_type or "video/octet-stream"
    return await _upload_to_kb(
        "video", [(fname, content, ct)], db,
        dataset_id=dataset_id, skip_duplicate_check=skip_duplicate_check,
    )


@router.post("/upload/video/batch", summary="[Admin] Batch upload videos to the video KB")
async def upload_videos_batch(
    files: list[UploadFile] = File(...),
    dataset_id: Optional[str] = Form(None),
    skip_duplicate_check: bool = Form(False),
    _=Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Upload multiple videos in one request. Each gets its own {video_name, url} index chunk."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    files_data = [(f.filename or "video", await f.read(), f.content_type or "video/octet-stream") for f in files]
    return await _upload_to_kb(
        "video", files_data, db,
        dataset_id=dataset_id, skip_duplicate_check=skip_duplicate_check,
    )


# ── File upload endpoints ─────────────────────────────────────────────────────


@router.post("/upload/file", summary="[Admin] Upload a single file to the file KB")
async def upload_file(
    file: UploadFile = File(...),
    dataset_id: Optional[str] = Form(None, description="Override auto-resolved file KB dataset"),
    skip_duplicate_check: bool = Form(False),
    _=Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Upload a single file and register it in the file KB with a {file_name, url} chunk."""
    fname = file.filename or "file"
    content = await file.read()
    ct = file.content_type or "application/octet-stream"
    return await _upload_to_kb(
        "file", [(fname, content, ct)], db,
        dataset_id=dataset_id, skip_duplicate_check=skip_duplicate_check,
    )


@router.post("/upload/file/batch", summary="[Admin] Batch upload files to the file KB")
async def upload_files_batch(
    files: list[UploadFile] = File(...),
    dataset_id: Optional[str] = Form(None),
    skip_duplicate_check: bool = Form(False),
    _=Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Upload multiple files in one request. Each gets its own {file_name, url} index chunk."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    files_data = [(f.filename or "file", await f.read(), f.content_type or "application/octet-stream") for f in files]
    return await _upload_to_kb(
        "file", files_data, db,
        dataset_id=dataset_id, skip_duplicate_check=skip_duplicate_check,
    )


# ── Media index query endpoints ───────────────────────────────────────────────


@router.get("", summary="List MediaIndex records")
async def list_media(
    kb_type: Optional[str] = None,
    filename: Optional[str] = None,
    dataset_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 30,
    _=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """List uploaded media records with optional filters.

    - *kb_type*: filter by "file", "image", "video", or "product"
    - *filename*: substring search on filename
    - *dataset_id*: filter by RAGFlow dataset
    """
    from models.media import MediaIndex

    stmt = select(MediaIndex)
    if kb_type:
        stmt = stmt.where(MediaIndex.kb_type == kb_type)
    if filename:
        stmt = stmt.where(MediaIndex.filename.ilike(f"%{filename}%"))
    if dataset_id:
        stmt = stmt.where(MediaIndex.dataset_id == dataset_id)

    stmt = stmt.order_by(MediaIndex.created_at.desc())

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total: int = total_result.scalar_one()

    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(stmt)
    records = result.scalars().all()

    return {
        "records": [
            {
                "id": str(r.id),
                "filename": r.filename,
                "url": r.url,
                "doc_id": r.doc_id,
                "dataset_id": r.dataset_id,
                "kb_type": r.kb_type,
                "chunk_id": r.chunk_id,
                "linked_file_doc_id": r.linked_file_doc_id,
                "linked_file_dataset_id": r.linked_file_dataset_id,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/stats", summary="Media upload statistics by KB type")
async def media_stats(
    _=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return document counts grouped by KB type."""
    from models.media import MediaIndex

    result = await db.execute(
        select(MediaIndex.kb_type, func.count(MediaIndex.id).label("count"))
        .group_by(MediaIndex.kb_type)
    )
    rows = result.all()
    stats = {row.kb_type: row.count for row in rows}
    return {"stats": stats, "total": sum(stats.values())}


@router.get("/{doc_id}", summary="Get a single MediaIndex record by RAGFlow doc_id")
async def get_media(
    doc_id: str,
    _=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    record = await get_media_record_by_doc_id(db, doc_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"No MediaIndex record for doc_id={doc_id}")
    return {
        "id": str(record.id),
        "filename": record.filename,
        "url": record.url,
        "doc_id": record.doc_id,
        "dataset_id": record.dataset_id,
        "kb_type": record.kb_type,
        "chunk_id": record.chunk_id,
        "linked_file_doc_id": record.linked_file_doc_id,
        "linked_file_dataset_id": record.linked_file_dataset_id,
        "created_at": record.created_at.isoformat(),
    }


@router.delete("/{doc_id}", summary="[Admin] Delete a media document from RAGFlow and MediaIndex")
async def delete_media(
    doc_id: str,
    _=Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Delete the RAGFlow document and its linked file KB cross-reference (if any),
    then remove the local MediaIndex record.
    """
    record = await get_media_record_by_doc_id(db, doc_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"No MediaIndex record for doc_id={doc_id}")

    base_url, hdrs = _ragflow_headers("application/json")
    warnings: list[str] = []

    # Delete from primary KB
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"{base_url}/api/v1/datasets/{record.dataset_id}/documents",
                headers=hdrs,
                json={"ids": [doc_id]},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                warnings.append(f"RAGFlow delete warning: {data.get('message')}")
    except Exception as exc:
        warnings.append(f"RAGFlow delete failed: {exc}")

    # Delete cross-reference from file KB (for product KB entries)
    if record.linked_file_doc_id and record.linked_file_dataset_id:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.delete(
                    f"{base_url}/api/v1/datasets/{record.linked_file_dataset_id}/documents",
                    headers=hdrs,
                    json={"ids": [record.linked_file_doc_id]},
                    timeout=30.0,
                )
                resp.raise_for_status()
        except Exception as exc:
            warnings.append(f"File KB cross-reference delete failed: {exc}")

    # Remove from local MediaIndex
    await delete_media_record(db, doc_id)

    result: dict[str, Any] = {
        "message": f"Media document {doc_id} deleted",
        "doc_id": doc_id,
        "filename": record.filename,
        "kb_type": record.kb_type,
    }
    if warnings:
        result["warnings"] = warnings
    return result


# ── Duplicate check endpoint ──────────────────────────────────────────────────


class DuplicateCheckRequest(BaseModel):
    filename: str = Field(..., description="Filename to check")
    dataset_id: Optional[str] = Field(None, description="Check in a specific dataset")
    kb_type: Optional[KBType] = Field(None, description="Check across all datasets of this KB type")


@router.post("/check-duplicate", summary="Check if a filename is already registered in MediaIndex")
async def check_duplicate(
    req: DuplicateCheckRequest,
    _=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return whether a filename already exists in MediaIndex.

    Provide either ``dataset_id`` (exact dataset) or ``kb_type`` (any dataset of that type).
    """
    from models.media import MediaIndex

    if not req.dataset_id and not req.kb_type:
        raise HTTPException(status_code=400, detail="Provide either dataset_id or kb_type")

    stmt = select(MediaIndex).where(MediaIndex.filename == req.filename)
    if req.dataset_id:
        stmt = stmt.where(MediaIndex.dataset_id == req.dataset_id)
    elif req.kb_type:
        stmt = stmt.where(MediaIndex.kb_type == req.kb_type)

    result = await db.execute(stmt)
    records = result.scalars().all()

    if not records:
        return {"exists": False, "filename": req.filename, "matches": []}

    return {
        "exists": True,
        "filename": req.filename,
        "matches": [
            {
                "doc_id": r.doc_id,
                "dataset_id": r.dataset_id,
                "kb_type": r.kb_type,
                "url": r.url,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ],
    }


# ── Reindex endpoint ──────────────────────────────────────────────────────────


class ReindexRequest(BaseModel):
    kb_types: list[KBType] = Field(
        default=["image", "video", "file"],
        description="KB types to scan. Defaults to image, video, file.",
    )


@router.post(
    "/reindex",
    summary="[Admin] Rebuild MediaIndex by scanning all KB datasets in RAGFlow",
)
async def reindex_media(
    req: ReindexRequest = ReindexRequest(),  # noqa: B008
    _=Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Scan RAGFlow datasets for each KB type and create missing MediaIndex records.

    This is a recovery/maintenance endpoint — it does NOT delete existing records.
    Use it after manually uploading files directly to RAGFlow outside this API.
    """
    from models.media import MediaIndex

    base_url, hdrs = _ragflow_headers()
    created = 0
    skipped = 0
    warnings: list[str] = []

    for kb_type in req.kb_types:
        datasets = await list_kb_datasets(base_url, hdrs, kb_type)
        for ds in datasets:
            ds_id = ds.get("id", "")
            # Fetch documents in this dataset
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{base_url}/api/v1/datasets/{ds_id}/documents",
                        headers=hdrs,
                        params={"page": 1, "page_size": 200},
                        timeout=30.0,
                    )
                    resp.raise_for_status()
                    doc_data = resp.json()
            except Exception as exc:
                warnings.append(f"Failed to list docs in dataset {ds_id}: {exc}")
                continue

            raw = doc_data.get("data", {})
            docs_list = raw.get("docs", raw) if isinstance(raw, dict) else raw
            if not isinstance(docs_list, list):
                continue

            for doc in docs_list:
                doc_id = doc.get("id", "")
                if not doc_id:
                    continue

                # Check if already indexed
                existing_result = await db.execute(
                    select(MediaIndex).where(MediaIndex.doc_id == doc_id)
                )
                if existing_result.scalar_one_or_none():
                    skipped += 1
                    continue

                doc_url = construct_doc_url(
                    base_url, doc_id, dataset_id=ds_id, location=doc.get("location"),
                )
                doc_filename = doc.get("name", doc_id)

                # Skip internal cross-reference documents created by this system
                if doc_filename.startswith("_ref_"):
                    skipped += 1
                    continue

                try:
                    await save_media_record(
                        db,
                        filename=doc_filename,
                        url=doc_url,
                        doc_id=doc_id,
                        dataset_id=ds_id,
                        kb_type=kb_type,
                    )
                    created += 1
                except Exception as exc:
                    warnings.append(f"Failed to index doc {doc_id}: {exc}")

    result: dict[str, Any] = {
        "message": f"Reindex complete — {created} new records created, {skipped} already indexed",
        "created": created,
        "skipped": skipped,
    }
    if warnings:
        result["warnings"] = warnings
    return result
