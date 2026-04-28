"""Batch upload router — upload files to RAGFlow KBs from a dictionary manifest.

Endpoint: ``POST /ragflow/batch-upload`` (JSON body)
   Uses a *server-side* folder_path — files are read from the server's local disk
   with recursive directory traversal. Accepts a manifest describing per-file target
   knowledge base, chunk method, and metadata. After uploading to the matching
   RAGFlow dataset they apply per-file settings, trigger parsing, and wait until
   every document is fully parsed (with retries on failure).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.database import get_db
from core.deps import get_current_admin
from core.media_helpers import (
    add_index_chunk,
    check_duplicate_filename,
    construct_doc_url,
    cross_ref_to_file_kb,
    detect_kb_type,
    save_media_record,
)

router = APIRouter(prefix="/ragflow", tags=["Batch Upload"])

# ── Map manifest knowledge_base values → config search substrings ────────────
KB_NAME_TO_SEARCH: dict[str, str] = {
    "产品库": settings.product_kb_name,
    "图片库": settings.image_kb_name,
    "视频库": settings.video_kb_name,
    "通用库": settings.general_kb_name,
    "程序库": settings.program_kb_name,
    "经验库": settings.experience_kb_name,
}

# ── Parser config defaults (mirrored from routers/ragflow.py) ─────────────────
PARSER_CONFIG_DEFAULTS: dict[str, Any] = {
    "naive": {
        "chunk_token_num": 128,
        "delimiter": "\\n",
        "html4excel": False,
        "layout_recognize": True,
        "raptor": {"use_raptor": False},
        "parent_child": {"use_parent_child": False, "children_delimiter": "\\n"},
    },
    "qa": {"raptor": {"use_raptor": False}},
    "manual": {"raptor": {"use_raptor": False}},
    "table": None,
    "paper": {"raptor": {"use_raptor": False}},
    "book": {"raptor": {"use_raptor": False}},
    "laws": {"raptor": {"use_raptor": False}},
    "presentation": {"raptor": {"use_raptor": False}},
    "picture": None,
    "one": None,
    "knowledge-graph": {
        "chunk_token_num": 128,
        "delimiter": "\\n",
        "entity_types": ["organization", "person", "location", "event", "time"],
    },
    "email": None,
}

# ── Polling / retry constants ─────────────────────────────────────────────────
PARSE_POLL_INTERVAL = 3.0       # seconds between status checks
PARSE_MAX_WAIT = 300.0          # max total wait per document (5 min)
PARSE_MAX_RETRIES = 2           # retries after the initial parse attempt
UPLOAD_MAX_RETRIES = 2          # retries for the upload HTTP call itself
UPLOAD_RETRY_BACKOFF = 2.0      # base seconds for exponential backoff on upload retry

# ── Global concurrency limiter ─────────────────────────────────────────────────
# Prevents concurrent batch upload requests from overwhelming RAGFlow.
# The semaphore value is set once from settings at startup (env var
# BATCH_UPLOAD_MAX_CONCURRENT).  Requests beyond the limit queue up FIFO.
_concurrency_sem: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Return the module-level semaphore, lazily initialised from settings."""
    global _concurrency_sem
    if _concurrency_sem is None:
        _concurrency_sem = asyncio.Semaphore(settings.batch_upload_max_concurrent)
    return _concurrency_sem


# ── Schemas ───────────────────────────────────────────────────────────────────


class ManifestItem(BaseModel):
    """A single entry in the batch-upload manifest."""

    file_name: str = Field(..., description="Exact filename to locate in folder_path")
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary meta_fields to attach to the RAGFlow document",
    )
    knowledge_base: str = Field(
        ...,
        description="Target KB: 产品库 | 图片库 | 视频库 | 通用库 | 程序库 | 经验库",
    )
    chunk_method: str = Field(
        "naive",
        description="Parsing method: naive | manual | qa | table | paper | book | "
        "laws | presentation | picture | one | knowledge-graph | email",
    )


class BatchUploadRequest(BaseModel):
    """Full manifest + server-side path for a batch upload operation.

    Provide **one** of:
    - ``manifest``: the JSON array inline in the request body
    - ``manifest_path``: absolute path to a JSON file on the server containing the array
    """

    manifest: Optional[list[ManifestItem]] = Field(
        None, min_length=1, description="Inline manifest array (alternative to manifest_path)"
    )
    manifest_path: Optional[str] = Field(
        None, description="Absolute path (on the server) to a JSON manifest file"
    )
    folder_path: str = Field(
        ...,
        description="Absolute path (on the server) to the folder containing the files",
    )
    skip_duplicate_check: bool = Field(
        False, description="Skip per-file duplicate-filename checks"
    )
    auto_parse: bool = Field(
        True,
        description="Trigger + wait for parsing after upload (default true)",
    )


class FileResult(BaseModel):
    """Per-file outcome."""

    file_name: str
    knowledge_base: str
    status: str  # "success" | "skipped" | "not_found" | "error"
    doc_id: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    parse_status: Optional[str] = None  # DONE / FAIL / CANCEL / SKIPPED / TIMEOUT / TRIGGER_FAILED
    doc_url: Optional[str] = None
    message: str = ""


class BatchUploadResponse(BaseModel):
    """Aggregate result for the whole batch."""

    total: int
    success: int
    skipped: int
    not_found: int
    error: int
    results: list[FileResult]


# ── Internal helpers ──────────────────────────────────────────────────────────


def _find_file_in_tree(folder: Path, file_name: str) -> Path | None:
    """Recursively search *folder* (and all subdirectories) for *file_name*.

    Returns the first match (depth-first).  Returns ``None`` if not found.
    Symlinks are followed.  The search is non-recursive within hidden directories
    (names starting with ``.``) to avoid crawling ``.git`` etc.
    """
    # Fast path — direct hit in the root
    direct = folder / file_name
    if direct.is_file():
        return direct

    # Slow path — walk subdirectories
    for root, dirs, files in os.walk(folder, followlinks=True):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if file_name in files:
            return Path(root) / file_name
    return None




def _headers(content_type: str | None = None) -> tuple[str, dict[str, str]]:
    """Build (base_url, headers) for a RAGFlow API call."""
    base_url = settings.ragflow_base_url.rstrip("/")
    hdrs: dict[str, str] = {"Authorization": f"Bearer {settings.ragflow_api_key}"}
    if content_type:
        hdrs["Content-Type"] = content_type
    return base_url, hdrs


async def _resolve_all_datasets(
    base_url: str, hdrs: dict, manifest: list[ManifestItem],
) -> dict[str, tuple[str | None, str | None]]:
    """Resolve every unique KB name in the manifest → (dataset_id, dataset_name)."""
    cache: dict[str, tuple[str | None, str | None]] = {}
    for item in manifest:
        kb = item.knowledge_base
        if kb not in cache:
            cache[kb] = await _find_dataset(base_url, hdrs, kb)
    return cache


async def _find_dataset(
    base_url: str, hdrs: dict, kb_name_cn: str
) -> tuple[str | None, str | None]:
    """Resolve a Chinese KB name (e.g. '产品库') → (dataset_id, actual_dataset_name)."""
    search = KB_NAME_TO_SEARCH.get(kb_name_cn, kb_name_cn)
    search_lower = search.lower()
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
                    ds_name = ds.get("name", "")
                    if search_lower in ds_name.lower():
                        return ds.get("id"), ds_name
    except Exception:
        pass
    return None, None


async def _get_doc_status(
    base_url: str, hdrs: dict, dataset_id: str, doc_id: str
) -> dict[str, Any]:
    """Fetch a single document's current info (including run_status) from RAGFlow."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{base_url}/api/v1/datasets/{dataset_id}/documents",
                headers=hdrs,
                params={"id": doc_id},
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                raw = data.get("data", {})
                docs = raw.get("docs", raw) if isinstance(raw, dict) else raw
                if isinstance(docs, list):
                    for d in docs:
                        if d.get("id") == doc_id:
                            return d
    except Exception:
        pass
    return {}


async def _trigger_parse(
    base_url: str, dataset_id: str, doc_ids: list[str]
) -> bool:
    """Start async parsing. Returns True if RAGFlow accepted the request."""
    _, json_hdrs = _headers("application/json")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/api/v1/datasets/{dataset_id}/chunks",
                headers=json_hdrs,
                json={"document_ids": doc_ids},
                timeout=30.0,
            )
            if resp.status_code == 200:
                return resp.json().get("code") == 0
    except Exception:
        pass
    return False


async def _wait_for_parse(
    base_url: str,
    hdrs: dict,
    dataset_id: str,
    doc_id: str,
    *,
    poll_interval: float = PARSE_POLL_INTERVAL,
    max_wait: float = PARSE_MAX_WAIT,
    max_retries: int = PARSE_MAX_RETRIES,
) -> tuple[str, str]:
    """Poll until document parsing reaches a terminal state.

    Returns (final_status, message).
    Terminal statuses: DONE, FAIL, CANCEL
    Retries parsing on FAIL up to *max_retries* times.
    """
    for attempt in range(max_retries + 1):
        elapsed = 0.0
        while elapsed < max_wait:
            doc = await _get_doc_status(base_url, hdrs, dataset_id, doc_id)
            run_status = doc.get("run_status", "")
            if run_status == "DONE":
                return "DONE", f"Parsing completed on attempt {attempt + 1}"
            if run_status == "CANCEL":
                return "CANCEL", "Parsing was cancelled"
            if run_status == "FAIL":
                break  # break inner poll loop → retry from top
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        if attempt < max_retries:
            ok = await _trigger_parse(base_url, dataset_id, [doc_id])
            if not ok:
                return "FAIL", f"Parse retry {attempt + 2} trigger failed"
            await asyncio.sleep(2.0)  # brief wait for RAGFlow to register new parse
        else:
            return "FAIL", f"Parsing failed after {max_retries + 1} attempt(s)"

    return "TIMEOUT", f"Parse did not complete within {max_wait}s"


# ── Core per-file processing (shared by both endpoints) ──────────────────────


async def _process_one_file(
    *,
    file_name: str,
    file_bytes: bytes,
    content_type: str,
    knowledge_base: str,
    chunk_method: str,
    meta: dict[str, Any],
    dataset_id: str,
    dataset_name: str | None,
    auto_parse: bool,
    skip_duplicate_check: bool,
    base_url: str,
    hdrs: dict[str, str],
    db: AsyncSession,
) -> FileResult:
    """Upload a single file to RAGFlow, apply settings, index, parse, and return result.

    This is the shared workhorse — both the server-side-path endpoint and the
    client-upload endpoint delegate to this function.
    """
    # ── Duplicate check ───────────────────────────────────────────────────
    if not skip_duplicate_check:
        existing = await check_duplicate_filename(db, file_name, dataset_id)
        if existing:
            return FileResult(
                file_name=file_name,
                knowledge_base=knowledge_base,
                status="skipped",
                doc_id=existing.doc_id,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                message=f"Duplicate — already exists as doc_id={existing.doc_id}",
            )

    # ── Upload to RAGFlow (with retry + backoff) ──────────────────────────
    upload_ok = False
    last_exc: Exception | None = None
    last_status: int | None = None
    last_body: str = ""

    for upload_attempt in range(UPLOAD_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{base_url}/api/v1/datasets/{dataset_id}/documents",
                    headers=hdrs,
                    files={"file": (file_name, file_bytes, content_type)},
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") == 0:
                    upload_ok = True
                    break
                else:
                    # RAGFlow returned an error code — may be transient
                    last_body = data.get("message", "unknown")
                    last_exc = None
        except httpx.HTTPStatusError as exc:
            last_status = exc.response.status_code
            last_body = exc.response.text[:500]
            last_exc = exc
        except Exception as exc:
            last_exc = exc

        if upload_attempt < UPLOAD_MAX_RETRIES:
            delay = UPLOAD_RETRY_BACKOFF * (2 ** upload_attempt)
            await asyncio.sleep(delay)

    if not upload_ok:
        if last_exc is not None:
            if isinstance(last_exc, httpx.HTTPStatusError):
                return FileResult(
                    file_name=file_name,
                    knowledge_base=knowledge_base,
                    status="error",
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    message=f"RAGFlow upload HTTP {last_status} after {UPLOAD_MAX_RETRIES + 1} attempts: {last_body}",
                )
            return FileResult(
                file_name=file_name,
                knowledge_base=knowledge_base,
                status="error",
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                message=f"Upload failed after {UPLOAD_MAX_RETRIES + 1} attempts: {last_exc}",
            )
        return FileResult(
            file_name=file_name,
            knowledge_base=knowledge_base,
            status="error",
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            message=f"RAGFlow error after {UPLOAD_MAX_RETRIES + 1} attempts: {last_body}",
        )

    doc_list = data.get("data", [])
    doc = (doc_list[0] if isinstance(doc_list, list) and doc_list else doc_list) or {}
    doc_id: str | None = doc.get("id")
    doc_location: str | None = doc.get("location")
    if not doc_id:
        return FileResult(
            file_name=file_name,
            knowledge_base=knowledge_base,
            status="error",
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            message="Upload succeeded but no doc_id returned",
        )

    # ── Apply metadata (meta_fields + chunk_method + parser_config) ───────
    meta_warnings: list[str] = []
    parser_config = PARSER_CONFIG_DEFAULTS.get(chunk_method)

    # Strip None values from meta
    clean_meta = {k: v for k, v in meta.items() if v is not None}

    body: dict[str, Any] = {}
    if chunk_method:
        body["chunk_method"] = chunk_method
    if parser_config is not None:
        body["parser_config"] = parser_config
    if clean_meta:
        body["meta_fields"] = clean_meta

    if body:
        try:
            _, json_hdrs = _headers("application/json")
            async with httpx.AsyncClient() as client:
                upd = await client.put(
                    f"{base_url}/api/v1/datasets/{dataset_id}/documents/{doc_id}",
                    headers=json_hdrs,
                    json=body,
                    timeout=30.0,
                )
                upd.raise_for_status()
                upd_data = upd.json()
                if upd_data.get("code") != 0:
                    meta_warnings.append(upd_data.get("message", "metadata update failed"))
        except Exception as exc:
            meta_warnings.append(str(exc))

    # ── Media indexing ────────────────────────────────────────────────────
    kb_type = detect_kb_type(dataset_name or "")
    doc_url = construct_doc_url(
        base_url, doc_id, dataset_id=dataset_id, location=doc_location,
    )

    if kb_type:
        chunk_id = await add_index_chunk(
            base_url, dataset_id, doc_id, file_name, doc_url, kb_type
        )

        linked_file_doc_id: str | None = None
        linked_file_dataset_id: str | None = None
        if kb_type == "product":
            cross = await cross_ref_to_file_kb(base_url, hdrs, file_name, doc_url, doc_id)
            if "warning" in cross:
                meta_warnings.append(cross["warning"])
            else:
                linked_file_doc_id = cross.get("file_kb_doc_id")
                linked_file_dataset_id = cross.get("file_kb_dataset_id")

        try:
            await save_media_record(
                db,
                filename=file_name,
                url=doc_url,
                doc_id=doc_id,
                dataset_id=dataset_id,
                kb_type=kb_type,
                chunk_id=chunk_id,
                linked_file_doc_id=linked_file_doc_id,
                linked_file_dataset_id=linked_file_dataset_id,
            )
        except Exception as exc:
            meta_warnings.append(f"MediaIndex save: {exc}")

    # ── Parse and wait ────────────────────────────────────────────────────
    parse_status: str = "SKIPPED"
    parse_message: str = ""
    if auto_parse:
        ok = await _trigger_parse(base_url, dataset_id, [doc_id])
        if not ok:
            parse_status = "TRIGGER_FAILED"
            parse_message = "Could not start parsing"
        else:
            parse_status, parse_message = await _wait_for_parse(
                base_url, hdrs, dataset_id, doc_id,
            )

    # ── Compose final message ─────────────────────────────────────────────
    msg_parts: list[str] = []
    if parse_status == "DONE":
        msg_parts.append("Uploaded and parsed successfully")
    elif parse_status == "FAIL":
        msg_parts.append(f"Uploaded but parsing FAILED: {parse_message}")
    elif parse_status == "CANCEL":
        msg_parts.append("Uploaded but parsing was cancelled")
    elif parse_status == "TIMEOUT":
        msg_parts.append(f"Uploaded but parse timed out: {parse_message}")
    elif parse_status == "SKIPPED":
        msg_parts.append("Uploaded (auto_parse disabled)")
    elif parse_status == "TRIGGER_FAILED":
        msg_parts.append(f"Uploaded but parse trigger failed: {parse_message}")
    if meta_warnings:
        msg_parts.append("Warnings: " + "; ".join(meta_warnings))

    return FileResult(
        file_name=file_name,
        knowledge_base=knowledge_base,
        status="success" if parse_status in ("DONE", "SKIPPED") else "error",
        doc_id=doc_id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        parse_status=parse_status,
        doc_url=doc_url,
        message=". ".join(msg_parts) if msg_parts else "Uploaded",
    )


# ── Endpoint 1 — server-side folder path (JSON body) ─────────────────────────


@router.post(
    "/batch-upload",
    summary="[Admin] Batch-upload from a server-side folder path",
)
async def batch_upload(
    req: BatchUploadRequest,
    _=Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> BatchUploadResponse:
    """Upload files described in a JSON manifest from a **server-side** folder.

    Two ways to provide the manifest:
    1. **manifest_path** — path to a JSON file on the server, e.g.
       ``/data/manifest.json``. The file is read from disk.
    2. **manifest** — inline JSON array in the request body.

    Files must already exist inside *folder_path* on the server.

    **Important**: *folder_path* / *manifest_path* are paths **on the server**.
    Use ``POST /ragflow/batch-upload/upload`` to send files from the client.
    """
    # ── Resolve manifest ───────────────────────────────────────────────────
    if req.manifest is not None and req.manifest_path is not None:
        raise HTTPException(status_code=400, detail="Provide manifest OR manifest_path, not both")
    if req.manifest is None and req.manifest_path is None:
        raise HTTPException(status_code=400, detail="Provide either manifest or manifest_path")

    if req.manifest_path:
        manifest_path = Path(req.manifest_path).expanduser().resolve()
        if not manifest_path.exists():
            raise HTTPException(status_code=400, detail=f"Manifest file not found: {manifest_path}")
        if manifest_path.suffix.lower() != ".json":
            raise HTTPException(status_code=400, detail=f"Manifest file must be .json: {manifest_path}")
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid JSON in manifest file: {exc}")
        if not isinstance(raw, list):
            raise HTTPException(status_code=422, detail="Manifest file must contain a JSON array")
        try:
            manifest_items = [ManifestItem(**obj) for obj in raw]
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid manifest entry: {exc}")
    else:
        manifest_items = req.manifest

    base_url, hdrs = _headers()

    # Validate folder_path
    folder = Path(req.folder_path).expanduser().resolve()
    if not folder.exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {folder}")

    dataset_cache = await _resolve_all_datasets(base_url, hdrs, manifest_items)
    results: list[FileResult] = []

    sem = _get_semaphore()
    async with sem:
        for idx, item in enumerate(manifest_items):
            file_path = _find_file_in_tree(folder, item.file_name)

            # File existence / type checks
            if file_path is None:
                results.append(FileResult(
                    file_name=item.file_name,
                    knowledge_base=item.knowledge_base,
                    status="not_found",
                    message=f"File not found recursively in: {folder}",
                ))
                continue

            dataset_id, dataset_name = dataset_cache[item.knowledge_base]
            if not dataset_id:
                results.append(FileResult(
                    file_name=item.file_name,
                    knowledge_base=item.knowledge_base,
                    status="error",
                    message=f"No RAGFlow dataset found matching '{item.knowledge_base}' "
                            f"(searched for '{KB_NAME_TO_SEARCH.get(item.knowledge_base, item.knowledge_base)}')",
                ))
                continue

            try:
                file_bytes = file_path.read_bytes()
            except Exception as exc:
                results.append(FileResult(
                    file_name=item.file_name,
                    knowledge_base=item.knowledge_base,
                    status="error",
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    message=f"Failed to read file: {exc}",
                ))
                continue

            # Cooldown between files to let RAGFlow breathe
            if idx > 0 and settings.batch_upload_cooldown_seconds > 0:
                await asyncio.sleep(settings.batch_upload_cooldown_seconds)

            content_type = _guess_content_type_from_name(item.file_name)
            result = await _process_one_file(
                file_name=item.file_name,
                file_bytes=file_bytes,
                content_type=content_type,
                knowledge_base=item.knowledge_base,
                chunk_method=item.chunk_method,
                meta=item.meta,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                auto_parse=req.auto_parse,
                skip_duplicate_check=req.skip_duplicate_check,
                base_url=base_url,
                hdrs=hdrs,
                db=db,
            )
            results.append(result)

    return _build_response(results)


# ── Response builder ─────────────────────────────────────────────────────────


def _build_response(results: list[FileResult]) -> BatchUploadResponse:
    """Aggregate per-file results into a summary response."""
    return BatchUploadResponse(
        total=len(results),
        success=sum(1 for r in results if r.status == "success"),
        skipped=sum(1 for r in results if r.status == "skipped"),
        not_found=sum(1 for r in results if r.status == "not_found"),
        error=sum(1 for r in results if r.status == "error"),
        results=results,
    )


# ── MIME type guesser ─────────────────────────────────────────────────────────

_MIME_MAP: dict[str, str] = {
    ".pdf": "application/pdf",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".md": "text/markdown",
    ".json": "application/json",
    ".xml": "application/xml",
    ".html": "text/html",
    ".htm": "text/html",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv",
    ".zip": "application/zip",
    ".7z": "application/x-7z-compressed",
    ".rar": "application/vnd.rar",
}


def _guess_content_type_from_name(filename: str) -> str:
    """Return a MIME type based on file extension, falling back to octet-stream."""
    suffix = Path(filename).suffix.lower()
    return _MIME_MAP.get(suffix, "application/octet-stream")
