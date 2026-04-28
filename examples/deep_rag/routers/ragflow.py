"""RAGFlow knowledge-base and document management router.

Dataset CRUD and document parsing require admin privileges.
Document upload/update/delete/list require admin privileges as well,
since the knowledge base is shared across all users.

Endpoints:
    POST   /ragflow/datasets                                    Create dataset         [admin]
    GET    /ragflow/datasets                                    List datasets          [auth]
    PUT    /ragflow/datasets/{dataset_id}                       Update dataset         [admin]
    DELETE /ragflow/datasets                                    Delete datasets        [admin]

    POST   /ragflow/datasets/{id}/documents/upload             Upload single doc      [admin]
    POST   /ragflow/datasets/{id}/documents/upload/batch       Upload multiple docs   [admin]
    GET    /ragflow/datasets/{id}/documents                    List documents         [auth]
    PUT    /ragflow/datasets/{id}/documents/{doc_id}           Update document        [admin]
    DELETE /ragflow/datasets/{id}/documents                    Delete documents       [admin]

    POST   /ragflow/datasets/{id}/documents/parse              Start parsing          [admin]
    DELETE /ragflow/datasets/{id}/documents/parse              Stop parsing           [admin]
"""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.database import get_db
from core.deps import get_current_admin, get_current_user
from core.media_helpers import (
    add_index_chunk,
    check_duplicate_filename,
    construct_doc_url,
    cross_ref_to_file_kb,
    delete_media_record,
    detect_kb_type,
    fetch_dataset_name,
    save_media_record,
)

router = APIRouter(prefix="/ragflow", tags=["RAGFlow"])


# ── Internal helpers ──────────────────────────────────────────────────────────


def _headers(content_type: str | None = None) -> tuple[str, dict[str, str]]:
    base_url = settings.ragflow_base_url.rstrip("/")
    hdrs: dict[str, str] = {
        "Authorization": f"Bearer {settings.ragflow_api_key}"
    }
    if content_type:
        hdrs["Content-Type"] = content_type
    return base_url, hdrs


def _check(data: dict) -> None:
    if data.get("code") != 0:
        raise HTTPException(
            status_code=400,
            detail=data.get("message", "RAGFlow API error"),
        )


# ── Types ─────────────────────────────────────────────────────────────────────

ChunkMethod = Literal[
    "naive", "manual", "qa", "table", "paper", "book",
    "laws", "presentation", "picture", "one", "knowledge-graph", "email",
]

PARSER_CONFIG_DEFAULTS: dict[str, Any] = {
    "naive": {
        "chunk_token_num": 128,
        "delimiter": "\\n",
        "html4excel": False,
        "layout_recognize": True,
        "raptor": {"use_raptor": False},
        "parent_child": {
            "use_parent_child": False,
            "children_delimiter": "\\n",
        },
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


# ── Schemas ───────────────────────────────────────────────────────────────────


class CreateDatasetRequest(BaseModel):
    """Parameters for creating a new RAGFlow knowledge-base dataset."""

    name: str = Field(..., max_length=128, description="Dataset name (max 128 chars)")
    avatar: Optional[str] = Field(None, description="Base64-encoded avatar image")
    description: Optional[str] = None
    embedding_model: Optional[str] = Field(
        None,
        description="e.g. 'BAAI/bge-large-zh-v1.5@BAAI'. Defaults to server default.",
    )
    permission: Literal["me", "team"] = Field(
        "me", description="'me' — private; 'team' — visible to team members"
    )
    chunk_method: ChunkMethod = Field(
        "naive", description="Default parsing method for documents in this dataset"
    )
    parser_config: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "Method-specific parsing config. "
            "Omit to use the built-in default for the chosen chunk_method."
        ),
    )


class UpdateDatasetRequest(BaseModel):
    name: Optional[str] = Field(None, max_length=128)
    description: Optional[str] = None
    embedding_model: Optional[str] = None
    permission: Optional[Literal["me", "team"]] = None
    chunk_method: Optional[ChunkMethod] = None
    parser_config: Optional[dict[str, Any]] = None


class DeleteDatasetsRequest(BaseModel):
    ids: list[str] = Field(..., description="Dataset IDs to delete")


class UpdateDocumentRequest(BaseModel):
    """Payload for updating a single document's metadata and parsing settings.

    chunk_method → compatible parser_config:

    | chunk_method     | parser_config |
    |---|---|
    | naive            | ``{"chunk_token_num":128,"delimiter":"\\\\n","html4excel":false,"layout_recognize":true,"raptor":{"use_raptor":false},"parent_child":{"use_parent_child":false,"children_delimiter":"\\\\n"}}`` |
    | qa / manual / paper / book / laws / presentation | ``{"raptor":{"use_raptor":false}}`` |
    | table / picture / one / email | ``null`` |
    | knowledge-graph  | ``{"chunk_token_num":128,"delimiter":"\\\\n","entity_types":["organization","person","location","event","time"]}`` |
    """

    display_name: Optional[str] = Field(None, description="New display name")
    meta_fields: Optional[dict[str, Any]] = Field(
        None, description="Arbitrary key-value metadata"
    )
    chunk_method: Optional[ChunkMethod] = None
    parser_config: Optional[dict[str, Any]] = None


class ParseDocumentsRequest(BaseModel):
    document_ids: list[str] = Field(..., description="Document IDs to parse / cancel")


class MetaFieldAlignRequest(BaseModel):
    """Configuration for meta_fields alignment on a batch of documents.

    ``field_mapping`` renames/remaps source keys to target keys:
        {"src_key": "dst_key", "旧字段": "新字段"}

    ``static_fields`` sets fixed values on every document:
        {"series": "X系列", "language": "zh"}

    ``field_mapping`` is applied first, then ``static_fields`` are merged in.
    Other existing meta_fields not mentioned in ``field_mapping`` are preserved
    unless ``drop_unmapped`` is True.

    You can also pass ``chunk_method`` and ``parser_config`` to update
    parsing settings for every document in the batch at the same time.
    """

    document_ids: list[str] = Field(
        ..., description="IDs of documents to update. Must belong to the dataset."
    )
    field_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Rename meta_field keys: {old_key: new_key}. Values are preserved.",
    )
    static_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Fixed meta_field key-value pairs applied to every document.",
    )
    drop_unmapped: bool = Field(
        False,
        description=(
            "If True, drop any meta_field key not mentioned in field_mapping. "
            "If False (default), unmapped keys are carried over unchanged."
        ),
    )
    chunk_method: Optional[ChunkMethod] = Field(
        None, description="Update parsing method for all documents (optional)."
    )
    parser_config: Optional[dict[str, Any]] = Field(
        None, description="Update parser config for all documents (optional)."
    )


class DeleteDocumentsRequest(BaseModel):
    ids: list[str] = Field(..., description="Document IDs to delete")


# ── Dataset endpoints ─────────────────────────────────────────────────────────


@router.post("/datasets", summary="[Admin] Create a knowledge-base dataset")
async def create_dataset(
    req: CreateDatasetRequest,
    _=Depends(get_current_admin),
) -> dict:
    base_url, hdrs = _headers("application/json")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/api/v1/datasets",
                headers=hdrs,
                json=req.model_dump(exclude_none=True),
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    _check(data)
    return {"dataset": data.get("data", {})}


@router.get("/datasets", summary="List all datasets")
async def list_datasets(
    page: int = 1,
    page_size: int = 30,
    orderby: str = "create_time",
    desc: bool = True,
    name: str = "",
    _=Depends(get_current_user),
) -> dict:
    base_url, hdrs = _headers()
    params: dict[str, Any] = {
        "page": page,
        "page_size": page_size,
        "orderby": orderby,
        "desc": str(desc).lower(),
    }
    if name:
        params["name"] = name
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{base_url}/api/v1/datasets",
                headers=hdrs,
                params=params,
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    _check(data)
    datasets = data.get("data", [])
    return {"datasets": datasets, "total": len(datasets), "page": page, "page_size": page_size}


@router.put("/datasets/{dataset_id}", summary="[Admin] Update a dataset")
async def update_dataset(
    dataset_id: str,
    req: UpdateDatasetRequest,
    _=Depends(get_current_admin),
) -> dict:
    payload = req.model_dump(exclude_none=True)
    if not payload:
        raise HTTPException(status_code=400, detail="No fields to update")
    base_url, hdrs = _headers("application/json")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.put(
                f"{base_url}/api/v1/datasets/{dataset_id}",
                headers=hdrs,
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    _check(data)
    return {"message": "Dataset updated", "data": data.get("data")}


@router.delete("/datasets", summary="[Admin] Delete datasets")
async def delete_datasets(
    req: DeleteDatasetsRequest,
    _=Depends(get_current_admin),
) -> dict:
    base_url, hdrs = _headers("application/json")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"{base_url}/api/v1/datasets",
                headers=hdrs,
                json={"ids": req.ids},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    _check(data)
    return {"message": "Datasets deleted", "deleted_ids": req.ids}


# ── Document upload endpoints ─────────────────────────────────────────────────


@router.post(
    "/datasets/{dataset_id}/documents/upload",
    summary="[Admin] Upload a single document",
)
async def upload_document(
    dataset_id: str,
    file: UploadFile = File(...),
    display_name: Optional[str] = Form(
        None, description="Display name override (defaults to filename)"
    ),
    chunk_method: Optional[str] = Form(
        None,
        description=(
            "Parsing method: naive | manual | qa | table | paper | book | laws | "
            "presentation | picture | one | knowledge-graph | email"
        ),
    ),
    parser_config: Optional[str] = Form(
        None, description="JSON string of method-specific parsing config"
    ),
    meta_fields: Optional[str] = Form(
        None, description="JSON object of arbitrary key-value metadata"
    ),
    skip_duplicate_check: bool = Form(
        False, description="Set True to allow re-uploading a filename already in this dataset"
    ),
    _=Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Upload a file and optionally set its parsing method, config, and metadata.

    For file / image / video / product KB datasets this endpoint will also:
    - Auto-create a ``{name, url}`` index chunk on the uploaded document.
    - For product KB uploads, additionally add a cross-reference entry to the file KB.
    - Record the upload in the local MediaIndex table for deduplication queries.

    After upload the document is ready for parsing. Trigger parsing via
    ``POST /ragflow/datasets/{id}/documents/parse``.
    """
    base_url, hdrs = _headers()
    file_bytes = await file.read()
    filename = display_name or file.filename or "document"
    content_type = file.content_type or "application/octet-stream"

    # Duplicate check before uploading
    if not skip_duplicate_check:
        existing = await check_duplicate_filename(db, filename, dataset_id)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"File '{filename}' already exists in this dataset "
                    f"(doc_id={existing.doc_id}). "
                    "Pass skip_duplicate_check=true to force re-upload."
                ),
            )

    # Step 1 — upload to RAGFlow
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/api/v1/datasets/{dataset_id}/documents",
                headers=hdrs,
                files={"file": (filename, file_bytes, content_type)},
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    _check(data)

    doc_list = data.get("data", [])
    doc = (doc_list[0] if isinstance(doc_list, list) and doc_list else doc_list) or {}
    doc_id: str | None = doc.get("id")

    # Step 2 — optional metadata update
    update_warnings: list[str] = []
    if doc_id and any([chunk_method, parser_config, meta_fields]):
        body: dict[str, Any] = {}
        if chunk_method:
            body["chunk_method"] = chunk_method
        if parser_config:
            try:
                body["parser_config"] = json.loads(parser_config)
            except json.JSONDecodeError:
                raise HTTPException(status_code=422, detail="parser_config must be valid JSON")
        if meta_fields:
            try:
                body["meta_fields"] = json.loads(meta_fields)
            except json.JSONDecodeError:
                raise HTTPException(status_code=422, detail="meta_fields must be valid JSON")

        try:
            _, upd_hdrs = _headers("application/json")
            async with httpx.AsyncClient() as client:
                upd = await client.put(
                    f"{base_url}/api/v1/datasets/{dataset_id}/documents/{doc_id}",
                    headers=upd_hdrs,
                    json=body,
                    timeout=30.0,
                )
                upd.raise_for_status()
                upd_data = upd.json()
                if upd_data.get("code") != 0:
                    update_warnings.append(upd_data.get("message", "metadata update failed"))
        except Exception as exc:
            update_warnings.append(str(exc))

    # Step 3 — media index: detect KB type, create index chunk, cross-reference
    media_info: dict[str, Any] = {}
    if doc_id:
        dataset_name = await fetch_dataset_name(base_url, hdrs, dataset_id)
        kb_type = detect_kb_type(dataset_name)

        if kb_type:
            doc_url = construct_doc_url(
                base_url, doc_id, dataset_id=dataset_id, location=doc.get("location"),
            )

            # Add {name, url} index chunk to the uploaded document
            chunk_id = await add_index_chunk(base_url, dataset_id, doc_id, filename, doc_url, kb_type)
            media_info["index_chunk_id"] = chunk_id
            media_info["doc_url"] = doc_url
            media_info["kb_type"] = kb_type

            # For product KB: also register in the file KB
            linked_file_doc_id: str | None = None
            linked_file_dataset_id: str | None = None
            if kb_type == "product":
                cross = await cross_ref_to_file_kb(base_url, hdrs, filename, doc_url, doc_id)
                if "warning" in cross:
                    update_warnings.append(cross["warning"])
                else:
                    linked_file_doc_id = cross.get("file_kb_doc_id")
                    linked_file_dataset_id = cross.get("file_kb_dataset_id")
                    media_info["file_kb_doc_id"] = linked_file_doc_id
                    media_info["file_kb_dataset_id"] = linked_file_dataset_id

            # Persist to local MediaIndex
            try:
                await save_media_record(
                    db,
                    filename=filename,
                    url=doc_url,
                    doc_id=doc_id,
                    dataset_id=dataset_id,
                    kb_type=kb_type,
                    chunk_id=chunk_id,
                    linked_file_doc_id=linked_file_doc_id,
                    linked_file_dataset_id=linked_file_dataset_id,
                )
            except Exception as exc:
                update_warnings.append(f"MediaIndex save failed: {exc}")

    result: dict[str, Any] = {
        "message": "Document uploaded successfully",
        "document": doc,
        "dataset_id": dataset_id,
    }
    if media_info:
        result["media_index"] = media_info
    if update_warnings:
        result["update_warnings"] = update_warnings
    return result


@router.post(
    "/datasets/{dataset_id}/documents/upload/batch",
    summary="[Admin] Upload multiple documents at once",
)
async def upload_documents_batch(
    dataset_id: str,
    files: list[UploadFile] = File(..., description="One or more files"),
    chunk_method: Optional[str] = Form(
        None,
        description=(
            "Parsing method applied to all files: naive | manual | qa | table | "
            "paper | book | laws | presentation | picture | one | knowledge-graph | email"
        ),
    ),
    parser_config: Optional[str] = Form(
        None, description="JSON string of parsing config applied to all files"
    ),
    skip_duplicate_check: bool = Form(
        False, description="Set True to skip per-file duplicate filename checks"
    ),
    _=Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Upload multiple files sharing the same chunk_method and parser_config.

    Each uploaded document automatically gets a ``{name, url}`` index chunk.
    Product KB uploads are also cross-referenced into the file KB.

    To apply different settings per file use the single-upload endpoint or
    call the update endpoint afterwards.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    base_url, hdrs = _headers()

    # Pre-read all files and run duplicate checks
    file_tuples: list[tuple[str, bytes, str]] = []
    update_warnings: list[str] = []
    for f in files:
        fname = f.filename or "document"
        if not skip_duplicate_check:
            existing = await check_duplicate_filename(db, fname, dataset_id)
            if existing:
                update_warnings.append(
                    f"Skipped duplicate '{fname}' (doc_id={existing.doc_id})"
                )
                continue
        file_tuples.append((fname, await f.read(), f.content_type or "application/octet-stream"))

    if not file_tuples:
        raise HTTPException(
            status_code=409,
            detail="All files are duplicates. Pass skip_duplicate_check=true to force.",
        )

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/api/v1/datasets/{dataset_id}/documents",
                headers=hdrs,
                files=[("file", (name, content, ct)) for name, content, ct in file_tuples],
                timeout=300.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    _check(data)

    docs: list[dict] = data.get("data", [])

    # Bulk metadata update if requested
    if docs and any([chunk_method, parser_config]):
        parsed_cfg: Any = None
        if parser_config:
            try:
                parsed_cfg = json.loads(parser_config)
            except json.JSONDecodeError:
                raise HTTPException(status_code=422, detail="parser_config must be valid JSON")

        update_payload = []
        for doc in docs:
            item: dict[str, Any] = {"id": doc["id"]}
            if chunk_method:
                item["chunk_method"] = chunk_method
            if parsed_cfg is not None:
                item["parser_config"] = parsed_cfg
            update_payload.append(item)

        try:
            _, upd_hdrs = _headers("application/json")
            async with httpx.AsyncClient() as client:
                upd = await client.put(
                    f"{base_url}/api/v1/datasets/{dataset_id}/documents",
                    headers=upd_hdrs,
                    json=update_payload,
                    timeout=60.0,
                )
                upd.raise_for_status()
                upd_data = upd.json()
                if upd_data.get("code") != 0:
                    update_warnings.append(upd_data.get("message", "bulk update failed"))
        except Exception as exc:
            update_warnings.append(str(exc))

    # Media indexing: add {name, url} chunks and persist MediaIndex records
    dataset_name = await fetch_dataset_name(base_url, hdrs, dataset_id)
    kb_type = detect_kb_type(dataset_name)
    media_results: list[dict[str, Any]] = []

    if kb_type and docs:
        for doc in docs:
            doc_id: str | None = doc.get("id")
            doc_filename: str = doc.get("name", "")
            if not doc_id:
                continue

            doc_url = construct_doc_url(
                base_url, doc_id, dataset_id=dataset_id, location=doc.get("location"),
            )
            chunk_id = await add_index_chunk(
                base_url, dataset_id, doc_id, doc_filename, doc_url, kb_type
            )

            linked_file_doc_id: str | None = None
            linked_file_dataset_id: str | None = None
            if kb_type == "product":
                cross = await cross_ref_to_file_kb(base_url, hdrs, doc_filename, doc_url, doc_id)
                if "warning" in cross:
                    update_warnings.append(cross["warning"])
                else:
                    linked_file_doc_id = cross.get("file_kb_doc_id")
                    linked_file_dataset_id = cross.get("file_kb_dataset_id")

            try:
                await save_media_record(
                    db,
                    filename=doc_filename,
                    url=doc_url,
                    doc_id=doc_id,
                    dataset_id=dataset_id,
                    kb_type=kb_type,
                    chunk_id=chunk_id,
                    linked_file_doc_id=linked_file_doc_id,
                    linked_file_dataset_id=linked_file_dataset_id,
                )
            except Exception as exc:
                update_warnings.append(f"MediaIndex save failed for {doc_filename}: {exc}")

            media_results.append(
                {"doc_id": doc_id, "filename": doc_filename, "url": doc_url, "chunk_id": chunk_id}
            )

    result: dict[str, Any] = {
        "message": f"{len(docs)} document(s) uploaded successfully",
        "documents": docs,
        "dataset_id": dataset_id,
    }
    if media_results:
        result["media_index"] = {"kb_type": kb_type, "indexed": media_results}
    if update_warnings:
        result["update_warnings"] = update_warnings
    return result


# ── Document management endpoints ─────────────────────────────────────────────


@router.get(
    "/datasets/{dataset_id}/documents",
    summary="List documents in a dataset",
)
async def list_documents(
    dataset_id: str,
    id: Optional[str] = None,
    keywords: str = "",
    page: int = 1,
    page_size: int = 30,
    orderby: str = "create_time",
    desc: bool = True,
    create_time_from: Optional[int] = None,
    create_time_to: Optional[int] = None,
    _=Depends(get_current_user),
) -> dict:
    """List documents with optional filters.

    - *id*: fetch a single document by its ID
    - *keywords*: filter by document name
    - *create_time_from* / *create_time_to*: Unix timestamp range
    """
    base_url, hdrs = _headers()
    params: dict[str, Any] = {
        "page": page,
        "page_size": page_size,
        "orderby": orderby,
        "desc": str(desc).lower(),
    }
    if id:
        params["id"] = id
    if keywords:
        params["keywords"] = keywords
    if create_time_from is not None:
        params["create_time_from"] = create_time_from
    if create_time_to is not None:
        params["create_time_to"] = create_time_to

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{base_url}/api/v1/datasets/{dataset_id}/documents",
                headers=hdrs,
                params=params,
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    _check(data)

    raw = data.get("data", {})
    if isinstance(raw, dict):
        documents = raw.get("docs", [])
        total = raw.get("total", len(documents))
    else:
        documents, total = raw, len(raw)

    return {"documents": documents, "total": total, "page": page, "page_size": page_size}


@router.put(
    "/datasets/{dataset_id}/documents/{document_id}",
    summary="[Admin] Update a document's metadata and parsing settings",
)
async def update_document(
    dataset_id: str,
    document_id: str,
    req: UpdateDocumentRequest,
    _=Depends(get_current_admin),
) -> dict:
    """Update display name, meta_fields, chunk_method, and/or parser_config.

    Chunk method / parser_config reference:

    | chunk_method | parser_config |
    |---|---|
    | naive | ``{"chunk_token_num":128,"delimiter":"\\\\n","html4excel":false,"layout_recognize":true,"raptor":{"use_raptor":false},"parent_child":{"use_parent_child":false,"children_delimiter":"\\\\n"}}`` |
    | qa / manual / paper / book / laws / presentation | ``{"raptor":{"use_raptor":false}}`` |
    | table / picture / one / email | ``null`` |
    | knowledge-graph | ``{"chunk_token_num":128,"delimiter":"\\\\n","entity_types":["organization","person","location","event","time"]}`` |
    """
    payload = req.model_dump(exclude_none=True)
    if not payload:
        raise HTTPException(status_code=400, detail="No fields to update")
    base_url, hdrs = _headers("application/json")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.put(
                f"{base_url}/api/v1/datasets/{dataset_id}/documents/{document_id}",
                headers=hdrs,
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    _check(data)
    return {"message": "Document updated", "data": data.get("data")}


@router.delete(
    "/datasets/{dataset_id}/documents",
    summary="[Admin] Delete documents",
)
async def delete_documents(
    dataset_id: str,
    req: DeleteDocumentsRequest,
    _=Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    base_url, hdrs = _headers("application/json")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"{base_url}/api/v1/datasets/{dataset_id}/documents",
                headers=hdrs,
                json={"ids": req.ids},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    _check(data)

    # Clean up MediaIndex records for deleted documents
    for doc_id in req.ids:
        await delete_media_record(db, doc_id)

    return {"message": "Documents deleted", "deleted_ids": req.ids}


# ── Meta-fields alignment endpoint ───────────────────────────────────────────


@router.post(
    "/datasets/{dataset_id}/documents/meta-align",
    summary="[Admin] Batch-align meta_fields (and optionally other fields) for multiple documents",
)
async def batch_meta_align(
    dataset_id: str,
    req: MetaFieldAlignRequest,
    _=Depends(get_current_admin),
) -> dict:
    """Apply a field-mapping dict and/or static values to the meta_fields of
    multiple documents in one call.

    Workflow
    --------
    1. Fetch the current metadata for every ``document_id`` in the request.
    2. For each document, build the new ``meta_fields``:
       a. Start from current meta_fields.
       b. Apply ``field_mapping`` (rename keys; keep values).
       c. If ``drop_unmapped=True``, remove keys not in ``field_mapping``.
       d. Merge in ``static_fields`` (overwriting existing values on conflict).
    3. PATCH each document via RAGFlow PUT endpoint with the merged payload.

    Use this endpoint to:
    - Normalise inconsistent field names after bulk upload.
    - Inject model/series tags into documents that were uploaded without them.
    - Rename fields to match your retrieval filter schema.
    - Update chunk_method / parser_config for a whole batch at once.

    Example
    -------
    ```json
    {
      "document_ids": ["doc-1", "doc-2"],
      "field_mapping": {"型号": "model", "系列": "series"},
      "static_fields": {"language": "zh", "category": "product"},
      "drop_unmapped": false,
      "chunk_method": "naive"
    }
    ```
    """
    if not req.document_ids:
        raise HTTPException(status_code=400, detail="document_ids must not be empty")
    if not req.field_mapping and not req.static_fields and not req.chunk_method and not req.parser_config:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: field_mapping, static_fields, chunk_method, parser_config",
        )

    base_url, hdrs = _headers()
    _, json_hdrs = _headers("application/json")

    results: list[dict[str, Any]] = []

    for doc_id in req.document_ids:
        # --- 1. Fetch current document metadata ---
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{base_url}/api/v1/datasets/{dataset_id}/documents",
                    headers=hdrs,
                    params={"id": doc_id},
                    timeout=15.0,
                )
                resp.raise_for_status()
                doc_data = resp.json()
        except Exception as exc:
            results.append({"id": doc_id, "status": "error", "detail": str(exc)})
            continue

        raw = doc_data.get("data", {})
        docs_list = raw.get("docs", raw) if isinstance(raw, dict) else raw
        current_doc = next(
            (d for d in docs_list if d.get("id") == doc_id),
            {},
        ) if isinstance(docs_list, list) else {}

        current_meta: dict[str, Any] = current_doc.get("meta_fields") or {}

        # --- 2. Build new meta_fields ---
        if req.drop_unmapped:
            new_meta: dict[str, Any] = {}
            for old_key, new_key in req.field_mapping.items():
                if old_key in current_meta:
                    new_meta[new_key] = current_meta[old_key]
        else:
            # Carry over all existing keys first
            new_meta = dict(current_meta)
            # Apply renames: add under new key, remove old key
            for old_key, new_key in req.field_mapping.items():
                if old_key in new_meta:
                    new_meta[new_key] = new_meta.pop(old_key)

        # Merge static fields (override existing values)
        new_meta.update(req.static_fields)

        # --- 3. Build update payload ---
        update_body: dict[str, Any] = {"meta_fields": new_meta}
        if req.chunk_method:
            update_body["chunk_method"] = req.chunk_method
        if req.parser_config:
            update_body["parser_config"] = req.parser_config

        # --- 4. PATCH via RAGFlow PUT endpoint ---
        try:
            async with httpx.AsyncClient() as client:
                upd = await client.put(
                    f"{base_url}/api/v1/datasets/{dataset_id}/documents/{doc_id}",
                    headers=json_hdrs,
                    json=update_body,
                    timeout=30.0,
                )
                upd.raise_for_status()
                upd_data = upd.json()
            if upd_data.get("code") != 0:
                results.append({
                    "id": doc_id,
                    "status": "ragflow_error",
                    "detail": upd_data.get("message", "unknown"),
                    "new_meta_fields": new_meta,
                })
            else:
                results.append({
                    "id": doc_id,
                    "status": "ok",
                    "new_meta_fields": new_meta,
                })
        except Exception as exc:
            results.append({"id": doc_id, "status": "error", "detail": str(exc)})

    ok_count = sum(1 for r in results if r["status"] == "ok")
    return {
        "message": f"Processed {len(req.document_ids)} documents — {ok_count} updated successfully.",
        "dataset_id": dataset_id,
        "results": results,
    }


# ── Parsing endpoints ─────────────────────────────────────────────────────────


@router.post(
    "/datasets/{dataset_id}/documents/parse",
    summary="[Admin] Start async parsing of documents",
)
async def parse_documents(
    dataset_id: str,
    req: ParseDocumentsRequest,
    _=Depends(get_current_admin),
) -> dict:
    """Trigger asynchronous parsing. Poll document ``run_status`` for progress:
    ``UNSTART`` → ``RUNNING`` → ``DONE`` / ``FAIL`` / ``CANCEL``.
    """
    base_url, hdrs = _headers("application/json")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/api/v1/datasets/{dataset_id}/chunks",
                headers=hdrs,
                json={"document_ids": req.document_ids},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    _check(data)
    return {"message": "Parsing started", "document_ids": req.document_ids, "data": data.get("data")}


@router.delete(
    "/datasets/{dataset_id}/documents/parse",
    summary="[Admin] Cancel parsing of documents",
)
async def stop_parsing(
    dataset_id: str,
    req: ParseDocumentsRequest,
    _=Depends(get_current_admin),
) -> dict:
    """Cancel in-progress parsing. Documents retain existing chunks (if any).
    ``run_status`` is set to ``CANCEL``.
    """
    base_url, hdrs = _headers("application/json")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"{base_url}/api/v1/datasets/{dataset_id}/chunks",
                headers=hdrs,
                json={"document_ids": req.document_ids},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    _check(data)
    return {"message": "Parsing cancelled", "document_ids": req.document_ids}
