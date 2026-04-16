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

from core.config import settings
from core.deps import get_current_admin, get_current_user

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
    _=Depends(get_current_admin),
) -> dict:
    """Upload a file and optionally set its parsing method, config, and metadata.

    After upload the document is ready for parsing. Trigger parsing via
    ``POST /ragflow/datasets/{id}/documents/parse``.
    """
    base_url, hdrs = _headers()
    file_bytes = await file.read()
    filename = display_name or file.filename or "document"
    content_type = file.content_type or "application/octet-stream"

    # Step 1 — upload
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

    result: dict[str, Any] = {
        "message": "Document uploaded successfully",
        "document": doc,
        "dataset_id": dataset_id,
    }
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
    _=Depends(get_current_admin),
) -> dict:
    """Upload multiple files sharing the same chunk_method and parser_config.

    To apply different settings per file use the single-upload endpoint or
    call the update endpoint afterwards.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    base_url, hdrs = _headers()
    file_tuples = [
        (f.filename or "document", await f.read(), f.content_type or "application/octet-stream")
        for f in files
    ]

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
    update_warnings: list[str] = []
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

    result: dict[str, Any] = {
        "message": f"{len(docs)} document(s) uploaded successfully",
        "documents": docs,
        "dataset_id": dataset_id,
    }
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
    return {"message": "Documents deleted", "deleted_ids": req.ids}


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
