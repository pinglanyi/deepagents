"""RAGFlow knowledge-base and document management API routes.

Provides FastAPI endpoints for creating datasets, uploading, updating,
deleting and parsing documents in RAGFlow.

Endpoints:
    POST   /ragflow/datasets                                    Create a dataset
    GET    /ragflow/datasets                                    List datasets
    DELETE /ragflow/datasets                                    Delete datasets

    POST   /ragflow/datasets/{dataset_id}/documents/upload       Upload single document
    POST   /ragflow/datasets/{dataset_id}/documents/upload/batch Upload multiple documents
    GET    /ragflow/datasets/{dataset_id}/documents              List documents
    PUT    /ragflow/datasets/{dataset_id}/documents/{doc_id}     Update a document
    DELETE /ragflow/datasets/{dataset_id}/documents              Delete documents

    POST   /ragflow/datasets/{dataset_id}/documents/parse        Start parsing
    DELETE /ragflow/datasets/{dataset_id}/documents/parse        Stop parsing
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

router = APIRouter(prefix="/ragflow", tags=["RAGFlow"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ragflow_headers(content_type: str | None = None) -> tuple[str, dict[str, str]]:
    """Return ``(base_url, headers)`` from environment variables."""
    base_url = os.getenv("RAGFLOW_BASE_URL", "http://localhost:9380").rstrip("/")
    api_key = os.getenv("RAGFLOW_API_KEY", "")
    headers: dict[str, str] = {"Authorization": f"Bearer {api_key}"}
    if content_type:
        headers["Content-Type"] = content_type
    return base_url, headers


def _check_ragflow_response(data: dict) -> None:
    """Raise HTTPException if RAGFlow returned a non-zero error code."""
    if data.get("code") != 0:
        raise HTTPException(
            status_code=400,
            detail=data.get("message", "RAGFlow API error (unknown)"),
        )


# ---------------------------------------------------------------------------
# Chunk method type and per-method parser_config defaults
# ---------------------------------------------------------------------------

ChunkMethod = Literal[
    "naive",
    "manual",
    "qa",
    "table",
    "paper",
    "book",
    "laws",
    "presentation",
    "picture",
    "one",
    "knowledge-graph",
    "email",
]

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


# ---------------------------------------------------------------------------
# Pydantic schemas — datasets
# ---------------------------------------------------------------------------

class CreateDatasetRequest(BaseModel):
    """Request body for creating a RAGFlow knowledge-base dataset."""

    name: str = Field(..., max_length=128, description="Dataset name (max 128 chars)")
    avatar: Optional[str] = Field(None, description="Base64-encoded avatar image")
    description: Optional[str] = Field(None, description="Dataset description")
    embedding_model: Optional[str] = Field(
        None,
        description="Embedding model name, e.g. 'BAAI/bge-large-zh-v1.5@BAAI'. "
                    "Defaults to the RAGFlow server default.",
    )
    permission: Literal["me", "team"] = Field(
        "me",
        description="'me' — private; 'team' — shared with team members",
    )
    chunk_method: ChunkMethod = Field(
        "naive",
        description="Default parsing method applied to documents in this dataset",
    )
    parser_config: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "Parsing configuration. Leave null to use the default for the selected "
            "chunk_method. See PARSER_CONFIG_DEFAULTS for per-method schemas."
        ),
    )


class UpdateDatasetRequest(BaseModel):
    """Request body for updating an existing dataset."""

    name: Optional[str] = Field(None, max_length=128)
    description: Optional[str] = None
    embedding_model: Optional[str] = None
    permission: Optional[Literal["me", "team"]] = None
    chunk_method: Optional[ChunkMethod] = None
    parser_config: Optional[dict[str, Any]] = None


class DeleteDatasetsRequest(BaseModel):
    """Request body for deleting one or more datasets."""

    ids: list[str] = Field(..., description="List of dataset IDs to delete")


# ---------------------------------------------------------------------------
# Pydantic schemas — documents
# ---------------------------------------------------------------------------

class UpdateDocumentRequest(BaseModel):
    """Payload for updating a single document's metadata and parsing settings.

    chunk_method → compatible parser_config:
      naive:          {"chunk_token_num":128,"delimiter":"\\n","html4excel":false,
                       "layout_recognize":true,"raptor":{"use_raptor":false},
                       "parent_child":{"use_parent_child":false,"children_delimiter":"\\n"}}
      qa:             {"raptor": {"use_raptor": false}}
      manual:         {"raptor": {"use_raptor": false}}
      table:          null
      paper:          {"raptor": {"use_raptor": false}}
      book:           {"raptor": {"use_raptor": false}}
      laws:           {"raptor": {"use_raptor": false}}
      presentation:   {"raptor": {"use_raptor": false}}
      picture:        null
      one:            null
      knowledge-graph: {"chunk_token_num":128,"delimiter":"\\n",
                        "entity_types":["organization","person","location","event","time"]}
      email:          null
    """

    display_name: Optional[str] = Field(None, description="New display name for the document")
    meta_fields: Optional[dict[str, Any]] = Field(
        None,
        description="Arbitrary key-value metadata to attach to the document",
    )
    chunk_method: Optional[ChunkMethod] = Field(
        None,
        description="Parsing/chunking method to apply",
    )
    parser_config: Optional[dict[str, Any]] = Field(
        None,
        description="Method-specific parsing configuration (see docstring for schemas)",
    )


class ParseDocumentsRequest(BaseModel):
    """Request body for starting or stopping document parsing."""

    document_ids: list[str] = Field(..., description="Document IDs to parse / cancel parsing")


class DeleteDocumentsRequest(BaseModel):
    """Request body for deleting documents."""

    ids: list[str] = Field(..., description="Document IDs to delete")


# ---------------------------------------------------------------------------
# Dataset endpoints
# ---------------------------------------------------------------------------

@router.post("/datasets", summary="Create a knowledge-base dataset")
async def create_dataset(req: CreateDatasetRequest) -> dict:
    """Create a new RAGFlow knowledge-base dataset.

    If *parser_config* is omitted the server-side default for the chosen
    *chunk_method* is used.  You can inspect ``PARSER_CONFIG_DEFAULTS`` in
    this module for the recommended per-method schemas.
    """
    base_url, headers = _ragflow_headers("application/json")
    payload = req.model_dump(exclude_none=True)

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/api/v1/datasets",
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    _check_ragflow_response(data)
    return {"dataset": data.get("data", {})}


@router.get("/datasets", summary="List all datasets")
async def list_datasets(
    page: int = 1,
    page_size: int = 30,
    orderby: str = "create_time",
    desc: bool = True,
    name: str = "",
) -> dict:
    """List RAGFlow datasets with optional pagination and name filter."""
    base_url, headers = _ragflow_headers()
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
                headers=headers,
                params=params,
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    _check_ragflow_response(data)
    datasets = data.get("data", [])
    return {
        "datasets": datasets,
        "total": len(datasets),
        "page": page,
        "page_size": page_size,
    }


@router.put("/datasets/{dataset_id}", summary="Update a dataset")
async def update_dataset(dataset_id: str, req: UpdateDatasetRequest) -> dict:
    """Update an existing dataset's name, description, or default parsing settings."""
    base_url, headers = _ragflow_headers("application/json")
    payload = req.model_dump(exclude_none=True)
    if not payload:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.put(
                f"{base_url}/api/v1/datasets/{dataset_id}",
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    _check_ragflow_response(data)
    return {"message": "Dataset updated successfully", "data": data.get("data")}


@router.delete("/datasets", summary="Delete one or more datasets")
async def delete_datasets(req: DeleteDatasetsRequest) -> dict:
    """Permanently delete datasets by ID.  All documents inside are also removed."""
    base_url, headers = _ragflow_headers("application/json")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"{base_url}/api/v1/datasets",
                headers=headers,
                json={"ids": req.ids},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    _check_ragflow_response(data)
    return {"message": "Datasets deleted successfully", "deleted_ids": req.ids}


# ---------------------------------------------------------------------------
# Document upload endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/datasets/{dataset_id}/documents/upload",
    summary="Upload a single document",
)
async def upload_document(
    dataset_id: str,
    file: UploadFile = File(..., description="Document file to upload"),
    display_name: Optional[str] = Form(
        None,
        description="Display name override; defaults to the original filename",
    ),
    chunk_method: Optional[str] = Form(
        None,
        description=(
            "Parsing method: naive | manual | qa | table | paper | book | laws | "
            "presentation | picture | one | knowledge-graph | email"
        ),
    ),
    parser_config: Optional[str] = Form(
        None,
        description="JSON string with method-specific parsing config (optional)",
    ),
    meta_fields: Optional[str] = Form(
        None,
        description="JSON object of arbitrary key-value metadata (optional)",
    ),
) -> dict:
    """Upload a single file and optionally set its parsing method and metadata.

    **Workflow:**
    1. The file is uploaded to RAGFlow and a document record is created.
    2. If *chunk_method*, *parser_config*, or *meta_fields* are supplied the
       document is immediately updated with those settings.
    3. To trigger actual parsing call
       ``POST /ragflow/datasets/{dataset_id}/documents/parse``.

    **parser_config examples by chunk_method:**

    | chunk_method | parser_config |
    |---|---|
    | naive | `{"chunk_token_num":128,"delimiter":"\\n","html4excel":false,"layout_recognize":true,"raptor":{"use_raptor":false},"parent_child":{"use_parent_child":false,"children_delimiter":"\\n"}}` |
    | qa / manual / paper / book / laws / presentation | `{"raptor":{"use_raptor":false}}` |
    | table / picture / one / email | `null` |
    | knowledge-graph | `{"chunk_token_num":128,"delimiter":"\\n","entity_types":["organization","person","location","event","time"]}` |
    """
    base_url, headers = _ragflow_headers()
    file_bytes = await file.read()
    filename = display_name or file.filename or "document"
    content_type = file.content_type or "application/octet-stream"

    # ── Step 1: upload ──────────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/api/v1/datasets/{dataset_id}/documents",
                headers=headers,
                files={"file": (filename, file_bytes, content_type)},
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    _check_ragflow_response(data)

    doc_list = data.get("data", [])
    doc = (doc_list[0] if isinstance(doc_list, list) and doc_list else doc_list) or {}
    doc_id: str | None = doc.get("id")

    # ── Step 2: optional metadata / parsing update ──────────────────────────
    update_errors: list[str] = []
    if doc_id and any([chunk_method, parser_config, meta_fields]):
        update_body: dict[str, Any] = {}
        if chunk_method:
            update_body["chunk_method"] = chunk_method
        if parser_config:
            try:
                update_body["parser_config"] = json.loads(parser_config)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=422,
                    detail="parser_config must be a valid JSON string",
                )
        if meta_fields:
            try:
                update_body["meta_fields"] = json.loads(meta_fields)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=422,
                    detail="meta_fields must be a valid JSON object string",
                )

        try:
            upd_headers = _ragflow_headers("application/json")[1]
            async with httpx.AsyncClient() as client:
                upd_resp = await client.put(
                    f"{base_url}/api/v1/datasets/{dataset_id}/documents/{doc_id}",
                    headers=upd_headers,
                    json=update_body,
                    timeout=30.0,
                )
                upd_resp.raise_for_status()
                upd_data = upd_resp.json()
                if upd_data.get("code") != 0:
                    update_errors.append(upd_data.get("message", "update failed"))
        except Exception as exc:
            update_errors.append(str(exc))

    result: dict[str, Any] = {
        "message": "Document uploaded successfully",
        "document": doc,
        "dataset_id": dataset_id,
    }
    if update_errors:
        result["update_warnings"] = update_errors
    return result


@router.post(
    "/datasets/{dataset_id}/documents/upload/batch",
    summary="Upload multiple documents at once",
)
async def upload_documents_batch(
    dataset_id: str,
    files: list[UploadFile] = File(..., description="One or more document files to upload"),
    chunk_method: Optional[str] = Form(
        None,
        description=(
            "Parsing method applied to all uploaded files: naive | manual | qa | "
            "table | paper | book | laws | presentation | picture | one | "
            "knowledge-graph | email"
        ),
    ),
    parser_config: Optional[str] = Form(
        None,
        description="JSON string of parsing config applied to all files (optional)",
    ),
) -> dict:
    """Upload multiple documents in a single request.

    All uploaded files share the same *chunk_method* and *parser_config*.
    To apply different settings per document use the single-upload endpoint
    and supply individual settings, or call the update endpoint afterwards.

    After uploading, trigger parsing with
    ``POST /ragflow/datasets/{dataset_id}/documents/parse``.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    base_url, headers = _ragflow_headers()

    # Read all files into memory
    file_tuples: list[tuple[str, bytes, str]] = []
    for f in files:
        content = await f.read()
        file_tuples.append((
            f.filename or "document",
            content,
            f.content_type or "application/octet-stream",
        ))

    # ── Step 1: upload all files in one multipart request ───────────────────
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/api/v1/datasets/{dataset_id}/documents",
                headers=headers,
                files=[("file", (name, content, ct)) for name, content, ct in file_tuples],
                timeout=300.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    _check_ragflow_response(data)
    docs: list[dict] = data.get("data", [])

    # ── Step 2: optional bulk metadata update ───────────────────────────────
    update_errors: list[str] = []
    if docs and any([chunk_method, parser_config]):
        parsed_config: Any = None
        if parser_config:
            try:
                parsed_config = json.loads(parser_config)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=422,
                    detail="parser_config must be a valid JSON string",
                )

        update_payload: list[dict] = []
        for doc in docs:
            doc_id = doc.get("id")
            if not doc_id:
                continue
            item: dict[str, Any] = {"id": doc_id}
            if chunk_method:
                item["chunk_method"] = chunk_method
            if parsed_config is not None:
                item["parser_config"] = parsed_config
            update_payload.append(item)

        if update_payload:
            try:
                upd_headers = _ragflow_headers("application/json")[1]
                async with httpx.AsyncClient() as client:
                    upd_resp = await client.put(
                        f"{base_url}/api/v1/datasets/{dataset_id}/documents",
                        headers=upd_headers,
                        json=update_payload,
                        timeout=60.0,
                    )
                    upd_resp.raise_for_status()
                    upd_data = upd_resp.json()
                    if upd_data.get("code") != 0:
                        update_errors.append(upd_data.get("message", "bulk update failed"))
            except Exception as exc:
                update_errors.append(str(exc))

    result: dict[str, Any] = {
        "message": f"{len(docs)} document(s) uploaded successfully",
        "documents": docs,
        "dataset_id": dataset_id,
    }
    if update_errors:
        result["update_warnings"] = update_errors
    return result


# ---------------------------------------------------------------------------
# Document management endpoints
# ---------------------------------------------------------------------------

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
) -> dict:
    """List documents in a dataset with optional filters and pagination.

    - *id*: fetch a single document by its ID
    - *keywords*: filter documents whose name contains the substring
    - *create_time_from* / *create_time_to*: Unix timestamp range filter
    """
    base_url, headers = _ragflow_headers()
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
                headers=headers,
                params=params,
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    _check_ragflow_response(data)
    raw = data.get("data", {})
    if isinstance(raw, dict):
        documents = raw.get("docs", [])
        total = raw.get("total", len(documents))
    else:
        documents = raw
        total = len(documents)

    return {
        "documents": documents,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.put(
    "/datasets/{dataset_id}/documents/{document_id}",
    summary="Update a single document",
)
async def update_document(
    dataset_id: str,
    document_id: str,
    req: UpdateDocumentRequest,
) -> dict:
    """Update a document's display name, metadata, chunk method, and/or parser config.

    **chunk_method → recommended parser_config:**

    | chunk_method | parser_config |
    |---|---|
    | `naive` | `{"chunk_token_num":128,"delimiter":"\\n","html4excel":false,"layout_recognize":true,"raptor":{"use_raptor":false},"parent_child":{"use_parent_child":false,"children_delimiter":"\\n"}}` |
    | `qa` | `{"raptor":{"use_raptor":false}}` |
    | `manual` | `{"raptor":{"use_raptor":false}}` |
    | `table` | `null` |
    | `paper` | `{"raptor":{"use_raptor":false}}` |
    | `book` | `{"raptor":{"use_raptor":false}}` |
    | `laws` | `{"raptor":{"use_raptor":false}}` |
    | `presentation` | `{"raptor":{"use_raptor":false}}` |
    | `picture` | `null` |
    | `one` | `null` |
    | `knowledge-graph` | `{"chunk_token_num":128,"delimiter":"\\n","entity_types":["organization","person","location","event","time"]}` |
    | `email` | `null` |
    """
    base_url, headers = _ragflow_headers("application/json")
    payload = req.model_dump(exclude_none=True)
    if not payload:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.put(
                f"{base_url}/api/v1/datasets/{dataset_id}/documents/{document_id}",
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    _check_ragflow_response(data)
    return {"message": "Document updated successfully", "data": data.get("data")}


@router.delete(
    "/datasets/{dataset_id}/documents",
    summary="Delete one or more documents",
)
async def delete_documents(dataset_id: str, req: DeleteDocumentsRequest) -> dict:
    """Permanently delete documents from a dataset by their IDs."""
    base_url, headers = _ragflow_headers("application/json")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"{base_url}/api/v1/datasets/{dataset_id}/documents",
                headers=headers,
                json={"ids": req.ids},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    _check_ragflow_response(data)
    return {"message": "Documents deleted successfully", "deleted_ids": req.ids}


# ---------------------------------------------------------------------------
# Document parsing endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/datasets/{dataset_id}/documents/parse",
    summary="Start parsing (chunking) documents",
)
async def parse_documents(dataset_id: str, req: ParseDocumentsRequest) -> dict:
    """Trigger asynchronous parsing of the specified documents.

    Parsing converts uploaded files into searchable chunks according to each
    document's *chunk_method* and *parser_config*.  The operation runs in the
    background — poll ``GET /ragflow/datasets/{dataset_id}/documents`` and
    check the ``run_status`` field (``RUNNING`` → ``DONE`` / ``FAIL``).
    """
    base_url, headers = _ragflow_headers("application/json")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/api/v1/datasets/{dataset_id}/chunks",
                headers=headers,
                json={"document_ids": req.document_ids},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    _check_ragflow_response(data)
    return {
        "message": "Parsing started",
        "document_ids": req.document_ids,
        "data": data.get("data"),
    }


@router.delete(
    "/datasets/{dataset_id}/documents/parse",
    summary="Stop parsing documents",
)
async def stop_parsing_documents(dataset_id: str, req: ParseDocumentsRequest) -> dict:
    """Cancel in-progress parsing for the specified documents.

    Documents whose parsing is cancelled retain their previous chunks (if any).
    Their ``run_status`` will be set to ``CANCEL``.
    """
    base_url, headers = _ragflow_headers("application/json")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"{base_url}/api/v1/datasets/{dataset_id}/chunks",
                headers=headers,
                json={"document_ids": req.document_ids},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    _check_ragflow_response(data)
    return {
        "message": "Parsing cancelled",
        "document_ids": req.document_ids,
        "data": data.get("data"),
    }
