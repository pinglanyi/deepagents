"""RAGFlow tools for the Deep RAG agent.

Knowledge base routing:
  - product KB  (meta_fields: model/series)  → Q&A intent
  - image KB    (image URL, description)      → image intent
  - file KB     (file URL, description)       → file intent
  - video KB    (video URL, description)      → video intent

Progressive disclosure: retrieves a large batch of chunks once, then
exposes them to the LLM incrementally without flooding the context window.

Buffer lifecycle per agent run:
  1. ragflow_retrieve()  → fetches batch_size chunks, stores sorted buffer,
                           returns first top_k to the LLM
  2. get_next_chunks()   → pops next top_k from buffer (no extra API call)
  3. If buffer empty     → caller should ragflow_retrieve(page=N+1)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import httpx
from langchain_core.tools import tool

from core.config import settings

# ---------------------------------------------------------------------------
# Global chunk buffer
#
# LangGraph runs each sync tool call in a thread-pool executor
# (run_in_executor), so threading.local() does NOT work — ragflow_retrieve
# writes in Thread-A while get_next_chunks reads in Thread-B, seeing an
# empty buffer. A plain module-level dict is the correct fix for a
# single-process deployment.
# ---------------------------------------------------------------------------
_buffer: dict[str, Any] = {
    "chunks": [],
    "offset": 0,
    "total_in_ragflow": 0,
    "question": "",
    "dataset_ids": [],
    "page": 1,
    "loaded": False,
}


def _reset_buffer(
    chunks: list[dict],
    total: int,
    question: str,
    dataset_ids: list[str],
    page: int,
    top_k: int,
) -> None:
    """Overwrite the global buffer with a fresh retrieval result."""
    _buffer["chunks"] = chunks
    _buffer["offset"] = top_k  # first top_k already returned to LLM
    _buffer["total_in_ragflow"] = total
    _buffer["question"] = question
    _buffer["dataset_ids"] = dataset_ids
    _buffer["page"] = page
    _buffer["loaded"] = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ragflow_base() -> tuple[str, dict[str, str]]:
    """Return (base_url, headers) from application settings."""
    base_url = settings.ragflow_base_url.rstrip("/")
    return base_url, {
        "Authorization": f"Bearer {settings.ragflow_api_key}",
        "Content-Type": "application/json",
    }


def _fmt_chunk(chunk: dict[str, Any], rank: int) -> str:
    """Format one chunk for display in the LLM prompt."""
    content = chunk.get("content", "").strip()
    doc = chunk.get("document_keyword") or chunk.get("doc_name") or "Unknown"
    score = chunk.get("similarity", 0.0)
    chunk_id = chunk.get("id", f"idx-{rank}")
    meta = chunk.get("meta_fields") or chunk.get("document_meta_fields") or {}
    meta_str = ""
    if meta:
        meta_str = f"meta: {json.dumps(meta, ensure_ascii=False)} | "
    return (
        f"### Chunk {rank} | score={score:.3f} | {meta_str}source={doc}\n"
        f"id: {chunk_id}\n\n"
        f"{content}\n\n"
        "---"
    )


def _get_kb_name(kb_type: str) -> str:
    """Return configured dataset name substring for a KB type."""
    mapping = {
        "product": settings.product_kb_name,
        "image": settings.image_kb_name,
        "file": settings.file_kb_name,
        "video": settings.video_kb_name,
    }
    return mapping.get(kb_type, kb_type)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool(parse_docstring=True)
def get_kb_datasets_by_type(
    kb_type: Literal["product", "image", "file", "video"],
) -> str:
    """Get RAGFlow dataset IDs for a specific knowledge base type.

    Call this instead of ragflow_list_datasets() when the user's intent is clear.
    Use the returned dataset IDs in ragflow_retrieve().

    KB type → intent mapping:
    - "product" : Q&A about product specs, features, parameters (supports meta_fields model filter)
    - "image"   : Find product images, drawings, photos, renderings
    - "file"    : Find documents, manuals, spec sheets, PDFs, downloads
    - "video"   : Find videos, tutorials, demos, installation guides

    Args:
        kb_type: The knowledge base type matching the user's intent.

    Returns:
        Table of matching dataset IDs and names. Use the ID column in ragflow_retrieve().
    """
    base_url, headers = _ragflow_base()
    name_filter = _get_kb_name(kb_type)

    try:
        resp = httpx.get(
            f"{base_url}/api/v1/datasets",
            headers=headers,
            params={"page": 1, "page_size": 100},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        return f"RAGFlow HTTP error {exc.response.status_code}: {exc.response.text[:400]}"
    except Exception as exc:  # noqa: BLE001
        return f"RAGFlow request failed: {exc}"

    if data.get("code") != 0:
        return f"RAGFlow API error: {data.get('message', 'unknown error')}"

    datasets: list[dict] = data.get("data", [])
    matched = [d for d in datasets if name_filter.lower() in d.get("name", "").lower()]

    if not matched:
        # Fallback: return all datasets so the LLM can pick the closest match
        all_names = [d.get("name", "") for d in datasets]
        return (
            f"No dataset found with '{name_filter}' in its name.\n"
            f"Available datasets: {all_names}\n\n"
            "Pick the closest match or call ragflow_list_datasets() for full details."
        )

    lines = [
        f"## {kb_type.title()} KB Datasets ({len(matched)} found)\n",
        "| ID | Name | Documents | Chunks |",
        "|----|------|-----------|--------|",
    ]
    for ds in matched:
        lines.append(
            f"| `{ds.get('id', '—')}` | {ds.get('name', '—')} "
            f"| {ds.get('document_count', ds.get('doc_num', '?'))} "
            f"| {ds.get('chunk_count', ds.get('chunk_num', '?'))} |"
        )
    lines.append("\nUse these IDs as `dataset_ids` in ragflow_retrieve().")
    return "\n".join(lines)


@tool(parse_docstring=True)
def complete_model_number(partial_model: str) -> str:
    """Look up a full model number from a partial name, alias, or abbreviation.

    Use this when the user's model reference looks incomplete, abbreviated, or
    ambiguous. The tool searches the configured model aliases database and
    returns candidates.

    IMPORTANT: After calling this tool, ALWAYS present the candidates to the
    user and ask for explicit confirmation before proceeding with retrieval.
    Do not assume a match — the user must confirm.

    Args:
        partial_model: The partial model name, alias, or abbreviation from the
                       user's question.

    Returns:
        Candidate full model numbers with their known aliases, and instructions
        to confirm with the user.
    """
    aliases_file = settings.model_aliases_file

    if not aliases_file or not Path(aliases_file).exists():
        return (
            f"No model aliases database configured (MODEL_ALIASES_FILE not set or file not found).\n"
            f"Using '{partial_model}' as-is.\n\n"
            "Ask the user: 'Could you confirm the full model number? I want to make sure "
            "I'm searching for the right product.'"
        )

    try:
        with open(aliases_file, encoding="utf-8") as f:
            db = json.load(f)
    except Exception as exc:  # noqa: BLE001
        return f"Failed to read aliases file '{aliases_file}': {exc}"

    # Expected format: {"aliases": {"FULL_MODEL": ["alias1", "alias2", ...], ...}}
    aliases: dict[str, list[str]] = db.get("aliases", {})
    partial_lower = partial_model.strip().lower()

    matches: list[dict[str, Any]] = []
    for full_model, alias_list in aliases.items():
        all_names = [full_model] + (alias_list or [])
        if any(
            partial_lower in name.lower() or name.lower() in partial_lower
            for name in all_names
        ):
            matches.append({"model": full_model, "aliases": alias_list or []})

    if not matches:
        return (
            f"No match found for '{partial_model}' in the aliases database.\n\n"
            "Ask the user to provide the complete model number before proceeding."
        )

    if len(matches) == 1:
        m = matches[0]
        alias_str = "、".join(m["aliases"]) if m["aliases"] else "无"
        return (
            f"Found 1 candidate: **{m['model']}** (别称: {alias_str})\n\n"
            f"请向用户确认: '您查询的是 **{m['model']}** 吗？确认后我将为您检索。'"
        )

    lines = [f"找到 {len(matches)} 个可能的型号匹配 '{partial_model}':\n"]
    for i, m in enumerate(matches, 1):
        alias_str = "、".join(m["aliases"]) if m["aliases"] else "无别称"
        lines.append(f"{i}. **{m['model']}** ({alias_str})")
    lines.append(
        "\n请向用户确认是哪个型号，确认后再进行检索。"
        "\n例如: '您查询的是以上哪个型号？请告诉我，我将为您精准检索。'"
    )
    return "\n".join(lines)


@tool(parse_docstring=True)
def ragflow_list_datasets(name_filter: str = "") -> str:
    """List all available knowledge-base datasets in RAGFlow.

    Use this when you need an overview of all available datasets, or when
    get_kb_datasets_by_type() returns no results.

    Args:
        name_filter: Optional substring to filter dataset names (case-insensitive).
                     Pass "" to list all datasets.

    Returns:
        Table of available datasets with their IDs, names, document counts,
        and chunk counts. Use the `id` column values as `dataset_ids` in
        ragflow_retrieve().
    """
    base_url, headers = _ragflow_base()

    try:
        resp = httpx.get(
            f"{base_url}/api/v1/datasets",
            headers=headers,
            params={"page": 1, "page_size": 100},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        return f"RAGFlow HTTP error {exc.response.status_code}: {exc.response.text[:400]}"
    except Exception as exc:  # noqa: BLE001
        return f"RAGFlow request failed: {exc}"

    if data.get("code") != 0:
        return f"RAGFlow API error: {data.get('message', 'unknown error')}"

    datasets: list[dict] = data.get("data", [])
    if not datasets:
        return "No datasets found in RAGFlow. Create a knowledge base first."

    if name_filter:
        datasets = [d for d in datasets if name_filter.lower() in d.get("name", "").lower()]
        if not datasets:
            return f"No datasets match filter '{name_filter}'. Remove the filter to see all."

    lines = [
        f"## Available RAGFlow Datasets ({len(datasets)} found)\n",
        "| ID | Name | Documents | Chunks | Status |",
        "|----|------|-----------|--------|--------|",
    ]
    for ds in datasets:
        ds_id = ds.get("id", "—")
        name = ds.get("name", "—")
        doc_count = ds.get("document_count", ds.get("doc_num", "?"))
        chunk_count = ds.get("chunk_count", ds.get("chunk_num", "?"))
        status = ds.get("status", ds.get("parse_status", "—"))
        lines.append(f"| `{ds_id}` | {name} | {doc_count} | {chunk_count} | {status} |")

    lines.append(
        "\nUse one or more values from the **ID** column as `dataset_ids` in ragflow_retrieve()."
    )
    return "\n".join(lines)


@tool(parse_docstring=True)
def ragflow_retrieve(
    question: str,
    dataset_ids: list[str],
    top_k: int = 6,
    batch_size: int = 32,
    page: int = 1,
    similarity_threshold: float = 0.2,
    vector_similarity_weight: float = 0.3,
    model_filter: str = "",
) -> str:
    """Retrieve a large batch of chunks from RAGFlow and return the most relevant ones.

    Fetches `batch_size` chunks from RAGFlow for `question`, ranks them by
    similarity, returns the top `top_k` to the LLM, and stores the rest in a
    progressive buffer. Call get_next_chunks() to access buffered chunks
    without another API round-trip.

    To get a completely fresh set of results (different documents), increment
    `page` (e.g., page=2, page=3).

    Args:
        question: The search query to send to RAGFlow.
        dataset_ids: One or more RAGFlow knowledge-base / dataset IDs.
        top_k: Number of chunks to return immediately (exposed to LLM).
        batch_size: Total chunks fetched from RAGFlow per call (buffer size).
                    Should be larger than top_k (default 32; use 64 for deeper search).
        page: Pagination index; increment to retrieve entirely new chunks.
        similarity_threshold: Minimum similarity score to include (0.0–1.0).
        vector_similarity_weight: Vector vs. keyword search blend (0.0–1.0).
        model_filter: If set, filter results to chunks whose document meta_fields
                      contain this model number or series (e.g. "G10", "X-200").
                      Only meaningful for the product KB. Leave "" for all other KBs
                      or when no model number is known.

    Returns:
        Formatted top_k chunks with scores/sources, plus buffer status info.
    """
    base_url, headers = _ragflow_base()

    payload: dict[str, Any] = {
        "question": question,
        "dataset_ids": dataset_ids,
        "top_k": batch_size,
        "similarity_threshold": similarity_threshold,
        "vector_similarity_weight": vector_similarity_weight,
        "page": page,
        "page_size": batch_size,
        "highlight": False,
    }

    # Pass model filter as a document-level condition when supported by RAGFlow
    if model_filter:
        payload["condition"] = {"fields": {"model": model_filter}}

    try:
        resp = httpx.post(
            f"{base_url}/api/v1/retrieval",
            headers=headers,
            json=payload,
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        return f"RAGFlow HTTP error {exc.response.status_code}: {exc.response.text[:400]}"
    except Exception as exc:  # noqa: BLE001
        return f"RAGFlow request failed: {exc}"

    if data.get("code") != 0:
        return f"RAGFlow API error: {data.get('message', 'unknown error')}"

    raw_chunks: list[dict] = data.get("data", {}).get("chunks", [])
    total_in_ragflow: int = data.get("data", {}).get("total", len(raw_chunks))

    if not raw_chunks:
        hint = "No chunks found. Check dataset_ids, try rephrasing the query, or use page=1."
        return f"[ragflow_retrieve] {hint}"

    # Sort by combined similarity score, highest first
    raw_chunks.sort(key=lambda c: float(c.get("similarity", 0.0)), reverse=True)

    # Post-filter by model_filter against meta_fields when set
    # (handles cases where RAGFlow doesn't support server-side condition filtering)
    if model_filter:
        model_lower = model_filter.lower()
        def _matches_model(chunk: dict) -> bool:
            meta = chunk.get("meta_fields") or chunk.get("document_meta_fields") or {}
            meta_str = json.dumps(meta, ensure_ascii=False).lower()
            doc_name = (chunk.get("doc_name") or chunk.get("document_keyword") or "").lower()
            return model_lower in meta_str or model_lower in doc_name

        filtered = [c for c in raw_chunks if _matches_model(c)]
        # Keep filtered results first; append unfiltered remainder so LLM still has context
        if filtered:
            unfiltered = [c for c in raw_chunks if not _matches_model(c)]
            raw_chunks = filtered + unfiltered

    # Populate the progressive buffer
    _reset_buffer(
        chunks=raw_chunks,
        total=total_in_ragflow,
        question=question,
        dataset_ids=dataset_ids,
        page=page,
        top_k=top_k,
    )

    top_chunks = raw_chunks[:top_k]
    buffered_remaining = len(raw_chunks) - top_k

    filter_note = f" (filtered by model='{model_filter}')" if model_filter else ""
    lines: list[str] = [
        f"## RAGFlow results for: '{question}'{filter_note}  (page={page})\n",
        f"Fetched {len(raw_chunks)} chunks from RAGFlow "
        f"(total matching in index: {total_in_ragflow}).\n",
        f"Showing top {len(top_chunks)} — {buffered_remaining} more are buffered.\n",
        "",
    ]
    lines += [_fmt_chunk(c, i + 1) for i, c in enumerate(top_chunks)]

    if buffered_remaining > 0:
        lines.append(
            f"\n**{buffered_remaining} chunks remain in buffer** — "
            "call get_next_chunks() to access them progressively."
        )
    else:
        lines.append(
            f"\nAll {len(raw_chunks)} fetched chunks shown. "
            f"Call ragflow_retrieve(page={page + 1}) for more results."
        )

    return "\n".join(lines)


@tool(parse_docstring=True)
def get_next_chunks(top_k: int = 5) -> str:
    """Return the next batch of chunks from the in-memory retrieval buffer.

    Use this after ragflow_retrieve() to progressively expose more evidence
    without making additional API calls. Each invocation advances the buffer
    cursor by `top_k` positions.

    Call this when the current answer is uncertain, incomplete, or requires
    more supporting evidence, before resorting to a new ragflow_retrieve() call.

    Args:
        top_k: Number of additional chunks to pop from the buffer.

    Returns:
        Next batch of chunks with scores/sources, plus remaining buffer count.
    """
    if not _buffer.get("loaded"):
        return "Buffer is empty — call ragflow_retrieve() first."

    chunks = _buffer["chunks"]
    offset = _buffer["offset"]

    if offset >= len(chunks):
        next_page = _buffer["page"] + 1
        return (
            f"Buffer exhausted ({len(chunks)} chunks processed). "
            f"Call ragflow_retrieve(page={next_page}) to fetch the next page."
        )

    end = min(offset + top_k, len(chunks))
    next_batch = chunks[offset:end]
    _buffer["offset"] = end
    remaining = len(chunks) - end

    lines: list[str] = [
        f"## Next chunks from buffer (#{offset + 1}–#{end})\n",
    ]
    lines += [_fmt_chunk(c, offset + i + 1) for i, c in enumerate(next_batch)]

    if remaining > 0:
        lines.append(f"\n**{remaining} chunks still in buffer** — call get_next_chunks() for more.")
    else:
        next_page = _buffer["page"] + 1
        lines.append(
            f"\nBuffer exhausted. Call ragflow_retrieve(page={next_page}) for fresh results."
        )

    return "\n".join(lines)


@tool(parse_docstring=True)
def evaluate_answer(
    question: str,
    current_answer: str,
    confidence: Literal["high", "medium", "low"],
    missing_aspects: list[str],
    chunks_used: int,
) -> str:
    """Record a structured self-evaluation of the current answer quality.

    Call this after drafting or refining an answer to decide whether to
    finalize or continue retrieving more chunks.

    Decision guide:
    - confidence=high, missing_aspects=[]  → FINALIZE the answer.
    - confidence=medium                    → call get_next_chunks() to verify.
    - confidence=low                       → call get_next_chunks() or ragflow_retrieve(page=N+1).

    Args:
        question: The original user question.
        current_answer: The answer draft being evaluated.
        confidence: Subjective confidence in the answer (high/medium/low).
        missing_aspects: List of question aspects not yet covered (empty if none).
        chunks_used: Number of chunks incorporated into the answer.

    Returns:
        Evaluation summary with a clear FINALIZE or CONTINUE recommendation.
    """
    if confidence == "high" and not missing_aspects:
        recommendation = "FINALIZE — answer is complete and well-supported."
    elif confidence == "low" or len(missing_aspects) > 2:
        recommendation = (
            "CONTINUE — significant gaps remain. "
            "Call get_next_chunks() or ragflow_retrieve(page=N+1)."
        )
    else:
        gap_str = ", ".join(missing_aspects) if missing_aspects else "none"
        recommendation = (
            f"CONTINUE (optional) — answer is usable but missing: {gap_str}. "
            "Call get_next_chunks() to attempt improvement, or finalize if pressed for time."
        )

    return (
        f"**Evaluation**\n"
        f"- Question: {question}\n"
        f"- Chunks used: {chunks_used}\n"
        f"- Answer length: {len(current_answer)} chars\n"
        f"- Confidence: {confidence}\n"
        f"- Missing aspects: {missing_aspects or 'none'}\n"
        f"- Recommendation: **{recommendation}**"
    )


@tool(parse_docstring=True)
def think(thought: str) -> str:
    """Record a private reasoning step to guide the retrieval-answer loop.

    Use this tool to pause and reflect between retrieval calls:
    - After reviewing a chunk batch: what did I learn? what's still missing?
    - Before deciding to get_next_chunks vs. finalize: is the answer good enough?
    - When reformulating the query: would a different phrasing surface better chunks?

    Args:
        thought: Your analysis of current evidence, gaps, and the next action.

    Returns:
        Acknowledgement that the thought was recorded.
    """
    return f"Thought recorded: {thought}"
