"""RAGFlow tools for the Deep RAG agent.

Implements progressive disclosure: retrieves a large batch of chunks from
RAGFlow once, then exposes them to the LLM incrementally (top-k at a time)
without flooding the model's context window.

Buffer lifecycle per agent run:
  1. ragflow_retrieve()  → fetches batch_size chunks, stores sorted buffer,
                           returns first top_k to the LLM
  2. get_next_chunks()   → pops next top_k from buffer (no extra API call)
  3. If buffer empty     → caller should ragflow_retrieve(page=N+1)
"""

from __future__ import annotations

import os
import threading
from typing import Any, Literal

import httpx
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Thread-local chunk buffer
# Each thread (= one agent run in LangGraph) gets its own isolated buffer.
# ---------------------------------------------------------------------------
_local = threading.local()


def _get_buffer() -> dict[str, Any]:
    """Return the thread-local chunk buffer, initialising if absent."""
    if not hasattr(_local, "buffer"):
        _local.buffer = {
            "chunks": [],
            "offset": 0,
            "total_in_ragflow": 0,
            "question": "",
            "dataset_ids": [],
            "page": 1,
            "loaded": False,
        }
    return _local.buffer


def _reset_buffer(
    chunks: list[dict],
    total: int,
    question: str,
    dataset_ids: list[str],
    page: int,
    top_k: int,
) -> None:
    """Overwrite the thread-local buffer with a fresh retrieval result."""
    buf = _get_buffer()
    buf["chunks"] = chunks
    buf["offset"] = top_k  # first top_k already returned to LLM
    buf["total_in_ragflow"] = total
    buf["question"] = question
    buf["dataset_ids"] = dataset_ids
    buf["page"] = page
    buf["loaded"] = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ragflow_base() -> tuple[str, dict[str, str]]:
    """Return (base_url, headers) from environment variables."""
    base_url = os.getenv("RAGFLOW_BASE_URL", "http://localhost:9380").rstrip("/")
    api_key = os.getenv("RAGFLOW_API_KEY", "")
    return base_url, {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _fmt_chunk(chunk: dict[str, Any], rank: int) -> str:
    """Format one chunk for display in the LLM prompt."""
    content = chunk.get("content", "").strip()
    doc = chunk.get("document_keyword") or chunk.get("doc_name") or "Unknown"
    score = chunk.get("similarity", 0.0)
    chunk_id = chunk.get("id", f"idx-{rank}")
    return (
        f"### Chunk {rank} | score={score:.3f} | source={doc}\n"
        f"id: {chunk_id}\n\n"
        f"{content}\n\n"
        "---"
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool(parse_docstring=True)
def ragflow_retrieve(
    question: str,
    dataset_ids: list[str],
    top_k: int = 6,
    batch_size: int = 64,
    page: int = 1,
    similarity_threshold: float = 0.2,
    vector_similarity_weight: float = 0.3,
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
                    Should be much larger than top_k (e.g. 64–128).
        page: Pagination index; increment to retrieve entirely new chunks.
        similarity_threshold: Minimum similarity score to include (0.0–1.0).
        vector_similarity_weight: Vector vs. keyword search blend (0.0–1.0).

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

    lines: list[str] = [
        f"## RAGFlow results for: '{question}'  (page={page})\n",
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
    buf = _get_buffer()

    if not buf.get("loaded"):
        return "Buffer is empty — call ragflow_retrieve() first."

    chunks = buf["chunks"]
    offset = buf["offset"]

    if offset >= len(chunks):
        next_page = buf["page"] + 1
        return (
            f"Buffer exhausted ({len(chunks)} chunks processed). "
            f"Call ragflow_retrieve(page={next_page}) to fetch the next page."
        )

    end = min(offset + top_k, len(chunks))
    next_batch = chunks[offset:end]
    buf["offset"] = end
    remaining = len(chunks) - end

    lines: list[str] = [
        f"## Next chunks from buffer (#{offset + 1}–#{end})\n",
    ]
    lines += [_fmt_chunk(c, offset + i + 1) for i, c in enumerate(next_batch)]

    if remaining > 0:
        lines.append(f"\n**{remaining} chunks still in buffer** — call get_next_chunks() for more.")
    else:
        next_page = buf["page"] + 1
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
