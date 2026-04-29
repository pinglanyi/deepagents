"""Context management for the Deep RAG agent.

Provides manual conversation compaction, token-tracking hooks, and
thread-level context statistics.

Architecture:
- ``create_context_middleware()`` — builds the `SummarizationToolMiddleware`
  that exposes the ``compact_conversation`` tool to the agent.
- ``ContextTracker`` — optional lightweight tracker that records compaction
  events for monitoring and debugging.
- ``compact_thread()`` — programmatic compaction trigger for the REST API.

Usage::

    from context_manager import create_context_middleware, ContextTracker

    tracker = ContextTracker()
    summ_mw, compact_mw = create_context_middleware(model, backend)
    agent = create_deep_agent(middleware=[summ_mw, compact_mw], ...)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    SummarizationToolMiddleware,
    create_summarization_middleware,
    create_summarization_tool_middleware,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context tracker — lightweight event log
# ---------------------------------------------------------------------------


class ContextTracker:
    """Tracks compaction and context events per thread.

    Thread-safe only for single-process use (not distributed).
    """

    def __init__(self) -> None:
        self._events: dict[str, list[dict[str, Any]]] = {}

    def record_compact(
        self,
        thread_id: str,
        messages_before: int,
        messages_after: int,
        tokens_before: int | None = None,
        tokens_after: int | None = None,
    ) -> None:
        """Record a compaction event for a thread.

        Args:
            thread_id: LangGraph thread ID.
            messages_before: Number of messages before compaction.
            messages_after: Number of messages after compaction.
            tokens_before: Estimated tokens before compaction.
            tokens_after: Estimated tokens after compaction.
        """
        events = self._events.setdefault(thread_id, [])
        events.append({
            "type": "compact",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "messages_before": messages_before,
            "messages_after": messages_after,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
        })

    def get_events(self, thread_id: str) -> list[dict[str, Any]]:
        """Return all context events for a thread."""
        return self._events.get(thread_id, [])

    def get_stats(self, thread_id: str) -> dict[str, Any]:
        """Return summary stats for a thread's context events.

        Args:
            thread_id: LangGraph thread ID.

        Returns:
            Dict with ``total_compactions`` and ``events``.
        """
        events = self._events.get(thread_id, [])
        compactions = [e for e in events if e["type"] == "compact"]
        return {
            "thread_id": thread_id,
            "total_compactions": len(compactions),
            "total_events": len(events),
            "events": events[-20:],  # last 20 events
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_context_middleware(
    model: Any,
    backend: Any,
) -> tuple[SummarizationMiddleware, SummarizationToolMiddleware]:
    """Create both summarization middleware layers with model-aware defaults.

    Args:
        model: A resolved LangChain chat model instance.
        backend: Backend for offloading conversation history.

    Returns:
        Tuple of ``(SummarizationMiddleware, SummarizationToolMiddleware)``.
    """
    summ_mw = create_summarization_middleware(model, backend)
    compact_mw = SummarizationToolMiddleware(summ_mw)
    return summ_mw, compact_mw


# ---------------------------------------------------------------------------
# Singleton tracker instance (process-level)
# ---------------------------------------------------------------------------

_context_tracker: ContextTracker | None = None


def get_context_tracker() -> ContextTracker:
    """Return the process-level ``ContextTracker`` singleton."""
    global _context_tracker
    if _context_tracker is None:
        _context_tracker = ContextTracker()
    return _context_tracker
