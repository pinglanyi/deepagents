"""Deep RAG agent package.

Exposes the three retrieval tools used by the Deep RAG agent,
plus the optional evaluate/think helpers for advanced usage.
"""

from rag_agent.prompts import (
    DEEP_RAG_ANSWER_FORMAT,
    DEEP_RAG_WORKFLOW_INSTRUCTIONS,
)
from rag_agent.tools import (
    evaluate_answer,
    get_next_chunks,
    ragflow_list_datasets,
    ragflow_retrieve,
    think,
)

__all__ = [
    # Core tools (used by default agent)
    "ragflow_list_datasets",
    "ragflow_retrieve",
    "get_next_chunks",
    # Optional tools (advanced / debugging)
    "evaluate_answer",
    "think",
    # Prompts
    "DEEP_RAG_WORKFLOW_INSTRUCTIONS",
    "DEEP_RAG_ANSWER_FORMAT",
]
