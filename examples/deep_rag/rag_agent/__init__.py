"""Deep RAG agent package.

Exposes the custom tools and prompt templates used by the Deep RAG agent.
"""

from rag_agent.prompts import (
    DEEP_RAG_ANSWER_FORMAT,
    DEEP_RAG_LOOP_INSTRUCTIONS,
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
    # Tools
    "ragflow_list_datasets",
    "ragflow_retrieve",
    "get_next_chunks",
    "evaluate_answer",
    "think",
    # Prompts
    "DEEP_RAG_WORKFLOW_INSTRUCTIONS",
    "DEEP_RAG_LOOP_INSTRUCTIONS",
    "DEEP_RAG_ANSWER_FORMAT",
]
