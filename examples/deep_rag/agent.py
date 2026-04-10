"""Deep RAG Agent — LangGraph entry point.

Creates a deep RAG agent that answers questions by progressively retrieving
document chunks from a RAGFlow knowledge base:

  0. Discover real dataset IDs  (ragflow_list_datasets)
  1. Retrieve a large batch     (ragflow_retrieve)
  2. Expose top-k to the LLM, keep the rest buffered
  3. Generate / refine an answer
  4. Evaluate quality           (evaluate_answer)
  5. Pop more chunks if needed  (get_next_chunks)
  6. Fetch the next page if the buffer is exhausted
  7. Repeat until confident or budget exhausted

Usage:
  langgraph dev          # start LangGraph Studio server
  uv run agent.py        # run interactively in the terminal
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from deepagents import create_deep_agent

from rag_agent.prompts import (
    DEEP_RAG_ANSWER_FORMAT,
    DEEP_RAG_LOOP_INSTRUCTIONS,
    DEEP_RAG_WORKFLOW_INSTRUCTIONS,
)
from rag_agent.tools import evaluate_answer, get_next_chunks, ragflow_list_datasets, ragflow_retrieve, think

load_dotenv()

# ---------------------------------------------------------------------------
# System prompt: workflow + loop rules + answer format
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "\n\n".join(
    [
        DEEP_RAG_WORKFLOW_INSTRUCTIONS,
        "=" * 72,
        DEEP_RAG_LOOP_INSTRUCTIONS,
        "=" * 72,
        DEEP_RAG_ANSWER_FORMAT,
    ]
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Default: DeepSeek (matches other examples in this repo).
# Override by setting DEEP_RAG_MODEL / DEEP_RAG_BASE_URL / DEEP_RAG_API_KEY.
# You can also pass any LangChain BaseChatModel instance directly.

_model_name = os.getenv("DEEP_RAG_MODEL", "deepseek-chat")
_api_key = os.getenv("DEEP_RAG_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
_base_url = os.getenv("DEEP_RAG_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "")

model_kwargs: dict = {"model": _model_name, "temperature": 0.0}
if _api_key:
    model_kwargs["api_key"] = _api_key
if _base_url:
    model_kwargs["base_url"] = _base_url

model = ChatOpenAI(**model_kwargs)

# ---------------------------------------------------------------------------
# RAG retrieval sub-agent (optional parallel retrieval)
# ---------------------------------------------------------------------------
# For simple single-dataset queries the main agent handles everything.
# For multi-dataset or multi-aspect questions the orchestrator can delegate
# parallel retrieval to this sub-agent.

rag_retrieval_subagent = {
    "name": "rag-retriever",
    "description": (
        "Delegate a focused retrieval task for a specific sub-question or dataset. "
        "Provide: the sub-question, dataset_ids, and any constraints. "
        "The sub-agent will retrieve chunks and return a structured finding summary."
    ),
    "system_prompt": (
        "You are a RAG retrieval specialist. Your only job is to:\n"
        "1. If dataset_ids are not provided or look like English words (not UUIDs),\n"
        "   call ragflow_list_datasets() first to discover real IDs.\n"
        "2. Call ragflow_retrieve() with the real dataset_ids and the given question.\n"
        "3. Call get_next_chunks() as needed (up to 3 times).\n"
        "4. Return a concise finding summary with inline citations [Chunk N | doc].\n\n"
        "Hard limits:\n"
        "- Max 2 ragflow_retrieve() calls per run (page=1 then page=2 if needed).\n"
        "- Max 3 get_next_chunks() calls per run.\n"
        "- Stop as soon as you can answer the sub-question confidently.\n"
        "- Do NOT write files — just return your findings as plain text."
    ),
    "tools": [ragflow_list_datasets, ragflow_retrieve, get_next_chunks, think],
}

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = create_deep_agent(
    model=model,
    tools=[ragflow_list_datasets, ragflow_retrieve, get_next_chunks, evaluate_answer, think],
    system_prompt=SYSTEM_PROMPT,
    subagents=[rag_retrieval_subagent],
)
