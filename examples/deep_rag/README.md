# Deep RAG

A deep retrieval-augmented generation agent built on [deepagents](../../libs/deepagents/).
It answers questions by **progressively** fetching chunks from a
[RAGFlow](https://github.com/infiniflow/ragflow) knowledge base — retrieving a large
batch once, then disclosing them to the LLM a few at a time until a confident answer
emerges.

## How it works

```
User question
      │
      ▼
ragflow_retrieve(batch_size=64, top_k=6)
      │  ← returns top 6 to LLM; 58 stay in buffer
      ▼
Draft answer + think()
      │
      ▼
evaluate_answer()  ─── confidence=high ──────────────► FINALIZE
      │
   confidence < high
      │
      ▼
get_next_chunks(top_k=5)   ← no extra API call; pops from buffer
      │
      ▼
Refine answer + evaluate_answer()
      │
   buffer exhausted?
      │  yes
      ▼
ragflow_retrieve(page=2)   ← fresh batch from RAGFlow
      │
      ▼  (repeat up to 3 pages / 10 iterations)
      │
      ▼
write_file("/rag_answer.md")  +  respond to user
```

### Key design principles

| Principle | Details |
|-----------|---------|
| **Large batch, small window** | Retrieve up to 128 chunks per API call; expose 4–6 at a time to the LLM |
| **Progressive disclosure** | `get_next_chunks()` reads from an in-memory buffer — zero extra API calls |
| **Structured self-evaluation** | `evaluate_answer()` gives a typed confidence signal that drives the loop |
| **Pagination** | `ragflow_retrieve(page=N)` fetches entirely new chunks when the buffer runs dry |
| **Optional sub-agent** | `rag-retriever` sub-agent can be delegated parallel sub-questions |

---

## Quickstart

**Prerequisites:** Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Navigate to this directory and install dependencies:

```bash
cd examples/deep_rag
uv sync
```

Copy and fill in environment variables:

```bash
cp .env.example .env
# Edit .env — set RAGFLOW_BASE_URL, RAGFLOW_API_KEY, and your LLM keys
```

### Option 1 — LangGraph Studio (recommended)

```bash
langgraph dev
```

LangGraph Studio opens in the browser. Submit your question together with the
`dataset_ids` you want to search, e.g.:

```
What are the main causes of supply chain disruptions?
(dataset_ids: ["kb_supply_chain_2024"])
```

### Option 2 — Python script

```python
import asyncio
from agent import agent

async def main():
    result = await agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    "What are the main causes of supply chain disruptions? "
                    "Use dataset_ids=[\"kb_supply_chain_2024\"]"
                ),
            }
        ]
    })
    for msg in result["messages"]:
        print(msg)

asyncio.run(main())
```

---

## Configuration

### RAGFlow

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGFLOW_BASE_URL` | `http://localhost:9380` | RAGFlow server URL |
| `RAGFLOW_API_KEY` | — | RAGFlow API key |

Find your dataset IDs in the RAGFlow UI under **Knowledge Base → Dataset**.

### LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEP_RAG_MODEL` | `deepseek-chat` | Model name (any OpenAI-compatible) |
| `DEEP_RAG_API_KEY` | — | API key for the LLM provider |
| `DEEP_RAG_BASE_URL` | — | Custom base URL (e.g. DeepSeek, vLLM) |

You can also use Anthropic or standard OpenAI — just set the appropriate env vars
and change the model initialisation in `agent.py`.

---

## Tools

| Tool | Purpose |
|------|---------|
| `ragflow_retrieve` | Fetch a large chunk batch from RAGFlow; return top-k; buffer the rest |
| `get_next_chunks` | Pop next batch from the in-memory buffer (no extra API call) |
| `evaluate_answer` | Structured self-evaluation: confidence + missing aspects + recommendation |
| `think` | Private reasoning step to guide retrieval decisions |

Built-in deepagents tools (`write_file`, `read_file`, `write_todos`, `task`, …) are
also available.

---

## Customisation

### Change retrieval parameters

In `agent.py` or directly in your prompt, adjust:

- `batch_size` — how many chunks to fetch per `ragflow_retrieve` call (default 64)
- `top_k` — how many chunks to show the LLM per call (default 6)
- `similarity_threshold` — minimum score to include a chunk (default 0.2)
- `vector_similarity_weight` — blend of vector vs. keyword search (default 0.3)

### Use a different LLM

```python
# Anthropic
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.0)

# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
```

Pass the model object to `create_deep_agent(model=model, ...)`.

### Add a reranker

After `ragflow_retrieve` returns chunks you can add a local reranking step by
registering a custom middleware or wrapping the tool to call a reranker API before
populating the buffer.

---

## Prompt customisation

| Prompt | Location | Purpose |
|--------|----------|---------|
| `DEEP_RAG_WORKFLOW_INSTRUCTIONS` | `rag_agent/prompts.py` | 6-step high-level workflow |
| `DEEP_RAG_LOOP_INSTRUCTIONS` | `rag_agent/prompts.py` | Per-iteration decision rules |
| `DEEP_RAG_ANSWER_FORMAT` | `rag_agent/prompts.py` | Final answer structure & citation style |
