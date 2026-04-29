# Deep RAG — Multi-user Commercial Backend

A production-ready multi-user RAG backend built on [deepagents](../../libs/deepagents/) and
[RAGFlow](https://github.com/infiniflow/ragflow). It combines an intent-routing retrieval
agent with a full REST API supporting authentication, per-user conversation isolation,
knowledge-base management, and batch document upload.

## Architecture

```
                    ┌──────────────────────────┐
                    │   FastAPI REST Server     │
                    │   (main.py)               │
                    │                           │
                    │  /auth/*    JWT auth      │
                    │  /chat/*    Chat + SSE    │
                    │  /threads/* Per-user      │
                    │  /ragflow/* KB mgmt       │
                    │  /users/*   Admin panel   │
                    └──────────┬───────────────┘
                               │
               ┌───────────────┴───────────────┐
               │                               │
       ┌───────▼────────┐            ┌────────▼────────┐
       │  LangGraph Agent│            │   PostgreSQL     │
       │  (agent.py)     │            │                  │
       │                 │            │  • User accounts │
       │  Intent routing │            │  • Threads       │
       │  Model matching │            │  • Checkpoints   │
       │  KB retrieval   │            │  • Media index   │
       │  Skills         │            └─────────────────┘
       │  Memory         │
       └───────┬─────────┘
               │
       ┌───────▼────────┐
       │    RAGFlow      │
       │  Knowledge Base │
       │  + MinIO (S3)   │
       └────────────────┘
```

### Agent workflow — intent routing (not a blind retrieval loop)

The agent follows a **6-step hard-budget** workflow with at most 6 tool calls per turn:

1. **Analyze intent** — detect whether the user wants Q&A, images, files, or videos
2. **Model completion** — resolve abbreviated/ambiguous product model numbers before searching
3. **Route to KB** — call `get_kb_datasets_by_type()` (one call, specific KB) or `ragflow_list_datasets()` (no model number, all KBs)
4. **Retrieve** — `ragflow_retrieve()` with optional `model_filter` for product KB (1–2 calls max)
5. **Enrich** — `get_next_chunks()` from in-memory buffer if a detail is missing (0–1 call)
6. **Answer inline** — respond directly in chat, no file writing

Key design principles:

| Principle | Detail |
|-----------|--------|
| **Intent-first routing** | Detect intent → route to the correct KB type (product/image/file/video) |
| **Model-aware search** | Extract + complete model numbers; filter chunks by `meta_fields.model` |
| **Large batch, small window** | Fetch 32+ chunks per API call; expose 6 at a time via progressive buffer |
| **Hard tool budget** | ≤ 6 tool calls per question; answer inline — no planning, no file writing |
| **Progressive disclosure** | `get_next_chunks()` reads from in-memory buffer — zero extra API calls |

---

## Quickstart

**Prerequisites:** Python ≥ 3.11, [uv](https://docs.astral.sh/uv/), and a running
[RAGFlow](https://github.com/infiniflow/ragflow) instance.

```bash
cd examples/deep_rag
uv sync
cp .env.example .env
# Edit .env — set DATABASE_URL, SECRET_KEY, RAGFLOW_API_KEY, and LLM keys
```

### Option 1 — LangGraph Studio (single-user, dev)

```bash
uv run langgraph dev --port 8122
```

Opens LangGraph Studio in the browser. Conversation history persists in SQLite
(`agent_data/checkpoints.sqlite`) or Postgres (set `POSTGRES_URI` in `.env`).

### Option 2 — Full multi-user backend (production)

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8123
# or with multiple workers:
uv run uvicorn main:app --host 0.0.0.0 --port 8123 --workers 4
```

Requires PostgreSQL. Tables are auto-created on startup.

---

## API overview

```
Authentication
    POST   /auth/register          Register new account
    POST   /auth/login             Login → access + refresh tokens
    POST   /auth/refresh           Rotate refresh token
    POST   /auth/logout            Revoke refresh token
    POST   /auth/logout/all        Revoke all refresh tokens

User profile (authenticated)
    GET    /users/me               Get own profile
    PUT    /users/me               Update profile / change password
    DELETE /users/me               Delete own account

User management (admin only)
    GET    /users                  List all users (paginated)
    GET    /users/stats            List users with thread-count stats
    GET    /users/{id}             Get user by ID
    PUT    /users/{id}             Update any user
    DELETE /users/{id}             Delete user

Chat (authenticated)
    POST   /chat                   Non-streaming chat
    POST   /chat/stream            Streaming chat (SSE)

Threads (authenticated — isolated per user)
    GET    /threads                List own threads
    GET    /threads/{id}           Full message history
    PATCH  /threads/{id}           Rename thread
    DELETE /threads/{id}           Delete thread
    DELETE /threads                Delete all own threads
    GET    /threads/{id}/export    Export as plain text

RAGFlow KB management (auth required, write = admin)
    POST   /ragflow/datasets                            Create dataset
    GET    /ragflow/datasets                            List datasets
    PUT    /ragflow/datasets/{id}                       Update dataset
    DELETE /ragflow/datasets                            Delete datasets
    POST   /ragflow/datasets/{id}/documents/upload      Upload single doc
    POST   /ragflow/datasets/{id}/documents/upload/batch  Batch upload
    GET    /ragflow/datasets/{id}/documents             List documents
    PUT    /ragflow/datasets/{id}/documents/{doc_id}    Update document
    DELETE /ragflow/datasets/{id}/documents             Delete documents
    POST   /ragflow/datasets/{id}/documents/parse       Start parsing
    DELETE /ragflow/datasets/{id}/documents/parse       Cancel parsing
    GET    /ragflow/media/{id}                          Get media file

Health
    GET    /health
```

Chat authentication: pass `Authorization: Bearer <access_token>` in headers.
Tokens expire per `ACCESS_TOKEN_EXPIRE_MINUTES` / `REFRESH_TOKEN_EXPIRE_DAYS`.

---

## Configuration

### Database (required for Option 2)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | asyncpg PostgreSQL URL (e.g. `postgresql+asyncpg://postgres:postgres@localhost:5432/deeprag`) |

### Auth / JWT

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | — | JWT signing secret (generate: `openssl rand -hex 32`) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Access token lifetime |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `30` | Refresh token lifetime |

### RAGFlow

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGFLOW_BASE_URL` | `http://localhost:9380` | RAGFlow server URL |
| `RAGFLOW_API_KEY` | — | RAGFlow API key |

### KB type naming (dataset name substrings)

| Variable | Default | Purpose |
|----------|---------|---------|
| `PRODUCT_KB_NAME` | `产品库` | Product specs, features, Q&A |
| `IMAGE_KB_NAME` | `图片库` | Product images, photos, drawings |
| `FILE_KB_NAME` | `文件库` | Documents, manuals, PDFs |
| `VIDEO_KB_NAME` | `视频库` | Video tutorials, demos |
| `GENERAL_KB_NAME` | `通用库` | Industry standards, general knowledge |
| `PROGRAM_KB_NAME` | `程序库` | Program/function parameters |
| `EXPERIENCE_KB_NAME` | `经验库` | Troubleshooting cases, history |

### Model aliases

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ALIASES_FILE` | — | Path to JSON file with model aliases for fuzzy matching |

### LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEP_RAG_MODEL` | `deepseek-chat` | Model name (any OpenAI-compatible) |
| `DEEP_RAG_API_KEY` | — | LLM API key |
| `DEEP_RAG_BASE_URL` | — | Custom base URL (DeepSeek, vLLM, Anthropic, etc.) |

### Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_DATA_DIR` | `./agent_data` | Agent files, long-term memory, SQLite checkpoints |

### MinIO / S3 (RAGFlow built-in, for document download URLs)

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO (S3) endpoint |
| `MINIO_ACCESS_KEY` | — | MinIO access key (empty = fallback to RAGFlow preview URL) |
| `MINIO_SECRET_KEY` | — | MinIO secret key |
| `MINIO_SECURE` | `false` | Use HTTPS |
| `MINIO_PRESIGNED_EXPIRY_SECONDS` | `3600` | Presigned URL validity (seconds) |

### Batch upload throttling

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_UPLOAD_MAX_CONCURRENT` | `1` | Max concurrent batch-upload requests (FIFO queue) |
| `BATCH_UPLOAD_COOLDOWN_SECONDS` | `2.0` | Seconds to wait between files within one batch |

### CORS

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `["*"]` | JSON list of allowed origins |

---

## Tools

| Tool | Purpose |
|------|---------|
| `get_kb_datasets_by_type` | Look up dataset IDs for a KB type (product/image/file/video) — one call, no guesswork |
| `ragflow_list_datasets` | List all available datasets in RAGFlow with IDs, doc counts, status |
| `ragflow_retrieve` | Fetch chunks from RAGFlow, return top-k, buffer the rest. Supports `model_filter` |
| `get_next_chunks` | Pop next batch from in-memory buffer (no API call) |
| `complete_model_number` | Resolve abbreviated/ambiguous product model numbers via aliases database |
| `evaluate_answer` | Structured self-evaluation (confidence + missing aspects) — use sparingly |
| `think` | Private reasoning step — use only when genuinely stuck |

Built-in deepagents tools (`write_file`, `read_file`, `edit_file`, `write_todos`, `task`, …)
are also available.

---

## Skills

Three RAG-specific skills are loaded from `./skills/` and available to the agent:

| Skill | Description | Key tools |
|-------|-------------|-----------|
| `rag-search` | Multi-source retrieval across all KB types | `ragflow_list_datasets`, `ragflow_retrieve`, `get_next_chunks`, `complete_model_number`, `get_kb_datasets_by_type` |
| `knowledge-base` | Manage RAGFlow knowledge bases (create, upload, parse, delete) | `ragflow_list_datasets`, `ragflow_retrieve` |
| `troubleshooting` | Diagnose product issues using experience KB + product specs | `ragflow_list_datasets`, `ragflow_retrieve`, `get_next_chunks`, `get_kb_datasets_by_type` |

Skills use progressive disclosure — only names and descriptions appear in the system prompt.
Full instructions are loaded on demand via `read_file`.

---

## Memory & persistence

### Long-term memory

Two levels, both persisted to disk via `FilesystemBackend`:

| Level | Path | Scope |
|-------|------|-------|
| Global | `/AGENTS.md` | System-wide knowledge shared across all users |
| Personal | `/users/{user_id}/AGENTS.md` | Per-user preferences and domain knowledge |

Global memory is loaded into the system prompt at startup. Personal memory is injected
at conversation start via `<user_memory>` tags. The agent updates memory with
`read_file()` + `edit_file()` — never `write_file()`.

### Conversation history

Checkpoints are managed by LangGraph Platform (configured in `langgraph.json`):
- **SQLite** (default): `agent_data/checkpoints.sqlite` — zero config
- **Postgres**: set `POSTGRES_URI` — survives restarts; shared across workers

For the production backend (`main.py`), PostgreSQL checkpointing is required and
configured automatically from `DATABASE_URL`.

### Context management

Automatic summarization via `SummarizationMiddleware` (built into `create_deep_agent`).
Manual compaction via the `compact_conversation` tool — use when switching topics or
when context grows too long.

---

## Batch upload

See [README_BATCH_UPLOAD.md](./README_BATCH_UPLOAD.md) for the full batch-upload API
documentation. Highlights:

- Recursive file search (files can be nested in subdirectories)
- Per-file `meta_fields`, `chunk_method`, and target KB configuration
- Duplicate detection (skip files already in the dataset)
- Upload retry with exponential backoff
- Parse polling with auto-retry on failure
- Global concurrency control + inter-file cooldown
- MinIO presigned download URLs in responses

---

## Conversation management CLI

`manage_history.py` provides a command-line interface for inspecting and managing
conversation threads:

```bash
uv run manage_history.py                    # list all threads
uv run manage_history.py --show <id>        # show messages in a thread
uv run manage_history.py --delete <id>      # delete a single thread
uv run manage_history.py --delete-all       # delete ALL threads (confirms first)
uv run manage_history.py --port 8122        # connect to a different port
```

Requires the LangGraph dev server to be running.

---

## Customisation

### Change retrieval parameters

In `agent.py` or directly in your prompt, adjust:
- `batch_size` — chunks fetched per `ragflow_retrieve` call (default 32)
- `top_k` — chunks shown to the LLM per call (default 6)
- `similarity_threshold` — minimum score to include (default 0.2)
- `vector_similarity_weight` — vector vs. keyword blend (default 0.3)

### Use a different LLM

Any OpenAI-compatible endpoint works:

```python
# Anthropic
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.0)

# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
```

Pass the model to `create_deep_agent(model=model, ...)`.

### Custom prompts

| Prompt | Location | Purpose |
|--------|----------|---------|
| `DEEP_RAG_WORKFLOW_INSTRUCTIONS` | `rag_agent/prompts.py` | 6-step intent-routing workflow + hard budget |
| `DEEP_RAG_ANSWER_FORMAT` | `rag_agent/prompts.py` | Answer structure, citation style, language rules |

### Add a new skill

Create a `skills/<name>/SKILL.md` file with YAML frontmatter and instructions.
The skill is automatically loaded from `./skills/` and available to the agent.
