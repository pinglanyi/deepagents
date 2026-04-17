"""Multi-user Deep RAG backend.

Production-ready FastAPI application with:
- JWT authentication (access + refresh tokens with rotation)
- Per-user conversation isolation via PostgreSQL
- LangGraph agent with PostgreSQL checkpointer
- RAGFlow knowledge-base management (admin-only write, auth-required read)

Start:
    uv run uvicorn main:app --host 0.0.0.0 --port 8123
    uv run uvicorn main:app --host 0.0.0.0 --port 8123 --workers 4

With gunicorn:
    uv run gunicorn main:app -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8123

API overview:
    Authentication
        POST /auth/register          Register a new account
        POST /auth/login             Login → access + refresh tokens
        POST /auth/refresh           Rotate refresh token → new access token
        POST /auth/logout            Revoke refresh token (current device)
        POST /auth/logout/all        Revoke all refresh tokens (all devices)

    User profile (authenticated)
        GET    /users/me             Get own profile
        PUT    /users/me             Update own profile / change password
        DELETE /users/me             Delete own account

    User management (admin only)
        GET    /users                List all users (paginated)
        GET    /users/stats          List users with thread-count stats
        GET    /users/{id}           Get user by ID
        PUT    /users/{id}           Update any user
        DELETE /users/{id}           Delete user

    Chat (authenticated)
        POST /chat                   Non-streaming chat
        POST /chat/stream            Streaming chat (SSE)

    Threads (authenticated — isolated per user)
        GET    /threads              List own threads
        GET    /threads/{id}         Full message history
        PATCH  /threads/{id}         Rename thread
        DELETE /threads/{id}         Delete thread
        DELETE /threads              Delete all own threads
        GET    /threads/{id}/export  Export as plain text

    RAGFlow knowledge-base (read: auth | write: admin)
        POST   /ragflow/datasets                              Create dataset
        GET    /ragflow/datasets                              List datasets
        PUT    /ragflow/datasets/{id}                         Update dataset
        DELETE /ragflow/datasets                              Delete datasets
        POST   /ragflow/datasets/{id}/documents/upload        Upload single doc
        POST   /ragflow/datasets/{id}/documents/upload/batch  Batch upload
        GET    /ragflow/datasets/{id}/documents               List documents
        PUT    /ragflow/datasets/{id}/documents/{doc_id}      Update document
        DELETE /ragflow/datasets/{id}/documents               Delete documents
        POST   /ragflow/datasets/{id}/documents/parse         Start parsing
        DELETE /ragflow/datasets/{id}/documents/parse         Cancel parsing

    Health
        GET /health

Environment variables (see .env.example):
    DATABASE_URL         asyncpg PostgreSQL URL (required)
    SECRET_KEY           JWT signing secret (required in production)
    DEEP_RAG_MODEL       LLM model name (default: deepseek-chat)
    DEEP_RAG_API_KEY     LLM API key
    DEEP_RAG_BASE_URL    LLM base URL
    RAGFLOW_BASE_URL     RAGFlow server URL (default: http://localhost:9380)
    RAGFLOW_API_KEY      RAGFlow API key
    AGENT_DATA_DIR       Local storage for agent files (default: ./agent_data)
    CORS_ORIGINS         JSON list of allowed origins (default: ["*"])
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from core.config import settings
from core.database import engine

# Arbitrary fixed integer — used as the PostgreSQL session-level advisory lock key
# to serialise schema creation across multiple workers on startup.
_SCHEMA_LOCK_ID = 0x44455052_41474442  # "DEPRAGDB" in hex

# Import all models so SQLAlchemy registers them before create_all
import models  # noqa: F401 — side-effect import

from routers import auth, chat, ragflow, threads, users
from services.agent import init_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Server startup: create DB tables and initialise the agent."""
    # Create all tables — guarded by a PostgreSQL advisory lock so that
    # concurrent workers (uvicorn --workers N / gunicorn) don't race on
    # CREATE TABLE and crash with a pg_type UniqueViolationError.
    # The lock is session-scoped: released automatically when the
    # connection is returned to the pool at the end of this block.
    from models.base import Base  # noqa: PLC0415

    async with engine.begin() as conn:
        await conn.execute(
            text("SELECT pg_advisory_lock(:lock_id)"),
            {"lock_id": _SCHEMA_LOCK_ID},
        )
        try:
            await conn.run_sync(Base.metadata.create_all)
        finally:
            await conn.execute(
                text("SELECT pg_advisory_unlock(:lock_id)"),
                {"lock_id": _SCHEMA_LOCK_ID},
            )

    # Initialise LangGraph agent with a shared PostgreSQL checkpointer
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: PLC0415

    async with AsyncPostgresSaver.from_conn_string(settings.langgraph_db_url) as cp:
        await cp.setup()
        init_agent(cp)
        yield

    await engine.dispose()


app = FastAPI(
    title="Deep RAG — Multi-user Backend",
    version="1.0.0",
    description=(
        "Production multi-user RAG chat backend.\n\n"
        "Each user has isolated conversation history. "
        "Admins manage the shared RAGFlow knowledge base."
    ),
    lifespan=lifespan,
)

# CORS — restrict origins in production via the CORS_ORIGINS env var
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(chat.router)
app.include_router(threads.router)
app.include_router(ragflow.router)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"], summary="Liveness probe")
async def health() -> dict:
    return {"status": "ok", "version": app.version}
