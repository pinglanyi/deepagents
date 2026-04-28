"""Centralised application settings loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Database ─────────────────────────────────────────────────────────────
    # asyncpg URL for SQLAlchemy ORM (users, threads, tokens).
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/deeprag"
    )

    # ── Auth / JWT ────────────────────────────────────────────────────────────
    secret_key: str = "CHANGE_ME_IN_PRODUCTION_use_openssl_rand_hex_32"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30

    # ── Agent / LLM ───────────────────────────────────────────────────────────
    agent_data_dir: Path = Path("./agent_data")
    deep_rag_model: str = "deepseek-chat"
    deep_rag_api_key: str = ""
    deep_rag_base_url: str = ""
    # LangGraph recursion limit per agent turn. Each graph node (LLM call,
    # tool call, middleware step) counts as one step. 50 is enough for the
    # deepagents middleware stack + up to ~8 tool calls per turn.
    agent_recursion_limit: int = 50

    # ── RAGFlow ───────────────────────────────────────────────────────────────
    ragflow_base_url: str = "http://localhost:9380"
    ragflow_api_key: str = ""

    # Knowledge base name substrings — used to identify each KB type by dataset name.
    # A dataset whose name contains the substring is treated as that KB type.
    product_kb_name: str = "产品库"     # product Q&A KB (has meta_fields: model/series)
    image_kb_name: str = "图片库"       # image library (image URL, description)
    file_kb_name: str = "文件库"         # document/file library (file URL, description)
    video_kb_name: str = "视频库"       # video library (video URL, description)
    general_kb_name: str = "通用库"     # general/industry standards KB
    program_kb_name: str = "程序库"     # program/function parameters KB
    experience_kb_name: str = "经验库"   # troubleshooting/case history KB

    # Path to a JSON file that maps full model numbers to known aliases/abbreviations.
    # Format: {"aliases": {"FULL_MODEL": ["alias1", "alias2"], ...}}
    # Leave empty to skip model completion.
    model_aliases_file: str = ""

    # ── Batch upload throttling ────────────────────────────────────────────────
    # Max concurrent batch-upload API requests. Additional requests will wait in
    # a FIFO queue until a slot frees up. Set to 1 to serialise all batch uploads.
    batch_upload_max_concurrent: int = 1

    # Seconds to wait between processing consecutive files within one batch.
    # Gives RAGFlow breathing room to start parsing before the next upload hits.
    batch_upload_cooldown_seconds: float = 2.0

    # ── MinIO / S3 (RAGFlow built-in MinIO) ──────────────────────────────────
    # These credentials are used to generate presigned download URLs for files
    # stored in RAGFlow's MinIO backend. Leave access_key empty to fall back
    # to the RAGFlow preview URL (backward-compatible behaviour).
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_secure: bool = False
    minio_presigned_expiry_seconds: int = 3600  # 1 hour

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["*"]

    @property
    def langgraph_db_url(self) -> str:
        """Plain psycopg URL for the LangGraph PostgreSQL checkpointer."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://")


settings = Settings()
