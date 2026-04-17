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

    # ── RAGFlow ───────────────────────────────────────────────────────────────
    ragflow_base_url: str = "http://localhost:9380"
    ragflow_api_key: str = ""

    # Knowledge base name substrings — used to identify each KB type by dataset name.
    # A dataset whose name contains the substring is treated as that KB type.
    product_kb_name: str = "product"   # product Q&A KB (has meta_fields: model/series)
    image_kb_name: str = "image"       # image library (image URL, description)
    file_kb_name: str = "file"         # document/file library (file URL, description)
    video_kb_name: str = "video"       # video library (video URL, description)

    # Path to a JSON file that maps full model numbers to known aliases/abbreviations.
    # Format: {"aliases": {"FULL_MODEL": ["alias1", "alias2"], ...}}
    # Leave empty to skip model completion.
    model_aliases_file: str = ""

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["*"]

    @property
    def langgraph_db_url(self) -> str:
        """Plain psycopg URL for the LangGraph PostgreSQL checkpointer."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://")


settings = Settings()
