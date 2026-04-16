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

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["*"]

    @property
    def langgraph_db_url(self) -> str:
        """Plain psycopg URL for the LangGraph PostgreSQL checkpointer."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://")


settings = Settings()
