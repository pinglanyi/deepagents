FROM python:3.12-slim

# ── uv ──────────────────────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ── System deps ─────────────────────────────────────────────────────────────────
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        curl \
        libpq-dev \
        gcc \
        netcat-openbsd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── App ─────────────────────────────────────────────────────────────────────────
WORKDIR /app

# Install deps first (layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source
COPY . .
RUN uv sync --frozen --no-dev

# ── Persistent data volume ──────────────────────────────────────────────────────
# AGENT_DATA_DIR — long-term memory, skills, checkpoint SQLite (if used).
# Mount a named volume or host path at /app/agent_data in production.
RUN mkdir -p /app/agent_data
VOLUME /app/agent_data

# ── Runtime ─────────────────────────────────────────────────────────────────────
EXPOSE 8123

ENV AGENT_DATA_DIR=/app/agent_data

# Wait for PostgreSQL, then start uvicorn.
# The entrypoint polls the DB until it's reachable (max 30 s).
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8123"]
