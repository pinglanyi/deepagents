#!/usr/bin/env bash
set -euo pipefail

# ── Wait for PostgreSQL ─────────────────────────────────────────────────────────
# Supports both DATABASE_URL (asyncpg) and POSTGRES_URI (psycopg).
# Extracts host:port from the URL and polls with pg_isready / nc.

DB_URL="${DATABASE_URL:-${POSTGRES_URI:-}}"

if [ -n "$DB_URL" ]; then
    # Strip the scheme and extract host:port
    # DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname
    WITHOUT_SCHEME="${DB_URL#*://}"          # user:pass@host:5432/dbname
    HOST_PORT="${WITHOUT_SCHEME#*@}"         # host:5432/dbname
    HOST_PORT="${HOST_PORT%%/*}"             # host:5432
    HOST="${HOST_PORT%:*}"                   # host
    PORT="${HOST_PORT##*:}"                  # 5432

    echo "Waiting for PostgreSQL at $HOST:$PORT ..."
    for i in $(seq 1 30); do
        if nc -z "$HOST" "$PORT" 2>/dev/null; then
            echo "PostgreSQL is ready."
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo "ERROR: PostgreSQL not reachable after 30 s — exiting."
            exit 1
        fi
        sleep 1
    done
fi

exec "$@"
