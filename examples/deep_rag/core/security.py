"""Password hashing and JWT utilities."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import jwt

from core.config import settings


# ── Passwords ─────────────────────────────────────────────────────────────────


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── JWT access tokens ─────────────────────────────────────────────────────────


def create_access_token(subject: str, extra: dict[str, Any] | None = None) -> str:
    """Create a signed JWT with ``sub=subject`` and an expiry claim."""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.access_token_expire_minutes)
    payload: dict[str, Any] = {"sub": subject, "iat": now, "exp": expire}
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT.  Raises ``jose.JWTError`` on failure."""
    return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])


# ── Opaque refresh tokens ─────────────────────────────────────────────────────


def generate_refresh_token() -> str:
    """Return a cryptographically secure random token (URL-safe base64)."""
    return secrets.token_urlsafe(48)
