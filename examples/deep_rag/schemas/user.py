"""Pydantic schemas for user-related request/response bodies."""

from __future__ import annotations

import re
import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator


class UserRegister(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def username_valid(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Username must contain only letters, digits, underscores, or hyphens"
            )
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    username: str
    is_active: bool
    is_admin: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UserUpdate(BaseModel):
    """Fields a user can change on their own profile."""

    username: str | None = Field(None, min_length=3, max_length=50)
    email: EmailStr | None = None
    current_password: str | None = Field(
        None, description="Required when changing password"
    )
    new_password: str | None = Field(None, min_length=8, max_length=128)


class AdminUserCreate(BaseModel):
    """Admin-only: create a user account directly (bypasses public registration)."""

    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)
    is_active: bool = True
    is_admin: bool = False

    @field_validator("username")
    @classmethod
    def username_valid(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Username must contain only letters, digits, underscores, or hyphens"
            )
        return v


class AdminUserUpdate(BaseModel):
    """Admin-only fields (superset of UserUpdate, no current_password check)."""

    username: str | None = Field(None, min_length=3, max_length=50)
    email: EmailStr | None = None
    password: str | None = Field(None, min_length=8, max_length=128)
    is_active: bool | None = None
    is_admin: bool | None = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Access token lifetime in seconds")


class RefreshRequest(BaseModel):
    refresh_token: str


class UserStats(BaseModel):
    """Admin view — aggregate stats for a user."""

    id: uuid.UUID
    username: str
    email: str
    is_active: bool
    is_admin: bool
    thread_count: int
    created_at: datetime
