"""Authentication router — register, login, token refresh, logout."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.deps import get_current_user
from core.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    verify_password,
)
from core.config import settings
from models.user import RefreshToken, User
from schemas.user import RefreshRequest, TokenResponse, UserLogin, UserOut, UserRegister

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(req: UserRegister, db: AsyncSession = Depends(get_db)) -> User:
    result = await db.execute(select(User).where(User.email == req.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    result = await db.execute(select(User).where(User.username == req.username))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already taken")

    user = User(
        email=req.email,
        username=req.username,
        hashed_password=hash_password(req.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="OAuth2-compatible login (Swagger UI 'Authorize' button)",
    description=(
        "Accepts `application/x-www-form-urlencoded` with `username` (= email) and `password`. "
        "This endpoint exists for Swagger UI / OAuth2 client compatibility. "
        "For regular API usage prefer `POST /auth/login` with a JSON body."
    ),
)
async def oauth2_token(
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    # OAuth2 spec uses 'username'; we treat it as the email field
    fake_req = UserLogin(email=form.username, password=form.password)
    return await login(fake_req, db)


@router.post("/login", response_model=TokenResponse, summary="Login and receive tokens")
async def login(req: UserLogin, db: AsyncSession = Depends(get_db)) -> TokenResponse:
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    access_token = create_access_token(str(user.id))
    refresh_token_str = generate_refresh_token()
    expires_at = datetime.now(timezone.utc) + timedelta(
        days=settings.refresh_token_expire_days
    )

    db.add(
        RefreshToken(
            user_id=user.id,
            token=refresh_token_str,
            expires_at=expires_at,
        )
    )
    await db.commit()

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token_str,
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Exchange a refresh token for a new access token (rotation)",
)
async def refresh_token(
    req: RefreshRequest, db: AsyncSession = Depends(get_db)
) -> TokenResponse:
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.token == req.refresh_token,
            RefreshToken.is_revoked.is_(False),
        )
    )
    token_obj = result.scalar_one_or_none()

    if not token_obj or token_obj.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    # Rotate: revoke old, issue new
    token_obj.is_revoked = True

    new_refresh = generate_refresh_token()
    new_expires = datetime.now(timezone.utc) + timedelta(
        days=settings.refresh_token_expire_days
    )
    db.add(
        RefreshToken(
            user_id=token_obj.user_id,
            token=new_refresh,
            expires_at=new_expires,
        )
    )

    access_token = create_access_token(str(token_obj.user_id))
    await db.commit()

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh,
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke the supplied refresh token (logout current device)",
)
async def logout(
    req: RefreshRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.token == req.refresh_token,
            RefreshToken.user_id == current_user.id,
        )
    )
    token_obj = result.scalar_one_or_none()
    if token_obj:
        token_obj.is_revoked = True
        await db.commit()


@router.post(
    "/logout/all",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke ALL refresh tokens for this user (sign out from all devices)",
)
async def logout_all(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.user_id == current_user.id,
            RefreshToken.is_revoked.is_(False),
        )
    )
    for tok in result.scalars().all():
        tok.is_revoked = True
    await db.commit()
