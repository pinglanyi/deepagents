"""User management router — self-service profile and admin CRUD."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.deps import get_current_admin, get_current_user
from core.security import hash_password, verify_password
from models.thread import Thread
from models.user import User
from schemas.user import AdminUserCreate, AdminUserUpdate, UserOut, UserStats, UserUpdate

router = APIRouter(prefix="/users", tags=["Users"])


# ── Self-service endpoints (any authenticated user) ───────────────────────────


@router.get("/me", response_model=UserOut, summary="Get current user profile")
async def get_me(current_user: User = Depends(get_current_user)) -> User:
    return current_user


@router.put("/me", response_model=UserOut, summary="Update current user profile")
async def update_me(
    req: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> User:
    if req.email and req.email != current_user.email:
        dup = await db.execute(select(User).where(User.email == req.email))
        if dup.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Email already in use")
        current_user.email = req.email

    if req.username and req.username != current_user.username:
        dup = await db.execute(select(User).where(User.username == req.username))
        if dup.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Username already taken")
        current_user.username = req.username

    if req.new_password:
        if not req.current_password or not verify_password(
            req.current_password, current_user.hashed_password
        ):
            raise HTTPException(
                status_code=400, detail="current_password is incorrect"
            )
        current_user.hashed_password = hash_password(req.new_password)

    await db.commit()
    await db.refresh(current_user)
    return current_user


@router.delete(
    "/me",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete own account (irreversible)",
)
async def delete_me(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    await db.delete(current_user)
    await db.commit()


# ── Admin-only endpoints ──────────────────────────────────────────────────────


@router.post(
    "",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="[Admin] Create a user account directly",
)
async def admin_create_user(
    req: AdminUserCreate,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_admin),
) -> User:
    """Create a user without going through the public /auth/register flow.

    Useful for provisioning accounts in bulk or creating admin accounts
    without exposing them to self-registration.
    """
    dup_email = await db.execute(select(User).where(User.email == req.email))
    if dup_email.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already in use")

    dup_username = await db.execute(select(User).where(User.username == req.username))
    if dup_username.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already taken")

    user = User(
        email=req.email,
        username=req.username,
        hashed_password=hash_password(req.password),
        is_active=req.is_active,
        is_admin=req.is_admin,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.get(
    "",
    response_model=list[UserOut],
    summary="[Admin] List all users",
)
async def list_users(
    page: int = 1,
    page_size: int = 50,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_admin),
) -> list[User]:
    result = await db.execute(
        select(User)
        .order_by(User.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(result.scalars().all())


@router.get(
    "/stats",
    response_model=list[UserStats],
    summary="[Admin] List users with aggregate stats",
)
async def list_users_stats(
    page: int = 1,
    page_size: int = 50,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_admin),
) -> list[UserStats]:
    # Count threads per user in one query
    thread_counts = await db.execute(
        select(Thread.user_id, func.count(Thread.id).label("cnt"))
        .group_by(Thread.user_id)
        .subquery()
    )
    # Simple approach: fetch users then look up count
    result = await db.execute(
        select(User)
        .order_by(User.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    users = result.scalars().all()

    # Thread count per user (separate query per user is acceptable for admin views)
    stats = []
    for u in users:
        cnt_result = await db.execute(
            select(func.count(Thread.id)).where(Thread.user_id == u.id)
        )
        cnt = cnt_result.scalar_one() or 0
        stats.append(
            UserStats(
                id=u.id,
                username=u.username,
                email=u.email,
                is_active=u.is_active,
                is_admin=u.is_admin,
                thread_count=cnt,
                created_at=u.created_at,
            )
        )
    return stats


@router.get(
    "/{user_id}",
    response_model=UserOut,
    summary="[Admin] Get user by ID",
)
async def get_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_admin),
) -> User:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.put(
    "/{user_id}",
    response_model=UserOut,
    summary="[Admin] Update any user's fields",
)
async def admin_update_user(
    user_id: uuid.UUID,
    req: AdminUserUpdate,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_admin),
) -> User:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if req.email is not None:
        user.email = req.email
    if req.username is not None:
        user.username = req.username
    if req.password is not None:
        user.hashed_password = hash_password(req.password)
    if req.is_active is not None:
        user.is_active = req.is_active
    if req.is_admin is not None:
        user.is_admin = req.is_admin

    await db.commit()
    await db.refresh(user)
    return user


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="[Admin] Delete a user account",
)
async def admin_delete_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin),
) -> None:
    if user_id == admin.id:
        raise HTTPException(
            status_code=400, detail="Cannot delete your own admin account"
        )
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    await db.delete(user)
    await db.commit()
