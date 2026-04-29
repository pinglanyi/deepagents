"""Management API — skills, memory, and context management endpoints.

Endpoints:
    GET  /skills              List all available skills
    GET  /skills/{name}       Get full SKILL.md content for a skill
    GET  /memory              Get current user's memory
    PUT  /memory              Update current user's memory
    POST /context/compact     Trigger manual conversation compaction
    GET  /context/stats       Get context/compaction statistics for a thread
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.database import get_db
from core.deps import get_current_admin, get_current_user
from core.context_manager import get_context_tracker
from core.memory_manager import MemoryManager
from models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Management"])

_memory_manager = MemoryManager(settings.agent_data_dir)
_skills_dir = (Path(__file__).parent.parent / "skills").resolve()


# ── Schemas ────────────────────────────────────────────────────────────────────


class SkillInfo(BaseModel):
    """Summary view of a skill (name + description only)."""

    name: str
    description: str
    path: str
    license: str | None = None


class SkillDetail(BaseModel):
    """Full skill information including the SKILL.md body."""

    name: str
    description: str
    path: str
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    body: str = ""


class MemoryUpdate(BaseModel):
    """Request body for updating user memory."""

    content: str = Field(..., min_length=1, max_length=100_000)


class MemoryResponse(BaseModel):
    """User memory content."""

    user_id: str
    path: str
    content: str
    exists: bool


class CompactRequest(BaseModel):
    """Request body for manual compaction."""

    thread_id: str = Field(..., description="LangGraph thread ID to compact")


class CompactResponse(BaseModel):
    """Result of a compaction request."""

    thread_id: str
    compacted: bool
    message: str


class ContextStatsResponse(BaseModel):
    """Context statistics for a thread."""

    thread_id: str
    total_compactions: int
    total_events: int
    recent_events: list[dict[str, Any]]


# ── Skills endpoints ──────────────────────────────────────────────────────────


def _list_skills() -> list[SkillInfo]:
    """Scan the skills directory for SKILL.md files and parse frontmatter."""
    if not _skills_dir.exists():
        return []

    skills: list[SkillInfo] = []
    for skill_dir in sorted(_skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue

        try:
            metadata = _parse_skill_frontmatter(skill_md.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to parse SKILL.md for %s", skill_dir.name)
            continue

        skills.append(SkillInfo(
            name=metadata.get("name", skill_dir.name),
            description=metadata.get("description", "(no description)"),
            path=str(skill_md),
            license=metadata.get("license"),
        ))

    return skills


def _parse_skill_frontmatter(content: str) -> dict[str, Any]:
    """Parse YAML frontmatter from a SKILL.md file.

    Args:
        content: Raw SKILL.md content.

    Returns:
        Dict of frontmatter fields.
    """
    if not content.startswith("---"):
        return {}

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}

    return yaml.safe_load(parts[1]) or {}


def _split_skill_body(content: str) -> tuple[dict[str, Any], str]:
    """Split SKILL.md into frontmatter dict and markdown body.

    Args:
        content: Raw SKILL.md content.

    Returns:
        Tuple of (frontmatter_dict, body_string).
    """
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    metadata = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()
    return metadata, body


def _parse_allowed_tools(metadata: dict[str, Any]) -> list[str]:
    """Parse allowed-tools from SKILL.md YAML frontmatter.

    The Agent Skills spec uses `allowed-tools` (hyphen) as the YAML key.
    This helper handles both ``allowed-tools`` and ``allowed_tools`` keys.
    """
    raw = metadata.get("allowed-tools") or metadata.get("allowed_tools") or ""
    if isinstance(raw, str):
        return raw.split()
    if isinstance(raw, list):
        return raw
    return []


@router.get(
    "/skills",
    response_model=list[SkillInfo],
    summary="List all available skills",
)
async def list_skills(
    current_user: User = Depends(get_current_user),
) -> list[SkillInfo]:
    """Return all skills discovered in the skills directory.

    Skills follow the Agent Skills specification (agentskills.io).
    Each skill is a directory with a SKILL.md file containing
    YAML frontmatter and markdown instructions.
    """
    return _list_skills()


@router.get(
    "/skills/{name}",
    response_model=SkillDetail,
    summary="Get full details for a skill",
)
async def get_skill(
    name: str,
    current_user: User = Depends(get_current_user),
) -> SkillDetail:
    """Return the complete SKILL.md for a skill, including all frontmatter
    fields and the full markdown body."""
    skill_dir = _skills_dir / name
    skill_md = skill_dir / "SKILL.md"

    if not skill_dir.is_dir() or not skill_md.exists():
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")

    content = skill_md.read_text(encoding="utf-8")
    metadata, body = _split_skill_body(content)

    return SkillDetail(
        name=metadata.get("name", name),
        description=metadata.get("description", "(no description)"),
        path=str(skill_md),
        license=metadata.get("license"),
        compatibility=metadata.get("compatibility"),
        metadata=metadata.get("metadata", {}),
        allowed_tools=_parse_allowed_tools(metadata),
        body=body,
    )


# ── Memory endpoints ──────────────────────────────────────────────────────────


@router.get(
    "/memory",
    response_model=MemoryResponse,
    summary="Get your personal agent memory",
)
async def get_my_memory(
    current_user: User = Depends(get_current_user),
) -> MemoryResponse:
    """Return the current user's AGENTS.md memory content.

    This is the per-user memory that persists across conversations.
    The agent uses this to remember preferences, context, and learnings.
    """
    user_id = str(current_user.id)
    _memory_manager.ensure_user_memory(user_id, current_user.email)

    content = _memory_manager.get_user_memory(user_id)
    path = _memory_manager.get_user_memory_path(user_id)

    return MemoryResponse(
        user_id=user_id,
        path=path,
        content=content or "",
        exists=content is not None,
    )


@router.put(
    "/memory",
    response_model=MemoryResponse,
    summary="Update your personal agent memory",
)
async def update_my_memory(
    body: MemoryUpdate,
    current_user: User = Depends(get_current_user),
) -> MemoryResponse:
    """Replace the current user's AGENTS.md memory content.

    Use this to set preferences, context, or instructions that the
    agent should remember across conversations.
    """
    user_id = str(current_user.id)
    _memory_manager.ensure_user_memory(user_id, current_user.email)

    memory_path = settings.agent_data_dir / "users" / user_id / "AGENTS.md"
    memory_path.write_text(body.content, encoding="utf-8")

    path = _memory_manager.get_user_memory_path(user_id)

    return MemoryResponse(
        user_id=user_id,
        path=path,
        content=body.content,
        exists=True,
    )


@router.get(
    "/admin/memories",
    response_model=list[dict[str, Any]],
    summary="[Admin] List all user memories",
)
async def list_all_memories(
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """Return a list of all users who have memory files (admin only)."""
    memories = _memory_manager.list_user_memories()
    return memories


# ── Context endpoints ─────────────────────────────────────────────────────────


@router.post(
    "/context/compact",
    response_model=CompactResponse,
    summary="Trigger manual conversation compaction",
)
async def compact_context(
    body: CompactRequest,
    current_user: User = Depends(get_current_user),
) -> CompactResponse:
    """Request manual compaction for a conversation thread.

    The agent's compact_conversation tool is invoked on the specified
    thread. This summarizes old messages and offloads full history to
    the backend. Compaction only proceeds if the context is above
    ~50% of the auto-compaction threshold.
    """
    from services.agent import get_agent

    agent = get_agent()
    config = {
        "configurable": {"thread_id": body.thread_id},
        "recursion_limit": settings.agent_recursion_limit,
    }

    try:
        result = await agent.ainvoke(
            {
                "messages": [{
                    "role": "user",
                    "content": (
                        "Please call the compact_conversation tool to compact "
                        "the conversation history. Do NOT respond with text — "
                        "only call the compact_conversation tool."
                    ),
                }],
            },
            config=config,
        )

        # Check if compaction happened by looking for the tool result
        messages = result.get("messages", [])
        compacted = any(
            hasattr(m, "name") and getattr(m, "name", "") == "compact_conversation"
            for m in messages
        )

        return CompactResponse(
            thread_id=body.thread_id,
            compacted=compacted,
            message=(
                "Conversation compacted successfully."
                if compacted
                else "Compaction was not needed (context below threshold)."
            ),
        )
    except Exception as exc:
        logger.exception("Compaction failed for thread %s", body.thread_id)
        return CompactResponse(
            thread_id=body.thread_id,
            compacted=False,
            message=f"Compaction failed: {exc}",
        )


@router.get(
    "/context/stats",
    response_model=ContextStatsResponse,
    summary="Get context statistics for a thread",
)
async def get_context_stats(
    thread_id: str = Query(..., description="LangGraph thread ID"),
    current_user: User = Depends(get_current_user),
) -> ContextStatsResponse:
    """Return context/compaction statistics for a thread.

    This shows how many times the conversation has been compacted
    and recent context events.
    """
    tracker = get_context_tracker()
    stats = tracker.get_stats(thread_id)

    return ContextStatsResponse(
        thread_id=thread_id,
        total_compactions=stats["total_compactions"],
        total_events=stats["total_events"],
        recent_events=stats["events"],
    )
