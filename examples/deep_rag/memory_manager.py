"""Per-user memory management for the Deep RAG agent.

Provides isolated AGENTS.md memory per user account.

Architecture:
- Global ``/AGENTS.md`` — system-wide knowledge loaded by MemoryMiddleware at startup
- Per-user ``/users/{user_id}/AGENTS.md`` — user-specific memory injected at request time

The agent is a singleton built at startup, so per-user memory cannot rely on
MemoryMiddleware's cached loading. Instead, this module provides helpers that
the chat endpoints call to inject user-specific memory into each request.

Usage::

    from memory_manager import MemoryManager

    mm = MemoryManager(agent_data_dir)

    # Get user memory content
    content = mm.get_user_memory(user_id)  # → str or None

    # Update user memory (typically done by the agent via edit_file)
    mm.ensure_user_memory(user_id)

    # Build a memory-augmented message list for agent invocation
    messages = mm.augment_messages(user_id, [{"role": "user", "content": "..."}])
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_USER_MEMORY_TEMPLATE = """# {user_label} — Agent Memory

This file stores what the agent learns about you across conversations.
It persists across threads and is loaded automatically.

## User Preferences

(none recorded yet)

## Learned Context

(none recorded yet)

## Frequently Asked Topics

(none recorded yet)

---

Last updated: {timestamp}
"""


class MemoryManager:
    """Manages per-user memory files on the filesystem.

    Each user gets a dedicated ``AGENTS.md`` under
    ``{agent_data_dir}/users/{user_id}/AGENTS.md``.

    Args:
        agent_data_dir: Root directory for agent data files.
    """

    def __init__(self, agent_data_dir: Path) -> None:
        self._root = agent_data_dir.resolve()
        self._users_dir = self._root / "users"
        self._global_memory_path = "/AGENTS.md"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_user_memory(self, user_id: str, user_label: str = "") -> Path:
        """Create per-user AGENTS.md if it does not exist.

        Args:
            user_id: User UUID string.
            user_label: Optional display label (e.g. email or username).

        Returns:
            Path to the user's AGENTS.md file.
        """
        user_dir = self._users_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        memory_file = user_dir / "AGENTS.md"

        if not memory_file.exists():
            label = user_label or user_id[:8]
            memory_file.write_text(
                _USER_MEMORY_TEMPLATE.format(
                    user_label=label,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                encoding="utf-8",
            )
            logger.info("Created memory file for user %s", user_id)

        return memory_file

    def get_user_memory(self, user_id: str) -> str | None:
        """Read a user's memory content.

        Args:
            user_id: User UUID string.

        Returns:
            Memory content as a string, or ``None`` if the file does not exist.
        """
        memory_file = self._users_dir / user_id / "AGENTS.md"
        if not memory_file.exists():
            return None
        return memory_file.read_text(encoding="utf-8")

    def get_user_memory_path(self, user_id: str) -> str:
        """Return the virtual-filesystem path for a user's memory.

        This is the path the agent uses with ``read_file`` / ``edit_file``
        via the FilesystemBackend.

        Args:
            user_id: User UUID string.

        Returns:
            Virtual path like ``/users/{user_id}/AGENTS.md``.
        """
        return f"/users/{user_id}/AGENTS.md"

    def get_global_memory_path(self) -> str:
        """Return the virtual path to the global AGENTS.md."""
        return self._global_memory_path

    def build_memory_context(self, user_id: str) -> str:
        """Build a memory context string for injection into system/user messages.

        Reads both global and per-user memory and combines them.

        Args:
            user_id: User UUID string.

        Returns:
            Formatted memory context ready for injection.
        """
        parts: list[str] = []

        # Global memory
        global_md = self._root / "AGENTS.md"
        if global_md.exists():
            global_content = global_md.read_text(encoding="utf-8")
            parts.append(f"<global_memory>\n{global_content}\n</global_memory>")

        # Per-user memory
        user_content = self.get_user_memory(user_id)
        if user_content:
            parts.append(f"<user_memory>\n{user_content}\n</user_memory>")
        else:
            parts.append("<user_memory>\n(No personal memory yet. Use edit_file to save.)\n</user_memory>")

        return "\n\n".join(parts)

    def augment_messages(
        self,
        user_id: str,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Prepend memory context to the first user message.

        Args:
            user_id: User UUID string.
            messages: Original message list.

        Returns:
            Augmented message list with memory context injected.
        """
        self.ensure_user_memory(user_id)
        memory_ctx = self.build_memory_context(user_id)

        augmented = list(messages)
        if augmented:
            first = augmented[0]
            original_content = first.get("content", "")
            first["content"] = f"{memory_ctx}\n\n---\n\n{original_content}"

        return augmented

    def list_user_memories(self) -> list[dict[str, Any]]:
        """List all users who have memory files.

        Returns:
            List of dicts with ``user_id``, ``has_memory``, ``size_bytes``, and ``updated_at``.
        """
        results: list[dict[str, Any]] = []
        if not self._users_dir.exists():
            return results

        for user_dir in sorted(self._users_dir.iterdir()):
            if not user_dir.is_dir():
                continue
            memory_file = user_dir / "AGENTS.md"
            if memory_file.exists():
                stat = memory_file.stat()
                # Try to extract last-updated from the file content
                updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                results.append({
                    "user_id": user_dir.name,
                    "has_memory": True,
                    "size_bytes": stat.st_size,
                    "updated_at": updated_at,
                })

        return results
