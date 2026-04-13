"""Conversation history manager for the Deep RAG agent.

Lists, inspects, and deletes LangGraph threads via the REST API.
Requires the langgraph dev server to be running.

Usage:
    uv run manage_history.py                      # list all threads
    uv run manage_history.py --show <thread_id>   # show messages in a thread
    uv run manage_history.py --delete <thread_id> # delete one thread
    uv run manage_history.py --delete-all         # delete ALL threads (asks for confirmation)
    uv run manage_history.py --port 8122          # connect to a different port (default: 8122)
"""

from __future__ import annotations

import argparse
import sys

import httpx


def _client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=10)


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


def list_threads(base_url: str) -> None:
    """Print all threads with creation time and last-message preview."""
    with _client(base_url) as c:
        resp = c.post("/threads/search", json={"limit": 100})
        resp.raise_for_status()
        threads = resp.json()

    if not threads:
        print("No conversation threads found.")
        return

    print(f"{'THREAD ID':<38}  {'UPDATED':<20}  LAST MESSAGE")
    print("-" * 100)
    for t in threads:
        tid = t.get("thread_id", "?")
        updated = (t.get("updated_at") or "")[:19].replace("T", " ")
        # Pull last human message from values if present
        values = t.get("values") or {}
        messages = values.get("messages") or []
        last_msg = ""
        for m in reversed(messages):
            if isinstance(m, dict):
                role = m.get("type") or m.get("role") or ""
                content = m.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )
                if role in ("human", "user") and content:
                    last_msg = content[:60].replace("\n", " ")
                    break
        print(f"{tid:<38}  {updated:<20}  {last_msg}")

    print(f"\nTotal: {len(threads)} thread(s)")


# ---------------------------------------------------------------------------
# Show
# ---------------------------------------------------------------------------


def show_thread(base_url: str, thread_id: str) -> None:
    """Print the full message history of a single thread."""
    with _client(base_url) as c:
        resp = c.get(f"/threads/{thread_id}/history")
        resp.raise_for_status()
        history = resp.json()

    if not history:
        print(f"Thread {thread_id}: no history found.")
        return

    # history is a list of state snapshots, newest-first; we want oldest-first
    snapshots = list(reversed(history))
    printed: set[str] = set()

    for snap in snapshots:
        values = snap.get("values") or {}
        messages = values.get("messages") or []
        for m in messages:
            if not isinstance(m, dict):
                continue
            msg_id = m.get("id") or ""
            if msg_id in printed:
                continue
            printed.add(msg_id)

            role = m.get("type") or m.get("role") or "unknown"
            content = m.get("content") or ""
            if isinstance(content, list):
                content = "\n".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )

            role_label = {"human": "You", "user": "You", "ai": "Agent", "assistant": "Agent"}.get(role, role)
            print(f"\n[{role_label}]")
            print(content)

    print(f"\n— end of thread {thread_id} —")


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def delete_thread(base_url: str, thread_id: str) -> None:
    """Delete a single thread and all its checkpoints."""
    with _client(base_url) as c:
        resp = c.delete(f"/threads/{thread_id}")
        if resp.status_code == 404:
            print(f"Thread {thread_id} not found.")
            return
        resp.raise_for_status()
    print(f"Deleted thread {thread_id}.")


def delete_all_threads(base_url: str) -> None:
    """Delete every thread after a confirmation prompt."""
    with _client(base_url) as c:
        resp = c.post("/threads/search", json={"limit": 1000})
        resp.raise_for_status()
        threads = resp.json()

    if not threads:
        print("No threads to delete.")
        return

    answer = input(f"Delete ALL {len(threads)} thread(s)? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    with _client(base_url) as c:
        for t in threads:
            tid = t.get("thread_id", "")
            c.delete(f"/threads/{tid}")
            print(f"  deleted {tid}")

    print(f"\nDeleted {len(threads)} thread(s).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(description="Manage Deep RAG conversation history")
    parser.add_argument("--port", type=int, default=8122, help="langgraph dev server port (default: 8122)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--show", metavar="THREAD_ID", help="show messages in a thread")
    group.add_argument("--delete", metavar="THREAD_ID", help="delete a specific thread")
    group.add_argument("--delete-all", action="store_true", help="delete ALL threads")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"

    try:
        if args.show:
            show_thread(base_url, args.show)
        elif args.delete:
            delete_thread(base_url, args.delete)
        elif args.delete_all:
            delete_all_threads(base_url)
        else:
            list_threads(base_url)
    except httpx.ConnectError:
        print(f"Cannot connect to {base_url}. Is `uv run langgraph dev --port {args.port}` running?", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"API error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
