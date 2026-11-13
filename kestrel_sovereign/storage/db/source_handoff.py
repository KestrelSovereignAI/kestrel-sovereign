"""The per-agent, per-source handoff lock shared by the durable ledger (#4041).

Durable admission, claim, acknowledgement and settlement of one source scope
serialize on this lock, and an A2A task write that arms a wake obligation
takes the same one first. Every path that touches both an obligation and a
task row therefore holds it before either row, so no two of them can wait on
each other's row locks.
"""

from __future__ import annotations

from typing import Any


def handoff_lock_key(agent_id: str, source: str) -> str:
    """The advisory-lock key of one ``(agent_id, source)`` scope."""
    return f"durable-signal:{agent_id}:{source}"


async def lock_source_handoff(
    backend: Any, *, agent_id: str, source: str, sqlite_reserve_table: str
) -> None:
    """Take the scope's handoff lock in the caller's open transaction.

    PostgreSQL holds a transaction-scoped advisory lock, released on commit or
    rollback. SQLite has one writer: a no-op write on ``sqlite_reserve_table``
    reserves it before the caller reads anything it will then write.
    """
    if backend.backend_type == "postgres":
        await backend.fetch_val(
            "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
            (handoff_lock_key(agent_id, source),),
        )
        return
    if backend.backend_type == "sqlite":
        await backend.execute(f"DELETE FROM {sqlite_reserve_table} WHERE 0")
        return
    raise RuntimeError(
        "Durable signal handoff serialization does not support backend "
        f"{backend.backend_type!r}"
    )
