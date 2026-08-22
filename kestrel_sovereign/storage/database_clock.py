"""Database-authoritative UTC clock helpers for durable shared state."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def database_backend_type(db: Any) -> str:
    """Return a concrete database type without trusting loose doubles."""

    backend_type = getattr(db, "backend_type", "")
    return backend_type.lower() if isinstance(backend_type, str) else ""


def database_now_sql(db: Any) -> str:
    """Return a portable statement-time UTC clock expression."""

    backend_type = database_backend_type(db)
    if backend_type == "postgres":
        return (
            "(to_char(clock_timestamp() AT TIME ZONE 'UTC', "
            "'YYYY-MM-DD\"T\"HH24:MI:SS.US') || '+00:00')"
        )
    if backend_type == "sqlite":
        return "strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')"
    raise RuntimeError("database statement clock is unavailable for this backend")


async def database_clock(db: Any) -> datetime:
    """Read a durable database's UTC wall clock.

    Concrete PostgreSQL and SQLite backends use statement time so replicas
    with skewed process clocks agree about shared timestamps and freshness.
    Deliberately minimal test adapters retain a host-clock fallback because
    they expose no database clock contract.
    """

    backend_type = database_backend_type(db)
    if backend_type == "postgres":
        value = await db.fetchval("SELECT clock_timestamp()")
    elif backend_type == "sqlite":
        value = await db.fetchval(
            "SELECT strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')"
        )
    else:
        return datetime.now(timezone.utc)

    if isinstance(value, datetime):
        return (
            value.replace(tzinfo=timezone.utc)
            if value.tzinfo is None
            else value.astimezone(timezone.utc)
        )
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        parsed = None
    if parsed is None:
        raise RuntimeError("database returned an invalid wall-clock timestamp")
    return (
        parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None
        else parsed.astimezone(timezone.utc)
    )


__all__ = ["database_backend_type", "database_clock", "database_now_sql"]
