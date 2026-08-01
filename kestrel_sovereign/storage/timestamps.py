"""Backend-neutral binding for durable UTC timestamp columns.

SQLite stores durable timestamps as explicit ISO-8601 text, while asyncpg
requires a :class:`datetime.datetime` for PostgreSQL ``TIMESTAMP`` parameters.
Storage code that owns a typed timestamp column must use this adapter rather
than relying on either driver's implicit coercions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_timestamp_parameter(backend_type: str, value: Any) -> datetime | str:
    """Return one UTC timestamp value suitable for the selected backend.

    The input contract is intentionally strict: durable timestamps identify an
    instant, so naive datetimes and strings without an offset are rejected.
    PostgreSQL's durable columns are ``TIMESTAMP`` (without timezone), so
    asyncpg receives a *naive UTC* ``datetime`` for its typed bind; SQLite
    receives the corresponding explicit ISO-8601 text.  Keeping this decision
    at the storage seam gives both backends the same instant and avoids
    SQLite's deprecated implicit datetime adapter.
    """
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str) and value:
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("timestamp must be an ISO-8601 UTC instant") from error
    else:
        raise TypeError("timestamp must be an aware datetime or ISO-8601 string")

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must include a timezone offset")
    timestamp = timestamp.astimezone(timezone.utc)
    if backend_type == "postgres":
        return timestamp.replace(tzinfo=None)
    if backend_type == "sqlite":
        return timestamp.isoformat()
    raise ValueError(f"unsupported timestamp backend: {backend_type!r}")


__all__ = ["utc_timestamp_parameter"]
