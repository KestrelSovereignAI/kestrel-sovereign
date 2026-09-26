"""Backend-neutral binding and reading of timestamp columns.

SQLite stores durable timestamps as explicit ISO-8601 text, while asyncpg
requires a :class:`datetime.datetime` for PostgreSQL ``TIMESTAMP`` parameters.
Storage code that owns a typed timestamp column must use this adapter rather
than relying on either driver's implicit coercions.

The read direction has the same split: a SQLite ``TIMESTAMP`` column comes back
as ISO-8601 text, while ``PostgresBackend.fetch_all`` returns asyncpg's native
``datetime``. :func:`timestamp_column_value` is the one place that accepts both.
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


def timestamp_column_value(value: Any) -> datetime:
    """Return a ``TIMESTAMP`` column value read from either backend as a datetime.

    SQLite yields ISO-8601 text (``CURRENT_TIMESTAMP`` stores
    ``YYYY-MM-DD HH:MM:SS``); asyncpg yields a native ``datetime``. Calling
    ``datetime.fromisoformat`` on the latter raises ``TypeError``, which is the
    PostgreSQL defect this helper exists to prevent. The value is returned as
    stored: no timezone is attached or removed. ``NULL`` is not a timestamp, so
    callers decide what an absent value means before calling this.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(
        "timestamp column value must be ISO-8601 text or a datetime, "
        f"not {type(value).__name__}"
    )


__all__ = ["timestamp_column_value", "utc_timestamp_parameter"]
