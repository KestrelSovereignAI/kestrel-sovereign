"""Database-authoritative UTC clock helpers for durable shared state."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

# The latest instant EVERY backend can store exactly. SQLite's statement clock
# writes milliseconds, so the last stored tick before ``datetime.max`` is
# ``.999``; a bound after it would have to round up past the end of the
# calendar ``datetime`` can represent. Callers that accept a caller-supplied
# bound refuse anything later than this before it reaches a store.
LATEST_TIMESTAMP_BOUND = datetime(
    9999, 12, 31, 23, 59, 59, 999000, tzinfo=timezone.utc
)


class TimestampBoundOutOfRange(ValueError):
    """A filter bound this backend cannot render at its stored precision."""


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


def database_timestamp_bound_text(db: Any, value: datetime) -> str:
    """Render one filter bound in the exact text shape this backend's clock writes.

    Receipt ``occurred_at`` columns are written by :func:`database_now_sql`, so
    every row in one database shares a fixed-width UTC format and therefore
    orders lexicographically. A filter bound must be rendered in that SAME
    width or it compares wrongly at the boundary: SQLite stores milliseconds,
    and the text ``...:00.500+00:00`` sorts BELOW a six-digit
    ``...:00.500000+00:00`` bound even though both name one instant.

    Precision the backend cannot store is rounded UP to the next stored tick,
    and that one rule is right for both bound shapes the feeds use. With
    stored values on a millisecond grid, ``x >= b`` holds exactly when
    ``x >= ceil(b)`` (an inclusive lower bound of ``.500001`` must exclude a
    row stored at ``.500``), and ``x < b`` holds exactly when ``x < ceil(b)``
    (an exclusive upper bound of ``.500001`` must include that same row).
    Truncating either bound would admit or drop a row on the wrong side of the
    requested instant.

    A bound later than :data:`LATEST_TIMESTAMP_BOUND` raises
    :class:`TimestampBoundOutOfRange` rather than an ``OverflowError`` from
    the rounding: it names a caller error, not a backend fault.

    The text is assembled field by field, not with ``strftime("%Y")``: that
    directive does not zero-pad years below 1000 on every platform (glibc
    renders year 5 as ``5``), and an unpadded year sorts wrongly against the
    four-digit years the database clock writes.
    """

    backend_type = database_backend_type(db)
    if backend_type not in {"postgres", "sqlite"}:
        raise RuntimeError(
            "database statement clock is unavailable for this backend"
        )
    try:
        utc = (
            value.replace(tzinfo=timezone.utc)
            if value.tzinfo is None
            else value.astimezone(timezone.utc)
        )
    except OverflowError as error:
        raise TimestampBoundOutOfRange(
            "timestamp bound falls outside the representable UTC range"
        ) from error
    if utc > LATEST_TIMESTAMP_BOUND:
        raise TimestampBoundOutOfRange(
            "timestamp bound is later than the latest storable instant"
        )
    if backend_type == "postgres":
        # Microsecond precision is exactly what a ``datetime`` carries.
        fraction = f"{utc.microsecond:06d}"
    else:
        remainder = utc.microsecond % 1000
        if remainder:
            utc = utc + timedelta(microseconds=1000 - remainder)
        fraction = f"{utc.microsecond // 1000:03d}"
    return (
        f"{utc.year:04d}-{utc.month:02d}-{utc.day:02d}T"
        f"{utc.hour:02d}:{utc.minute:02d}:{utc.second:02d}.{fraction}+00:00"
    )


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


__all__ = [
    "LATEST_TIMESTAMP_BOUND",
    "TimestampBoundOutOfRange",
    "database_backend_type",
    "database_clock",
    "database_now_sql",
    "database_timestamp_bound_text",
]
