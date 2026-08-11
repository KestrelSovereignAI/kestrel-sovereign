"""Explicit timestamp bind intent shared by unified stores and PostgreSQL."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TimestamptzParameter:
    """A datetime intended for a PostgreSQL ``TIMESTAMPTZ`` column.

    ``asyncpg`` correctly preserves an aware datetime's absolute instant when
    it receives the aware value.  The generic backend compatibility layer also
    serves legacy naive ``TIMESTAMP`` callers, so this small marker lets it
    distinguish the two contracts without changing those callers globally.
    """

    value: datetime
