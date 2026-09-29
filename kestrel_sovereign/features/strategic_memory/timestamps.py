"""The graph ``created_at`` contract, applied at the projection seam (#3255).

``storage/async_graph_store.py`` requires every persisted
``properties.created_at`` to be a UTC ISO-8601 string (the shape
``datetime.now(timezone.utc).isoformat()`` produces) because range filters,
ordering and the scoped EPHEMERAL purge compare it lexicographically. The
strategic-memory ledger and decision file carry day-granular dates
(``YYYY-MM-DD``) written by ``date.today()``, and some rows carry no date at
all. Three projections used to copy those values straight into
``created_at`` — a bare date sorts before every timestamp of its day, and an
empty string either aborts the Postgres purge cast or, after #3227, drops the
row out of leak coverage for good.

One rule, in one place: a date becomes midnight UTC of that date; a full
timestamp is normalised to the contract; nothing usable leaves the key
ABSENT, never empty.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, Optional


def contract_created_at(value: Any) -> Optional[str]:
    """Return ``value`` in the graph ``created_at`` contract, or ``None``.

    - ``YYYY-MM-DD`` → ``YYYY-MM-DDT00:00:00+00:00`` (midnight UTC of that
      calendar date: the ledger records days, and the contract needs an
      instant that sorts with every other stamp of that day).
    - an ISO-8601 datetime → the same instant in UTC with a ``+00:00``
      offset; a naive datetime is taken as UTC, ``Z`` is accepted.
    - ``None``, an empty or whitespace string, a non-string, or text that is
      neither of the above → ``None``. The caller leaves the key absent.
    """
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if len(text) == 10:
                parsed = date.fromisoformat(text)
                moment = datetime(parsed.year, parsed.month, parsed.day, tzinfo=timezone.utc)
            else:
                moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat()


def stamp_created_at(properties: Dict[str, Any], value: Any) -> Dict[str, Any]:
    """Set ``properties["created_at"]`` from ``value`` under the contract, or
    leave the key absent when ``value`` carries no usable instant."""
    stamped = contract_created_at(value)
    if stamped is None:
        properties.pop("created_at", None)
    else:
        properties["created_at"] = stamped
    return properties
