"""Conversation identifier normalization helpers."""

from typing import Optional


def coerce_persistent_message_id(message_id: object) -> Optional[int]:
    """Return a database row id for persistent conversation lookups."""
    try:
        row_id = int(message_id)  # API path parameters arrive as strings.
    except (TypeError, ValueError):
        return None

    if row_id < 1:
        return None
    return row_id
