"""
Shared utilities for A2A datastores.
"""

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def generate_id() -> str:
    """Generate a unique ID for store records."""
    return uuid4().hex


def now_utc() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def datetime_to_iso(dt: datetime) -> str:
    """Convert datetime to ISO 8601 string."""
    return dt.isoformat()


def iso_to_datetime(iso_str: str) -> datetime:
    """Parse ISO 8601 string to datetime."""
    return datetime.fromisoformat(iso_str)


def json_dumps(obj: Any) -> str:
    """Serialize object to JSON string."""
    return json.dumps(obj, default=str)


def json_loads(s: str) -> Any:
    """Deserialize JSON string to object."""
    if not s:
        return None
    return json.loads(s)
