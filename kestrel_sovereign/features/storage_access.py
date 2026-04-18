"""Safe storage resolvers for feature-internal persistence access.

Feature tables are intentionally internal infrastructure. When an agent is
privacy-wrapped, features should use the raw storage object supplied by the
agent rather than touching deprecated PrivacyEnforcingStorage passthrough
properties, which emit warnings and bypass the privacy API by accident.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

_MISSING = object()


def _is_mock_object(value: Any) -> bool:
    return type(value).__module__ == "unittest.mock"


def _instance_attr(value: Any, name: str) -> Any:
    try:
        attrs = vars(value)
    except TypeError:
        return _MISSING
    return attrs.get(name, _MISSING)


def _safe_attr(value: Any, name: str) -> Any:
    """Read an attribute without invoking properties or MagicMock fabrication."""
    if value is None:
        return None

    instance_value = _instance_attr(value, name)
    if instance_value is not _MISSING:
        return instance_value

    if _is_mock_object(value):
        return None

    try:
        static_value = inspect.getattr_static(value, name)
    except AttributeError:
        return None

    if isinstance(static_value, property):
        return None

    return getattr(value, name, None)


def resolve_feature_database(agent: Any) -> Optional[Any]:
    """Resolve the database handle a feature should use for its own tables.

    Resolution order:
    1. ``agent._raw_storage.db`` when available.
    2. ``agent.storage._storage.db`` for privacy-wrapped storage.
    3. Explicit ``database``, ``db``, or ``_db`` attributes on unwrapped test
       or legacy storage objects, without invoking deprecated wrapper properties.
    """
    raw_storage = _safe_attr(agent, "_raw_storage")
    db = _safe_attr(raw_storage, "db")
    if db is not None:
        return db

    storage = _safe_attr(agent, "storage")
    wrapped_storage = _safe_attr(storage, "_storage")
    db = _safe_attr(wrapped_storage, "db")
    if db is not None:
        return db

    for name in ("database", "db", "_db"):
        db = _safe_attr(storage, name)
        if db is not None:
            return db

    return None


def resolve_feature_conversation_store(agent: Any) -> Optional[Any]:
    """Resolve the raw conversation store without touching wrapper properties."""
    raw_storage = _safe_attr(agent, "_raw_storage")
    conversation = _safe_attr(raw_storage, "conversation")
    if conversation is not None:
        return conversation

    storage = _safe_attr(agent, "storage")
    wrapped_storage = _safe_attr(storage, "_storage")
    conversation = _safe_attr(wrapped_storage, "conversation")
    if conversation is not None:
        return conversation

    return _safe_attr(storage, "conversation")
