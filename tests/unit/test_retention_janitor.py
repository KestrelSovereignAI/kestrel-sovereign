"""Unit tests for the retention helpers (#764).

Retention runs on the existing per-agent cron scheduler — see the
``trash_retention`` built-in handler in
``kestrel_sovereign/features/scheduler/feature.py`` and its dedicated
test file ``tests/unit/test_scheduler_trash_retention.py``. This file
covers the reusable helpers in
``kestrel_sovereign/storage/retention.py``.

The storage-layer purge primitive itself is exercised end-to-end
against SQLite in ``tests/integration/test_retention_purge_primitive.py``.
"""
from __future__ import annotations

from types import SimpleNamespace

from kestrel_sovereign.storage.retention import (
    agent_privacy_mode,
    resolve_retention_days,
)


# ---------------------------------------------------------------------------
# resolve_retention_days
# ---------------------------------------------------------------------------


def test_resolve_uses_privacy_override_when_present():
    config = {
        "conversation_history_days": 30,
        "privacy_overrides": {"isolated": 7},
    }
    assert resolve_retention_days(
        config=config, privacy_mode="isolated", fallback=14,
    ) == 7


def test_resolve_falls_back_to_global_when_no_override():
    config = {"conversation_history_days": 14}
    assert resolve_retention_days(
        config=config, privacy_mode="normal", fallback=30,
    ) == 14


def test_resolve_falls_back_to_compiled_default():
    """No config at all → use the supplied fallback."""
    assert resolve_retention_days(
        config={}, privacy_mode="normal", fallback=30,
    ) == 30


def test_resolve_returns_none_for_zero_or_negative_retention():
    """Zero/negative would purge instantly — the resolver signals "skip
    this agent" by returning None. The handler logs a warning and
    skips the purge call entirely.
    """
    assert resolve_retention_days(
        config={"conversation_history_days": 0}, privacy_mode="normal",
        fallback=30,
    ) is None
    assert resolve_retention_days(
        config={"conversation_history_days": -1}, privacy_mode="normal",
        fallback=30,
    ) is None


def test_resolve_ignores_garbage_values():
    """Operator-edited TOML can carry strings instead of ints. Don't
    crash; fall through to the next priority level."""
    config = {
        "conversation_history_days": "thirty",
        "privacy_overrides": {"normal": "seven"},
    }
    assert resolve_retention_days(
        config=config, privacy_mode="normal", fallback=21,
    ) == 21


def test_resolve_privacy_mode_match_is_case_insensitive():
    """Privacy mode names normalize to lowercase before lookup so
    ``PrivacyMode.ISOLATED.value`` and ``"isolated"`` both match the
    override row."""
    config = {"privacy_overrides": {"isolated": 5}}
    assert resolve_retention_days(
        config=config, privacy_mode="ISOLATED", fallback=30,
    ) == 5
    assert resolve_retention_days(
        config=config, privacy_mode="Isolated", fallback=30,
    ) == 5


def test_resolve_with_no_privacy_mode_uses_global_default():
    """An agent without a resolvable privacy mode (early init, mock
    fixture) just gets the global default — never None unless the
    operator's intent is "no retention."""
    config = {"conversation_history_days": 14}
    assert resolve_retention_days(
        config=config, privacy_mode=None, fallback=30,
    ) == 14


# ---------------------------------------------------------------------------
# agent_privacy_mode
# ---------------------------------------------------------------------------


def test_agent_privacy_mode_handles_enum_and_string():
    # Enum-like with a .value
    a = SimpleNamespace(_privacy_mode=SimpleNamespace(value="ISOLATED"))
    assert agent_privacy_mode(a) == "isolated"
    # Plain string on the public attr
    a = SimpleNamespace(privacy_mode="Normal")
    assert agent_privacy_mode(a) == "normal"
    # Missing
    a = SimpleNamespace()
    assert agent_privacy_mode(a) is None


# ---------------------------------------------------------------------------
# resolve_cognition_retention_days (#1674) — opt-in, no compiled default
# ---------------------------------------------------------------------------

from kestrel_sovereign.storage.retention import resolve_cognition_retention_days


def test_cognition_resolve_returns_none_when_key_absent():
    """Opt-in: an unset window keeps episodes forever (skip)."""
    assert resolve_cognition_retention_days(config={}, key="episodes_days") is None


def test_cognition_resolve_reads_configured_window():
    assert resolve_cognition_retention_days(
        config={"episodes_days": 180}, key="episodes_days",
    ) == 180


def test_cognition_resolve_none_for_non_positive():
    assert resolve_cognition_retention_days(
        config={"episodes_days": 0}, key="episodes_days",
    ) is None
    assert resolve_cognition_retention_days(
        config={"episodes_days": -5}, key="episodes_days",
    ) is None


def test_cognition_resolve_none_for_non_int():
    assert resolve_cognition_retention_days(
        config={"episodes_days": "soon"}, key="episodes_days",
    ) is None
