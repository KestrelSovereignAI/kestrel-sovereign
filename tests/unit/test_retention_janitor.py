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
# load_forgetting_config (#1674) — [forgetting] deletion tier, opt-in/off,
# always fully-defaulted so callers never branch on missing keys.
# ---------------------------------------------------------------------------

from kestrel_sovereign.storage.retention import (
    load_forgetting_config,
    DEFAULT_FORGETTING_DELETE_THRESHOLD,
    DEFAULT_FORGETTING_GRACE_DAYS,
)


def _patch_forgetting_section(monkeypatch, section):
    """Make ``load_section('forgetting')`` return ``section``."""
    import kestrel_sovereign.config as cfg
    monkeypatch.setattr(
        cfg, "load_section",
        lambda name: section if name == "forgetting" else {},
    )


def test_forgetting_absent_section_is_opt_out_with_defaults(monkeypatch):
    """No [forgetting] section → disabled, but the threshold/grace defaults are
    still populated so the (skipped) call site never sees missing keys."""
    _patch_forgetting_section(monkeypatch, {})
    cfg = load_forgetting_config()
    assert cfg["enabled"] is False
    assert cfg["delete_threshold"] == DEFAULT_FORGETTING_DELETE_THRESHOLD
    assert cfg["grace_days"] == DEFAULT_FORGETTING_GRACE_DAYS


def test_forgetting_reads_configured_values(monkeypatch):
    _patch_forgetting_section(monkeypatch, {
        "enabled": True, "delete_threshold": 0.05, "delete_grace_days": 30,
    })
    cfg = load_forgetting_config()
    assert cfg == {"enabled": True, "delete_threshold": 0.05, "grace_days": 30}


def test_forgetting_garbage_threshold_falls_back(monkeypatch):
    _patch_forgetting_section(monkeypatch, {"enabled": True, "delete_threshold": "low"})
    cfg = load_forgetting_config()
    assert cfg["delete_threshold"] == DEFAULT_FORGETTING_DELETE_THRESHOLD


def test_forgetting_threshold_above_one_falls_back(monkeypatch):
    """delete_threshold is compared to a decay strength in (0, 1]; a typo like
    `2` would make every past-grace episode eligible, so it must fail safe."""
    _patch_forgetting_section(monkeypatch, {"enabled": True, "delete_threshold": 2})
    assert load_forgetting_config()["delete_threshold"] == DEFAULT_FORGETTING_DELETE_THRESHOLD


def test_forgetting_non_positive_values_fall_back(monkeypatch):
    _patch_forgetting_section(monkeypatch, {
        "enabled": True, "delete_threshold": 0, "delete_grace_days": -5,
    })
    cfg = load_forgetting_config()
    assert cfg["delete_threshold"] == DEFAULT_FORGETTING_DELETE_THRESHOLD
    assert cfg["grace_days"] == DEFAULT_FORGETTING_GRACE_DAYS


def test_forgetting_non_bool_enabled_is_disabled(monkeypatch):
    """A non-bool ``enabled`` must fail safe to OFF, never truthy-coerce a
    string like "false" into deletion being on."""
    _patch_forgetting_section(monkeypatch, {"enabled": "false"})
    assert load_forgetting_config()["enabled"] is False
