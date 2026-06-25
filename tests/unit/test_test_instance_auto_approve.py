"""Tests for #1936 — non-interactive approval for tagged test instances.

A headless test agent cannot answer the interactive approval queue, so without
this it hangs on the ~20-minute approval timeout (#406) for every ASK-level
tool. The fix reuses the framework's existing, audited global auto-mode, gated
strictly behind BOTH ``is_test_instance`` AND an explicit operator opt-in.
"""

import pytest
import pytest_asyncio
from unittest.mock import MagicMock

from kestrel_sovereign.features.security.feature import (
    SecurityFeature,
    _test_auto_approve_opt_in,
    _TEST_AUTO_APPROVE_ENV_VARS,
)
from kestrel_sovereign.features.security.permissions import (
    PermissionLevel,
    PermissionStore,
)


# --------------------------------------------------------------------------- #
# Env opt-in parsing
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clear_opt_in_env(monkeypatch):
    """Start every test with the opt-in env vars unset."""
    for var in _TEST_AUTO_APPROVE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "On"])
def test_opt_in_truthy(monkeypatch, value):
    monkeypatch.setenv("KESTREL_TEST_AUTO_APPROVE", value)
    assert _test_auto_approve_opt_in() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_opt_in_falsy(monkeypatch, value):
    monkeypatch.setenv("KESTREL_TEST_AUTO_APPROVE", value)
    assert _test_auto_approve_opt_in() is False


def test_opt_in_unset_is_false():
    assert _test_auto_approve_opt_in() is False


def test_opt_in_alias_var(monkeypatch):
    """The alternate env var name also opts in."""
    monkeypatch.setenv("KESTREL_TEST_INSTANCE_AUTO_APPROVE", "1")
    assert _test_auto_approve_opt_in() is True


# --------------------------------------------------------------------------- #
# Enabling logic — the strict two-gate scope
# --------------------------------------------------------------------------- #
def _feature_with_store(store, *, is_test_instance):
    agent = MagicMock()
    agent.is_test_instance = is_test_instance
    agent._agent_name = "kite"
    feature = SecurityFeature(agent)
    feature.permission_store = store
    return feature


@pytest_asyncio.fixture
async def store(tmp_path):
    s = PermissionStore(str(tmp_path / "perms.db"))
    await s.initialize()
    return s


@pytest.mark.asyncio
async def test_test_instance_with_opt_in_enables_auto_mode(monkeypatch, store):
    monkeypatch.setenv("KESTREL_TEST_AUTO_APPROVE", "1")
    feature = _feature_with_store(store, is_test_instance=True)

    enabled = await feature._maybe_enable_test_instance_auto_approve()

    assert enabled is True
    assert store.get_global_auto_mode() is True
    # An otherwise ASK-by-default tool now resolves to AUTO (no human queue).
    level = await store.get_permission("ComputeFeature", "run_script")
    assert level == PermissionLevel.AUTO


@pytest.mark.asyncio
async def test_test_instance_without_opt_in_stays_interactive(store):
    # No env opt-in set (autouse fixture clears it).
    feature = _feature_with_store(store, is_test_instance=True)

    enabled = await feature._maybe_enable_test_instance_auto_approve()

    assert enabled is False
    assert store.get_global_auto_mode() is False
    level = await store.get_permission("ComputeFeature", "run_script")
    assert level == PermissionLevel.ASK


@pytest.mark.asyncio
async def test_non_test_agent_never_enables_even_with_opt_in(monkeypatch, store):
    """Production/sovereign agents are unaffected even if the env is set."""
    monkeypatch.setenv("KESTREL_TEST_AUTO_APPROVE", "1")
    feature = _feature_with_store(store, is_test_instance=False)

    enabled = await feature._maybe_enable_test_instance_auto_approve()

    assert enabled is False
    assert store.get_global_auto_mode() is False
    level = await store.get_permission("ComputeFeature", "run_script")
    assert level == PermissionLevel.ASK


@pytest.mark.asyncio
async def test_deny_and_always_ask_remain_hard_rails_under_auto_mode(monkeypatch, store):
    """Auto-mode promotes ASK→AUTO but DENY/ALWAYS_ASK stay hard policy rails,
    so an operator can still block a specific dangerous tool on a test agent."""
    monkeypatch.setenv("KESTREL_TEST_AUTO_APPROVE", "1")
    await store.set_permission("KeyManagementFeature", "export_key", PermissionLevel.DENY)
    await store.set_permission("ComputeFeature", "run_script", PermissionLevel.ALWAYS_ASK)

    feature = _feature_with_store(store, is_test_instance=True)
    assert await feature._maybe_enable_test_instance_auto_approve() is True

    assert (
        await store.get_permission("KeyManagementFeature", "export_key")
        == PermissionLevel.DENY
    )
    assert (
        await store.get_permission("ComputeFeature", "run_script")
        == PermissionLevel.ALWAYS_ASK
    )


@pytest.mark.asyncio
async def test_enabling_writes_audit_row(monkeypatch, store):
    monkeypatch.setenv("KESTREL_TEST_AUTO_APPROVE", "1")
    feature = _feature_with_store(store, is_test_instance=True)

    await feature._maybe_enable_test_instance_auto_approve()

    history = await store.get_audit_log(limit=10)
    assert any(
        row.get("decision") == "auto_mode_allowed"
        and row.get("user_choice") == "is_test_instance_opt_in"
        for row in history
    ), f"expected an is_test_instance audit row, got: {history}"
