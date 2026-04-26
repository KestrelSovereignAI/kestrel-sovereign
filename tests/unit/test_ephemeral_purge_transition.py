"""Unit tests for the EPHEMERAL hard-purge transition wiring (#767).

Covers the kestrel_agent side of the defense-in-depth — the storage
layer purge primitive itself is exercised end-to-end against SQLite in
``tests/integration/test_ephemeral_hard_purge.py``.

Key contract: leaving EPHEMERAL → anything fires
``storage.purge_ephemeral_session``, and if the result reports any
leaks, the agent writes a ``security_audit_log`` entry through its
SecurityFeature.
"""
from __future__ import annotations

import json
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode


def _make_agent(*, initial_mode=PrivacyMode.EPHEMERAL, leak_breakdown=None):
    """Build a minimally-wired KestrelAgent for transition tests.

    Side-steps __init__ (which requires storage paths, LLM service, etc.)
    and constructs only the surface the transition method touches.
    """
    leak_breakdown = leak_breakdown or {"conversation_history": 0, "graph_nodes": 0}

    permission_store = MagicMock()
    permission_store.log_decision = AsyncMock()
    security_feature = MagicMock(permission_store=permission_store)

    agent = KestrelAgent.__new__(KestrelAgent)
    agent._privacy_mode = initial_mode
    agent.did = "did:test:agent"

    storage = MagicMock()
    storage.purge_ephemeral_session = AsyncMock(return_value=leak_breakdown)
    storage.set_privacy_mode = MagicMock()
    agent.storage = storage

    privacy_agent = MagicMock()
    privacy_agent.set_mode = MagicMock(return_value="Privacy mode changed.")
    agent.privacy_agent = privacy_agent

    agent.features = {"Security": security_feature}
    agent.llm_service = MagicMock()
    agent.llm_service.providers = []
    agent.llm_service.get_model_preference = MagicMock(return_value={})
    agent.llm_service.set_model_preference = MagicMock()
    return agent, storage, permission_store


@pytest.mark.asyncio
async def test_clean_ephemeral_exit_calls_purge_and_skips_audit():
    """Happy path: agent leaves EPHEMERAL with no leaks. Purge fires
    (defense-in-depth must always run) but audit log is silent —
    nothing to report.
    """
    agent, storage, permission_store = _make_agent(
        initial_mode=PrivacyMode.EPHEMERAL,
        leak_breakdown={"conversation_history": 0, "graph_nodes": 0},
    )

    await agent._set_privacy_mode_with_effects_locked(PrivacyMode.NORMAL)

    storage.purge_ephemeral_session.assert_awaited_once()
    call_kwargs = storage.purge_ephemeral_session.await_args.kwargs
    assert "ephemeral-mode-exit-to-normal" in call_kwargs["reason"]

    # No leaks -> no audit entry
    permission_store.log_decision.assert_not_awaited()


@pytest.mark.asyncio
async def test_leaked_ephemeral_exit_writes_audit_with_breakdown():
    """When the storage purge reports leaks, the agent records a
    security_audit_log entry with the agent DID, the reason, and the
    per-table breakdown so the operator can investigate."""
    agent, storage, permission_store = _make_agent(
        initial_mode=PrivacyMode.EPHEMERAL,
        leak_breakdown={"conversation_history": 3, "graph_nodes": 1},
    )

    await agent._set_privacy_mode_with_effects_locked(PrivacyMode.ANONYMOUS)

    storage.purge_ephemeral_session.assert_awaited_once()
    permission_store.log_decision.assert_awaited_once()

    kwargs = permission_store.log_decision.await_args.kwargs
    assert kwargs["feature_name"] == "ephemeral_purge"
    assert kwargs["decision"] == "leak_purged"
    assert kwargs["action"] == "ephemeral_session_close"
    summary = json.loads(kwargs["args_summary"])
    assert summary["agent_did"] == "did:test:agent"
    assert summary["breakdown"] == {"conversation_history": 3, "graph_nodes": 1}
    assert summary["reason"].startswith("ephemeral-mode-exit-to-")


@pytest.mark.asyncio
async def test_normal_to_ephemeral_does_not_purge():
    """Entering EPHEMERAL has nothing to purge — the rail only fires on
    EXIT. Going INTO EPHEMERAL is a privacy contract change, not a
    session close."""
    agent, storage, permission_store = _make_agent(
        initial_mode=PrivacyMode.NORMAL,
    )

    await agent._set_privacy_mode_with_effects_locked(PrivacyMode.EPHEMERAL)

    storage.purge_ephemeral_session.assert_not_awaited()
    permission_store.log_decision.assert_not_awaited()


@pytest.mark.asyncio
async def test_normal_to_isolated_does_not_purge():
    """Non-EPHEMERAL transitions never trigger the rail. Soft-deleted
    NORMAL data must not be touched by an unrelated mode switch."""
    agent, storage, permission_store = _make_agent(
        initial_mode=PrivacyMode.NORMAL,
    )

    await agent._set_privacy_mode_with_effects_locked(PrivacyMode.ISOLATED)

    storage.purge_ephemeral_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_ephemeral_to_ephemeral_does_not_purge():
    """A no-op EPHEMERAL → EPHEMERAL transition (e.g. UI re-confirms
    the same mode) shouldn't fire the rail. The session is still
    open."""
    agent, storage, permission_store = _make_agent(
        initial_mode=PrivacyMode.EPHEMERAL,
    )

    await agent._set_privacy_mode_with_effects_locked(PrivacyMode.EPHEMERAL)

    storage.purge_ephemeral_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_audit_failure_does_not_block_purge_or_transition():
    """If the audit write itself blows up (db locked, table missing),
    the agent must still complete the transition. The leak has already
    been scrubbed at this point — the audit is the breadcrumb, not
    the safety net.
    """
    agent, storage, permission_store = _make_agent(
        initial_mode=PrivacyMode.EPHEMERAL,
        leak_breakdown={"conversation_history": 1, "graph_nodes": 0},
    )
    permission_store.log_decision.side_effect = RuntimeError("audit table missing")

    # Should not raise
    result = await agent._set_privacy_mode_with_effects_locked(PrivacyMode.NORMAL)

    assert result.message == "Privacy mode changed."
    storage.purge_ephemeral_session.assert_awaited_once()
    storage.set_privacy_mode.assert_called_once_with(PrivacyMode.NORMAL)


@pytest.mark.asyncio
async def test_purge_failure_does_not_block_transition():
    """If the storage layer purge itself fails (corrupt DB, lock), the
    transition still completes. We log a warning but don't strand the
    agent in EPHEMERAL forever because of a downstream issue.
    """
    agent, storage, permission_store = _make_agent(
        initial_mode=PrivacyMode.EPHEMERAL,
    )
    storage.purge_ephemeral_session.side_effect = RuntimeError("DB locked")

    result = await agent._set_privacy_mode_with_effects_locked(PrivacyMode.NORMAL)

    assert result.message == "Privacy mode changed."
    storage.set_privacy_mode.assert_called_once_with(PrivacyMode.NORMAL)


@pytest.mark.asyncio
async def test_audit_skipped_when_security_feature_missing():
    """Slim test setups don't load SecurityFeature. The rail still
    purges and logs a WARNING, but writes no audit entry.
    """
    agent, storage, _ = _make_agent(
        initial_mode=PrivacyMode.EPHEMERAL,
        leak_breakdown={"conversation_history": 1, "graph_nodes": 0},
    )
    agent.features = {}  # No SecurityFeature

    # Should not raise
    await agent._set_privacy_mode_with_effects_locked(PrivacyMode.NORMAL)

    storage.purge_ephemeral_session.assert_awaited_once()
