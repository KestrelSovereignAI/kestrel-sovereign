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
from kestrel_sovereign.features.privacy.feature import PrivacyTransitionDecision
from kestrel_sovereign.storage.privacy_wrapper import (
    EphemeralPurgeReport,
    StorePurgeResult,
    PurgeOutcome,
)


def _report_from_counts(counts):
    """Build the structured purge report the storage layer now returns (#2673).

    A positive count models a purged leak; zero models a clean sweep. Failures
    are modelled separately (via a raised ``side_effect`` or an explicit FAILED
    ``StorePurgeResult``) so a clean zero is never confused with a failure.
    """
    return EphemeralPurgeReport(
        StorePurgeResult(
            store,
            PurgeOutcome.PURGED if count > 0 else PurgeOutcome.CLEAN,
            rows=count,
        )
        for store, count in counts.items()
    )


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
    storage.purge_ephemeral_session = AsyncMock(
        return_value=_report_from_counts(leak_breakdown)
    )
    storage.set_privacy_mode = MagicMock()
    agent.storage = storage

    privacy_agent = MagicMock()
    privacy_agent.set_mode = MagicMock(return_value="Privacy mode changed.")
    # New contract: the agent consults evaluate_transition before applying.
    # These exits are all non-destructive (never PUBLIC→EPHEMERAL).
    privacy_agent.evaluate_transition = MagicMock(
        side_effect=lambda m: PrivacyTransitionDecision(target=m, requires_confirmation=False)
    )
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
async def test_public_to_ephemeral_stages_pending_and_confirm_applies_atomically():
    """PUBLIC→EPHEMERAL is data-destructive: the transition must be STAGED, not
    applied, until confirmed — the split-state fix. Before confirm no state
    holder changes (the bug flipped agent + wrapper while the privacy agent
    stayed PUBLIC and kept persisting). After confirm all holders move together.
    """
    agent, storage, permission_store = _make_agent(initial_mode=PrivacyMode.PUBLIC)
    # Simulate the real privacy agent flagging this specific transition.
    agent.privacy_agent.evaluate_transition = MagicMock(
        return_value=PrivacyTransitionDecision(
            target=PrivacyMode.EPHEMERAL, requires_confirmation=True, warning="WARNING ... confirm"
        )
    )

    staged = await agent._set_privacy_mode_with_effects_locked(PrivacyMode.EPHEMERAL)

    # Staged, not applied: result flags confirmation and NOTHING flipped.
    assert staged.requires_confirmation is True
    assert staged.pending_mode == PrivacyMode.EPHEMERAL.value
    assert agent._privacy_mode == PrivacyMode.PUBLIC
    storage.set_privacy_mode.assert_not_called()
    agent.privacy_agent.set_mode.assert_not_called()
    storage.purge_ephemeral_session.assert_not_awaited()
    assert agent._pending_privacy_transition == PrivacyMode.EPHEMERAL

    # Confirm applies atomically to all three holders.
    applied = await agent.confirm_privacy_transition()
    assert applied.requires_confirmation is False
    assert applied.applied is True
    assert agent._privacy_mode == PrivacyMode.EPHEMERAL
    storage.set_privacy_mode.assert_called_once_with(PrivacyMode.EPHEMERAL)
    agent.privacy_agent.set_mode.assert_called_once_with(PrivacyMode.EPHEMERAL)
    assert agent._pending_privacy_transition is None


@pytest.mark.asyncio
async def test_confirm_with_nothing_pending_is_a_safe_noop():
    agent, storage, _ = _make_agent(initial_mode=PrivacyMode.NORMAL)
    result = await agent.confirm_privacy_transition()
    assert result.requires_confirmation is False
    assert result.applied is False  # no-op is distinguishable from an apply
    assert "No pending" in result.message
    storage.set_privacy_mode.assert_not_called()


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
async def test_required_purge_failure_blocks_transition_and_stays_ephemeral():
    """#2673: if a required no-trace purge sweep fails (corrupt DB, lock), the
    EPHEMERAL exit must be REFUSED — the agent cannot claim a clean transition
    it could not certify. It stays in EPHEMERAL (the safe, more restrictive
    state) and the failure is audited as ``purge_failed`` (distinct from a
    leak).
    """
    agent, storage, permission_store = _make_agent(
        initial_mode=PrivacyMode.EPHEMERAL,
    )
    storage.purge_ephemeral_session.side_effect = RuntimeError("DB locked")

    result = await agent._set_privacy_mode_with_effects_locked(PrivacyMode.NORMAL)

    # Not applied, explicitly flagged as a purge failure — never reported as
    # a successful mode change.
    assert result.applied is False
    assert result.purge_failed is True
    # The mode was NOT flipped and no state holder changed.
    assert agent._privacy_mode == PrivacyMode.EPHEMERAL
    storage.set_privacy_mode.assert_not_called()

    # A distinct purge-failure audit was written (not a leak_purged audit).
    permission_store.log_decision.assert_awaited_once()
    kwargs = permission_store.log_decision.await_args.kwargs
    assert kwargs["decision"] == "purge_failed"
    summary = json.loads(kwargs["args_summary"])
    assert summary["reason"].startswith("ephemeral-mode-exit-to-")
    assert set(summary["failed_stores"]) == {
        "conversation_history",
        "graph_nodes",
        "channel_messages",
    }


@pytest.mark.asyncio
async def test_partial_required_sweep_failure_blocks_transition():
    """#2673: even a SINGLE required content-store failure (with the others
    clean) must block the exit — a partial certification is no certification.
    A FAILED sweep is distinguishable from a clean zero via the structured
    report, so it cannot be waved through as ``conversation_history: 0``.
    """
    agent, storage, permission_store = _make_agent(
        initial_mode=PrivacyMode.EPHEMERAL,
    )
    # conversation_history clean, graph_nodes FAILED, channel_messages clean.
    storage.purge_ephemeral_session = AsyncMock(return_value=EphemeralPurgeReport([
        StorePurgeResult("conversation_history", PurgeOutcome.CLEAN, rows=0),
        StorePurgeResult("graph_nodes", PurgeOutcome.FAILED, error="disk I/O error"),
        StorePurgeResult("channel_messages", PurgeOutcome.CLEAN, rows=0),
    ]))

    result = await agent._set_privacy_mode_with_effects_locked(PrivacyMode.NORMAL)

    assert result.applied is False
    assert result.purge_failed is True
    assert agent._privacy_mode == PrivacyMode.EPHEMERAL
    storage.set_privacy_mode.assert_not_called()
    kwargs = permission_store.log_decision.await_args.kwargs
    assert kwargs["decision"] == "purge_failed"
    summary = json.loads(kwargs["args_summary"])
    assert summary["failed_stores"] == ["graph_nodes"]


@pytest.mark.asyncio
async def test_shutdown_purge_failure_records_durable_audit_and_does_not_raise():
    """#2673 shutdown-time failure: an EPHEMERAL agent exiting while its purge
    fails must NOT report false success. Shutdown stays bounded and the failure
    is recorded as durable ``purge_failed`` audit evidence rather than swallowed.
    """
    agent, storage, permission_store = _make_agent(
        initial_mode=PrivacyMode.EPHEMERAL,
    )
    storage.purge_ephemeral_session.side_effect = RuntimeError("DB locked at shutdown")

    # Must not raise — shutdown continues.
    await agent._purge_ephemeral_on_shutdown(timeout=5.0)

    permission_store.log_decision.assert_awaited_once()
    kwargs = permission_store.log_decision.await_args.kwargs
    assert kwargs["decision"] == "purge_failed"
    summary = json.loads(kwargs["args_summary"])
    assert summary["reason"] == "ephemeral-agent-shutdown"
    assert set(summary["failed_stores"]) == {
        "conversation_history",
        "graph_nodes",
        "channel_messages",
    }


@pytest.mark.asyncio
async def test_shutdown_purge_timeout_records_durable_audit():
    """#2673 shutdown-time timeout: a purge that hangs past the shutdown budget
    is bounded by the timeout and reported as a durable ``purge_failed`` audit
    (failure=shutdown-purge-timeout), never a best-effort success.
    """
    import asyncio

    agent, storage, permission_store = _make_agent(
        initial_mode=PrivacyMode.EPHEMERAL,
    )

    async def _never_returns(*args, **kwargs):
        await asyncio.sleep(10)

    # Replace the mock with a real hanging coroutine so wait_for times out.
    storage.purge_ephemeral_session = _never_returns

    await agent._purge_ephemeral_on_shutdown(timeout=0.05)

    permission_store.log_decision.assert_awaited_once()
    kwargs = permission_store.log_decision.await_args.kwargs
    assert kwargs["decision"] == "purge_failed"
    summary = json.loads(kwargs["args_summary"])
    assert summary["failure"] == "shutdown-purge-timeout"


@pytest.mark.asyncio
async def test_shutdown_audit_write_is_bounded_by_budget():
    """#2673: a hung/locked audit DB after a purge timeout must NOT overrun the
    shutdown budget. The durable audit tail is carved from the supplied budget,
    so ``_purge_ephemeral_on_shutdown`` stays bounded even when BOTH the purge
    and the audit write hang. Before the fix the post-timeout audit was awaited
    unbounded and a locked audit DB could exceed the shutdown deadline.
    """
    import asyncio
    import time

    agent, storage, permission_store = _make_agent(
        initial_mode=PrivacyMode.EPHEMERAL,
    )

    async def _never_returns(*args, **kwargs):
        await asyncio.sleep(10)

    # Both the purge AND the durable audit write hang far past the budget.
    storage.purge_ephemeral_session = _never_returns
    permission_store.log_decision = AsyncMock(side_effect=_never_returns)

    start = time.monotonic()
    await agent._purge_ephemeral_on_shutdown(timeout=0.1)
    elapsed = time.monotonic() - start

    # Bounded by the 0.1s budget (generous CI headroom), NOT the 10s audit hang.
    assert elapsed < 1.0, f"shutdown purge overran its budget: {elapsed:.2f}s"


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
