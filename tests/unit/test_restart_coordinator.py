"""Tests for the durable restart coordinator (#1512).

Covers the store layer (table init, request lifecycle, status
transitions, race-protection on cancel) and the feature surface
(request/list/cancel @tools, the executor cron task's safety gate,
and the post-restart sweep that wakes the requesting agent).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.restart_coordinator import (
    RestartCoordinatorFeature,
)
from kestrel_sovereign.features.restart_coordinator.store import (
    ensure_restart_requests_table,
    get_request,
    insert_request,
    list_requests,
    record_update_log,
    update_status,
)
from kestrel_sovereign.features.restart_coordinator.update_profiles import (
    get_update_profile,
    is_valid_target_ref,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _CapturingDispatcher:
    def __init__(self):
        self.signals = []

    def enqueue_signal(self, signal):
        self.signals.append(signal)
        return None


class _StubRegistry:
    def __init__(self):
        self.registered = []
        self.by_name = {}

    def register(self, reg):
        if reg.name in self.by_name:
            raise RuntimeError(f"duplicate source {reg.name!r}")
        self.registered.append(reg)
        self.by_name[reg.name] = reg

    def get(self, name):
        return self.by_name.get(name)


async def _backend(tmp_path):
    """Wrap SQLiteBackend in AsyncDatabase to match the production
    surface ``resolve_feature_database`` returns. ``AsyncDatabase``
    exposes ``fetchall`` (no underscore) while the bare backend uses
    ``fetch_all`` — store code must work against the wrapper.
    """
    raw = SQLiteBackend(str(tmp_path / "restart.db"))
    await raw.connect()
    db = AsyncDatabase(raw)
    await ensure_restart_requests_table(db)
    return db


def _make_agent(backend, did="did:test:agent", dispatcher=None,
                registry=None):
    raw_storage = SimpleNamespace(db=backend)
    return SimpleNamespace(
        did=did,
        agent_id=did,
        _raw_storage=raw_storage,
        storage=None,
        dispatcher=dispatcher,
        signal_registry=registry,
        # An idle agent — empty active-request set and no background
        # tasks. Tests that need to model "busy" override these.
        _active_request_ids=set(),
        _background_tasks=set(),
        features={"RestartCoordinatorFeature": True},
    )


async def _make_feature(tmp_path, **kwargs):
    backend = await _backend(tmp_path)
    agent = _make_agent(backend, **kwargs)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    return feat, backend


# ---------------------------------------------------------------------------
# Store layer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_table_is_idempotent(tmp_path):
    backend = await _backend(tmp_path)
    # Second call must NOT error or wipe data.
    await ensure_restart_requests_table(backend)
    await insert_request(
        backend, requested_by_agent="a", reason="r",
    )
    await ensure_restart_requests_table(backend)
    rows = await list_requests(backend)
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_insert_then_list_then_get(tmp_path):
    backend = await _backend(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="did:t:a", reason="config landed",
        urgency="high", policy="idle_agents_only",
    )
    rows = await list_requests(backend)
    assert len(rows) == 1
    assert rows[0].id == req.id
    assert rows[0].status == "pending"
    assert rows[0].urgency == "high"
    # get_request returns the same row.
    fetched = await get_request(backend, req.id)
    assert fetched is not None
    assert fetched.reason == "config landed"


@pytest.mark.asyncio
async def test_update_status_gated_on_expected_current(tmp_path):
    backend = await _backend(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="a", reason="r",
    )
    # Transition gated on the wrong expected status must NOT update.
    ok = await update_status(
        backend, req.id, status="executing",
        expected_current_status="executing",  # wrong — row is pending
    )
    assert ok is False
    fetched = await get_request(backend, req.id)
    assert fetched.status == "pending"
    # Correct expected status: transition lands.
    ok = await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )
    assert ok is True


@pytest.mark.asyncio
async def test_list_requests_filters_by_status_and_agent(tmp_path):
    backend = await _backend(tmp_path)
    await insert_request(backend, requested_by_agent="a", reason="r1")
    await insert_request(backend, requested_by_agent="a", reason="r2")
    await insert_request(backend, requested_by_agent="b", reason="r3")
    # Filter by agent.
    a_rows = await list_requests(backend, agent_id="a")
    assert len(a_rows) == 2
    # Filter by status (all pending).
    pending = await list_requests(backend, status="pending")
    assert len(pending) == 3


# ---------------------------------------------------------------------------
# Feature surface — @tool methods
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_restart_creates_pending_row(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    result = await feat.request_restart(
        reason="kestrel.toml change landed",
    )
    assert result.status is ToolResultStatus.OK
    req = result.data["request"]
    assert req["status"] == "pending"
    assert req["requested_by_agent"] == "did:test:agent"
    # Persisted to the table.
    rows = await list_requests(backend)
    assert any(r.id == req["id"] for r in rows)


@pytest.mark.asyncio
async def test_request_restart_rejects_unknown_urgency(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    result = await feat.request_restart(reason="r", urgency="bogus")
    assert result.error is not None
    assert "urgency" in result.error


@pytest.mark.asyncio
async def test_request_restart_rejects_unknown_policy(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    result = await feat.request_restart(reason="r", policy="anything")
    assert result.error is not None
    assert "policy" in result.error


@pytest.mark.asyncio
async def test_request_restart_rejects_empty_reason(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    result = await feat.request_restart(reason="   ")
    assert result.error is not None
    assert "reason" in result.error


@pytest.mark.asyncio
async def test_list_restart_requests_returns_all_then_filtered(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    await feat.request_restart(reason="r1")
    await feat.request_restart(reason="r2", urgency="high")
    # No filter.
    r = await feat.list_restart_requests()
    assert r.data["count"] == 2
    # Filter to pending — still both.
    r = await feat.list_restart_requests(status="pending")
    assert r.data["count"] == 2
    # Filter to completed — none yet.
    r = await feat.list_restart_requests(status="completed")
    assert r.data["count"] == 0


@pytest.mark.asyncio
async def test_cancel_pending_request(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    created = await feat.request_restart(reason="r")
    req_id = created.data["request"]["id"]
    cancel = await feat.cancel_restart_request(req_id, reason="never mind")
    assert cancel.status is ToolResultStatus.OK
    # Row reflects terminal canceled state.
    row = await get_request(backend, req_id)
    assert row.status == "canceled"
    assert "never mind" in row.status_reason


@pytest.mark.asyncio
async def test_cannot_cancel_executing_request(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    created = await feat.request_restart(reason="r")
    req_id = created.data["request"]["id"]
    # Move it to executing manually.
    await update_status(
        backend, req_id, status="executing",
        expected_current_status="pending",
    )
    cancel = await feat.cancel_restart_request(req_id)
    assert cancel.error is not None
    assert "executing" in cancel.error.lower()


@pytest.mark.asyncio
async def test_cancel_unknown_id_errors(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    cancel = await feat.cancel_restart_request("does-not-exist")
    assert cancel.error is not None
    assert "No restart request" in cancel.error


# ---------------------------------------------------------------------------
# Executor cron task — restart_coordinator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_executor_no_pending_returns_no_op(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()
    assert result.data["executed"] == [] if isinstance(
        result.data["executed"], list
    ) else result.data["executed"] is False
    assert mock_spawn.call_count == 0


@pytest.mark.asyncio
async def test_executor_spawns_subprocess_for_idle_agent(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    created = await feat.request_restart(reason="ship")
    req_id = created.data["request"]["id"]

    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()

    assert mock_spawn.call_count == 1
    assert result.data["executed"][0]["request_id"] == req_id
    # Row moved to executing.
    row = await get_request(backend, req_id)
    assert row.status == "executing"


@pytest.mark.asyncio
async def test_executor_defers_when_agent_reports_active_request(tmp_path):
    # Agent has at least one active request — executor must defer.
    backend = await _backend(tmp_path)
    agent = _make_agent(backend)
    agent._active_request_ids.add("req-1")
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    await feat.request_restart(reason="r")

    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()

    assert mock_spawn.call_count == 0
    assert len(result.data["deferred"]) == 1
    assert "busy" in result.data["deferred"][0]["reason"]


@pytest.mark.asyncio
async def test_executor_executes_on_busy_with_timeout_policy(tmp_path):
    backend = await _backend(tmp_path)
    agent = _make_agent(backend)
    agent._active_request_ids.add("req-busy")
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    # File a request, then back-date requested_at past the 5-min
    # timeout so the policy allows execution despite a busy agent.
    req = await insert_request(
        backend,
        requested_by_agent=agent.did,
        reason="r",
        policy="allow_busy_after_timeout",
    )
    aged = (
        datetime.now(timezone.utc) - timedelta(seconds=600)
    ).isoformat()
    await backend.execute(
        "UPDATE restart_requests SET requested_at = ? WHERE id = ?",
        (aged, req.id),
    )

    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()

    assert mock_spawn.call_count == 1
    assert result.data["executed"][0]["request_id"] == req.id


@pytest.mark.asyncio
async def test_executor_never_runs_manual_only_policy(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    await feat.request_restart(reason="r", policy="manual_only")
    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()
    assert mock_spawn.call_count == 0
    assert len(result.data["deferred"]) == 1
    assert "manual_only" in result.data["deferred"][0]["reason"]


@pytest.mark.asyncio
async def test_executor_processes_higher_urgency_first(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    low = await feat.request_restart(reason="lo", urgency="low")
    high = await feat.request_restart(reason="hi", urgency="critical")
    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()
    # Only the highest-urgency one is dispatched per poll.
    assert len(result.data["executed"]) == 1
    assert (
        result.data["executed"][0]["request_id"]
        == high.data["request"]["id"]
    )
    # Low-urgency one is still pending for the next poll.
    rows = await list_requests(backend, status="pending")
    assert len(rows) == 1
    assert rows[0].id == low.data["request"]["id"]


@pytest.mark.asyncio
async def test_executor_recovers_on_spawn_failure(tmp_path):
    """If the detached subprocess spawn raises, the row must move
    back from executing to pending so the next poll retries —
    not be left stuck in executing forever.
    """
    feat, backend = await _make_feature(tmp_path)
    created = await feat.request_restart(reason="r")
    req_id = created.data["request"]["id"]

    with patch.object(
        RestartCoordinatorFeature,
        "_spawn_restart_subprocess",
        side_effect=OSError("kestrel binary missing"),
    ):
        await feat.restart_coordinator()

    row = await get_request(backend, req_id)
    assert row.status == "pending"
    assert "spawn failed" in row.status_reason


# ---------------------------------------------------------------------------
# Post-restart wakeup sweep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_restart_sweep_marks_completed_and_emits_signal(tmp_path):
    """A fresh feature constructed after restart must mark any
    ``executing`` row owned by this agent as ``completed`` and emit
    one ``restart.completed`` COGNITION signal so the requesting
    agent wakes.
    """
    backend = await _backend(tmp_path)
    # Pre-seed an executing row as if a prior poll spawned the
    # subprocess that took us down and brought us back up.
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="pre-restart",
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )

    dispatcher = _CapturingDispatcher()
    registry = _StubRegistry()
    agent = _make_agent(backend, dispatcher=dispatcher, registry=registry)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()

    row = await get_request(backend, req.id)
    assert row.status == "completed"
    assert len(dispatcher.signals) == 1
    sig = dispatcher.signals[0]
    assert sig.payload["request_id"] == req.id
    assert sig.payload["reason"] == "pre-restart"
    assert sig.source == "restart.completed"


@pytest.mark.asyncio
async def test_post_restart_sweep_only_touches_this_agents_rows(tmp_path):
    """Multi-agent: the sweep must NOT mark another agent's
    in-flight executing row as completed.
    """
    backend = await _backend(tmp_path)
    other = await insert_request(
        backend, requested_by_agent="did:test:other", reason="other",
    )
    await update_status(
        backend, other.id, status="executing",
        expected_current_status="pending",
    )

    agent = _make_agent(backend, did="did:test:agent")
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()

    row = await get_request(backend, other.id)
    assert row.status == "executing"


@pytest.mark.asyncio
async def test_update_status_race_only_winner_returns_true(tmp_path):
    """Two coordinators both call UPDATE ... WHERE status='pending';
    only the row whose pre-image actually matched updated, so only
    that caller must observe True. The pre-codex-P1 fix returned
    True for both because the int rowcount was misinterpreted as
    "no cursor" and a fallback SELECT was used.
    """
    backend = await _backend(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="a", reason="r",
    )
    # First update wins.
    a_ok = await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )
    # Second update sees status already executing — must NOT update.
    b_ok = await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )
    assert a_ok is True
    assert b_ok is False


@pytest.mark.asyncio
async def test_executor_defers_when_no_introspection_surface(tmp_path):
    """If the agent doesn't expose ANY in-flight surface, the
    safety check must conservatively report busy. Pre-codex-P1-fix
    the absence of introspection was treated as "idle", which
    defeated the idle_agents_only policy on production agents.
    """
    backend = await _backend(tmp_path)
    raw_storage = SimpleNamespace(db=backend)
    bare_agent = SimpleNamespace(
        did="did:test:agent",
        agent_id="did:test:agent",
        _raw_storage=raw_storage,
        storage=None,
        dispatcher=None,
        signal_registry=None,
        features={"RestartCoordinatorFeature": True},
    )
    feat = RestartCoordinatorFeature(bare_agent)
    await feat.initialize()
    await feat.request_restart(reason="r")
    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()
    assert mock_spawn.call_count == 0
    assert len(result.data["deferred"]) == 1
    assert "no idleness" in result.data["deferred"][0]["reason"]


@pytest.mark.asyncio
async def test_post_restart_sweep_retries_when_dispatcher_raises(tmp_path):
    """If the dispatcher is present but ``enqueue_signal`` raises,
    the row must STAY in executing so a future agent boot's sweep
    can retry. Pre-codex-P2-fix the row was already terminalized
    so the wake was permanently lost.
    """
    backend = await _backend(tmp_path)
    req = await insert_request(
        backend, requested_by_agent="did:test:agent", reason="r",
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )

    class _BrokenDispatcher:
        def enqueue_signal(self, signal):
            raise RuntimeError("dispatcher down")

    agent = _make_agent(backend, dispatcher=_BrokenDispatcher())
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()

    row = await get_request(backend, req.id)
    assert row.status == "executing", (
        f"row must stay executing for retry; got {row.status!r}"
    )


# ---------------------------------------------------------------------------
# update_then_restart — audited update-and-restart (#1539)
# ---------------------------------------------------------------------------


def _git_checkout(tmp_path):
    """Make ``tmp_path`` look like a git checkout for repo validation."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    return str(tmp_path)


@pytest.mark.asyncio
async def test_request_update_then_restart_creates_row(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    repo = _git_checkout(tmp_path)
    result = await feat.request_restart(
        reason="ship merged fix",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        target_ref="main",
        repo_path=repo,
    )
    assert result.status is ToolResultStatus.OK
    req = result.data["request"]
    assert req["operation"] == "update_then_restart"
    assert req["update_profile"] == "sovereign_local_uv_sync"
    assert req["update_target_ref"] == "main"
    assert req["update_repo_path"] == repo
    # Persisted with the new fields intact.
    row = await get_request(backend, req["id"])
    assert row.operation == "update_then_restart"
    assert row.update_target_ref == "main"


@pytest.mark.asyncio
async def test_request_update_then_restart_rejects_unknown_profile(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    repo = _git_checkout(tmp_path)
    result = await feat.request_restart(
        reason="r",
        operation="update_then_restart",
        update_profile="rm_rf_everything",
        target_ref="main",
        repo_path=repo,
    )
    assert result.error is not None
    assert "update_profile" in result.error


@pytest.mark.asyncio
async def test_request_update_then_restart_rejects_bad_target_ref(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    repo = _git_checkout(tmp_path)
    # Missing ref.
    missing = await feat.request_restart(
        reason="r",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        target_ref="",
        repo_path=repo,
    )
    assert missing.error is not None
    assert "target_ref" in missing.error
    # Option-injection-style ref must be rejected.
    crafted = await feat.request_restart(
        reason="r",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        target_ref="--upload-pack=evil",
        repo_path=repo,
    )
    assert crafted.error is not None
    assert "target_ref" in crafted.error


@pytest.mark.asyncio
async def test_request_update_then_restart_rejects_non_git_repo(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    result = await feat.request_restart(
        reason="r",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        target_ref="main",
        repo_path=str(tmp_path / "not-a-checkout"),
    )
    assert result.error is not None
    assert "repo_path" in result.error


@pytest.mark.asyncio
async def test_request_restart_rejects_unknown_operation(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    result = await feat.request_restart(reason="r", operation="nuke")
    assert result.error is not None
    assert "operation" in result.error


@pytest.mark.asyncio
async def test_list_and_cancel_update_then_restart(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    repo = _git_checkout(tmp_path)
    created = await feat.request_restart(
        reason="r",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        target_ref="v1.2.3",
        repo_path=repo,
    )
    req_id = created.data["request"]["id"]
    listed = await feat.list_restart_requests()
    assert listed.data["count"] == 1
    assert listed.data["requests"][0]["operation"] == "update_then_restart"
    cancel = await feat.cancel_restart_request(req_id, reason="stand down")
    assert cancel.status is ToolResultStatus.OK
    row = await get_request(backend, req_id)
    assert row.status == "canceled"


@pytest.mark.asyncio
async def test_coordinator_rejects_unknown_profile_terminal(tmp_path):
    """A row carrying an unknown profile (inserted outside the tool's
    validation) must be terminally rejected by the coordinator, not
    retried forever or executed.
    """
    feat, backend = await _make_feature(tmp_path)
    req = await insert_request(
        backend,
        requested_by_agent="did:test:agent",
        reason="r",
        operation="update_then_restart",
        update_profile="bogus_profile",
        update_target_ref="main",
        update_repo_path=_git_checkout(tmp_path),
    )
    with patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        await feat.restart_coordinator()
    assert mock_spawn.call_count == 0
    row = await get_request(backend, req.id)
    assert row.status == "rejected"
    assert "unknown update profile" in row.status_reason


@pytest.mark.asyncio
async def test_coordinator_runs_update_then_spawns_restart(tmp_path):
    feat, backend = await _make_feature(tmp_path)
    repo = _git_checkout(tmp_path)
    created = await feat.request_restart(
        reason="ship",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        target_ref="main",
        repo_path=repo,
    )
    req_id = created.data["request"]["id"]

    fake_update = {
        "ok": True,
        "profile": "sovereign_local_uv_sync",
        "repo_path": repo,
        "target_ref": "main",
        "resolved_ref": "abc1234",
        "steps": [],
        "migration": {"ran": False, "reason": "additive"},
        "failed_step": None,
    }

    async def _fake_run_update(self, req, profile):
        return fake_update

    with patch.object(
        RestartCoordinatorFeature, "_run_update", _fake_run_update,
    ), patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()

    assert mock_spawn.call_count == 1
    assert result.data["executed"][0]["request_id"] == req_id
    row = await get_request(backend, req_id)
    assert row.status == "executing"
    # Update audit log persisted on the row.
    assert row.update_log_dict()["resolved_ref"] == "abc1234"


@pytest.mark.asyncio
async def test_coordinator_update_failure_leaves_retryable(tmp_path):
    """If the update fails before restart, the row must NOT restart and
    must be left retryable (back to pending) with a clear reason.
    """
    feat, backend = await _make_feature(tmp_path)
    repo = _git_checkout(tmp_path)
    created = await feat.request_restart(
        reason="ship",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        target_ref="main",
        repo_path=repo,
    )
    req_id = created.data["request"]["id"]

    async def _fake_run_update(self, req, profile):
        return {
            "ok": False,
            "profile": "sovereign_local_uv_sync",
            "repo_path": repo,
            "target_ref": "main",
            "resolved_ref": "",
            "steps": [{"step": "install", "ok": False}],
            "migration": {"ran": False, "reason": "additive"},
            "failed_step": "install",
        }

    with patch.object(
        RestartCoordinatorFeature, "_run_update", _fake_run_update,
    ), patch.object(
        RestartCoordinatorFeature, "_spawn_restart_subprocess",
    ) as mock_spawn:
        result = await feat.restart_coordinator()

    assert mock_spawn.call_count == 0
    row = await get_request(backend, req_id)
    assert row.status == "pending"
    assert "install" in row.status_reason
    assert row.update_log_dict()["failed_step"] == "install"
    assert len(result.data["deferred"]) == 1


@pytest.mark.asyncio
async def test_run_update_records_steps_and_resolved_ref(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    profile = get_update_profile("sovereign_local_uv_sync")
    req = SimpleNamespace(
        update_repo_path=str(tmp_path),
        update_target_ref="main",
        update_allow_migrations=False,
    )

    async def _fake_step(self, step):
        out = "deadbeef" if step.name == "resolve_ref" else ""
        return {
            "step": step.name,
            "argv": list(step.argv),
            "returncode": 0,
            "ok": True,
            "stdout_tail": out,
            "stderr_tail": "",
        }

    with patch.object(
        RestartCoordinatorFeature, "_run_update_step", _fake_step,
    ):
        update = await feat._run_update(req, profile)

    assert update["ok"] is True
    assert update["resolved_ref"] == "deadbeef"
    assert {s["step"] for s in update["steps"]} >= {
        "fetch", "checkout", "install", "resolve_ref",
    }
    assert update["migration"]["ran"] is False


@pytest.mark.asyncio
async def test_run_update_stops_at_first_failing_step(tmp_path):
    feat, _ = await _make_feature(tmp_path)
    profile = get_update_profile("sovereign_local_uv_sync")
    req = SimpleNamespace(
        update_repo_path=str(tmp_path),
        update_target_ref="main",
        update_allow_migrations=False,
    )

    async def _fake_step(self, step):
        ok = step.name == "fetch"  # checkout fails
        return {
            "step": step.name,
            "argv": list(step.argv),
            "returncode": 0 if ok else 1,
            "ok": ok,
            "stdout_tail": "",
            "stderr_tail": "boom" if not ok else "",
        }

    with patch.object(
        RestartCoordinatorFeature, "_run_update_step", _fake_step,
    ):
        update = await feat._run_update(req, profile)

    assert update["ok"] is False
    assert update["failed_step"] == "checkout"
    # Stopped before install/resolve_ref ran.
    assert {s["step"] for s in update["steps"]} == {"fetch", "checkout"}


@pytest.mark.asyncio
async def test_post_restart_sweep_signal_includes_update_metadata(tmp_path):
    """The completion signal for an update_then_restart row must carry
    operation + target ref + the resolved commit so the agent can verify
    it booted into the requested ref.
    """
    backend = await _backend(tmp_path)
    req = await insert_request(
        backend,
        requested_by_agent="did:test:agent",
        reason="merged fix",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        update_target_ref="main",
        update_repo_path=str(tmp_path),
    )
    await record_update_log(
        backend, req.id, json.dumps({"resolved_ref": "cafef00d"}),
    )
    await update_status(
        backend, req.id, status="executing",
        expected_current_status="pending",
    )

    dispatcher = _CapturingDispatcher()
    agent = _make_agent(backend, dispatcher=dispatcher)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()

    assert len(dispatcher.signals) == 1
    payload = dispatcher.signals[0].payload
    assert payload["operation"] == "update_then_restart"
    assert payload["target_ref"] == "main"
    assert payload["resolved_ref"] == "cafef00d"
    assert payload["update_profile"] == "sovereign_local_uv_sync"


def test_is_valid_target_ref_guards():
    assert is_valid_target_ref("main")
    assert is_valid_target_ref("v1.2.3")
    assert is_valid_target_ref("feature/foo-bar")
    assert is_valid_target_ref("a1b2c3d4")
    assert not is_valid_target_ref("")
    assert not is_valid_target_ref("--upload-pack=x")
    assert not is_valid_target_ref("a..b")
    assert not is_valid_target_ref("foo; rm -rf /")


# ---------------------------------------------------------------------------
# Real-git profile behaviour (#1539) — the update profile must actually
# advance a branch checkout, not silently no-op. This exercises the real
# fetch/checkout/resolve_ref steps (install is skipped — no `uv sync` in a
# unit test) against a local bare-remote + clone, the kestrel-talon
# test_git_worktree.py pattern.
# ---------------------------------------------------------------------------


def _git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=str(cwd), check=True, capture_output=True,
    )


def _head(repo):
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.mark.asyncio
async def test_profile_checkout_lands_on_fetched_branch_commit(tmp_path):
    """A named ``git checkout main`` after a fetch stays on the STALE
    local commit (fetch updates origin/main, not local main); the profile
    must instead land the working checkout on the freshly-fetched commit.
    This is the headline use case — update local checkout to a branch ref
    then restart — and is invisible to the fully-mocked unit tests.
    """
    if shutil.which("git") is None:
        pytest.skip("git not available")

    remote = tmp_path / "remote.git"
    remote.mkdir()
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))

    # Seed main with commit v1 and push.
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", str(remote), str(seed))
    (seed / "file.txt").write_text("v1")
    _git(seed, "add", "file.txt")
    _git(seed, "commit", "-m", "v1")
    _git(seed, "push", "origin", "main")

    # The local checkout the coordinator will update — currently at v1.
    work = tmp_path / "work"
    _git(tmp_path, "clone", str(remote), str(work))
    head_before = _head(work)

    # A new commit v2 lands on remote main AFTER the local checkout exists.
    (seed / "file.txt").write_text("v2")
    _git(seed, "add", "file.txt")
    _git(seed, "commit", "-m", "v2")
    _git(seed, "push", "origin", "main")
    remote_head = _head(seed)
    assert remote_head != head_before

    feat, _ = await _make_feature(tmp_path)
    profile = get_update_profile("sovereign_local_uv_sync")
    steps = profile.build_steps(
        repo_path=str(work), target_ref="main", allow_migrations=False,
    )
    for step in steps:
        if step.name == "install":
            continue  # do not run `uv sync` in a unit test
        outcome = await feat._run_update_step(step)
        assert outcome["ok"], f"step {step.name!r} failed: {outcome}"

    head_after = _head(work)
    assert head_after == remote_head, (
        "checkout must land on the fetched commit; a named branch "
        "checkout would have left it at the stale local commit"
    )
    assert head_after != head_before


@pytest.mark.asyncio
async def test_boot_resets_interrupted_updating_row(tmp_path):
    """A row left in ``updating`` by a host that went down mid-update must
    be reset to ``pending`` for retry on the next boot — NOT reported as a
    completed restart (no restart.completed signal), since the update
    never finished and we never restarted into the new code.
    """
    backend = await _backend(tmp_path)
    req = await insert_request(
        backend,
        requested_by_agent="did:test:agent",
        reason="merged fix",
        operation="update_then_restart",
        update_profile="sovereign_local_uv_sync",
        update_target_ref="main",
        update_repo_path=str(tmp_path),
    )
    await update_status(
        backend, req.id, status="updating",
        expected_current_status="pending",
    )

    dispatcher = _CapturingDispatcher()
    agent = _make_agent(backend, dispatcher=dispatcher)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()

    row = await get_request(backend, req.id)
    assert row.status == "pending", (
        f"interrupted update must be retryable; got {row.status!r}"
    )
    assert "mid-update" in row.status_reason
    # No completion signal — nothing actually restarted.
    assert dispatcher.signals == []


@pytest.mark.asyncio
async def test_signal_source_registered_idempotent(tmp_path):
    """A second feature init must not double-register the signal
    source. The registry uses ``get`` to short-circuit.
    """
    backend = await _backend(tmp_path)
    registry = _StubRegistry()
    agent = _make_agent(backend, registry=registry)
    feat1 = RestartCoordinatorFeature(agent)
    await feat1.initialize()
    assert "restart.completed" in registry.by_name
    # Second init in the same process — no duplicate, no warning.
    feat2 = RestartCoordinatorFeature(agent)
    await feat2.initialize()
    # Still exactly one registration.
    assert len(registry.registered) == 1
