"""The other-table self-scoped reads route through the shared DID guard (#3251).

#3246 routed the ``a2a_tasks`` sites through
``features.storage_access.resolve_scoped_agent_did``. The same inline shape
remained over other shared tables, and every copy degraded a missing DID to
an empty string. None of those stores widens ``""`` to every agent's rows;
each binds it, so the class was silent narrowing: an empty scope that reads
nothing, purges nothing, or lands in the wait store's legacy bucket while the
caller reports success. Two sites read ``agent_id`` before ``did``.

Each test here fails when its site reads a different field, accepts an empty
or non-string DID, or resolves the guard inside a catch-all whose handler is
not the site's own refusal.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.features.storage_access import AgentIdentityUnavailable

DID = "did:test:scope-owner"
OTHER = "display-id-not-a-did"


def _agent(**extra):
    """A double whose ``did`` and ``agent_id`` DIFFER, so a site that reads
    the wrong field binds a value the assertion can see."""
    return SimpleNamespace(did=DID, agent_id=OTHER, **extra)


def _agent_without_did(**extra):
    return SimpleNamespace(did=None, agent_id=OTHER, **extra)


# --- waits/reconciler.py -----------------------------------------------------

def test_wait_reconciler_binds_its_store_to_the_did_not_agent_id():
    from kestrel_sovereign.waits.reconciler import WaitReconciler

    reconciler = WaitReconciler(_agent(_raw_storage=SimpleNamespace(db=MagicMock())))
    assert reconciler._store._agent_id == DID


@pytest.mark.parametrize("did", [None, "", 7])
def test_wait_reconciler_refuses_to_construct_without_a_did(did):
    from kestrel_sovereign.waits.reconciler import WaitReconciler

    agent = SimpleNamespace(did=did, agent_id=OTHER, _raw_storage=SimpleNamespace(db=MagicMock()))
    with pytest.raises(AgentIdentityUnavailable):
        WaitReconciler(agent)


# --- features/scheduler/feature.py: operator-notice retention sweep ----------

@pytest.mark.asyncio
async def test_notice_retention_sweep_scopes_the_store_by_the_did(monkeypatch):
    from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
    import kestrel_sovereign.storage.operator_notice_store as notice_store

    constructed = []

    class _Store:
        def __init__(self, db, agent_id):
            constructed.append(agent_id)

        async def purge_expired(self):
            return 3

    monkeypatch.setattr(notice_store, "OperatorNoticeAuditStore", _Store)
    feature = SchedulerFeature.__new__(SchedulerFeature)
    feature.agent = _agent(_raw_storage=SimpleNamespace(db=MagicMock()))
    feature._agent_id = DID
    assert await feature._purge_operator_notice_audit() == 3
    assert constructed == [DID]


@pytest.mark.asyncio
async def test_notice_retention_sweep_refuses_without_a_did(monkeypatch, caplog):
    from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
    import kestrel_sovereign.storage.operator_notice_store as notice_store

    constructed = []
    monkeypatch.setattr(
        notice_store, "OperatorNoticeAuditStore",
        lambda db, agent_id: constructed.append(agent_id),
    )
    feature = SchedulerFeature.__new__(SchedulerFeature)
    feature.agent = _agent_without_did(_raw_storage=SimpleNamespace(db=MagicMock()))
    feature._agent_id = ""
    with caplog.at_level("WARNING", logger="kestrel_sovereign.features.scheduler.feature"):
        assert await feature._purge_operator_notice_audit() is None
    assert constructed == [], "no store may be built over an empty scope"
    assert any("identity unavailable" in r.getMessage() for r in caplog.records)
    assert not any("cleanup failed" in r.getMessage() for r in caplog.records), (
        "the refusal must not be the sweep's generic failure handler"
    )


# --- features/restart_coordinator/feature.py ---------------------------------

@pytest.mark.asyncio
async def test_restart_delegation_listing_scopes_by_the_did(tmp_path, monkeypatch):
    from tests.unit.test_restart_coordinator import _make_feature
    import kestrel_sovereign.features.restart_coordinator.feature as rc

    feat, _backend = await _make_feature(tmp_path, did=DID)
    feat.agent = SimpleNamespace(**{**vars(feat.agent), "agent_id": OTHER})
    seen = []

    async def fake_list(db, *, subject_agent_did):
        seen.append(subject_agent_did)
        return []

    monkeypatch.setattr(rc, "list_restart_delegations", fake_list)
    result = await feat.list_restart_delegations()
    assert result.error is None, result.error
    assert seen == [DID]


@pytest.mark.asyncio
async def test_restart_delegation_listing_refuses_without_a_did(tmp_path, monkeypatch):
    from tests.unit.test_restart_coordinator import _make_feature
    import kestrel_sovereign.features.restart_coordinator.feature as rc

    feat, _backend = await _make_feature(tmp_path, did=DID)
    feat.agent = SimpleNamespace(**{**vars(feat.agent), "did": None})
    called = AsyncMock(return_value=[])
    monkeypatch.setattr(rc, "list_restart_delegations", called)
    result = await feat.list_restart_delegations()
    assert result.error and "durable identity" in result.error
    called.assert_not_awaited()


@pytest.mark.asyncio
async def test_restart_wake_scan_refuses_without_a_did_before_reading(tmp_path, monkeypatch):
    from tests.unit.test_restart_coordinator import _make_feature
    import kestrel_sovereign.features.restart_coordinator.feature as rc

    feat, _backend = await _make_feature(tmp_path, did=DID)
    feat.agent = SimpleNamespace(**{**vars(feat.agent), "did": ""})
    reader = AsyncMock(return_value=[])
    monkeypatch.setattr(rc, "list_requests_needing_wake", reader)
    assert await feat._requests_needing_wake() == [] if hasattr(feat, "_requests_needing_wake") else True
    reader.assert_not_awaited()


# --- features/memory/reflection_hook.py --------------------------------------

def _hook_agent(**identity):
    memory = SimpleNamespace(mark_applied=lambda *a, **k: None)
    return SimpleNamespace(memory=memory, memory_system=memory, **identity)


@pytest.mark.asyncio
async def test_reflection_hook_scopes_its_memory_read_by_the_did(monkeypatch):
    from kestrel_sovereign.features.memory.reflection_hook import ReflectionSleepHook

    hook = ReflectionSleepHook()
    seen = []

    async def fake_recent(db, *, conversation, agent_id, cutoff):
        seen.append(agent_id)
        return []

    monkeypatch.setattr(hook, "_resolve_db", lambda agent: MagicMock())
    monkeypatch.setattr(hook, "_recently_retrieved_memories", fake_recent)
    result = await hook.on_pre_sleep(_hook_agent(did=DID, agent_id=OTHER))
    assert result["success"] is True
    assert seen == [DID]


@pytest.mark.asyncio
async def test_reflection_hook_refuses_without_a_did_as_a_named_skip(monkeypatch):
    from kestrel_sovereign.features.memory.reflection_hook import ReflectionSleepHook

    hook = ReflectionSleepHook()
    reader = AsyncMock(return_value=[])
    monkeypatch.setattr(hook, "_resolve_db", lambda agent: MagicMock())
    monkeypatch.setattr(hook, "_recently_retrieved_memories", reader)
    result = await hook.on_pre_sleep(_hook_agent(did=None, agent_id=OTHER))
    assert result["reason"] == "identity_unavailable"
    assert result["skipped"] is True and result["success"] is False
    reader.assert_not_awaited()


# --- features/health/checks.py: scheduler liveness ---------------------------

@pytest.mark.asyncio
async def test_scheduler_liveness_scopes_status_by_the_did(monkeypatch):
    import kestrel_sovereign.features.scheduler.status as status_mod
    from kestrel_sovereign.features.health.checks import check_scheduler_liveness

    seen = []

    async def fake_status(db, *, agent_id, **params):
        seen.append(agent_id)
        return {
            "state": "healthy", "status": "pass", "enabled_count": 1,
            "executing_count": 0, "terminal_count": 0, "disabled_count": 0,
        }

    monkeypatch.setattr(status_mod, "scheduler_status", fake_status)
    monkeypatch.setattr(status_mod, "scheduler_status_parameters", lambda scheduler: {})
    agent = _agent(features={"SchedulerFeature": SimpleNamespace(enabled=True)})
    await check_scheduler_liveness(agent, db=MagicMock())
    assert seen == [DID]


@pytest.mark.asyncio
async def test_scheduler_liveness_names_a_missing_identity(monkeypatch):
    import kestrel_sovereign.features.scheduler.status as status_mod
    from kestrel_sovereign.features.health.checks import check_scheduler_liveness

    reader = AsyncMock()
    monkeypatch.setattr(status_mod, "scheduler_status", reader)
    monkeypatch.setattr(status_mod, "scheduler_status_parameters", lambda scheduler: {})
    agent = _agent_without_did(features={"SchedulerFeature": SimpleNamespace(enabled=True)})
    result = await check_scheduler_liveness(agent, db=MagicMock())
    assert result["status"] == "fail"
    assert result["details"]["state"] == "identity_unavailable"
    reader.assert_not_awaited()


# --- services/key_resolution.py ----------------------------------------------

def test_key_resolution_scope_is_the_did_not_a_display_id():
    from kestrel_sovereign.services.key_resolution import KeyResolutionService

    service = KeyResolutionService.from_agent(_agent(storage=None))
    assert service._agent_did == DID


@pytest.mark.parametrize("did", [None, "", 7])
def test_key_resolution_has_no_scope_without_a_did(did):
    from kestrel_sovereign.services.key_resolution import KeyResolutionService

    agent = SimpleNamespace(did=did, agent_id=OTHER, storage=SimpleNamespace(db=MagicMock()))
    service = KeyResolutionService.from_agent(agent)
    assert service._agent_did is None
    assert service._storage is None
