"""Host-provided ``stalled_work_rescue`` workflow sources (#2192).

Two layers of coverage:

1. Registration + handler behavior against the real SignalDispatcher — every
   source the built-in workflow names is registered as an ACTION source, and the
   handlers preserve evidence boundaries (quote observation before claims, fail
   closed on missing evidence, never infer merged/shipped/resolved state).
2. A controlled start of ``stalled_work_rescue`` through the real workflow runner
   (skipped when ``kestrel_feature_workflows`` is not installed): with the sources
   registered the start-contract validation passes and a run record is created,
   where before it failed with "references unregistered source".
"""

from __future__ import annotations

import pytest

from kestrel_sdk.signals import (
    Signal,
    SignalMode,
    Status,
    Trust,
    Urgency,
    Visibility,
)
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.workflow_rescue import (
    SOURCE_NAMES,
    build_workflow_rescue_registrations,
    register_workflow_rescue_sources,
)
from kestrel_sovereign.storage.db import SQLiteBackend


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_builds_every_source_the_builtin_names():
    regs = build_workflow_rescue_registrations()
    names = {r.name for r in regs}
    assert names == set(SOURCE_NAMES)
    # The built-in ``stalled_work_rescue`` references exactly these six.
    assert names == {
        "fleet_stalled_sweep",
        "governance_review",
        "a2a_repair_dispatch",
        "evidence_verify",
        "close_resolved_todos",
        "reopen_resolved_todos",
    }


def test_all_sources_are_action_mode_with_handlers():
    for reg in build_workflow_rescue_registrations():
        assert SignalMode.ACTION in reg.allowed_modes, reg.name
        assert reg.default_mode == SignalMode.ACTION, reg.name
        assert reg.handler is not None, reg.name
        assert reg.trust == Trust.TRUSTED, reg.name
        assert reg.log_redaction is not None, reg.name


def test_register_is_idempotent_and_reports_new_names():
    registry = SourceRegistry()
    first = register_workflow_rescue_sources(registry)
    assert set(first) == set(SOURCE_NAMES)
    for name in SOURCE_NAMES:
        assert registry.get(name) is not None
    # A second call registers nothing new (no duplicate-registration error).
    second = register_workflow_rescue_sources(registry)
    assert second == []


def test_register_skips_a_preexisting_source():
    registry = SourceRegistry()
    # Pretend the host already provided a richer fleet_stalled_sweep.
    registry.register(build_workflow_rescue_registrations()[0])
    registered = register_workflow_rescue_sources(registry)
    assert "fleet_stalled_sweep" not in registered
    assert set(registered) == set(SOURCE_NAMES) - {"fleet_stalled_sweep"}


def test_register_no_registry_is_noop():
    assert register_workflow_rescue_sources(None) == []


# ---------------------------------------------------------------------------
# Handler behavior through the real dispatcher
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Minimal agent stub: the dispatcher fires log writes through
    ``_track_background_task`` (ACTION sources never touch the constitution
    path, so nothing else is needed)."""

    def __init__(self):
        self.background_tasks: list = []

    def _track_background_task(self, coro, *, name):
        import asyncio

        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


@pytest.fixture
async def dispatcher(tmp_path):
    import asyncio

    backend = SQLiteBackend(str(tmp_path / "rescue.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    register_workflow_rescue_sources(registry)
    agent = _FakeAgent()
    disp = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=store,
    )
    try:
        yield disp
    finally:
        pending = [t for t in agent.background_tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await backend.close()


def _signal(source: str, payload: dict) -> Signal:
    return Signal(
        source=source,
        kind="workflow.stage",
        mode=SignalMode.ACTION,
        payload=payload,
        target_agent="did:web:k.example",
        visibility=Visibility.INTERNAL,
        urgency=Urgency.NORMAL,
    )


async def test_fleet_stalled_sweep_observes_and_quotes(dispatcher):
    result = await dispatcher.dispatch_signal(
        _signal("fleet_stalled_sweep", {"stale_days": 3, "stalled_items": ["#1", "#2"]})
    )
    assert result.status == Status.OK
    data = result.action_result
    assert data["stalled_count"] == 2
    # Observation is quoted verbatim before any downstream claim.
    assert "`fleet_stalled_sweep` reported `stalled_count: 2`." == data["observation"]


async def test_fleet_stalled_sweep_zero_is_ok(dispatcher):
    result = await dispatcher.dispatch_signal(_signal("fleet_stalled_sweep", {}))
    assert result.status == Status.OK
    assert result.action_result["stalled_count"] == 0


async def test_governance_review_records_intent_not_authorization(dispatcher):
    result = await dispatcher.dispatch_signal(
        _signal("governance_review", {"scope": "proactive_work_rescue", "stalled_count": 2})
    )
    assert result.status == Status.OK
    data = result.action_result
    assert data["intent"] == "request_consent"
    # The source records intent only; approval is the consent gate's job.
    assert data["authorized"] is False


async def test_governance_review_preserves_explicit_consent_marker(dispatcher):
    result = await dispatcher.dispatch_signal(
        _signal(
            "governance_review",
            {
                "scope": "proactive_work_rescue",
                "stalled_count": 2,
                "approved": True,
                "approved_by": "did:web:k.operator",
            },
        )
    )
    assert result.status == Status.OK
    data = result.action_result
    assert data["approved"] is True
    assert data["approved_by"] == "did:web:k.operator"
    assert data["authorized"] is False


async def test_a2a_repair_dispatch_fails_closed_without_targets(dispatcher):
    result = await dispatcher.dispatch_signal(_signal("a2a_repair_dispatch", {}))
    assert result.status == Status.FAILED


async def test_a2a_repair_dispatch_records_dispatched_not_merged(dispatcher):
    result = await dispatcher.dispatch_signal(
        _signal("a2a_repair_dispatch", {"repairs": ["#1", "#2"]})
    )
    assert result.status == Status.OK
    data = result.action_result
    assert data["dispatched_count"] == 2
    # Dispatch must NOT be reported as merged/shipped — only dispatched.
    assert data["state"] == "dispatched"
    assert "merged" not in data
    assert "shipped" not in data


async def test_evidence_verify_fails_closed_without_evidence(dispatcher):
    result = await dispatcher.dispatch_signal(_signal("evidence_verify", {}))
    assert result.status == Status.FAILED


async def test_evidence_verify_ok_only_on_real_evidence(dispatcher):
    result = await dispatcher.dispatch_signal(
        _signal("evidence_verify", {"evidence": {"ci_status": "green"}})
    )
    assert result.status == Status.OK
    data = result.action_result
    assert data["verified"] is True
    assert "`evidence_verify` reported `evidence:" in data["observation"]


async def test_close_resolved_fails_closed_without_evidence(dispatcher):
    # No resolved todos at all.
    assert (
        await dispatcher.dispatch_signal(_signal("close_resolved_todos", {}))
    ).status == Status.FAILED
    # A todo that carries no resolution evidence must not be closed.
    result = await dispatcher.dispatch_signal(
        _signal("close_resolved_todos", {"resolved_todos": [{"id": 7}]})
    )
    assert result.status == Status.FAILED


async def test_close_resolved_closes_only_with_evidence(dispatcher):
    result = await dispatcher.dispatch_signal(
        _signal(
            "close_resolved_todos",
            {"resolved_todos": [{"id": 7, "evidence": {"pr": 42, "merged_sha": "abc"}}]},
        )
    )
    assert result.status == Status.OK
    assert result.action_result["closed"] == [7]


async def test_reopen_is_idempotent_noop(dispatcher):
    result = await dispatcher.dispatch_signal(
        _signal("reopen_resolved_todos", {"closed": [7, 8]})
    )
    assert result.status == Status.OK
    assert result.action_result["reopened"] == [7, 8]


# ---------------------------------------------------------------------------
# Controlled start of the built-in workflow through the real runner
# ---------------------------------------------------------------------------


async def _make_runner(tmp_path, registry):
    """Build a real WorkflowRunner backed by SQLite over ``registry``."""
    wf_runner = pytest.importorskip("kestrel_feature_workflows.runner")
    from kestrel_feature_workflows.store import WorkflowStore
    from kestrel_sovereign.identity.runtime_identity import AgentIdentity
    from kestrel_sovereign.security.crypto_suite import (
        ALG_ECDSA_SECP256K1_SHA256,
        get_suite,
    )

    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    identity = AgentIdentity(
        legacy_did="did:web:k.example",
        legacy_keypair=suite.generate_keypair(),
        legacy_did_document={},
    )

    def resolve_public_key(did: str) -> bytes:
        if did != identity.legacy_did:
            raise KeyError(did)
        return suite.serialize_public_key(identity.legacy_keypair.public_key)

    def resolve_verification_methods(did: str) -> list:
        raise KeyError(did)

    backend = SQLiteBackend(str(tmp_path / "wf.db"))
    await backend.connect()
    signal_store = SignalLogStore(backend)
    await signal_store.initialize()
    store = WorkflowStore(backend)
    await store.initialize()
    dispatcher = SignalDispatcher(
        agent=None,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=signal_store,
    )
    runner = wf_runner.WorkflowRunner(
        store=store,
        dispatcher=dispatcher,
        registry=registry,
        agent_identity=identity,
        public_key_resolver=resolve_public_key,
        verification_methods_resolver=resolve_verification_methods,
    )
    return runner, store, identity, backend


async def _define_builtin(store, identity):
    from kestrel_feature_workflows.library import get_builtin_workflow
    from kestrel_feature_workflows.signing import sign_workflow_spec

    spec = get_builtin_workflow("stalled_work_rescue")
    signed = sign_workflow_spec(spec, identity)
    await store.put_definition(signed)


async def test_controlled_start_creates_a_run_with_sources_registered(tmp_path):
    pytest.importorskip("kestrel_feature_workflows")
    registry = SourceRegistry()
    register_workflow_rescue_sources(registry)
    runner, store, identity, backend = await _make_runner(tmp_path, registry)
    try:
        await _define_builtin(store, identity)
        run = await runner.start_run(name="stalled_work_rescue", params={})
        assert run.run_id
        # A run record is now persisted (the pre-fix bug created none).
        assert await store.get_run(run.run_id) is not None
    finally:
        await backend.close()


async def test_start_fails_closed_when_sources_are_missing(tmp_path):
    pytest.importorskip("kestrel_feature_workflows")
    from kestrel_feature_workflows.runner import WorkflowRunnerError

    registry = SourceRegistry()  # deliberately empty — no rescue sources
    runner, store, identity, backend = await _make_runner(tmp_path, registry)
    try:
        await _define_builtin(store, identity)
        with pytest.raises(WorkflowRunnerError, match="unregistered source"):
            await runner.start_run(name="stalled_work_rescue", params={})
        # And no run record leaked from the failed start.
        assert await store.list_runs(limit=10) == []
    finally:
        await backend.close()
