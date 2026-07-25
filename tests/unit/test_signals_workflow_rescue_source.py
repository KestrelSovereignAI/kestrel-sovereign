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
    RECURRING_CONSENT_SCOPE,
    RECURRING_DEFAULT_CRON,
    RECURRING_SCHEDULE_TASK_NAME,
    RECURRING_WORKFLOW_NAME,
    SOURCE_NAMES,
    UnsafeRecurringScheduleError,
    assert_safe_recurring_params,
    build_fleet_stalled_sweep_registration,
    build_recurring_schedule_request,
    build_workflow_rescue_registrations,
    is_safe_recurring_params,
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

    did = "did:web:k.example"

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


async def test_a2a_repair_dispatch_fails_closed_with_only_stalled_items(dispatcher):
    # Detected stalled_items must NOT auto-become dispatch targets: a recurring
    # loop that forwards the observation stage's output cannot dispatch A2A
    # repairs without explicit per-run repair targets + fresh approval (#2200,
    # acceptance criterion 2).
    result = await dispatcher.dispatch_signal(
        _signal("a2a_repair_dispatch", {"stalled_items": ["#1", "#2"]})
    )
    assert result.status == Status.FAILED


async def test_a2a_repair_dispatch_recurring_tick_no_targets_is_clean_noop(dispatcher):
    # A recurring observation-only tick reaches dispatch with no per-run approval
    # having selected any target. It must complete cleanly (no-op), not fail the
    # unattended run (#2249). Direct fail-closed calls above are unaffected.
    result = await dispatcher.dispatch_signal(
        _signal("a2a_repair_dispatch", {"recurring": True, "stale_days": 3})
    )
    assert result.status == Status.OK
    data = result.action_result
    assert data["skipped"] is True
    assert data["dispatched_count"] == 0
    assert data["state"] == "skipped"
    # A no-op must never look like real work — no merged/shipped/dispatched list.
    assert "merged" not in data
    assert "shipped" not in data


async def test_a2a_repair_dispatch_recurring_with_stalled_items_still_no_dispatch(
    dispatcher,
):
    # Even carrying detected stalled_items, a recurring tick with no explicit
    # repair targets must not dispatch them — it skips cleanly (#2249 + #2200).
    result = await dispatcher.dispatch_signal(
        _signal(
            "a2a_repair_dispatch",
            {"recurring": True, "stalled_items": ["#1", "#2"]},
        )
    )
    assert result.status == Status.OK
    data = result.action_result
    assert data["skipped"] is True
    assert data["dispatched_count"] == 0
    # The detected items were NOT promoted to dispatched targets.
    assert data.get("dispatched", []) == []


async def test_a2a_repair_dispatch_recurring_with_targets_still_dispatches(dispatcher):
    # When a per-run approval DOES select targets mid-loop, a recurring tick
    # dispatches them normally — the no-op branch only triggers on empty targets.
    result = await dispatcher.dispatch_signal(
        _signal("a2a_repair_dispatch", {"recurring": True, "repairs": ["#9"]})
    )
    assert result.status == Status.OK
    data = result.action_result
    assert data["dispatched_count"] == 1
    assert data["state"] == "dispatched"
    assert data.get("skipped") is not True


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


async def test_evidence_verify_recurring_tick_no_evidence_is_clean_noop(dispatcher):
    # A recurring tick that dispatched nothing has nothing to verify — it must
    # complete cleanly, not fail closed (#2249).
    result = await dispatcher.dispatch_signal(
        _signal("evidence_verify", {"recurring": True})
    )
    assert result.status == Status.OK
    data = result.action_result
    assert data["skipped"] is True
    # Must not claim anything was verified.
    assert data.get("verified") is not True


async def test_evidence_verify_recurring_with_dispatched_repairs_still_fails_closed(
    dispatcher,
):
    # #2249 P1: the runner merges run params into every stage payload, so a
    # recurring run that DID select explicit repair targets carries them into
    # evidence_verify. A real dispatch happened — the recurring no-op branch must
    # NOT skip verification. Without evidence this must fail closed, not pass as a
    # no-op (which would complete the run with an unverified irreversible action).
    result = await dispatcher.dispatch_signal(
        _signal("evidence_verify", {"recurring": True, "repairs": ["#9"]})
    )
    assert result.status == Status.FAILED


async def test_evidence_verify_recurring_with_dispatched_marker_fails_closed(
    dispatcher,
):
    # The dispatch stage's forwarded output (dispatched/dispatched_count) also
    # marks a real action — a recurring tick that sees it must be proven, not
    # skipped.
    result = await dispatcher.dispatch_signal(
        _signal(
            "evidence_verify",
            {"recurring": True, "dispatched": ["#9"], "dispatched_count": 1},
        )
    )
    assert result.status == Status.FAILED


async def test_evidence_verify_recurring_after_noop_dispatch_still_skips(dispatcher):
    # A recurring tick whose dispatch was itself a no-op forwards
    # ``skipped: True`` / ``dispatched_count: 0`` — that is not a selected action,
    # so evidence_verify may still complete cleanly as a no-op.
    result = await dispatcher.dispatch_signal(
        _signal(
            "evidence_verify",
            {"recurring": True, "skipped": True, "dispatched_count": 0},
        )
    )
    assert result.status == Status.OK
    assert result.action_result["skipped"] is True


async def test_close_resolved_recurring_with_dispatched_repairs_fails_closed(
    dispatcher,
):
    # #2249 P1 mirror for the close stage: a recurring run that selected explicit
    # repair targets must not slip past the close gate as a no-op.
    result = await dispatcher.dispatch_signal(
        _signal("close_resolved_todos", {"recurring": True, "repairs": ["#9"]})
    )
    assert result.status == Status.FAILED


async def test_close_resolved_recurring_tick_no_todos_is_clean_noop(dispatcher):
    # A recurring tick that resolved nothing has nothing to close — clean no-op,
    # not a failed run (#2249).
    result = await dispatcher.dispatch_signal(
        _signal("close_resolved_todos", {"recurring": True})
    )
    assert result.status == Status.OK
    data = result.action_result
    assert data["skipped"] is True
    assert data["closed_count"] == 0


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
# Safe recurring scheduling (#2200)
# ---------------------------------------------------------------------------


import json


def test_recurring_schedule_request_shape_defaults():
    req = build_recurring_schedule_request()
    # Maps 1:1 onto SchedulerFeature.schedule_add kwargs.
    assert set(req) == {"cron_expression", "task_name", "args_json"}
    assert req["cron_expression"] == RECURRING_DEFAULT_CRON
    # Runs the schedulable workflows-feature tool, not a bespoke task name.
    assert req["task_name"] == RECURRING_SCHEDULE_TASK_NAME == "workflow_run"
    args = json.loads(req["args_json"])
    assert args["name"] == RECURRING_WORKFLOW_NAME == "stalled_work_rescue"
    # Observation-only params: no repair targets, no evidence, no approval.
    assert args["params"] == {"stale_days": 3, "recurring": True}


def test_recurring_schedule_request_custom_cron_and_stale_days():
    req = build_recurring_schedule_request(cron="0 3 * * *", stale_days=7)
    assert req["cron_expression"] == "0 3 * * *"
    assert json.loads(req["args_json"])["params"]["stale_days"] == 7


def test_recurring_default_params_are_safe():
    assert is_safe_recurring_params({"stale_days": 3, "recurring": True})
    assert assert_safe_recurring_params(None) == {}
    assert assert_safe_recurring_params({}) == {}


@pytest.mark.parametrize(
    "unsafe",
    [
        {"repairs": ["#1"]},
        {"repair_targets": ["#1"]},
        {"resolved_todos": [{"id": 7}]},
        {"evidence": {"ci_status": "green"}},
    ],
)
def test_recurring_rejects_preseeded_irreversible_targets(unsafe):
    # No A2A dispatch / close may be pre-authorized by a blanket schedule —
    # targets and evidence are supplied per-run (criteria 2 & 3).
    assert not is_safe_recurring_params(unsafe)
    with pytest.raises(UnsafeRecurringScheduleError):
        assert_safe_recurring_params(unsafe)
    with pytest.raises(UnsafeRecurringScheduleError):
        build_recurring_schedule_request(extra_params=unsafe)


@pytest.mark.parametrize(
    "marker",
    [
        {"approved": True},
        {"approved_by": "did:web:k.operator"},
        {"consent": True},
        {"decision": "approve"},
        {"approval_id": "req-1"},
    ],
)
def test_recurring_rejects_blanket_approval_marker(marker):
    # Consent is collected fresh per run by the govern_intent gate; a schedule
    # must never carry a standing approval (criterion 1: no blanket approval).
    assert not is_safe_recurring_params(marker)
    with pytest.raises(UnsafeRecurringScheduleError):
        build_recurring_schedule_request(extra_params=marker)


def test_recurring_scope_matches_consent_gate():
    # The recurring loop's consent scope must match the built-in gate's scope
    # (kestrel_feature_workflows library _stalled_work_rescue govern_intent).
    assert RECURRING_CONSENT_SCOPE == "proactive_work_rescue"


async def test_fleet_stalled_sweep_discovers_without_preseeded_items():
    # Integration-style: the scheduled workflow_run input (observation-only
    # params, no pre-seeded repair targets) yields non-empty observed candidates
    # via live discovery (#2200, acceptance criterion 1 + P2 review finding).
    req = build_recurring_schedule_request(stale_days=5)
    params = json.loads(req["args_json"])["params"]
    assert params == {"stale_days": 5, "recurring": True}

    seen_stale_days = []

    async def discover(stale_days):
        seen_stale_days.append(stale_days)
        return [{"id": "job-42", "kind": "talon_job", "status": "running"}]

    reg = build_fleet_stalled_sweep_registration(discover)
    result = await reg.handler(params)
    assert result["stalled_count"] == 1
    assert result["discovered"] is True
    assert result["stalled_items"] == [
        {"id": "job-42", "kind": "talon_job", "status": "running"}
    ]
    # The stale-days threshold from the schedule reaches the survey.
    assert seen_stale_days == [5]


async def test_fleet_stalled_sweep_preseeded_items_skip_discovery():
    called = False

    async def discover(stale_days):
        nonlocal called
        called = True
        return [{"id": "should-not-appear"}]

    reg = build_fleet_stalled_sweep_registration(discover)
    result = await reg.handler({"stalled_items": ["#1"]})
    assert result["stalled_count"] == 1
    assert result["discovered"] is False
    assert called is False


async def test_fleet_stalled_sweep_discovery_error_degrades_to_zero():
    async def discover(stale_days):
        raise RuntimeError("survey backend unavailable")

    reg = build_fleet_stalled_sweep_registration(discover)
    result = await reg.handler({"stale_days": 3, "recurring": True})
    # Observation degrades to "observed nothing" rather than aborting the loop.
    assert result["stalled_count"] == 0
    assert result["discovered"] is False


def test_recurring_falsey_marker_values_are_allowed():
    # A benign falsey value for a marker key does not trip the guard — only a
    # truthy standing approval / pre-seeded target is unsafe.
    assert is_safe_recurring_params({"approved": False, "repairs": []})


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
