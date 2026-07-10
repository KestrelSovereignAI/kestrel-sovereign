"""Built-in ``fleet_coding_pipeline`` workflow + its host sources (#2303).

Three layers of coverage:

1. The WorkflowSpec definition — stage/gate shape, the irreversible talon stage,
   the consent gate, the ci_green verify gate — and that it validates against the
   wire schema and injects into the workflows package's built-in registry
   (no package file edited).
2. The two support sources (``fleet_coding_approval``, ``fleet_ci_probe``) as
   registered ACTION sources with evidence-preserving handlers.
3. A controlled start of the built-in through the real workflow runner
   (skipped when ``kestrel_feature_workflows`` is absent): the consent gate parks
   the run in WAITING; ``{repo, issue}`` args thread through to the
   ``talon_pipeline_dispatch`` params; and the observability correlation keys
   hold through the definition's ``talon_run`` stage.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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
from kestrel_sovereign.signals.sources.fleet_coding_pipeline import (
    APPROVE_DISPATCH_STAGE,
    CONSENT_SCOPE,
    FLEET_CI_PROBE,
    FLEET_CODING_APPROVAL,
    SOURCE_NAMES,
    TALON_RUN_STAGE,
    VERIFY_CI_STAGE,
    WORKFLOW_DESCRIPTION,
    WORKFLOW_NAME,
    build_fleet_coding_pipeline_registrations,
    build_fleet_coding_pipeline_spec,
    register_fleet_coding_pipeline_builtin,
    register_fleet_coding_pipeline_sources,
)
from kestrel_sovereign.signals.sources.talon_pipeline import (
    SOURCE_NAME as TALON_PIPELINE_SOURCE,
    register_talon_pipeline_source,
)
from kestrel_sovereign.storage.db import SQLiteBackend

wf = pytest.importorskip("kestrel_feature_workflows")


# ---------------------------------------------------------------------------
# WorkflowSpec definition
# ---------------------------------------------------------------------------


def test_spec_stage_shape():
    spec = build_fleet_coding_pipeline_spec()
    assert [s.name for s in spec.stages] == [
        APPROVE_DISPATCH_STAGE,
        TALON_RUN_STAGE,
        VERIFY_CI_STAGE,
    ]
    by_name = {s.name: s for s in spec.stages}

    # Approve — human consent before any code is written (default-closed).
    approve = by_name[APPROVE_DISPATCH_STAGE]
    assert approve.signal_source == FLEET_CODING_APPROVAL
    assert approve.gate.type == "consent_collect"
    assert approve.gate.params["scope"] == CONSENT_SCOPE

    # Dispatch — the actual talon coding run, irreversible + record-only.
    talon = by_name[TALON_RUN_STAGE]
    assert talon.signal_source == TALON_PIPELINE_SOURCE
    assert talon.irreversible is True
    assert talon.compensate == "compensate_record_only"
    # wait: false (durable cli_background, no 3600s cap) + self_review on.
    assert talon.params["wait"] is False
    assert talon.params["self_review"] is True

    # Verify — the talon PR's CI is green. A plain signal_status_ok gate over
    # the coordinator-bound probe, which binds to the dispatched run's own
    # output (never a caller-supplied branch, #2303).
    verify = by_name[VERIFY_CI_STAGE]
    assert verify.signal_source == FLEET_CI_PROBE
    assert verify.gate.type == "signal_status_ok"
    assert verify.read_only is True


def test_manual_trigger_only():
    spec = build_fleet_coding_pipeline_spec()
    kinds = {t.kind.value for t in spec.triggers}
    assert kinds == {"manual"}


def test_autonomous_variant_omits_consent_stage():
    spec = build_fleet_coding_pipeline_spec(autonomous=True)
    names = [s.name for s in spec.stages]
    assert APPROVE_DISPATCH_STAGE not in names
    assert names == [TALON_RUN_STAGE, VERIFY_CI_STAGE]


@pytest.mark.parametrize("autonomous", [False, True])
def test_spec_validates_against_wire_schema(autonomous):
    from kestrel_feature_workflows.schema import validate_spec_payload

    validate_spec_payload(build_fleet_coding_pipeline_spec(autonomous).to_dict())


def test_spec_round_trips_through_the_wire_form():
    from kestrel_feature_workflows.models import WorkflowSpec

    spec = build_fleet_coding_pipeline_spec()
    payload = spec.to_dict()
    assert WorkflowSpec.from_dict(payload).to_dict() == payload


# ---------------------------------------------------------------------------
# Built-in registry injection (no kestrel-feature-workflows file edited)
# ---------------------------------------------------------------------------


def test_builtin_injection_is_discoverable_and_idempotent():
    from kestrel_feature_workflows.library import (
        BUILTIN_BUILDERS,
        BUILTIN_DESCRIPTIONS,
        get_builtin_workflow,
        list_builtin_workflows,
    )

    # Idempotent: pre-existing (from a prior test/import) reports False, absent
    # reports True. Either way it ends up registered exactly once.
    BUILTIN_BUILDERS.pop(WORKFLOW_NAME, None)
    BUILTIN_DESCRIPTIONS.pop(WORKFLOW_NAME, None)
    assert register_fleet_coding_pipeline_builtin() is True
    assert register_fleet_coding_pipeline_builtin() is False

    assert WORKFLOW_NAME in list_builtin_workflows()
    assert BUILTIN_DESCRIPTIONS[WORKFLOW_NAME] == WORKFLOW_DESCRIPTION
    # Loads through the package's canonical path with the default (consent) shape.
    loaded = get_builtin_workflow(WORKFLOW_NAME)
    assert loaded.stages[0].name == APPROVE_DISPATCH_STAGE


# ---------------------------------------------------------------------------
# Support sources
# ---------------------------------------------------------------------------


def test_builds_exactly_the_support_sources():
    regs = build_fleet_coding_pipeline_registrations()
    assert {r.name for r in regs} == set(SOURCE_NAMES)
    assert set(SOURCE_NAMES) == {FLEET_CODING_APPROVAL, FLEET_CI_PROBE}


def test_support_sources_are_action_mode_with_handlers():
    for reg in build_fleet_coding_pipeline_registrations():
        assert SignalMode.ACTION in reg.allowed_modes, reg.name
        assert reg.default_mode == SignalMode.ACTION, reg.name
        assert reg.handler is not None, reg.name
        assert reg.trust == Trust.TRUSTED, reg.name


def test_register_support_sources_idempotent():
    registry = SourceRegistry()
    first = register_fleet_coding_pipeline_sources(registry)
    assert set(first) == set(SOURCE_NAMES)
    assert register_fleet_coding_pipeline_sources(registry) == []


def test_register_support_sources_no_registry_is_noop():
    assert register_fleet_coding_pipeline_sources(None) == []


class _FakeAgent:
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

    backend = SQLiteBackend(str(tmp_path / "fleet.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    coordinator = _FakeCoordinator()
    register_fleet_coding_pipeline_sources(registry, coordinator=coordinator)
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


async def test_approval_records_intent_not_authorization(dispatcher):
    result = await dispatcher.dispatch_signal(
        _signal(FLEET_CODING_APPROVAL, {"repo": "org/repo", "issue": 5})
    )
    assert result.status == Status.OK
    data = result.action_result
    assert data["intent"] == "request_consent"
    # The source records intent only; approval is the consent gate's job.
    assert data["authorized"] is False
    assert data["scope"] == CONSENT_SCOPE


async def test_approval_ignores_caller_supplied_consent_markers(dispatcher):
    # A caller who can start the workflow must NOT be able to self-approve by
    # putting approval-ish fields in the run params (which the runner merges into
    # the stage payload). The handler strips every consent-marker field, so none
    # can reach the consent_collect gate as a self-granted approval (#2303).
    from kestrel_sovereign.signals.sources.fleet_coding_pipeline import (
        CONSENT_MARKER_FIELDS,
    )

    result = await dispatcher.dispatch_signal(
        _signal(
            FLEET_CODING_APPROVAL,
            {
                "repo": "org/repo",
                "issue": 5,
                "approved": True,
                "consent": "granted",
                "status": "approved",
                "approved_by": "did:web:k.orchestrator",
            },
        )
    )
    data = result.action_result
    # None of the caller-supplied approval markers survived into the result.
    for field in CONSENT_MARKER_FIELDS:
        assert field not in data, field
    # It still records the intent, never authorization.
    assert data["intent"] == "request_consent"
    assert data["authorized"] is False


async def test_ci_probe_binds_to_dispatch_ignoring_caller_branch(dispatcher):
    # The probe verifies the talon PR the dispatch produced. Even though the
    # caller stuffs ``branch: attacker`` into the payload, the probe never
    # forwards it — verification uses the talon PR head the coordinator reports.
    result = await dispatcher.dispatch_signal(
        _signal(
            FLEET_CI_PROBE,
            {"repo": "org/repo", "issue": 9, "branch": "attacker-branch"},
        )
    )
    assert result.status == Status.OK
    data = result.action_result
    assert data["state"] == "green"
    # The verified branch is the talon PR head, NOT the caller-supplied branch.
    assert data["branch"] == "talon/fix"
    assert data["branch"] != "attacker-branch"


async def test_ci_probe_forwards_dispatched_job_id(tmp_path):
    # The dispatch stage's job_id flows here in the verify payload; the probe
    # must forward it to verify_pipeline_ci so verification binds to THIS run's
    # own talon job, not a (repo, issue) correlation a concurrent run could
    # poison (#2303).
    coord = _FakeCoordinator()
    registry = SourceRegistry()
    register_fleet_coding_pipeline_sources(registry, coordinator=coord)
    agent = _FakeAgent()
    backend = SQLiteBackend(str(tmp_path / "jobid.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    disp = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=store,
    )
    try:
        result = await disp.dispatch_signal(
            _signal(
                FLEET_CI_PROBE,
                {"repo": "org/repo", "issue": 9, "job_id": "job-abc"},
            )
        )
        assert result.status == Status.OK
        assert len(coord.verify_calls) == 1
        assert coord.verify_calls[0]["job_id"] == "job-abc"
    finally:
        await backend.close()


async def test_ci_probe_fails_closed_when_ci_not_green(tmp_path):
    # A verdict that is not green must make the probe fail closed (raise), so the
    # signal_status_ok gate never passes on an unverified PR.
    coord = _FakeCoordinator(
        ci_verdict={"ci_green": False, "reason": "pr_not_found"}
    )
    registry = SourceRegistry()
    register_fleet_coding_pipeline_sources(registry, coordinator=coord)
    agent = _FakeAgent()
    backend = SQLiteBackend(str(tmp_path / "probe.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    disp = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=store,
    )
    try:
        result = await disp.dispatch_signal(
            _signal(FLEET_CI_PROBE, {"repo": "org/repo", "issue": 9})
        )
        assert result.status != Status.OK
    finally:
        await backend.close()


# ---------------------------------------------------------------------------
# Controlled start through the real workflow runner
# ---------------------------------------------------------------------------


class _FakeCoordinator:
    """Records dispatch_pipeline kwargs; returns a dispatched cli_background job.

    Also stubs ``verify_pipeline_ci`` — the seam ``fleet_ci_probe`` delegates to.
    It records the kwargs it was called with (so a test can assert the probe
    never forwards a caller-supplied ``branch``) and returns a canned verdict
    whose ``branch`` is the talon PR head the dispatch produced, NOT any caller
    input (#2303).
    """

    def __init__(self, *, ci_verdict=None):
        self.calls: list = []
        self.verify_calls: list = []
        # Default: CI verified green against the talon PR head branch.
        self._ci_verdict = ci_verdict or {
            "ci_green": True,
            "repo": "org/repo",
            "pr": 4242,
            "pr_url": "https://github.com/org/repo/pull/4242",
            "branch": "talon/fix",
            "head_sha": "deadbeef",
            "job_id": "job-fleet-1",
            "reason": None,
        }

    async def dispatch_pipeline(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "dispatched": True,
            "job_id": "job-fleet-1",
            "method": "cli_background",
            "log_path": None,
        }

    async def verify_pipeline_ci(self, **kwargs):
        # Signature deliberately accepts no ``branch`` kwarg — if the probe ever
        # tried to forward a caller-supplied branch this would TypeError.
        self.verify_calls.append(kwargs)
        return dict(self._ci_verdict)


async def _make_runner(tmp_path, registry, *, consent_provider=None):
    from kestrel_feature_workflows.runner import WorkflowRunner
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
        agent=_FakeAgent(),
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=signal_store,
    )
    runner = WorkflowRunner(
        store=store,
        dispatcher=dispatcher,
        registry=registry,
        agent_identity=identity,
        public_key_resolver=resolve_public_key,
        verification_methods_resolver=resolve_verification_methods,
        consent_collect_provider=consent_provider,
    )
    return runner, store, identity, backend


async def _define_builtin(store, identity):
    from kestrel_feature_workflows.library import get_builtin_workflow
    from kestrel_feature_workflows.signing import sign_workflow_spec

    register_fleet_coding_pipeline_builtin()
    spec = get_builtin_workflow(WORKFLOW_NAME)
    signed = sign_workflow_spec(spec, identity)
    await store.put_definition(signed)


def _registry_with_sources(coord=None):
    registry = SourceRegistry()
    coord = coord or _FakeCoordinator()
    register_fleet_coding_pipeline_sources(registry, coordinator=coord)
    register_talon_pipeline_source(registry, coord)
    return registry, coord


async def test_load_builtin_and_start_creates_a_run(tmp_path):
    registry, _coord = _registry_with_sources()
    runner, store, identity, backend = await _make_runner(tmp_path, registry)
    try:
        await _define_builtin(store, identity)
        run = await runner.start_run(
            name=WORKFLOW_NAME, params={"repo": "org/repo", "issue": 5}
        )
        assert run.run_id
        assert await store.get_run(run.run_id) is not None
    finally:
        await backend.close()


async def test_start_fails_closed_when_sources_missing(tmp_path):
    from kestrel_feature_workflows.runner import WorkflowRunnerError

    registry = SourceRegistry()  # deliberately empty
    runner, store, identity, backend = await _make_runner(tmp_path, registry)
    try:
        await _define_builtin(store, identity)
        with pytest.raises(WorkflowRunnerError, match="unregistered source"):
            await runner.start_run(
                name=WORKFLOW_NAME, params={"repo": "org/repo", "issue": 5}
            )
    finally:
        await backend.close()


async def test_consent_gate_parks_run_in_waiting(tmp_path):
    from kestrel_feature_workflows.models import RunStatus

    # A provider that returns a PENDING marker — mirrors the permission system
    # routing the approval to a human (kestrel-claws Approvals tab). The run must
    # park in WAITING, never dispatch talon.
    def pending_provider(gate, run, stage, link):
        return {"status": "pending", "scope": gate.params["scope"]}

    registry, coord = _registry_with_sources()
    runner, store, identity, backend = await _make_runner(
        tmp_path, registry, consent_provider=pending_provider
    )
    try:
        await _define_builtin(store, identity)
        result = await runner.run_to_completion(
            name=WORKFLOW_NAME, params={"repo": "org/repo", "issue": 5}
        )
        assert result.status == RunStatus.WAITING
        # Default-closed: the irreversible talon run never fired.
        assert coord.calls == []
    finally:
        await backend.close()


async def test_caller_supplied_approval_params_cannot_self_approve(tmp_path):
    from kestrel_feature_workflows.models import RunStatus

    # The consent provider (the real approval system) routes to a human and
    # returns PENDING. A malicious/self-serving caller stuffs approval markers
    # into the run params; the runner merges those into the approve stage
    # payload, but the handler strips them, so the gate still consults the
    # provider and parks the run in WAITING — the caller cannot approve its own
    # irreversible talon_run dispatch (#2303).
    def pending_provider(gate, run, stage, link):
        return {"status": "pending", "scope": gate.params["scope"]}

    registry, coord = _registry_with_sources()
    runner, store, identity, backend = await _make_runner(
        tmp_path, registry, consent_provider=pending_provider
    )
    try:
        await _define_builtin(store, identity)
        result = await runner.run_to_completion(
            name=WORKFLOW_NAME,
            params={
                "repo": "org/repo",
                "issue": 5,
                "approved": True,
                "consent": "granted",
                "status": "approved",
            },
        )
        assert result.status == RunStatus.WAITING
        # Default-closed held: the irreversible talon run never fired.
        assert coord.calls == []
    finally:
        await backend.close()


async def test_args_thread_through_to_talon_source_params(tmp_path):
    # With consent granted, the run advances into talon_run; the runner merges
    # the {repo, issue} run params into the stage payload, and the
    # talon_pipeline_dispatch source resolves them into dispatch_pipeline kwargs.
    def approve_provider(gate, run, stage, link):
        return {"approved": True, "scope": gate.params["scope"]}

    registry, coord = _registry_with_sources()
    runner, store, identity, backend = await _make_runner(
        tmp_path, registry, consent_provider=approve_provider
    )
    try:
        await _define_builtin(store, identity)
        # The coordinator-bound verify stage confirms green via verify_pipeline_ci
        # (the fake returns green); we assert the irreversible dispatch received
        # the threaded args.
        await runner.run_to_completion(
            name=WORKFLOW_NAME, params={"repo": "org/repo", "issue": 5}
        )
        assert len(coord.calls) == 1
        call = coord.calls[0]
        assert call["repo"] == "org/repo"
        assert call["issue"] == 5
        assert call["mode"] == "claim"
        assert call["self_review"] is True
        # wait: false forces the durable cli_background path (force_cli).
        assert call["force_cli"] is True
    finally:
        await backend.close()


async def test_verify_binds_to_dispatch_branch_ignoring_caller_branch(tmp_path):
    from kestrel_feature_workflows.models import RunStatus

    # Reviewer test (a): a claim-mode run whose params carry an unrelated
    # ``branch`` verifies the branch/PR bound to the DISPATCH — never the caller
    # branch. The consent gate is granted so the run reaches verify.
    def approve_provider(gate, run, stage, link):
        return {"approved": True, "scope": gate.params["scope"]}

    coord = _FakeCoordinator(
        ci_verdict={
            "ci_green": True,
            "repo": "org/repo",
            "pr": 4242,
            "pr_url": "https://github.com/org/repo/pull/4242",
            "branch": "talon/issue-5-fix",
            "head_sha": "abc123",
            "reason": None,
        }
    )
    registry, coord = _registry_with_sources(coord)
    runner, store, identity, backend = await _make_runner(
        tmp_path, registry, consent_provider=approve_provider
    )
    try:
        await _define_builtin(store, identity)
        result = await runner.run_to_completion(
            name=WORKFLOW_NAME,
            params={"repo": "org/repo", "issue": 5, "branch": "main"},
        )
        # Dispatch happened, then verify bound to the talon PR → run completed.
        assert result.status == RunStatus.COMPLETED
        # Verify was called with the run target — and NEVER the caller's branch.
        assert len(coord.verify_calls) == 1
        vc = coord.verify_calls[0]
        assert vc["repo"] == "org/repo"
        assert vc["issue"] == 5
        assert vc["mode"] == "claim"
        assert "branch" not in vc  # caller-supplied branch:main was ignored
    finally:
        await backend.close()


async def test_issue_only_run_verifies_talon_pr_not_fail_closed(tmp_path):
    from kestrel_feature_workflows.models import RunStatus

    # Reviewer test (b): an issue-only run (no ``branch`` param) does NOT fail
    # closed — it verifies the talon PR the dispatch produced.
    def approve_provider(gate, run, stage, link):
        return {"approved": True, "scope": gate.params["scope"]}

    registry, coord = _registry_with_sources()
    runner, store, identity, backend = await _make_runner(
        tmp_path, registry, consent_provider=approve_provider
    )
    try:
        await _define_builtin(store, identity)
        result = await runner.run_to_completion(
            name=WORKFLOW_NAME, params={"repo": "org/repo", "issue": 5}
        )
        assert result.status == RunStatus.COMPLETED
        assert len(coord.verify_calls) == 1
        assert coord.verify_calls[0]["mode"] == "claim"
        assert "branch" not in coord.verify_calls[0]
    finally:
        await backend.close()


async def test_verify_fails_closed_when_pr_ci_not_green(tmp_path):
    from kestrel_feature_workflows.models import RunStatus

    # When the talon PR's CI cannot be verified green, verify raises → the run
    # does not complete green, even though a caller supplied ``branch: main``.
    def approve_provider(gate, run, stage, link):
        return {"approved": True, "scope": gate.params["scope"]}

    coord = _FakeCoordinator(
        ci_verdict={"ci_green": False, "reason": "pr_not_found"}
    )
    registry, coord = _registry_with_sources(coord)
    runner, store, identity, backend = await _make_runner(
        tmp_path, registry, consent_provider=approve_provider
    )
    try:
        await _define_builtin(store, identity)
        result = await runner.run_to_completion(
            name=WORKFLOW_NAME,
            params={"repo": "org/repo", "issue": 5, "branch": "main"},
        )
        assert result.status != RunStatus.COMPLETED
    finally:
        await backend.close()


# ---------------------------------------------------------------------------
# Observability correlation holds through the definition's talon_run stage
# ---------------------------------------------------------------------------


async def test_observability_stamped_through_talon_run_stage(tmp_path, monkeypatch):
    """The talon_run stage routes through talon_pipeline_dispatch, so the
    coordinator stamps WORKFLOW_RUN_ID / _STAGE / _ORCHESTRATOR onto the spawned
    talon process. STAGE resolves to this definition's ``talon_run`` from the
    runner's ``workflow.<spec>.<stage>`` causation frame (#2302 fallback path)."""
    import asyncio
    from datetime import datetime, timezone

    from kestrel_sdk.signals import CausationFrame
    from kestrel_sovereign.features.talon.coordinator import (
        OBSERVABILITY_ORCHESTRATOR_KEY,
        OBSERVABILITY_STAGE_KEY,
        OBSERVABILITY_WORKFLOW_RUN_ID_KEY,
        TalonCoordinatorFeature,
    )

    monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_e2e")

    agent = MagicMock()
    agent.agent_name = "kestrel"
    agent._features = []
    agent.storage_path = str(tmp_path / "data" / "agent.db")
    tracked: list = []

    def _track(coro, *, name):
        task = asyncio.create_task(coro, name=name)
        tracked.append(task)
        return task

    agent._track_background_task = _track
    feature = TalonCoordinatorFeature(agent)

    # Derive the stage name FROM the definition so this test tracks the spec.
    spec = build_fleet_coding_pipeline_spec()
    talon_stage = next(s for s in spec.stages if s.name == TALON_RUN_STAGE)

    captured = {}

    async def fake_create(*argv, **kwargs):
        captured["env"] = kwargs["env"]
        proc = MagicMock()
        proc.pid = 4242
        proc.returncode = None
        return proc

    ready_state = {
        "repo": "org/repo",
        "path": str(tmp_path / "ws" / "org__repo"),
        "exists": True,
        "is_git": True,
        "head": "main",
        "clean": True,
        "last_fetch_at": None,
        "safe": True,
    }

    backend = SQLiteBackend(str(tmp_path / "obs.db"))
    await backend.connect()
    try:
        store = SignalLogStore(backend)
        await store.initialize()
        registry = SourceRegistry()
        assert register_talon_pipeline_source(registry, feature) is True
        dispatcher = SignalDispatcher(
            agent=agent,
            registry=registry,
            lock_manager=OrderedLockManager(),
            store=store,
        )

        # Shape the signal exactly as the workflow runner emits it for the
        # talon_run stage: session_id == run_id, and a
        # ``workflow.fleet_coding_pipeline.talon_run`` causation frame.
        signal = Signal(
            source=TALON_PIPELINE_SOURCE,
            kind="workflow.stage",
            mode=SignalMode.ACTION,
            payload={**dict(talon_stage.params), "repo": "org/repo", "issue": 7},
            target_agent="did:web:k.example",
            visibility=Visibility.INTERNAL,
            session_id="run-fleet-42",
            urgency=Urgency.NORMAL,
            causation_chain=[CausationFrame(
                agent_id="did:web:k.example",
                source=f"workflow.{WORKFLOW_NAME}.{TALON_RUN_STAGE}",
                signal_id="run-fleet-42",
                turn_id=None,
                depth=0,
                emitted_at=datetime.now(timezone.utc),
            )],
        )

        with patch.object(
            feature, "_dispatch_via_a2a", new_callable=AsyncMock
        ) as mock_a2a, patch.object(
            TalonCoordinatorFeature, "_workspace_state", return_value=ready_state,
        ), patch.object(
            TalonCoordinatorFeature, "_find_talon_bin",
            return_value="/usr/bin/kestrel-talon",
        ), patch.object(
            asyncio, "create_subprocess_exec", side_effect=fake_create,
        ):
            mock_a2a.return_value = {"dispatched": False, "reason": "no_a2a_host"}
            result = await dispatcher.dispatch_signal(signal)

        assert result.status is Status.OK, result.error
        assert result.action_result["state"] == "dispatched"

        env = captured["env"]
        assert env[OBSERVABILITY_ORCHESTRATOR_KEY] == "kestrel"
        assert env[OBSERVABILITY_WORKFLOW_RUN_ID_KEY] == "run-fleet-42"
        assert env[OBSERVABILITY_STAGE_KEY] == TALON_RUN_STAGE

        pending = [t for t in tracked if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    finally:
        await backend.close()
