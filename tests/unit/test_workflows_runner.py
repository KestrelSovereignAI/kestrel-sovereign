"""Phase 1 foundation tests for WorkflowRunner."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from kestrel_sdk.signals import (
    RedactionPolicy,
    SignalMode,
    SourceRegistration,
)

from kestrel_sovereign.features.workflows import (
    Edge,
    EdgeKind,
    Gate,
    RunStatus,
    Stage,
    WorkflowRunner,
    WorkflowRunnerError,
    WorkflowSpec,
    derive_stage_idempotency_key,
)
from kestrel_sovereign.features.workflows.signing import sign_workflow_spec
from kestrel_sovereign.features.workflows.store import WorkflowStore
from kestrel_sovereign.identity.runtime_identity import AgentIdentity
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    get_suite,
)
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.storage.db import SQLiteBackend


class _FakeAgent:
    def __init__(self, did: str):
        self._did = did
        self.background_tasks: list[asyncio.Task] = []

    @property
    def did(self) -> str:
        return self._did

    async def process_input(self, prompt: str):
        return "ok"

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


def _identity(did: str = "did:web:k.example") -> AgentIdentity:
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    return AgentIdentity(
        legacy_did=did,
        legacy_keypair=suite.generate_keypair(),
        legacy_did_document={},
    )


def _resolver_for(identity: AgentIdentity):
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    pub = suite.serialize_public_key(identity.legacy_keypair.public_key)

    def resolve(did: str) -> bytes:
        if did != identity.legacy_did:
            raise KeyError(did)
        return pub

    return resolve


def _redaction() -> RedactionPolicy:
    return RedactionPolicy(summarize=lambda payload: "<redacted>")


def _action_source(name: str, handler) -> SourceRegistration:
    return SourceRegistration(
        name=name,
        schema=dict,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=handler,
        log_redaction=_redaction(),
    )


def _stage(name: str, source: str) -> Stage:
    return Stage(
        name=name,
        signal_source=source,
        signal_mode=SignalMode.ACTION,
        read_only=True,
    )


@pytest.fixture
async def runner_components(tmp_path):
    identity = _identity()
    backend = SQLiteBackend(str(tmp_path / "workflows.db"))
    await backend.connect()
    signal_store = SignalLogStore(backend)
    await signal_store.initialize()
    workflow_store = WorkflowStore(backend)
    await workflow_store.initialize()
    registry = SourceRegistry()
    agent = _FakeAgent(identity.legacy_did)
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=signal_store,
    )
    runner = WorkflowRunner(
        store=workflow_store,
        dispatcher=dispatcher,
        registry=registry,
        agent_identity=identity,
        public_key_resolver=_resolver_for(identity),
    )
    yield SimpleNamespace(
        backend=backend,
        dispatcher=dispatcher,
        identity=identity,
        registry=registry,
        runner=runner,
        store=workflow_store,
    )
    pending = [task for task in agent.background_tasks if not task.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await backend.close()


async def _put_signed(c, spec: WorkflowSpec) -> WorkflowSpec:
    signed = sign_workflow_spec(spec, c.identity)
    await c.store.put_definition(signed)
    return signed


@pytest.mark.asyncio
async def test_runner_walks_sequential_signal_status_ok_flow(runner_components):
    c = runner_components
    calls: list[dict] = []

    async def handler(payload):
        calls.append(payload)
        return {"ok": True}

    c.registry.register(_action_source("ci.lint", handler))
    c.registry.register(_action_source("ci.tests", handler))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[
                _stage("lint", "ci.lint"),
                _stage("tests", "ci.tests"),
            ],
            edges=[
                Edge(
                    kind=EdgeKind.SEQUENTIAL,
                    from_stage="lint",
                    to_stage="tests",
                )
            ],
        ),
    )

    result = await c.runner.run_to_completion(name="release")

    assert result.status == RunStatus.COMPLETED
    assert calls == [{}, {}]
    run = await c.store.get_run(result.run_id)
    assert run is not None
    assert run.status == RunStatus.COMPLETED
    assert run.engine_nonce and len(run.engine_nonce) == 32
    links = await c.store.list_stage_links(result.run_id)
    assert [link.stage_name for link in links] == ["lint", "tests"]
    assert all(link.gate_outcome.value == "pass" for link in links)


@pytest.mark.asyncio
async def test_runner_refuses_unregistered_stage_before_signal(runner_components):
    c = runner_components
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[_stage("lint", "ci.lint")],
        ),
    )

    with pytest.raises(WorkflowRunnerError, match="unregistered source"):
        await c.runner.run_to_completion(name="release")

    rows = await c.backend.fetch_all(
        f"SELECT run_id FROM {c.store.RUNS_TABLE}"
    )
    assert rows == []


@pytest.mark.asyncio
async def test_runner_refuses_unsupported_gate_before_signal(runner_components):
    c = runner_components
    calls = 0

    async def handler(payload):
        nonlocal calls
        calls += 1
        return {"ok": True}

    c.registry.register(_action_source("ci.lint", handler))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[
                Stage(
                    name="lint",
                    signal_source="ci.lint",
                    signal_mode=SignalMode.ACTION,
                    read_only=True,
                    gate=Gate(
                        type="red_team_clear",
                        params={"prompt_pack_constraint": ">=1,<2"},
                    ),
                )
            ],
        ),
    )

    with pytest.raises(WorkflowRunnerError, match="not implemented"):
        await c.runner.run_to_completion(name="release")

    assert calls == 0
    rows = await c.backend.fetch_all(
        f"SELECT run_id FROM {c.store.RUNS_TABLE}"
    )
    assert rows == []


@pytest.mark.asyncio
async def test_runner_refuses_cyclic_sequential_graph_before_signal(
    runner_components,
):
    c = runner_components
    calls = 0

    async def handler(payload):
        nonlocal calls
        calls += 1
        return {"ok": True}

    c.registry.register(_action_source("do.a", handler))
    c.registry.register(_action_source("do.b", handler))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[
                _stage("a", "do.a"),
                _stage("b", "do.b"),
            ],
            edges=[
                Edge(kind=EdgeKind.SEQUENTIAL, from_stage="a", to_stage="b"),
                Edge(kind=EdgeKind.SEQUENTIAL, from_stage="b", to_stage="b"),
            ],
        ),
    )

    with pytest.raises(WorkflowRunnerError, match="cycle"):
        await c.runner.run_to_completion(name="release")

    assert calls == 0
    rows = await c.backend.fetch_all(
        f"SELECT run_id FROM {c.store.RUNS_TABLE}"
    )
    assert rows == []


@pytest.mark.asyncio
async def test_runner_refuses_unreachable_stages_before_signal(
    runner_components,
):
    c = runner_components
    calls = 0

    async def handler(payload):
        nonlocal calls
        calls += 1
        return {"ok": True}

    c.registry.register(_action_source("do.a", handler))
    c.registry.register(_action_source("do.b", handler))
    c.registry.register(_action_source("do.c", handler))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[
                _stage("a", "do.a"),
                _stage("b", "do.b"),
                _stage("c", "do.c"),
            ],
            edges=[
                Edge(kind=EdgeKind.SEQUENTIAL, from_stage="b", to_stage="c"),
                Edge(kind=EdgeKind.SEQUENTIAL, from_stage="c", to_stage="b"),
            ],
        ),
    )

    with pytest.raises(WorkflowRunnerError, match="unreachable"):
        await c.runner.run_to_completion(name="release")

    assert calls == 0
    rows = await c.backend.fetch_all(
        f"SELECT run_id FROM {c.store.RUNS_TABLE}"
    )
    assert rows == []


@pytest.mark.asyncio
async def test_runner_refuses_sequential_fanout_before_signal(
    runner_components,
):
    c = runner_components
    calls = 0

    async def handler(payload):
        nonlocal calls
        calls += 1
        return {"ok": True}

    for source in ("do.a", "do.b", "do.c", "do.d"):
        c.registry.register(_action_source(source, handler))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[
                _stage("a", "do.a"),
                _stage("b", "do.b"),
                _stage("c", "do.c"),
                _stage("d", "do.d"),
            ],
            edges=[
                Edge(kind=EdgeKind.SEQUENTIAL, from_stage="a", to_stage="b"),
                Edge(kind=EdgeKind.SEQUENTIAL, from_stage="a", to_stage="c"),
                Edge(kind=EdgeKind.SEQUENTIAL, from_stage="b", to_stage="d"),
                Edge(kind=EdgeKind.SEQUENTIAL, from_stage="c", to_stage="d"),
            ],
        ),
    )

    with pytest.raises(WorkflowRunnerError, match="fan-out"):
        await c.runner.run_to_completion(name="release")

    assert calls == 0
    rows = await c.backend.fetch_all(
        f"SELECT run_id FROM {c.store.RUNS_TABLE}"
    )
    assert rows == []


@pytest.mark.asyncio
async def test_runner_validates_run_params_schema_before_insert(
    runner_components,
):
    c = runner_components
    calls = 0

    async def handler(payload):
        nonlocal calls
        calls += 1
        return {"ok": True}

    c.registry.register(_action_source("ci.lint", handler))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[_stage("lint", "ci.lint")],
            params_schema={
                "type": "object",
                "required": ["branch"],
                "properties": {"branch": {"type": "string"}},
            },
        ),
    )

    with pytest.raises(WorkflowRunnerError, match="params failed validation"):
        await c.runner.run_to_completion(name="release", params={})

    assert calls == 0
    rows = await c.backend.fetch_all(
        f"SELECT run_id FROM {c.store.RUNS_TABLE}"
    )
    assert rows == []


@pytest.mark.asyncio
async def test_runner_passes_run_params_to_stage_payload(runner_components):
    c = runner_components
    calls: list[dict] = []

    async def handler(payload):
        calls.append(payload)
        return {"ok": True}

    c.registry.register(_action_source("ci.lint", handler))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[
                Stage(
                    name="lint",
                    signal_source="ci.lint",
                    signal_mode=SignalMode.ACTION,
                    read_only=True,
                    params={"branch": "main", "suite": "lint"},
                )
            ],
            params_schema={
                "type": "object",
                "required": ["branch"],
                "properties": {"branch": {"type": "string"}},
            },
        ),
    )

    result = await c.runner.run_to_completion(
        name="release",
        params={"branch": "feature/workflows", "repo": "kestrel"},
    )

    assert result.status == RunStatus.COMPLETED
    assert calls == [
        {"branch": "feature/workflows", "suite": "lint", "repo": "kestrel"}
    ]


@pytest.mark.asyncio
async def test_runner_rejects_falsy_non_object_params(runner_components):
    c = runner_components
    c.registry.register(_action_source("ci.lint", lambda payload: {"ok": True}))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[_stage("lint", "ci.lint")],
            params_schema={"type": "object"},
        ),
    )

    with pytest.raises(WorkflowRunnerError, match="params must be an object"):
        await c.runner.run_to_completion(name="release", params=[])

    rows = await c.backend.fetch_all(
        f"SELECT run_id FROM {c.store.RUNS_TABLE}"
    )
    assert rows == []


@pytest.mark.asyncio
async def test_runner_inserts_idempotency_link_before_handler(runner_components):
    c = runner_components
    observed_link_count: list[int] = []

    async def handler(payload):
        rows = await c.backend.fetch_all(
            f"SELECT idempotency_key FROM {c.store.STAGE_LINKS_TABLE}"
        )
        observed_link_count.append(len(rows))
        return {"ok": True}

    c.registry.register(_action_source("ci.lint", handler))
    spec = await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[_stage("lint", "ci.lint")],
        ),
    )

    result = await c.runner.run_to_completion(name="release")
    run = await c.store.get_run(result.run_id)
    link = (await c.store.list_stage_links(result.run_id))[0]
    expected = derive_stage_idempotency_key(
        run_id=result.run_id,
        stage=spec.stages[0],
        attempt_number=1,
        engine_nonce=run.engine_nonce,
    )

    assert observed_link_count == [1]
    assert link.idempotency_key == expected


@pytest.mark.asyncio
async def test_runner_rejects_revoked_definition(runner_components):
    c = runner_components
    c.registry.register(_action_source("ci.lint", lambda payload: {"ok": True}))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[_stage("lint", "ci.lint")],
        ),
    )
    assert await c.store.revoke_definition(
        "release", 1, reason="retired"
    )

    with pytest.raises(WorkflowRunnerError, match="not found"):
        await c.runner.run_to_completion(name="release")


@pytest.mark.asyncio
async def test_runner_records_failed_gate_when_dispatch_fails(runner_components):
    c = runner_components

    async def handler(payload):
        raise RuntimeError("boom")

    c.registry.register(_action_source("ci.lint", handler))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[_stage("lint", "ci.lint")],
        ),
    )

    result = await c.runner.run_to_completion(name="release")
    links = await c.store.list_stage_links(result.run_id)

    assert result.status == RunStatus.FAILED
    assert links[0].gate_outcome.value == "fail"
    assert "boom" in links[0].gate_reason
    assert links[0].signal_id is not None


@pytest.mark.asyncio
async def test_runner_compensates_passed_stages_when_later_gate_fails(
    runner_components,
):
    c = runner_components
    events: list[str] = []

    async def do_one(payload):
        events.append("do_one")
        return {"ok": True}

    async def do_two(payload):
        events.append("do_two")
        raise RuntimeError("boom")

    async def undo_one(payload):
        events.append("undo_one")
        return {"ok": True}

    async def undo_two(payload):
        events.append("undo_two")
        return {"ok": True}

    c.registry.register(_action_source("do.one", do_one))
    c.registry.register(_action_source("do.two", do_two))
    c.registry.register(_action_source("undo.one", undo_one))
    c.registry.register(_action_source("undo.two", undo_two))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[
                Stage(
                    name="one",
                    signal_source="do.one",
                    signal_mode=SignalMode.ACTION,
                    compensate="undo.one",
                ),
                Stage(
                    name="two",
                    signal_source="do.two",
                    signal_mode=SignalMode.ACTION,
                    compensate="undo.two",
                ),
            ],
            edges=[
                Edge(kind=EdgeKind.SEQUENTIAL, from_stage="one", to_stage="two")
            ],
        ),
    )

    result = await c.runner.run_to_completion(name="release")

    assert result.status == RunStatus.FAILED
    assert events == ["do_one", "do_two", "undo_one"]
    stored = await c.store.get_run(result.run_id)
    assert stored.status == RunStatus.FAILED
    links = await c.store.list_stage_links(result.run_id)
    assert [link.compensate_state for link in links] == ["complete", "pending"]


@pytest.mark.asyncio
async def test_runner_cancel_compensates_completed_stages_reverse_order(
    runner_components,
):
    c = runner_components
    events: list[str] = []

    async def do_one(payload):
        events.append("do_one")
        return {"ok": True}

    async def do_two(payload):
        events.append("do_two")
        return {"ok": True}

    async def undo_one(payload):
        events.append("undo_one")
        return {"ok": True}

    async def undo_two(payload):
        events.append("undo_two")
        return {"ok": True}

    c.registry.register(_action_source("do.one", do_one))
    c.registry.register(_action_source("do.two", do_two))
    c.registry.register(_action_source("undo.one", undo_one))
    c.registry.register(_action_source("undo.two", undo_two))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[
                Stage(
                    name="one",
                    signal_source="do.one",
                    signal_mode=SignalMode.ACTION,
                    compensate="undo.one",
                ),
                Stage(
                    name="two",
                    signal_source="do.two",
                    signal_mode=SignalMode.ACTION,
                    compensate="undo.two",
                ),
            ],
            edges=[
                Edge(
                    kind=EdgeKind.SEQUENTIAL,
                    from_stage="one",
                    to_stage="two",
                )
            ],
        ),
    )

    run = await c.runner.start_run(name="release")
    spec = await c.store.get_definition("release", 1)
    await c.runner._dispatch_stage(run, spec, spec.stages[0])
    await c.runner._dispatch_stage(run, spec, spec.stages[1])
    await c.store.update_run_status(run.run_id, RunStatus.PAUSED)

    status = await c.runner.cancel_run(run.run_id)

    assert status == RunStatus.CANCELLED
    assert events == ["do_one", "do_two", "undo_two", "undo_one"]
    links = await c.store.list_stage_links(run.run_id)
    assert [link.compensate_state for link in links] == ["complete", "complete"]


@pytest.mark.asyncio
async def test_runner_cancel_active_run_sets_barrier_without_compensating_twice(
    runner_components,
):
    c = runner_components
    events: list[str] = []

    async def do_one(payload):
        events.append("do_one")
        return {"ok": True}

    async def undo_one(payload):
        events.append("undo_one")
        return {"ok": True}

    c.registry.register(_action_source("do.one", do_one))
    c.registry.register(_action_source("undo.one", undo_one))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[
                Stage(
                    name="one",
                    signal_source="do.one",
                    signal_mode=SignalMode.ACTION,
                    compensate="undo.one",
                )
            ],
        ),
    )
    run = await c.runner.start_run(name="release")
    spec = await c.store.get_definition("release", 1)
    await c.runner._dispatch_stage(run, spec, spec.stages[0])

    status = await c.runner.cancel_run(run.run_id)

    assert status == RunStatus.COMPENSATING
    assert events == ["do_one"]

    result = await c.runner.continue_run(run.run_id)

    assert result.status == RunStatus.CANCELLED
    assert events == ["do_one", "undo_one"]


@pytest.mark.asyncio
async def test_runner_cancel_during_final_stage_compensates(runner_components):
    c = runner_components
    events: list[str] = []
    run_id = None

    async def do_one(payload):
        events.append("do_one")
        await c.runner.cancel_run(run_id)
        return {"ok": True}

    async def undo_one(payload):
        events.append("undo_one")
        return {"ok": True}

    c.registry.register(_action_source("do.one", do_one))
    c.registry.register(_action_source("undo.one", undo_one))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[
                Stage(
                    name="one",
                    signal_source="do.one",
                    signal_mode=SignalMode.ACTION,
                    compensate="undo.one",
                )
            ],
        ),
    )

    run = await c.runner.start_run(name="release")
    run_id = run.run_id
    result = await c.runner.continue_run(run.run_id)

    assert result.status == RunStatus.CANCELLED
    assert events == ["do_one", "undo_one"]


@pytest.mark.asyncio
async def test_runner_pause_during_stage_stops_at_boundary(runner_components):
    c = runner_components
    events: list[str] = []
    run_id = None

    async def do_one(payload):
        events.append("one")
        await c.store.update_run_status(run_id, RunStatus.PAUSED)
        return {"ok": True}

    async def do_two(payload):
        events.append("two")
        return {"ok": True}

    c.registry.register(_action_source("do.one", do_one))
    c.registry.register(_action_source("do.two", do_two))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[
                _stage("one", "do.one"),
                _stage("two", "do.two"),
            ],
            edges=[
                Edge(
                    kind=EdgeKind.SEQUENTIAL,
                    from_stage="one",
                    to_stage="two",
                )
            ],
        ),
    )

    run = await c.runner.start_run(name="release")
    run_id = run.run_id
    paused = await c.runner.continue_run(run.run_id)
    stored = await c.store.get_run(run.run_id)

    assert paused.status == RunStatus.PAUSED
    assert stored.status == RunStatus.PAUSED
    assert list(stored.current_stages) == ["two"]
    assert events == ["one"]

    resumed = await c.runner.continue_run(run.run_id)

    assert resumed.status == RunStatus.COMPLETED
    assert events == ["one", "two"]


@pytest.mark.asyncio
async def test_runner_cancel_refuses_completed_run(runner_components):
    c = runner_components
    c.registry.register(_action_source("ci.lint", lambda payload: {"ok": True}))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[_stage("lint", "ci.lint")],
        ),
    )
    result = await c.runner.run_to_completion(name="release")

    with pytest.raises(WorkflowRunnerError, match="terminal"):
        await c.runner.cancel_run(result.run_id)


@pytest.mark.asyncio
async def test_runner_resume_continues_paused_run(runner_components):
    c = runner_components
    calls: list[dict] = []

    async def handler(payload):
        calls.append(payload)
        return {"ok": True}

    c.registry.register(_action_source("ci.lint", handler))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[_stage("lint", "ci.lint")],
        ),
    )
    run = await c.runner.start_run(name="release")
    await c.store.update_run_status(run.run_id, RunStatus.PAUSED)

    result = await c.runner.continue_run(run.run_id)

    assert result.status == RunStatus.COMPLETED
    assert calls == [{}]
    stored = await c.store.get_run(run.run_id)
    assert stored.status == RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_runner_reverifies_pinned_definition_before_resume(
    runner_components,
):
    c = runner_components
    calls: list[str] = []

    async def handler(payload):
        calls.append("original")
        return {"ok": True}

    async def tampered_handler(payload):
        calls.append("tampered")
        return {"ok": True}

    c.registry.register(_action_source("ci.lint", handler))
    c.registry.register(_action_source("evil.run", tampered_handler))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[_stage("lint", "ci.lint")],
        ),
    )
    run = await c.runner.start_run(name="release")
    await c.store.update_run_status(run.run_id, RunStatus.PAUSED)
    row = await c.store.get_definition_row("release", 1)
    payload = json.loads(row["spec_json"])
    payload["stages"][0]["signal_source"] = "evil.run"
    await c.backend.execute(
        f"UPDATE {c.store.DEFINITIONS_TABLE} SET spec_json = ? "
        "WHERE name = ? AND version = ?",
        (json.dumps(payload, sort_keys=True, separators=(",", ":")), "release", 1),
    )

    with pytest.raises(WorkflowRunnerError, match="signature failed"):
        await c.runner.continue_run(run.run_id)

    assert calls == []


@pytest.mark.asyncio
async def test_runner_marks_post_revocation_resume_for_audit(
    runner_components,
):
    c = runner_components
    calls: list[str] = []

    async def handler(payload):
        calls.append("lint")
        return {"ok": True}

    c.registry.register(_action_source("ci.lint", handler))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[_stage("lint", "ci.lint")],
        ),
    )
    run = await c.runner.start_run(name="release")
    await c.store.update_run_status(run.run_id, RunStatus.PAUSED)
    assert await c.store.revoke_definition("release", 1, reason="retired")

    result = await c.runner.continue_run(run.run_id)

    assert result.status == RunStatus.COMPLETED
    assert calls == ["lint"]
    stored = await c.store.get_run(run.run_id)
    assert stored.signature_post_revocation is True


@pytest.mark.asyncio
async def test_runner_refuses_to_continue_terminal_run(runner_components):
    c = runner_components
    c.registry.register(_action_source("ci.lint", lambda payload: {"ok": True}))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[_stage("lint", "ci.lint")],
        ),
    )
    result = await c.runner.run_to_completion(name="release")

    with pytest.raises(WorkflowRunnerError, match="terminal"):
        await c.runner.continue_run(result.run_id)


@pytest.mark.asyncio
async def test_runner_retries_failed_stage_with_fresh_attempt_number(
    runner_components,
):
    c = runner_components
    attempts = 0

    async def handler(payload):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("first attempt fails")
        return {"ok": True}

    c.registry.register(_action_source("ci.lint", handler))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[_stage("lint", "ci.lint")],
        ),
    )
    failed = await c.runner.run_to_completion(name="release")

    retried = await c.runner.retry_stage(failed.run_id, "lint")

    assert retried.status == RunStatus.COMPLETED
    links = await c.store.list_stage_links(failed.run_id)
    assert [link.attempt_number for link in links] == [1, 2]
    assert links[0].gate_outcome.value == "fail"
    assert links[1].gate_outcome.value == "pass"
    assert links[0].idempotency_key != links[1].idempotency_key


@pytest.mark.asyncio
async def test_runner_retry_reruns_compensated_prerequisites(
    runner_components,
):
    c = runner_components
    events: list[str] = []
    configure_attempts = 0

    async def create(payload):
        events.append("create")
        return {"ok": True}

    async def configure(payload):
        nonlocal configure_attempts
        configure_attempts += 1
        events.append("configure")
        if configure_attempts == 1:
            raise RuntimeError("configure failed")
        return {"ok": True}

    async def undo_create(payload):
        events.append("undo_create")
        return {"ok": True}

    c.registry.register(_action_source("resource.create", create))
    c.registry.register(_action_source("resource.configure", configure))
    c.registry.register(_action_source("resource.undo_create", undo_create))
    await _put_signed(
        c,
        WorkflowSpec(
            name="resource",
            version=1,
            stages=[
                Stage(
                    name="create",
                    signal_source="resource.create",
                    signal_mode=SignalMode.ACTION,
                    compensate="resource.undo_create",
                ),
                _stage("configure", "resource.configure"),
            ],
            edges=[
                Edge(
                    kind=EdgeKind.SEQUENTIAL,
                    from_stage="create",
                    to_stage="configure",
                )
            ],
        ),
    )
    failed = await c.runner.run_to_completion(name="resource")
    assert failed.status == RunStatus.FAILED
    assert events == ["create", "configure", "undo_create"]

    retried = await c.runner.retry_stage(failed.run_id, "configure")

    assert retried.status == RunStatus.COMPLETED
    assert events == [
        "create",
        "configure",
        "undo_create",
        "create",
        "configure",
    ]
    links = await c.store.list_stage_links(failed.run_id)
    assert [link.stage_name for link in links] == [
        "create",
        "configure",
        "create",
        "configure",
    ]


@pytest.mark.asyncio
async def test_runner_rejects_retrying_stage_other_than_failed_one(
    runner_components,
):
    c = runner_components
    events: list[str] = []

    async def lint(payload):
        events.append("lint")
        raise RuntimeError("lint failed")

    async def deploy(payload):
        events.append("deploy")
        return {"ok": True}

    c.registry.register(_action_source("ci.lint", lint))
    c.registry.register(_action_source("release.deploy", deploy))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[
                _stage("lint", "ci.lint"),
                _stage("deploy", "release.deploy"),
            ],
            edges=[
                Edge(
                    kind=EdgeKind.SEQUENTIAL,
                    from_stage="lint",
                    to_stage="deploy",
                )
            ],
        ),
    )
    failed = await c.runner.run_to_completion(name="release")

    with pytest.raises(WorkflowRunnerError, match="failed at stage 'lint'"):
        await c.runner.retry_stage(failed.run_id, "deploy")

    assert events == ["lint"]
    links = await c.store.list_stage_links(failed.run_id)
    assert [link.stage_name for link in links] == ["lint"]


@pytest.mark.asyncio
async def test_runner_retry_clears_stale_finished_at_before_dispatch(
    runner_components,
):
    c = runner_components
    attempts = 0
    finished_at_during_retry = []

    async def handler(payload):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("first attempt fails")
        run = await c.store.get_run(failed.run_id)
        finished_at_during_retry.append(run.finished_at)
        return {"ok": True}

    c.registry.register(_action_source("ci.lint", handler))
    await _put_signed(
        c,
        WorkflowSpec(
            name="release",
            version=1,
            stages=[_stage("lint", "ci.lint")],
        ),
    )
    failed = await c.runner.run_to_completion(name="release")
    failed_run = await c.store.get_run(failed.run_id)
    assert failed_run.finished_at is not None

    retried = await c.runner.retry_stage(failed.run_id, "lint")

    assert retried.status == RunStatus.COMPLETED
    assert finished_at_during_retry == [None]
