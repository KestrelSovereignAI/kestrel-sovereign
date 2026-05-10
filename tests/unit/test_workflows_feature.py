"""Tests for the agent-callable WorkflowsFeature surface."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sdk.signals import RedactionPolicy, SignalMode, SourceRegistration

from kestrel_sovereign.features.workflows.feature import WorkflowsFeature
from kestrel_sovereign.features.security.approval_queue import ApprovalQueue
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
    def __init__(self, backend: SQLiteBackend):
        self.identity = _identity()
        self.did = self.identity.legacy_did
        self.storage = SimpleNamespace(db=backend)
        self.background_tasks: list[asyncio.Task] = []
        self.signal_registry = SourceRegistry()
        self.dispatcher = None

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


def _redaction() -> RedactionPolicy:
    return RedactionPolicy(summarize=lambda payload: "<redacted>")


def _action_source(name: str, calls: list[dict]) -> SourceRegistration:
    async def handler(payload):
        calls.append(payload)
        return {"ok": True}

    return SourceRegistration(
        name=name,
        schema=dict,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=handler,
        log_redaction=_redaction(),
    )


@pytest.fixture
async def feature_components(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "workflows-feature.db"))
    await backend.connect()
    agent = _FakeAgent(backend)
    signal_store = SignalLogStore(backend)
    await signal_store.initialize()
    agent.dispatcher = SignalDispatcher(
        agent=agent,
        registry=agent.signal_registry,
        lock_manager=OrderedLockManager(),
        store=signal_store,
    )
    feature = WorkflowsFeature(agent)
    await feature.initialize()
    yield SimpleNamespace(agent=agent, backend=backend, feature=feature)
    pending = [task for task in agent.background_tasks if not task.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await backend.close()


def _spec() -> dict:
    return {
        "name": "release",
        "version": 1,
        "stages": [
            {
                "name": "lint",
                "signal_source": "ci.lint",
                "signal_mode": "action",
                "compensate": "noop_idempotent",
                "read_only": True,
            }
        ],
    }


@pytest.mark.asyncio
async def test_workflow_define_signs_and_lists_definition(feature_components):
    c = feature_components
    result = await c.feature.workflow_define(_spec())

    assert result.status.value == "ok"
    assert result.data["name"] == "release"
    assert result.data["spec_hash"]

    listed = await c.feature.workflow_list_definitions()
    assert listed.status.value == "ok"
    assert listed.data["definitions"][0]["name"] == "release"
    assert listed.data["definitions"][0]["spec_hash"] == result.data["spec_hash"]


@pytest.mark.asyncio
async def test_workflow_run_status_history_and_list_runs(feature_components):
    c = feature_components
    calls: list[dict] = []
    c.agent.signal_registry.register(_action_source("ci.lint", calls))
    await c.feature.workflow_define(_spec())

    run_result = await c.feature.workflow_run("release", params={"branch": "main"})

    assert run_result.status.value == "ok"
    assert run_result.data["status"] == "completed"
    assert calls == [{"branch": "main"}]

    status = await c.feature.workflow_status(run_result.data["run_id"])
    assert status.status.value == "ok"
    assert status.data["status"] == "completed"
    assert status.data["params"] == {"branch": "main"}

    history = await c.feature.workflow_history(run_result.data["run_id"])
    assert history.status.value == "ok"
    assert history.data["links"][0]["stage_name"] == "lint"

    runs = await c.feature.workflow_list_runs(workflow_name="release")
    assert runs.status.value == "ok"
    assert runs.data["runs"][0]["run_id"] == run_result.data["run_id"]


@pytest.mark.asyncio
async def test_workflow_resume_resolves_pending_consent_from_approval_queue(
    feature_components,
):
    c = feature_components
    queue = ApprovalQueue()
    c.agent.features = {"SecurityFeature": SimpleNamespace(approval_queue=queue)}
    approval_task = asyncio.create_task(
        queue.request_approval(
            "WorkflowsFeature",
            "publish",
            {"scope": "publish_pr"},
        )
    )
    while not queue.pending_requests:
        await asyncio.sleep(0)
    approval_id = queue.pending_requests[0].id

    async def handler(payload):
        return {
            "scope": payload["scope"],
            "status": "pending",
            "approval_id": approval_id,
        }

    c.agent.signal_registry.register(
        SourceRegistration(
            name="hooks.consent",
            schema=dict,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=handler,
            log_redaction=_redaction(),
        )
    )
    spec = _spec()
    spec["stages"][0]["name"] = "approve"
    spec["stages"][0]["signal_source"] = "hooks.consent"
    spec["stages"][0]["read_only"] = False
    spec["stages"][0]["gate"] = {
        "type": "consent_collect",
        "params": {"scope": "publish_pr"},
    }
    await c.feature.workflow_define(spec)

    run = await c.feature.workflow_run("release")
    assert run.status.value == "ok"
    assert run.data["status"] == "waiting"

    assert queue.submit_decision(approval_id, True, "once") is True
    resumed = await c.feature.workflow_resume(run.data["run_id"])
    approved, scope = await approval_task

    assert approved is True
    assert scope == "once"
    assert resumed.status.value == "ok"
    assert resumed.data["status"] == "completed"


@pytest.mark.asyncio
async def test_workflow_resume_rejects_unrelated_consent_approval(
    feature_components,
):
    c = feature_components
    queue = ApprovalQueue()
    c.agent.features = {"SecurityFeature": SimpleNamespace(approval_queue=queue)}
    approval_task = asyncio.create_task(
        queue.request_approval(
            "WorkflowsFeature",
            "other",
            {"scope": "delete_prod"},
        )
    )
    while not queue.pending_requests:
        await asyncio.sleep(0)
    approval_id = queue.pending_requests[0].id

    async def handler(payload):
        return {
            "scope": payload["scope"],
            "status": "pending",
            "approval_id": approval_id,
        }

    c.agent.signal_registry.register(
        SourceRegistration(
            name="hooks.consent",
            schema=dict,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=handler,
            log_redaction=_redaction(),
        )
    )
    spec = _spec()
    spec["stages"][0]["name"] = "approve"
    spec["stages"][0]["signal_source"] = "hooks.consent"
    spec["stages"][0]["read_only"] = False
    spec["stages"][0]["gate"] = {
        "type": "consent_collect",
        "params": {"scope": "publish_pr"},
    }
    await c.feature.workflow_define(spec)

    run = await c.feature.workflow_run("release")
    assert queue.submit_decision(approval_id, True, "once") is True
    resumed = await c.feature.workflow_resume(run.data["run_id"])
    await approval_task

    assert resumed.status.value == "ok"
    assert resumed.data["status"] == "failed"


@pytest.mark.asyncio
async def test_workflow_run_reports_unregistered_source(feature_components):
    c = feature_components
    await c.feature.workflow_define(_spec())

    result = await c.feature.workflow_run("release")

    assert result.status.value == "error"
    assert "unregistered source" in result.error


@pytest.mark.asyncio
async def test_workflow_run_preserves_invalid_falsy_params(feature_components):
    c = feature_components
    calls: list[dict] = []
    c.agent.signal_registry.register(_action_source("ci.lint", calls))
    await c.feature.workflow_define(_spec())

    result = await c.feature.workflow_run("release", params=[])

    assert result.status.value == "error"
    assert "params must be an object" in result.error
    assert calls == []


@pytest.mark.asyncio
async def test_workflow_pause_unknown_run_fails(feature_components):
    result = await feature_components.feature.workflow_pause("missing")

    assert result.status.value == "error"
    assert "Unknown workflow run" in result.error


@pytest.mark.asyncio
async def test_workflow_resume_continues_paused_run(feature_components):
    c = feature_components
    calls: list[dict] = []
    c.agent.signal_registry.register(_action_source("ci.lint", calls))
    await c.feature.workflow_define(_spec())
    run = await c.feature.runner.start_run(name="release")
    paused = await c.feature.workflow_pause(run.run_id)

    resumed = await c.feature.workflow_resume(run.run_id)

    assert paused.status.value == "ok"
    assert resumed.status.value == "ok"
    assert resumed.data["status"] == "completed"
    assert calls == [{}]


@pytest.mark.asyncio
async def test_workflow_pause_completed_run_fails(feature_components):
    c = feature_components
    calls: list[dict] = []
    c.agent.signal_registry.register(_action_source("ci.lint", calls))
    await c.feature.workflow_define(_spec())
    run_result = await c.feature.workflow_run("release")

    pause = await c.feature.workflow_pause(run_result.data["run_id"])

    assert pause.status.value == "error"
    assert "cannot be paused" in pause.error


@pytest.mark.asyncio
async def test_workflow_remediate_retry(feature_components):
    c = feature_components
    attempts = 0

    async def handler(payload):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("first attempt fails")
        return {"ok": True}

    c.agent.signal_registry.register(
        SourceRegistration(
            name="ci.lint",
            schema=dict,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=handler,
            log_redaction=_redaction(),
        )
    )
    await c.feature.workflow_define(_spec())
    failed = await c.feature.workflow_run("release")

    retried = await c.feature.workflow_remediate(
        failed.data["run_id"], "lint", "retry"
    )

    assert failed.data["status"] == "failed"
    assert retried.status.value == "ok"
    assert retried.data["status"] == "completed"
    history = await c.feature.workflow_history(failed.data["run_id"])
    assert [l["attempt_number"] for l in history.data["links"]] == [1, 2]
