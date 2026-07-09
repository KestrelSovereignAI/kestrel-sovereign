"""Orchestrator identity + workflow correlation stamping (kestrel-talon#53).

The coordinator stamps three FROZEN keys onto every outgoing talon
invocation at the dispatch funnels:

    KESTREL_OBSERVABILITY_ORCHESTRATOR
    KESTREL_OBSERVABILITY_WORKFLOW_RUN_ID
    KESTREL_OBSERVABILITY_STAGE

CLI dispatches carry them as process env vars on the spawned talon
process; A2A dispatches carry the same keys as structured metadata fields
on the message. Workflow correlation is read off the in-flight Signal
published by the dispatcher's per-task context.
"""

from __future__ import annotations

import asyncio
import asyncio as _asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sdk.signals import (
    CausationFrame,
    Signal,
    SignalMode,
    Status,
    Urgency,
    Visibility,
)
from kestrel_sovereign.features.talon.coordinator import (
    OBSERVABILITY_ORCHESTRATOR_KEY,
    OBSERVABILITY_STAGE_KEY,
    OBSERVABILITY_WORKFLOW_RUN_ID_KEY,
    TalonCoordinatorFeature,
)
from kestrel_sovereign.signals.context import (
    get_current_signal,
    reset_current_signal,
    set_current_signal,
)


def _make_agent(tmp_path=None, name="kestrel"):
    agent = MagicMock()
    agent.agent_name = name
    agent._features = []
    if tmp_path is not None:
        agent.storage_path = str(tmp_path / "data" / "agent.db")
    return agent


def _workflow_signal(
    run_id="run-abc123",
    stage_in_payload="dispatch_pipeline",
    kind="workflow.stage",
    chain_stage=None,
):
    payload = {"repo": "org/repo", "issue": 5}
    if stage_in_payload:
        payload["workflow_stage_name"] = stage_in_payload
    chain = []
    if chain_stage:
        chain.append(CausationFrame(
            agent_id="did:web:k.example",
            source=f"workflow.stalled_work_rescue.{chain_stage}",
            signal_id=run_id,
            turn_id=None,
            depth=0,
            emitted_at=datetime.now(timezone.utc),
        ))
    return Signal(
        source="talon_pipeline_dispatch",
        kind=kind,
        mode=SignalMode.ACTION,
        payload=payload,
        target_agent="did:web:k.example",
        visibility=Visibility.INTERNAL,
        session_id=run_id,
        urgency=Urgency.NORMAL,
        causation_chain=chain,
    )


class _SignalContext:
    """Set/reset the dispatcher's current-signal context around a test."""

    def __init__(self, signal):
        self.signal = signal
        self._token = None

    def __enter__(self):
        self._token = set_current_signal(self.signal)
        return self

    def __exit__(self, *exc):
        reset_current_signal(self._token)
        return False


# ---------------------------------------------------------------------------
# _observability_context: workflow-stage vs direct-call vs no-agent
# ---------------------------------------------------------------------------


class TestObservabilityContext:
    def test_direct_call_has_orchestrator_only(self):
        feature = TalonCoordinatorFeature(_make_agent())
        assert get_current_signal() is None
        ctx = feature._observability_context()
        assert ctx == {OBSERVABILITY_ORCHESTRATOR_KEY: "kestrel"}

    def test_workflow_stage_sets_all_three(self):
        feature = TalonCoordinatorFeature(_make_agent())
        with _SignalContext(_workflow_signal()):
            ctx = feature._observability_context()
        assert ctx[OBSERVABILITY_ORCHESTRATOR_KEY] == "kestrel"
        assert ctx[OBSERVABILITY_WORKFLOW_RUN_ID_KEY] == "run-abc123"
        assert ctx[OBSERVABILITY_STAGE_KEY] == "dispatch_pipeline"

    def test_stage_name_falls_back_to_causation_chain(self):
        feature = TalonCoordinatorFeature(_make_agent())
        signal = _workflow_signal(
            stage_in_payload=None, chain_stage="dispatch_repairs"
        )
        with _SignalContext(signal):
            ctx = feature._observability_context()
        assert ctx[OBSERVABILITY_STAGE_KEY] == "dispatch_repairs"

    def test_dotted_stage_name_survives_causation_parse(self):
        # The workflows name grammar permits dots in stage names; the
        # fallback must strip only the "workflow.<spec>." prefix, never
        # truncate the stage itself ("deploy.v2" must not become "v2").
        feature = TalonCoordinatorFeature(_make_agent())
        signal = _workflow_signal(
            stage_in_payload=None, chain_stage="deploy.v2"
        )
        with _SignalContext(signal):
            ctx = feature._observability_context()
        assert ctx[OBSERVABILITY_STAGE_KEY] == "deploy.v2"

    def test_non_workflow_signal_sets_no_correlation(self):
        feature = TalonCoordinatorFeature(_make_agent())
        signal = _workflow_signal(kind="scheduler.tick")
        with _SignalContext(signal):
            ctx = feature._observability_context()
        assert OBSERVABILITY_WORKFLOW_RUN_ID_KEY not in ctx
        assert OBSERVABILITY_STAGE_KEY not in ctx
        assert ctx[OBSERVABILITY_ORCHESTRATOR_KEY] == "kestrel"

    def test_no_agent_identity_leaves_orchestrator_unset(self):
        # A MagicMock agent_name is not a real (str) identity — downstream
        # treats a missing key as "Direct".
        agent = MagicMock()
        agent._features = []
        assert not isinstance(agent.agent_name, str)
        feature = TalonCoordinatorFeature(agent)
        ctx = feature._observability_context()
        assert OBSERVABILITY_ORCHESTRATOR_KEY not in ctx

        agent.agent_name = "   "  # blank is not an identity either
        assert OBSERVABILITY_ORCHESTRATOR_KEY not in feature._observability_context()

    def test_workflow_correlation_without_agent_name_still_stamped(self):
        agent = MagicMock()
        agent._features = []
        feature = TalonCoordinatorFeature(agent)
        with _SignalContext(_workflow_signal()):
            ctx = feature._observability_context()
        assert OBSERVABILITY_ORCHESTRATOR_KEY not in ctx
        assert ctx[OBSERVABILITY_WORKFLOW_RUN_ID_KEY] == "run-abc123"
        assert ctx[OBSERVABILITY_STAGE_KEY] == "dispatch_pipeline"


# ---------------------------------------------------------------------------
# CLI transport: env vars on the spawned talon process
# ---------------------------------------------------------------------------


class TestCliEnvStamping:
    @staticmethod
    def _fake_proc():
        proc = MagicMock()
        proc.pid = 4242
        proc.returncode = None
        return proc

    @pytest.mark.asyncio
    async def test_background_cli_env_carries_all_three_in_workflow(
        self, tmp_path, monkeypatch
    ):
        feature = TalonCoordinatorFeature(_make_agent(tmp_path))
        captured = {}

        async def fake_create(*argv, **kwargs):
            captured["env"] = kwargs["env"]
            return self._fake_proc()

        with _SignalContext(_workflow_signal()), \
             patch.object(TalonCoordinatorFeature, "_find_talon_bin",
                          return_value="/usr/bin/kestrel-talon"), \
             patch.object(_asyncio, "create_subprocess_exec",
                          side_effect=fake_create):
            result = await feature._dispatch_via_cli_background(
                ["claim", "--repo", "org/repo", "--issue", "5"],
                label="claim:org/repo#5",
                env={"GITHUB_TOKEN": "ghp_x"},
            )

        assert result["dispatched"] is True
        env = captured["env"]
        assert env[OBSERVABILITY_ORCHESTRATOR_KEY] == "kestrel"
        assert env[OBSERVABILITY_WORKFLOW_RUN_ID_KEY] == "run-abc123"
        assert env[OBSERVABILITY_STAGE_KEY] == "dispatch_pipeline"
        # The caller's env survives underneath.
        assert env["GITHUB_TOKEN"] == "ghp_x"

    @pytest.mark.asyncio
    async def test_background_cli_env_direct_call_has_orchestrator_only(
        self, tmp_path
    ):
        feature = TalonCoordinatorFeature(_make_agent(tmp_path))
        captured = {}

        async def fake_create(*argv, **kwargs):
            captured["env"] = kwargs["env"]
            return self._fake_proc()

        with patch.object(TalonCoordinatorFeature, "_find_talon_bin",
                          return_value="/usr/bin/kestrel-talon"), \
             patch.object(_asyncio, "create_subprocess_exec",
                          side_effect=fake_create):
            result = await feature._dispatch_via_cli_background(
                ["claim", "--repo", "org/repo", "--issue", "5"],
                label="claim:org/repo#5",
                env={"GITHUB_TOKEN": "ghp_x"},
            )

        assert result["dispatched"] is True
        env = captured["env"]
        assert env[OBSERVABILITY_ORCHESTRATOR_KEY] == "kestrel"
        assert OBSERVABILITY_WORKFLOW_RUN_ID_KEY not in env
        assert OBSERVABILITY_STAGE_KEY not in env

    @pytest.mark.asyncio
    async def test_background_cli_env_no_agent_leaves_all_unset(
        self, tmp_path
    ):
        agent = MagicMock()
        agent._features = []
        agent.storage_path = str(tmp_path / "data" / "agent.db")
        feature = TalonCoordinatorFeature(agent)
        captured = {}

        async def fake_create(*argv, **kwargs):
            captured["env"] = kwargs["env"]
            return self._fake_proc()

        with patch.object(TalonCoordinatorFeature, "_find_talon_bin",
                          return_value="/usr/bin/kestrel-talon"), \
             patch.object(_asyncio, "create_subprocess_exec",
                          side_effect=fake_create):
            await feature._dispatch_via_cli_background(
                ["claim", "--repo", "org/repo", "--issue", "5"],
                label="claim:org/repo#5",
                env={"GITHUB_TOKEN": "ghp_x"},
            )

        env = captured["env"]
        assert OBSERVABILITY_ORCHESTRATOR_KEY not in env
        assert OBSERVABILITY_WORKFLOW_RUN_ID_KEY not in env
        assert OBSERVABILITY_STAGE_KEY not in env


# ---------------------------------------------------------------------------
# AMP/A2A transport: structured metadata fields on the message
# ---------------------------------------------------------------------------


class TestA2aMetadataStamping:
    @staticmethod
    def _capture_urlopen(captured):
        def fake_urlopen(req, timeout=10):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            response = MagicMock()
            response.read.return_value = b"{}"
            return response
        return fake_urlopen

    @pytest.mark.asyncio
    async def test_a2a_metadata_carries_all_three_in_workflow(self):
        feature = TalonCoordinatorFeature(_make_agent())
        captured = {}
        with _SignalContext(_workflow_signal()), \
             patch.object(feature, "_discover_host_url",
                          return_value="http://localhost:8888"), \
             patch(
                 "kestrel_sovereign.features.talon.coordinator."
                 "urllib.request.urlopen",
                 side_effect=self._capture_urlopen(captured),
             ):
            result = await feature._dispatch_via_a2a("org/repo", 5)

        assert result["dispatched"] is True
        metadata = captured["body"]["metadata"]
        assert metadata[OBSERVABILITY_ORCHESTRATOR_KEY] == "kestrel"
        assert metadata[OBSERVABILITY_WORKFLOW_RUN_ID_KEY] == "run-abc123"
        assert metadata[OBSERVABILITY_STAGE_KEY] == "dispatch_pipeline"
        # The pre-existing metadata contract is intact.
        assert metadata["repo"] == "org/repo"
        assert metadata["issue_number"] == 5

    @pytest.mark.asyncio
    async def test_a2a_metadata_direct_call_has_orchestrator_only(self):
        feature = TalonCoordinatorFeature(_make_agent())
        captured = {}
        with patch.object(feature, "_discover_host_url",
                          return_value="http://localhost:8888"), \
             patch(
                 "kestrel_sovereign.features.talon.coordinator."
                 "urllib.request.urlopen",
                 side_effect=self._capture_urlopen(captured),
             ):
            result = await feature._dispatch_via_a2a("org/repo", 5)

        assert result["dispatched"] is True
        metadata = captured["body"]["metadata"]
        assert metadata[OBSERVABILITY_ORCHESTRATOR_KEY] == "kestrel"
        assert OBSERVABILITY_WORKFLOW_RUN_ID_KEY not in metadata
        assert OBSERVABILITY_STAGE_KEY not in metadata


# ---------------------------------------------------------------------------
# End-to-end: real dispatcher -> real source -> real coordinator -> spawn env
# ---------------------------------------------------------------------------
#
# Pins the load-bearing ordering: the dispatcher's current-signal ContextVar
# hook must be set when the coordinator builds the subprocess env, i.e. the
# stamping happens synchronously inside the handler's dispatch (same task),
# BEFORE the background talon process is spawned. Deleting the dispatcher
# hook — or deferring _dispatch_via_cli_background onto a fresh task after
# the handler returns — must fail this test.


class TestEndToEndWorkflowStageStamping:
    @pytest.mark.asyncio
    async def test_workflow_stage_signal_stamps_spawned_env(
        self, tmp_path, monkeypatch
    ):
        from kestrel_sovereign.signals import (
            OrderedLockManager,
            SignalDispatcher,
            SignalLogStore,
            SourceRegistry,
        )
        from kestrel_sovereign.signals.sources.talon_pipeline import (
            SOURCE_NAME,
            register_talon_pipeline_source,
        )
        from kestrel_sovereign.storage.db import SQLiteBackend

        monkeypatch.setenv("KESTREL_TALON_WORKSPACE_ROOT", str(tmp_path / "ws"))
        monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "home"))
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_e2e")

        agent = _make_agent(tmp_path)
        tracked: list = []

        def _track(coro, *, name):
            task = asyncio.create_task(coro, name=name)
            tracked.append(task)
            return task

        agent._track_background_task = _track
        feature = TalonCoordinatorFeature(agent)

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

        captured = {}

        async def fake_create(*argv, **kwargs):
            captured["env"] = kwargs["env"]
            proc = MagicMock()
            proc.pid = 999
            proc.returncode = None
            return proc

        backend = SQLiteBackend(str(tmp_path / "e2e.db"))
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

            signal = Signal(
                source=SOURCE_NAME,
                kind="workflow.stage",
                mode=SignalMode.ACTION,
                payload={"repo": "org/repo", "issue": 7, "wait": False},
                target_agent="did:web:k.example",
                visibility=Visibility.INTERNAL,
                session_id="run-e2e-42",
                urgency=Urgency.NORMAL,
                causation_chain=[CausationFrame(
                    agent_id="did:web:k.example",
                    source="workflow.rescue_flow.deploy.v2",
                    signal_id="run-e2e-42",
                    turn_id=None,
                    depth=0,
                    emitted_at=datetime.now(timezone.utc),
                )],
            )

            with patch.object(
                feature, "_dispatch_via_a2a", new_callable=AsyncMock
            ) as mock_a2a, patch.object(
                TalonCoordinatorFeature, "_workspace_state",
                return_value=ready_state,
            ), patch.object(
                TalonCoordinatorFeature, "_find_talon_bin",
                return_value="/usr/bin/kestrel-talon",
            ), patch.object(
                _asyncio, "create_subprocess_exec", side_effect=fake_create,
            ):
                mock_a2a.return_value = {
                    "dispatched": False, "reason": "no_a2a_host",
                }
                result = await dispatcher.dispatch_signal(signal)

            assert result.status is Status.OK, result.error
            assert result.action_result["state"] == "dispatched"

            env = captured["env"]
            assert env[OBSERVABILITY_ORCHESTRATOR_KEY] == "kestrel"
            assert env[OBSERVABILITY_WORKFLOW_RUN_ID_KEY] == "run-e2e-42"
            assert env[OBSERVABILITY_STAGE_KEY] == "deploy.v2"

            pending = [t for t in tracked if not t.done()]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        finally:
            await backend.close()
