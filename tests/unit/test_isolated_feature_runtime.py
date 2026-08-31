"""Tests for isolated feature runtime proxy behavior."""

import asyncio
import base64
import errno
import gc
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import traceback
import types
import weakref
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from kestrel_sdk.features import ContributionContractError
from kestrel_sdk.isolated_feature import (
    MAX_HOST_INGRESS_PAYLOAD_BYTES,
    ConfigTransitionError,
    ConfigTransitionResult,
    HostIngressCapabilities,
    HostIngressError,
    HostIngressUnknownNameError,
    HostIngressUnsupportedError,
)
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sovereign.features import isolated_runtime
from kestrel_sovereign.features.channels.route_ownership import (
    ChannelRouteOwnershipStore,
)
from kestrel_sovereign.features.contribution_runtime import (
    FeatureContributionCollectionError,
    FeatureContributionRuntime,
)
from kestrel_sovereign.features.isolated_runtime import (
    HostedTelegramRouteAttestation,
    IsolatedRuntimeConfigurationError,
    IsolatedRuntimeNamespaceError,
    IsolatedRuntimePreparationError,
    IsolatedRuntimeTelemetrySnapshot,
    ProxyFeature,
    SchedulerExecutionContextUnavailable,
    SchedulerTerminalAdmissionError,
    canonical_telegram_bot_id,
    configure_hosted_isolated_runtime_lifecycle,
    derive_isolated_runtime_namespace,
    resolve_agent_runtime_dir,
    resolve_isolated_runtime_namespace,
    set_hosted_telegram_route_attestation_resolver,
)
from kestrel_sovereign.features.scheduler.runner import (
    SCHEDULER_PROTOCOL_VERSION,
    ScheduledTask,
    SchedulerExecution,
    SchedulerRunner,
    _current_execution,
    _SchedulerExecutionScope,
    get_current_scheduler_execution,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.operator import OperatorRuntimeRegistry
from kestrel_sovereign.signals import SourceRegistry
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from kestrel_sovereign.ui_contributions import compute_ui_manifest
from kestrel_sovereign.waits import WaitRegistry

_TEST_AGENT_DID = "did:test:isolated-runtime"
_TEST_CONFIG_NODE_ID = f"feature_config:v2:{_TEST_AGENT_DID}:TestFeature"


@pytest.fixture(autouse=True)
def _clear_process_wide_channel_credentials(monkeypatch):
    """Unit behavior must not depend on an operator's loaded channel secrets."""

    for key in (
        "KESTREL_TELEGRAM_TOKEN",
        "TELEGRAM_BOT_TOKEN",
        "KESTREL_WHATSAPP_PROVIDER",
        "KESTREL_WHATSAPP_SESSION_DB",
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_WHATSAPP_FROM",
        "KESTREL_FEATURE_DATA_DIR",
        "KESTREL_DATA_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    # Historical proxy tests use inert absolute strings because their fake
    # clients never execute a subprocess. Keep those fixtures focused on their
    # lifecycle concern while the dedicated prebuilt-override tests below use
    # real filesystem shapes and exercise the production validator.
    validate_overrides = isolated_runtime._validate_hosted_process_prebuilt_overrides
    placeholders = {
        "/bin/test-service",
        "/bin/wa-service",
        "/operator/test-service",
    }

    def validate_non_placeholder_overrides(feature_name, *, runtime_venv=None):
        hidden = {}
        for suffix in ("BIN", "VENV"):
            key = isolated_runtime._env_key(feature_name, suffix)
            if os.environ.get(key) in placeholders:
                hidden[key] = os.environ.pop(key)
        try:
            return validate_overrides(feature_name, runtime_venv=runtime_venv)
        finally:
            os.environ.update(hidden)

    monkeypatch.setattr(
        isolated_runtime,
        "_validate_hosted_process_prebuilt_overrides",
        validate_non_placeholder_overrides,
    )
_TELEGRAM_ATTEMPT_TOKEN = "t" * 43


def _child_distribution_probe(version: str):
    """Build the classified child-distribution result used by venv tests."""

    if version == "missing":
        return isolated_runtime._FeatureDistributionProbe.missing()
    if version == "probe-failed":
        return isolated_runtime._FeatureDistributionProbe.failed()
    if version == "unknown":
        return isolated_runtime._FeatureDistributionProbe.present_unversioned()
    return isolated_runtime._FeatureDistributionProbe.versioned(version)


def test_proxy_contribution_owners_are_stable_and_runtime_specific():
    agent = Mock(did=_TEST_AGENT_DID, agent_id=_TEST_AGENT_DID)
    first_runtime = InstalledFeatureRuntime(
        class_name="FirstIsolatedFeature",
        entry_point="shared.feature:FirstIsolatedFeature",
        distribution="shared-isolated-package",
        runtime="isolated-venv",
        service="first-service",
    )
    second_runtime = InstalledFeatureRuntime(
        class_name="SecondIsolatedFeature",
        entry_point="shared.feature:SecondIsolatedFeature",
        distribution="shared-isolated-package",
        runtime="isolated-venv",
        service="second-service",
    )
    first = ProxyFeature(agent, first_runtime)
    restarted_first = ProxyFeature(agent, first_runtime)
    second = ProxyFeature(agent, second_runtime)

    assert first.contribution_owner == restarted_first.contribution_owner
    assert first.contribution_owner != second.contribution_owner
    assert first.contribution_owner.startswith("isolated-runtime:")
    assert len(first.contribution_owner) <= 256

    runtime = FeatureContributionRuntime(
        operator_registry=OperatorRuntimeRegistry(),
        wait_registry=WaitRegistry(),
        source_registry=SourceRegistry(),
    )
    prepared = runtime.prepare_transition((first, second))
    assert tuple(item.owner for item in prepared) == (
        first.contribution_owner,
        second.contribution_owner,
    )


def test_proxy_duplicate_runtime_owner_is_rejected_before_transition_mutation():
    agent = Mock(did=_TEST_AGENT_DID, agent_id=_TEST_AGENT_DID)
    runtime_metadata = InstalledFeatureRuntime(
        class_name="ConflictingIsolatedFeature",
        entry_point="conflict.feature:ConflictingIsolatedFeature",
        distribution="conflicting-isolated-package",
        runtime="isolated-venv",
        service="conflict-service",
    )
    first = ProxyFeature(agent, runtime_metadata)
    duplicate = ProxyFeature(agent, runtime_metadata)
    runtime = FeatureContributionRuntime(
        operator_registry=OperatorRuntimeRegistry(),
        wait_registry=WaitRegistry(),
        source_registry=SourceRegistry(),
    )

    with pytest.raises(FeatureContributionCollectionError) as exc_info:
        runtime.prepare_transition((first, duplicate))

    error = exc_info.value
    assert error.feature is duplicate
    assert error.stage == "contribution validation"
    assert error.getter == "validate_contribution_owner_uniqueness"
    assert str(error) == (
        "feature contribution failure during contribution validation "
        "(validate_contribution_owner_uniqueness)"
    )
    assert duplicate.contribution_owner not in str(error)
    assert "duplicate active feature contribution_owner" not in str(error)
    assert type(error.__cause__) is ContributionContractError
    assert "duplicate active feature contribution_owner" in str(error.__cause__)
    assert runtime.active_owners() == ()


class FakeIsolatedClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.event_handler = None
        self.calls = []

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def health(self):
        return True

    @property
    def capabilities(self):
        # Most tests model an outbound-only utility feature. Production
        # children must make the same explicit declaration before the default
        # hosted idle policy may retire an otherwise metadata-poor child.
        return {"inbound_producer": False}

    async def list_tools(self):
        # Real wire contract: services advertise a JSON-Schema ``input_schema``
        # (kestrel_sdk.isolated_feature.protocol.ToolMetadata), NOT a bare
        # ``parameters`` dict. The host must convert it into ToolParameters.
        return [
            {
                "name": "ping",
                "description": "Ping the isolated service",
                "category": "utility",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "text to echo",
                        }
                    },
                    "required": ["message"],
                },
                "command_prefix": "!ping",
            }
        ]

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return {"echo": args}

    def on_event(self, handler):
        self.event_handler = handler


class TelegramChannelClient(FakeIsolatedClient):
    """An isolated client which explicitly negotiates the Telegram bridge."""

    @property
    def capabilities(self):
        return {
            "channel": {
                "channel_type": "telegram",
                "send_tool": "telegram_send",
            }
        }


def _idle_test_runtime() -> InstalledFeatureRuntime:
    return InstalledFeatureRuntime(
        class_name="TestFeature",
        entry_point="test_pkg.feature:TestFeature",
        distribution="test-pkg",
        runtime="isolated-venv",
        service="test_service",
        description="Test proxy",
    )


def _configure_idle_lifecycle(agent, tmp_path, **kwargs) -> None:
    agent.isolated_runtime_root = tmp_path / "runtimes"
    agent.isolated_runtime_namespace = "tenant/agent"
    agent.isolated_runtime_scope = resolve_isolated_runtime_namespace(
        agent.isolated_runtime_root,
        agent.isolated_runtime_namespace,
    )
    configure_hosted_isolated_runtime_lifecycle(agent, **kwargs)


@pytest.mark.asyncio
async def test_idle_retirement_cold_starts_exactly_one_new_generation(
    monkeypatch, tmp_path
):
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    snapshots: list[IsolatedRuntimeTelemetrySnapshot] = []
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=snapshots.append,
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=client_factory)
    await feature.initialize()
    assert feature._last_used_monotonic is not None
    feature._last_used_monotonic -= 7200

    retired = await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )

    assert retired is True
    assert clients[0].stopped is True
    assert feature.runtime_telemetry_snapshot().state == "idle"
    assert [tool.name for tool in feature.get_tools()] == ["ping"]

    results = await asyncio.gather(
        feature.call_isolated_tool("ping", {"message": "first"}),
        feature.call_isolated_tool("ping", {"message": "second"}),
    )

    assert len(clients) == 2
    assert all(result["success"] is True for result in results)
    snapshot = feature.runtime_telemetry_snapshot()
    assert snapshot.state == "running"
    assert snapshot.restart_count == 0
    assert snapshot.idle_wake_count == 1
    assert snapshot.lifecycle_generation == 2
    assert snapshot.last_used_at is not None
    assert not hasattr(snapshot, "command")
    assert not hasattr(snapshot, "environment")
    # Observer delivery is intentionally off-loop and best-effort. Under the
    # full xdist load the worker can settle after the lifecycle assertions, so
    # synchronize with the public observation instead of assuming immediate
    # executor service.
    for _ in range(200):
        if snapshots:
            break
        await asyncio.sleep(0.01)
    assert {item.feature for item in snapshots} == {"TestFeature"}
    await feature.shutdown()


@pytest.mark.asyncio
async def test_idle_deadline_loses_to_already_admitted_tool(monkeypatch, tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingClient(FakeIsolatedClient):
        async def call_tool(self, name, args):
            entered.set()
            await release.wait()
            return await super().call_tool(name, args)

    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=BlockingClient)
    await feature.initialize()
    assert feature._last_used_monotonic is not None
    feature._last_used_monotonic -= 7200
    stale_generation = feature._activity_generation
    stale_last_used = feature._last_used_monotonic

    call = asyncio.create_task(
        feature.call_isolated_tool("ping", {"message": "wins"})
    )
    await entered.wait()
    retirement = asyncio.create_task(
        feature._retire_idle_generation(
            expected_activity_generation=stale_generation,
            expected_last_used=stale_last_used,
        )
    )
    await asyncio.sleep(0)
    release.set()

    assert (await call)["success"] is True
    assert await retirement is False
    assert feature.runtime_telemetry_snapshot().state == "running"
    await feature.shutdown()


@pytest.mark.asyncio
async def test_idle_retirement_refuses_work_admitted_before_monitor_baseline(
    monkeypatch, tmp_path
):
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingClient(FakeIsolatedClient):
        async def call_tool(self, name, args):
            entered.set()
            await release.wait()
            return await super().call_tool(name, args)

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=BlockingClient)
    await feature.initialize()

    call = asyncio.create_task(feature.call_isolated_tool("ping", {"message": "long"}))
    await entered.wait()
    # Model a call that outlives the idle deadline. Capture the monitor baseline
    # only after admission so generation/deadline fences match and the atomic
    # active-reader guard is the sole reason retirement loses.
    feature._last_used_monotonic -= 7200
    monitor_generation = feature._activity_generation
    monitor_last_used = feature._last_used_monotonic

    assert not await feature._retire_idle_generation(
        expected_activity_generation=monitor_generation,
        expected_last_used=monitor_last_used,
    )
    assert feature._traffic_gate.closed is False
    assert feature._client is not None

    release.set()
    assert (await call)["success"] is True
    await feature.shutdown()


@pytest.mark.asyncio
async def test_idle_stop_failure_seals_and_retains_exact_client_for_retry(
    monkeypatch, tmp_path
):
    class RetryStopClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise RuntimeError("private child failure")
            await super().stop()

    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = RetryStopClient()
    feature = ProxyFeature(
        agent, _idle_test_runtime(), client_factory=lambda **_kwargs: client
    )
    await feature.initialize()
    assert feature._last_used_monotonic is not None
    feature._last_used_monotonic -= 7200

    assert not await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    assert feature._terminal_lifecycle_latched is True
    assert feature._traffic_gate.sealed is True
    assert feature._client is None
    assert feature._terminal_retirement_clients == []
    assert client.stop_calls == 2
    assert client.stopped is True
    stopped = feature.runtime_telemetry_snapshot()
    assert stopped.state == "stopped"
    assert stopped.active_processes == 0
    assert stopped.cleanup_eligible is False
    assert (await feature.call_isolated_tool("ping", {}))["status"] == "error"

    await feature.shutdown()
    assert client.stop_calls == 2
    assert client.stopped is True


@pytest.mark.asyncio
async def test_idle_monitor_reports_terminal_retirement_failure_not_producer(
    monkeypatch, tmp_path, caplog
):
    class RetryStopClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise RuntimeError("private child failure")
            await super().stop()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=0.01)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = RetryStopClient()
    feature = ProxyFeature(
        agent, _idle_test_runtime(), client_factory=lambda **_kwargs: client
    )

    with caplog.at_level("WARNING"):
        await feature.initialize()
        feature._last_used_monotonic -= 1
        for _ in range(200):
            if any(
                "retirement entered terminal cleanup" in message
                for message in caplog.messages
            ):
                break
            await asyncio.sleep(0.01)

    assert feature._terminal_lifecycle_latched is True
    assert feature._idle_monitor_task is None
    assert feature._owns_inbound_producer() is False
    assert any(
        "retirement entered terminal cleanup" in message
        for message in caplog.messages
    )
    assert not any(
        "owns an inbound producer" in message for message in caplog.messages
    )
    await feature.shutdown()


@pytest.mark.asyncio
async def test_shutdown_latch_beats_queued_idle_retirement(monkeypatch, tmp_path):
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = FakeIsolatedClient()
    feature = ProxyFeature(
        agent, _idle_test_runtime(), client_factory=lambda **_kwargs: client
    )
    await feature.initialize()
    assert feature._last_used_monotonic is not None
    feature._last_used_monotonic -= 7200

    async with feature._reload_lock:
        retirement = asyncio.create_task(
            feature._retire_idle_generation(
                expected_activity_generation=feature._activity_generation,
                expected_last_used=feature._last_used_monotonic,
            )
        )
        await asyncio.sleep(0)
        shutdown = asyncio.create_task(feature.shutdown())
        await asyncio.sleep(0)
        assert feature._terminal_lifecycle_latched is True

    assert await retirement is False
    await shutdown
    assert client.stopped is True
    assert feature.runtime_telemetry_snapshot().state == "stopped"


@pytest.mark.asyncio
async def test_idle_monitor_emits_retirement_without_clock_polling(monkeypatch, tmp_path):
    retired = asyncio.Event()

    def observe(snapshot):
        if snapshot.state == "idle":
            retired.set()

    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    _configure_idle_lifecycle(
        agent, tmp_path, idle_timeout_seconds=0.01, telemetry_observer=observe
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = FakeIsolatedClient()
    feature = ProxyFeature(
        agent, _idle_test_runtime(), client_factory=lambda **_kwargs: client
    )

    await feature.initialize()
    await asyncio.wait_for(retired.wait(), timeout=1)

    assert client.stopped is True
    assert feature.runtime_telemetry_snapshot().cleanup_eligible is True
    await feature.shutdown()
    stopped = feature.runtime_telemetry_snapshot()
    assert stopped.state == "stopped"
    assert stopped.cleanup_eligible is False
    with pytest.raises(
        IsolatedRuntimePreparationError,
        match="not idle and reclaimable",
    ):
        await feature.reclaim_idle_workspace()


@pytest.mark.asyncio
async def test_async_telemetry_observer_cannot_hang_child_lifecycle(
    monkeypatch, tmp_path
):
    observer_started = asyncio.Event()
    release_observer = asyncio.Event()

    async def observe(_snapshot):
        observer_started.set()
        while True:
            try:
                await release_observer.wait()
                return
            except asyncio.CancelledError:
                continue

    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=observe,
    )
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_OBSERVER_TIMEOUT", 0.01)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = FakeIsolatedClient()
    feature = ProxyFeature(
        agent, _idle_test_runtime(), client_factory=lambda **_kwargs: client
    )

    await asyncio.wait_for(feature.initialize(), timeout=1)

    await asyncio.wait_for(observer_started.wait(), timeout=1)
    assert feature._telemetry_observer_tasks
    assert client.started is True
    await feature._emit_runtime_telemetry()
    assert len(feature._telemetry_observer_tasks) == 1
    release_observer.set()
    await asyncio.wait_for(feature.shutdown(), timeout=1)


@pytest.mark.asyncio
async def test_reinitialize_after_idle_retirement_restores_live_supervision(
    monkeypatch, tmp_path
):
    probed = asyncio.Event()

    class ProbedClient(FakeIsolatedClient):
        async def health(self):
            probed.set()
            return True

    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = ProbedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=client_factory)
    await feature.initialize()
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    await feature.shutdown()
    probed.clear()

    await feature.initialize()
    await asyncio.wait_for(probed.wait(), timeout=2)

    snapshot = feature.runtime_telemetry_snapshot()
    assert len(clients) == 2
    assert feature._idle_retired is False
    assert snapshot.state == "running"
    assert snapshot.active_processes == 1
    assert snapshot.cleanup_eligible is False
    await feature.shutdown()


@pytest.mark.asyncio
async def test_ordinary_inflight_health_probe_does_not_report_retirement_uncertain(
    monkeypatch, tmp_path
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    release_probe = asyncio.Event()
    probe = asyncio.create_task(release_probe.wait())
    feature._own_health_probe_task(probe)

    snapshot = feature.runtime_telemetry_snapshot()

    assert snapshot.state == "running"
    assert snapshot.active_processes == 1
    release_probe.set()
    await probe
    await feature.shutdown()


@pytest.mark.asyncio
async def test_transient_idle_wake_failure_reopens_for_later_retry(monkeypatch, tmp_path):
    class FailingStartClient(FakeIsolatedClient):
        async def start(self):
            raise RuntimeError("private transient start failure")

    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = (
            FailingStartClient(**kwargs)
            if len(clients) == 1
            else FakeIsolatedClient(**kwargs)
        )
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=client_factory)
    await feature.initialize()
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )

    failed = await feature.call_isolated_tool("ping", {"message": "first"})
    recovered = await feature.call_isolated_tool("ping", {"message": "retry"})

    assert failed == {
        "status": "error",
        "error": "isolated feature could not start",
        "tool": "ping",
        "success": False,
    }
    assert recovered["success"] is True
    assert len(clients) == 3
    assert feature._traffic_gate.sealed is False
    assert feature._terminal_lifecycle_latched is False
    await feature.shutdown()


@pytest.mark.asyncio
async def test_post_publication_wake_failure_reparks_idle_supervisor(
    monkeypatch, tmp_path
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    monkeypatch.setattr(
        feature,
        "_register_event_handler",
        AsyncMock(side_effect=RuntimeError("private registration failure")),
    )

    failed = await feature.call_isolated_tool("ping", {"message": "wake"})

    assert failed["success"] is False
    assert feature._idle_retired is True
    assert feature._client is None
    assert feature._idle_resume_event.is_set() is False
    await feature.shutdown()


@pytest.mark.asyncio
async def test_idle_set_config_wakes_and_runs_negotiated_validation(monkeypatch, tmp_path):
    old_config = {"enabled": True, "token": "old-token"}
    rejected_config = {"enabled": True, "token": "rejected-token"}
    prepared = []

    class RejectingTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            prepared.append(dict(config))
            raise ConfigTransitionError("feature rejected this config")

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = RejectingTransitionClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    await feature.persist_config(old_config)
    await feature.initialize()
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )

    with pytest.raises(ConfigTransitionError, match="feature rejected"):
        await feature.set_config(rejected_config)

    assert len(clients) == 2
    assert prepared == [rejected_config]
    assert feature.runtime_telemetry_snapshot().state == "running"
    assert agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties["config"] == old_config
    await feature.shutdown()


@pytest.mark.asyncio
async def test_idle_unstartable_child_allows_durable_config_repair(monkeypatch, tmp_path):
    bad_config = {"enabled": True, "token": "bad-token"}
    fixed_config = {"enabled": True, "token": "fixed-token"}

    class FailingWakeClient(FakeIsolatedClient):
        async def start(self):
            raise RuntimeError("bad active config prevents startup")

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = (
            FakeIsolatedClient(**kwargs)
            if not clients
            else FailingWakeClient(**kwargs)
        )
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    await feature.persist_config(bad_config)
    await feature.initialize()
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )

    await feature.set_config(fixed_config)

    assert len(clients) == 2
    assert feature._client is None
    assert feature._idle_retired is True
    assert agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties["config"] == fixed_config
    await feature.shutdown()


@pytest.mark.asyncio
async def test_terminal_latch_during_idle_wake_routes_config_to_repair(
    monkeypatch, tmp_path
):
    old_config = {"enabled": True, "token": "old-token"}
    repaired_config = {"enabled": True, "token": "new-token"}
    wake_started = asyncio.Event()
    release_wake = asyncio.Event()

    class SlowWakeClient(FakeIsolatedClient):
        async def start(self):
            wake_started.set()
            await release_wake.wait()
            await super().start()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = (
            FakeIsolatedClient(**kwargs)
            if not clients
            else SlowWakeClient(**kwargs)
        )
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    await feature.persist_config(old_config)
    await feature.initialize()
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    update = asyncio.create_task(feature.set_config(repaired_config))
    await asyncio.wait_for(wake_started.wait(), timeout=1)
    feature._latch_terminal_lifecycle()
    release_wake.set()

    await update

    assert feature._terminal_lifecycle_latched is True
    assert feature._idle_retired is False
    assert agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties["config"] == repaired_config
    await feature.shutdown()


@pytest.mark.asyncio
async def test_successful_recovery_clears_stale_nonterminal_uncertainty(
    monkeypatch, tmp_path
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    # Model the sticky marker left after a non-terminal facade timeout whose
    # exact late task and retained facade subsequently settled.
    feature._terminal_cleanup_uncertain = True
    assert feature.runtime_telemetry_snapshot().state == "running"

    async with feature._reload_lock:
        await feature._replace_client(feature._host_config)

    assert feature._terminal_cleanup_uncertain is False
    assert feature.runtime_telemetry_snapshot().state == "running"
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    idle = feature.runtime_telemetry_snapshot()
    assert idle.state == "idle"
    assert idle.cleanup_eligible is True
    await feature.reclaim_idle_workspace()
    await feature.shutdown()


@pytest.mark.asyncio
async def test_reclaim_and_telemetry_share_live_retirement_evidence(
    monkeypatch, tmp_path
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    retained = FakeIsolatedClient()
    feature._retain_terminal_retirement_client(retained)

    uncertain = feature.runtime_telemetry_snapshot()
    assert uncertain.state == "retirement-uncertain"
    assert uncertain.cleanup_eligible is False
    with pytest.raises(
        IsolatedRuntimePreparationError,
        match="not idle and reclaimable",
    ):
        await feature.reclaim_idle_workspace()

    feature._release_terminal_retirement_client(retained)
    assert feature.runtime_telemetry_snapshot().cleanup_eligible is True
    await feature.reclaim_idle_workspace()
    await feature.shutdown()


@pytest.mark.asyncio
async def test_contained_immutable_venv_override_is_never_cleanup_eligible(
    monkeypatch, tmp_path
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    runtime_dir = feature._prepare_runtime_workspace()
    immutable_venv = runtime_dir / ".venv"
    _write_prebuilt_venv_shape(immutable_venv)
    monkeypatch.setenv(
        "KESTREL_FEATURE_TESTFEATURE_VENV",
        str(immutable_venv),
    )
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    feature._idle_retired = True
    feature._client = None

    snapshot = await feature.sample_runtime_telemetry(refresh_disk=True)

    assert snapshot.state == "idle"
    assert snapshot.cleanup_eligible is False
    assert snapshot.environment_bytes is None
    assert feature._runtime_venv_is_core_managed() is False
    with pytest.raises(
        IsolatedRuntimePreparationError,
        match=(
            "contains the immutable venv selected by "
            "KESTREL_FEATURE_TESTFEATURE_VENV"
        ),
    ) as raised:
        await feature.reclaim_idle_workspace()
    assert str(tmp_path) not in str(raised.value)
    assert immutable_venv.is_dir()
    assert isolated_runtime._venv_python(immutable_venv).is_file()


@pytest.mark.asyncio
async def test_external_immutable_venv_with_bin_does_not_block_workspace_reclaim(
    monkeypatch, tmp_path
):
    external_venv = tmp_path / "operator" / "prebuilt-venv"
    external_python = _write_prebuilt_venv_shape(external_venv)
    external_bin = tmp_path / "operator" / "test-service"
    external_bin.write_text("#!/bin/sh\nexit 0\n")
    external_bin.chmod(0o700)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_VENV", str(external_venv))
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", str(external_bin))

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    runtime_dir = feature._prepare_runtime_workspace()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    feature._idle_retired = True
    feature._client = None

    assert feature._venv_path == runtime_dir / ".venv"
    assert feature._bin_path == external_bin.resolve()
    assert feature._validated_hosted_immutable_venv_path == external_venv.resolve()
    assert feature.runtime_telemetry_snapshot().cleanup_eligible is True
    assert (
        await feature.reclaim_idle_workspace()
        is isolated_runtime.RuntimeNamespaceCleanupOutcome.REMOVED
    )
    assert not runtime_dir.exists()
    assert external_venv.is_dir()
    assert external_python.is_file()
    assert external_bin.is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ("process-env", "runtime-metadata"))
async def test_failed_idle_wake_revalidation_keeps_immutable_venv_custody(
    monkeypatch, tmp_path, selection
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    probe = ProxyFeature(
        agent,
        _idle_test_runtime(),
        client_factory=FakeIsolatedClient,
    )
    runtime_dir = probe._prepare_runtime_workspace()
    immutable_venv = runtime_dir / ".venv"
    immutable_python = _write_prebuilt_venv_shape(immutable_venv)

    if selection == "process-env":
        setting = "KESTREL_FEATURE_TESTFEATURE_VENV"
        monkeypatch.setenv(setting, str(immutable_venv))
        feature = probe
    else:
        setting = "runtime.venv"
        runtime = _idle_test_runtime()
        feature = ProxyFeature(
            agent,
            InstalledFeatureRuntime(
                class_name=runtime.class_name,
                entry_point=runtime.entry_point,
                distribution=runtime.distribution,
                runtime=runtime.runtime,
                service=runtime.service,
                project=runtime.project,
                description=runtime.description,
                venv=str(immutable_venv),
            ),
            client_factory=FakeIsolatedClient,
        )
        feature._prepare_runtime_workspace()

    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    feature._idle_retired = True
    feature._client = None
    assert feature.runtime_telemetry_snapshot().cleanup_eligible is False
    validated_path = feature._validated_hosted_immutable_venv_path

    real_unsafe = isolated_runtime._hosted_immutable_metadata_is_unsafe
    monkeypatch.setattr(
        isolated_runtime,
        "_hosted_immutable_metadata_is_unsafe",
        lambda _metadata: True,
    )
    with pytest.raises(
        IsolatedRuntimePreparationError,
        match="could not be prepared after idle retirement",
    ):
        await feature._wake_idle_runtime()
    monkeypatch.setattr(
        isolated_runtime,
        "_hosted_immutable_metadata_is_unsafe",
        real_unsafe,
    )

    snapshot = feature.runtime_telemetry_snapshot()
    assert snapshot.state == "idle"
    assert snapshot.cleanup_eligible is False
    assert feature._hosted_immutable_venv_custody_unproven is True
    assert feature._validated_hosted_immutable_venv_path == validated_path
    with pytest.raises(
        IsolatedRuntimePreparationError,
        match=f"custody selected by {re.escape(setting)} could not be proven",
    ) as raised:
        await feature.reclaim_idle_workspace()
    assert str(tmp_path) not in str(raised.value)
    assert immutable_venv.is_dir()
    assert immutable_python.is_file()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    assert feature._hosted_immutable_venv_custody_unproven is False
    assert feature.runtime_telemetry_snapshot().cleanup_eligible is False


@pytest.mark.asyncio
async def test_idle_ui_manifest_retains_last_published_contribution(monkeypatch, tmp_path):
    static_dir = None

    class UIClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            assert static_dir is not None
            return {
                "inbound_producer": False,
                "ui_contributions": {
                    "modules": ["panel.mjs"],
                    "css": [],
                    "static_dir": str(static_dir),
                    "capability": "testfeature",
                }
            }

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=UIClient)
    static_dir = (
        feature._feature_runtime_dir()
        / ".venv"
        / "lib"
        / "python3.14"
        / "site-packages"
        / "test_pkg"
        / "static"
    )
    static_dir.mkdir(parents=True)
    (static_dir / "panel.mjs").write_text("export default {};\n")
    agent.features = {"TestFeature": feature}
    await feature.initialize()
    running_manifest = compute_ui_manifest(agent)
    feature._last_used_monotonic -= 7200

    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )

    assert compute_ui_manifest(agent) == running_manifest
    assert running_manifest == [
        {
            "feature": "TestFeature",
            "capability": "testfeature",
            "modules": ["/features/testfeature/static/panel.mjs"],
            "css": [],
        }
    ]
    assert (
        await feature.reclaim_idle_workspace()
        is isolated_runtime.RuntimeNamespaceCleanupOutcome.REMOVED
    )
    assert static_dir.exists() is False
    assert compute_ui_manifest(agent) == []
    await feature.shutdown()
    stopped = feature.runtime_telemetry_snapshot()
    assert stopped.state == "stopped"
    assert compute_ui_manifest(agent) == []


@pytest.mark.asyncio
async def test_post_connect_wake_telemetry_failure_never_marks_live_child_idle(
    monkeypatch, tmp_path
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=client_factory)
    await feature.initialize()
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    monkeypatch.setattr(
        feature,
        "_emit_runtime_telemetry",
        AsyncMock(side_effect=RuntimeError("synthetic telemetry failure")),
    )

    result = await feature.call_isolated_tool("ping", {"message": "first"})
    await asyncio.sleep(0)

    assert result["success"] is True
    assert len(clients) == 2
    assert feature._client is clients[1]
    assert feature._idle_retired is False
    snapshot = feature.runtime_telemetry_snapshot()
    assert snapshot.state == "running"
    assert snapshot.cleanup_eligible is False
    assert (await feature.call_isolated_tool("ping", {"message": "retry"}))["success"]
    await feature.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        IsolatedRuntimeNamespaceError("namespace changed"),
        IsolatedRuntimeConfigurationError("immutable venv changed"),
        RuntimeError("uv unavailable"),
    ],
)
async def test_idle_wake_preparation_failures_use_stable_tool_envelope(
    monkeypatch, tmp_path, failure
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    monkeypatch.setattr(
        feature,
        "_prepare_runtime_workspace",
        Mock(side_effect=failure),
    )

    result = await feature.call_isolated_tool("ping", {"message": "wake"})

    assert result == {
        "status": "error",
        "error": "isolated feature could not start",
        "tool": "ping",
        "success": False,
    }
    assert feature._idle_retired is True
    assert feature._traffic_gate.sealed is False
    await feature.shutdown()


@pytest.mark.asyncio
async def test_idle_wake_recreates_missing_core_managed_venv(monkeypatch, tmp_path):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.delenv("KESTREL_FEATURE_TESTFEATURE_BIN", raising=False)
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    provision_calls = 0

    async def provision():
        nonlocal provision_calls
        provision_calls += 1
        assert feature._venv_path is not None
        (feature._venv_path / "bin").mkdir(parents=True, exist_ok=True)
        (feature._venv_path / "bin" / "python").write_text("managed")
        feature._last_cache_hit = False

    monkeypatch.setattr(feature, "_ensure_venv_without_blocking_event_loop", provision)
    await feature.initialize()
    assert feature._venv_path is not None
    original_venv = feature._venv_path
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    shutil.rmtree(original_venv)

    result = await feature.call_isolated_tool("ping", {"message": "wake"})

    assert result["success"] is True
    assert provision_calls == 2
    assert (original_venv / "bin" / "python").read_text() == "managed"
    await feature.shutdown()


@pytest.mark.asyncio
async def test_idle_wake_reloads_durable_config_changed_by_another_replica(
    monkeypatch, tmp_path
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    durable_config = {"api_token": "old-token"}
    clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=client_factory)

    async def load_host_config():
        return dict(durable_config)

    monkeypatch.setattr(feature, "_load_host_config", load_host_config)
    await feature.initialize()
    assert clients[0].kwargs["config"] == {"api_token": "old-token"}
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )

    durable_config["api_token"] = "new-token"
    assert (await feature.call_isolated_tool("ping", {"message": "wake"}))["success"]

    assert len(clients) == 2
    assert clients[1].kwargs["config"] == {"api_token": "new-token"}
    await feature.shutdown()


@pytest.mark.asyncio
async def test_idle_disk_telemetry_never_holds_reload_lock(monkeypatch, tmp_path):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=lambda _snapshot: None,
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    refresh_started = asyncio.Event()

    async def stalled_refresh(**_kwargs):
        refresh_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(feature, "_refresh_disk_telemetry", stalled_refresh)
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    await asyncio.wait_for(refresh_started.wait(), timeout=1)

    await asyncio.wait_for(feature.shutdown(), timeout=1)
    assert feature._reload_lock.locked() is False


@pytest.mark.asyncio
async def test_slow_observer_coalesces_idle_cleanup_snapshot(monkeypatch, tmp_path):
    running_started = asyncio.Event()
    release_running = asyncio.Event()
    snapshots = []

    async def observe(snapshot):
        snapshots.append(snapshot)
        if snapshot.state == "running":
            running_started.set()
            await release_running.wait()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=observe,
    )
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_OBSERVER_TIMEOUT", 0.01)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    await asyncio.wait_for(running_started.wait(), timeout=1)
    for _ in range(100):
        if not feature._telemetry_emit_tasks:
            break
        await asyncio.sleep(0.01)
    schedule_calls = 0
    original_schedule = feature._schedule_runtime_telemetry

    def count_schedule(**kwargs):
        nonlocal schedule_calls
        schedule_calls += 1
        return original_schedule(**kwargs)

    monkeypatch.setattr(feature, "_schedule_runtime_telemetry", count_schedule)
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    for _ in range(100):
        if feature._telemetry_observer_emit_pending:
            break
        await asyncio.sleep(0.01)
    assert feature._telemetry_observer_emit_pending is True
    await asyncio.sleep(0.05)
    assert schedule_calls == 1

    release_running.set()
    for _ in range(100):
        if any(snapshot.cleanup_eligible for snapshot in snapshots):
            break
        await asyncio.sleep(0.01)

    assert [snapshot.state for snapshot in snapshots] == ["running", "idle"]
    assert snapshots[-1].cleanup_eligible is True
    assert schedule_calls == 2
    await feature.shutdown()


@pytest.mark.asyncio
async def test_saturated_observer_executor_retries_forced_idle_snapshot(
    monkeypatch, tmp_path, caplog
):
    executor = isolated_runtime._BoundedDaemonExecutor(
        max_workers=1,
        queue_capacity=1,
    )
    running = threading.Event()
    release = threading.Event()

    def block():
        running.set()
        assert release.wait(timeout=2)

    active = executor.submit(block)
    assert running.wait(timeout=1)
    queued = executor.submit(lambda: None)
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_OBSERVER_EXECUTOR", executor)
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_RETRY_BASE_SECONDS", 0.01)
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_RETRY_MAX_SECONDS", 0.02)
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_FORCED_RETRY_LIMIT", 1)

    snapshots = []
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=snapshots.append,
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    for _ in range(200):
        if (
            feature._telemetry_retry_attempt
            > isolated_runtime._TELEMETRY_FORCED_RETRY_LIMIT
            and feature._telemetry_retry_task is not None
        ):
            break
        await asyncio.sleep(0.01)
    assert feature._telemetry_retry_task is not None
    assert (
        feature._telemetry_retry_attempt
        > isolated_runtime._TELEMETRY_FORCED_RETRY_LIMIT
    )
    assert caplog.messages.count(
        "Hosted isolated runtime telemetry observer capacity was unavailable "
        "for TestFeature"
    ) == 1
    assert not any("telemetry observer failed" in line for line in caplog.messages)

    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    assert snapshots == []

    release.set()
    await asyncio.to_thread(active.result, 1)
    await asyncio.to_thread(queued.result, 1)
    for _ in range(200):
        if snapshots and snapshots[-1].cleanup_eligible:
            break
        await asyncio.sleep(0.01)

    assert snapshots[-1].state == "idle"
    assert snapshots[-1].cleanup_eligible is True
    await feature.shutdown()
    executor.shutdown()


@pytest.mark.asyncio
async def test_forced_idle_snapshot_retries_after_builder_failure(monkeypatch, tmp_path):
    snapshots = []
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=snapshots.append,
    )
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_RETRY_BASE_SECONDS", 0.01)
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_RETRY_MAX_SECONDS", 0.02)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    for _ in range(100):
        if snapshots and not feature._telemetry_emit_tasks:
            break
        await asyncio.sleep(0.01)
    snapshots.clear()

    original_builder = feature._build_runtime_telemetry_snapshot
    failures_remaining = 1

    def fail_once(*args):
        nonlocal failures_remaining
        if failures_remaining:
            failures_remaining -= 1
            raise LookupError("hostile facade snapshot failure")
        return original_builder(*args)

    monkeypatch.setattr(feature, "_build_runtime_telemetry_snapshot", fail_once)
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    for _ in range(200):
        if snapshots and snapshots[-1].cleanup_eligible:
            break
        await asyncio.sleep(0.01)

    assert failures_remaining == 0
    assert snapshots[-1].state == "idle"
    assert snapshots[-1].cleanup_eligible is True
    await feature.shutdown()


def test_telemetry_observer_submission_lock_is_agent_scoped(monkeypatch, tmp_path):
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    first_runtime_root = tmp_path / "first-runtime"
    second_runtime_root = tmp_path / "second-runtime"
    first_runtime_root.mkdir()
    second_runtime_root.mkdir()
    first_agent = Mock(did=_TEST_AGENT_DID, features={})
    first_agent.storage_path = str(tmp_path / "first" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        first_agent,
        first_runtime_root,
        idle_timeout_seconds=3600,
        telemetry_observer=lambda _snapshot: None,
    )
    first = ProxyFeature(
        first_agent, _idle_test_runtime(), client_factory=FakeIsolatedClient
    )
    sibling = ProxyFeature(
        first_agent, _idle_test_runtime(), client_factory=FakeIsolatedClient
    )

    second_agent = Mock(did="did:key:zSecond", features={})
    second_agent.storage_path = str(tmp_path / "second" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        second_agent,
        second_runtime_root,
        idle_timeout_seconds=3600,
        telemetry_observer=lambda _snapshot: None,
    )
    other = ProxyFeature(
        second_agent, _idle_test_runtime(), client_factory=FakeIsolatedClient
    )

    assert first._telemetry_observer_agent_lock is sibling._telemetry_observer_agent_lock
    assert first._telemetry_observer_agent_lock is not other._telemetry_observer_agent_lock
    assert (
        first._telemetry_observer_agent_admission
        is sibling._telemetry_observer_agent_admission
    )
    assert (
        first._telemetry_observer_agent_admission
        is not other._telemetry_observer_agent_admission
    )


@pytest.mark.asyncio
async def test_terminal_cancellation_keeps_sync_observer_admission_until_settlement(
    monkeypatch, tmp_path
):
    executor = isolated_runtime._BoundedDaemonExecutor(
        max_workers=2,
        queue_capacity=2,
    )
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_OBSERVER_EXECUTOR", executor)
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_OBSERVER_TIMEOUT", 0.01)
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_RETRY_BASE_SECONDS", 1.0)
    release = threading.Event()
    first_started = threading.Event()
    first_finished = threading.Event()
    same_agent_calls = 0
    calls_lock = threading.Lock()

    def blocking_observer(_snapshot):
        nonlocal same_agent_calls
        with calls_lock:
            same_agent_calls += 1
        first_started.set()
        try:
            assert release.wait(timeout=2)
        finally:
            first_finished.set()

    first_agent = Mock(did=_TEST_AGENT_DID, features={})
    first_agent.storage_path = str(tmp_path / "first" / "kestrel_prime.db")
    (tmp_path / "first-runtime").mkdir()
    _configure_idle_lifecycle(
        first_agent,
        tmp_path / "first-runtime",
        idle_timeout_seconds=3600,
        telemetry_observer=blocking_observer,
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    monkeypatch.setenv("KESTREL_FEATURE_SIBLINGFEATURE_BIN", "/bin/test-service")
    first = ProxyFeature(
        first_agent,
        _idle_test_runtime(),
        client_factory=FakeIsolatedClient,
    )
    sibling_runtime = InstalledFeatureRuntime(
        class_name="SiblingFeature",
        entry_point="test_pkg.feature:SiblingFeature",
        distribution="sibling-pkg",
        runtime="isolated-venv",
        service="sibling_service",
    )
    sibling = ProxyFeature(
        first_agent,
        sibling_runtime,
        client_factory=FakeIsolatedClient,
    )

    other_snapshots = []
    other_agent = Mock(did="did:key:zOtherTenant", features={})
    other_agent.storage_path = str(tmp_path / "other" / "kestrel_prime.db")
    (tmp_path / "other-runtime").mkdir()
    _configure_idle_lifecycle(
        other_agent,
        tmp_path / "other-runtime",
        idle_timeout_seconds=3600,
        telemetry_observer=other_snapshots.append,
    )
    other = ProxyFeature(
        other_agent,
        _idle_test_runtime(),
        client_factory=FakeIsolatedClient,
    )

    first_emit = asyncio.create_task(first._emit_runtime_telemetry())
    try:
        assert await asyncio.to_thread(first_started.wait, 1)
        first._latch_terminal_lifecycle()
        await asyncio.wait_for(first_emit, timeout=1)

        await sibling._emit_runtime_telemetry()
        await asyncio.sleep(0.05)
        assert same_agent_calls == 1

        await other._emit_runtime_telemetry()
        for _ in range(100):
            if other_snapshots:
                break
            await asyncio.sleep(0.01)
        assert len(other_snapshots) == 1

        release.set()
        assert await asyncio.to_thread(first_finished.wait, 1)
        await sibling._emit_runtime_telemetry()
        for _ in range(100):
            if same_agent_calls == 2:
                break
            await asyncio.sleep(0.01)
        assert same_agent_calls == 2
    finally:
        sibling._latch_terminal_lifecycle()
        other._latch_terminal_lifecycle()
        release.set()
        await asyncio.gather(first_emit, return_exceptions=True)
        executor.shutdown()


@pytest.mark.asyncio
async def test_inflight_sync_emit_coalesces_idle_cleanup_snapshot(monkeypatch, tmp_path):
    snapshots = []
    build_started = threading.Event()
    release_build = threading.Event()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=snapshots.append,
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    for _ in range(100):
        if snapshots and not feature._telemetry_emit_tasks:
            break
        await asyncio.sleep(0.01)
    snapshots.clear()

    original_builder = feature._build_runtime_telemetry_snapshot

    def slow_builder(*args):
        build_started.set()
        assert release_build.wait(timeout=2)
        return original_builder(*args)

    monkeypatch.setattr(feature, "_build_runtime_telemetry_snapshot", slow_builder)
    feature._schedule_runtime_telemetry(force=True)
    assert await asyncio.to_thread(build_started.wait, 1)
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    assert feature._telemetry_emit_pending is True

    release_build.set()
    for _ in range(100):
        if snapshots and snapshots[-1].cleanup_eligible:
            break
        await asyncio.sleep(0.01)

    assert snapshots[-1].state == "idle"
    assert snapshots[-1].cleanup_eligible is True
    assert feature._telemetry_emit_pending is False
    await feature.shutdown()


@pytest.mark.asyncio
async def test_sync_observer_never_blocks_initialize_or_event_loop(monkeypatch, tmp_path):
    loop_thread = threading.get_ident()
    observer_threads = []
    observer_started = threading.Event()
    observer_finished = threading.Event()
    release_observer = threading.Event()
    heartbeat_ticks = 0
    heartbeat_done = asyncio.Event()

    def observe(_snapshot):
        observer_threads.append(threading.get_ident())
        observer_started.set()
        assert release_observer.wait(timeout=2)
        observer_finished.set()

    async def heartbeat():
        nonlocal heartbeat_ticks
        for _ in range(5):
            await asyncio.sleep(0)
            heartbeat_ticks += 1
        heartbeat_done.set()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=observe,
    )
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_OBSERVER_TIMEOUT", 0.01)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)

    heartbeat_task = asyncio.create_task(heartbeat())
    await feature.initialize()
    assert await asyncio.to_thread(observer_started.wait, 1)
    await asyncio.wait_for(heartbeat_done.wait(), timeout=1)

    assert heartbeat_ticks == 5
    assert observer_threads
    assert all(thread != loop_thread for thread in observer_threads)
    assert not observer_finished.is_set()
    release_observer.set()
    assert await asyncio.to_thread(observer_finished.wait, 1)
    await heartbeat_task
    await feature.shutdown()


@pytest.mark.asyncio
async def test_sync_observer_never_occupies_lifecycle_default_executor(
    monkeypatch, tmp_path
):
    observer_started = threading.Event()
    release_observer = threading.Event()
    default_executor = ThreadPoolExecutor(max_workers=1)
    asyncio.get_running_loop().set_default_executor(default_executor)

    def observe(_snapshot):
        observer_started.set()
        assert release_observer.wait(timeout=2)

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=observe,
    )
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_OBSERVER_TIMEOUT", 0.01)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    for _ in range(100):
        if observer_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert observer_started.is_set()
    assert isolated_runtime._TELEMETRY_OBSERVER_EXECUTOR._threads
    assert all(
        thread.daemon
        for thread in isolated_runtime._TELEMETRY_OBSERVER_EXECUTOR._threads
    )

    prepared = threading.Event()

    def ensure_venv():
        prepared.set()
        return False

    monkeypatch.setattr(feature, "ensure_venv", ensure_venv)
    await asyncio.wait_for(
        feature._ensure_venv_without_blocking_event_loop(),
        timeout=0.5,
    )

    assert prepared.is_set()
    release_observer.set()
    await feature.shutdown()


@pytest.mark.asyncio
async def test_sync_observer_failure_is_logged(monkeypatch, tmp_path, caplog):
    def observe(_snapshot):
        raise RuntimeError("private observer failure")

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=observe,
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)

    with caplog.at_level("WARNING"):
        await feature.initialize()
        for _ in range(100):
            if any("telemetry observer failed" in line for line in caplog.messages):
                break
            await asyncio.sleep(0.01)

    assert any("telemetry observer failed" in line for line in caplog.messages)
    await feature.shutdown()


def test_bounded_observer_executor_shutdown_cancels_full_queue_without_blocking():
    executor = isolated_runtime._BoundedDaemonExecutor(
        max_workers=1,
        queue_capacity=1,
    )
    running = threading.Event()
    release = threading.Event()

    def block():
        running.set()
        assert release.wait(timeout=2)

    active = executor.submit(block)
    assert running.wait(timeout=1)
    queued = executor.submit(lambda: None)

    executor.shutdown()

    assert queued.cancelled()
    assert active.done() is False
    rejected = executor.submit(lambda: None)
    with pytest.raises(RuntimeError, match="executor stopped"):
        rejected.result()
    release.set()
    active.result(timeout=1)


def test_bounded_observer_executor_fails_closed_on_partial_worker_start(
    monkeypatch,
):
    executor = isolated_runtime._BoundedDaemonExecutor(
        max_workers=2,
        queue_capacity=2,
    )
    original_start = threading.Thread.start
    starts = 0

    def fail_second_start(thread):
        nonlocal starts
        starts += 1
        if starts == 2:
            raise RuntimeError("synthetic worker start failure")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_second_start)

    failed = executor.submit(lambda: None)
    with pytest.raises(RuntimeError, match="worker start failure"):
        failed.result()

    assert executor._shutdown is False
    assert executor._threads == []
    recovered = executor.submit(lambda: "recovered")
    assert recovered.result(timeout=1) == "recovered"
    executor.shutdown()


def test_bounded_observer_executor_reclaims_cancelled_queue_slots_before_retry():
    executor = isolated_runtime._BoundedDaemonExecutor(
        max_workers=1,
        queue_capacity=2,
    )
    running = threading.Event()
    release = threading.Event()

    def block():
        running.set()
        assert release.wait(timeout=2)

    active = executor.submit(block)
    assert running.wait(timeout=1)
    queued = executor.submit(lambda: "stale")
    for attempt in range(12):
        assert queued.cancel() is True
        queued = executor.submit(lambda attempt=attempt: attempt)
        with executor._work.mutex:
            retained = list(executor._work.queue)
        assert len(retained) == 1
        assert retained[0][0] is queued

    release.set()
    active.result(timeout=1)
    assert queued.result(timeout=1) == 11
    executor.shutdown()


@pytest.mark.asyncio
async def test_terminal_latch_cancels_queued_observer_before_host_invocation(
    monkeypatch, tmp_path
):
    executor = isolated_runtime._BoundedDaemonExecutor(
        max_workers=1,
        queue_capacity=2,
    )
    running = threading.Event()
    release = threading.Event()
    observer_called = threading.Event()

    def block():
        running.set()
        assert release.wait(timeout=2)

    active = executor.submit(block)
    assert running.wait(timeout=1)
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_OBSERVER_EXECUTOR", executor)
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_OBSERVER_TIMEOUT", 2.0)

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=lambda _snapshot: observer_called.set(),
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    queued_future = None
    for _ in range(200):
        with executor._work.mutex:
            queued_items = list(executor._work.queue)
        if queued_items:
            queued_future = queued_items[0][0]
            break
        await asyncio.sleep(0.01)
    assert queued_future is not None
    assert not queued_future.done()

    feature._latch_terminal_lifecycle()
    for _ in range(100):
        if queued_future.cancelled() and not feature._telemetry_observer_tasks:
            break
        await asyncio.sleep(0.01)

    assert queued_future.cancelled()
    assert not feature._telemetry_observer_tasks
    release.set()
    active.result(timeout=1)
    await asyncio.sleep(0.05)
    assert not observer_called.is_set()
    await feature.shutdown()
    executor.shutdown()


@pytest.mark.asyncio
async def test_queued_observer_future_becomes_terminal_at_delivery_deadline(
    monkeypatch, tmp_path
):
    executor = isolated_runtime._BoundedDaemonExecutor(
        max_workers=1,
        queue_capacity=2,
    )
    running = threading.Event()
    release = threading.Event()

    def block():
        running.set()
        assert release.wait(timeout=2)

    active = executor.submit(block)
    assert running.wait(timeout=1)
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_OBSERVER_EXECUTOR", executor)
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_OBSERVER_TIMEOUT", 0.01)
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_RETRY_BASE_SECONDS", 1.0)

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=lambda _snapshot: None,
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    for _ in range(200):
        if not feature._telemetry_observer_tasks and feature._telemetry_retry_task:
            break
        await asyncio.sleep(0.01)

    with executor._work.mutex:
        queued_items = list(executor._work.queue)
    assert len(queued_items) == 1
    queued_future = queued_items[0][0]
    assert queued_future.cancelled()
    assert feature._telemetry_retry_task is not None

    await feature.shutdown()
    release.set()
    active.result(timeout=1)
    executor.shutdown()


def test_bounded_observer_executor_serializes_submit_against_shutdown(monkeypatch):
    executor = isolated_runtime._BoundedDaemonExecutor(
        max_workers=1,
        queue_capacity=1,
    )
    enqueue_started = threading.Event()
    release_enqueue = threading.Event()
    original_put = executor._work.put_nowait

    def blocked_put(item):
        enqueue_started.set()
        assert release_enqueue.wait(timeout=2)
        original_put(item)

    monkeypatch.setattr(executor._work, "put_nowait", blocked_put)
    submitted = []
    submit_thread = threading.Thread(
        target=lambda: submitted.append(executor.submit(lambda: "delivered")),
    )
    submit_thread.start()
    assert enqueue_started.wait(timeout=1)

    shutdown_thread = threading.Thread(target=executor.shutdown)
    shutdown_thread.start()
    assert shutdown_thread.is_alive()
    release_enqueue.set()
    submit_thread.join(timeout=1)
    shutdown_thread.join(timeout=1)

    assert submit_thread.is_alive() is False
    assert shutdown_thread.is_alive() is False
    assert submitted[0].done()
    assert submitted[0].cancelled() or submitted[0].result() == "delivered"


@pytest.mark.asyncio
async def test_idle_workspace_reclaim_serializes_racing_wake(monkeypatch, tmp_path):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=client_factory)
    await feature.initialize()
    runtime_dir = feature._feature_runtime_dir()
    (runtime_dir / "data" / "reclaim-me").write_text("private")
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    feature._idle_ui_contributions = {
        "modules": ["panel.mjs"],
        "static_dir": str(runtime_dir / ".venv" / "feature_pkg" / "static"),
    }
    feature._environment_bytes = 100
    feature._private_writable_bytes = 20
    feature._downloaded_bytes = 5
    feature._disk_telemetry_status = "complete"

    delete_started = threading.Event()
    release_delete = threading.Event()
    original_remove = isolated_runtime._remove_isolated_feature_runtime

    def slow_remove(*args):
        delete_started.set()
        assert release_delete.wait(timeout=2)
        return original_remove(*args)

    monkeypatch.setattr(isolated_runtime, "_remove_isolated_feature_runtime", slow_remove)
    reclaim = asyncio.create_task(feature.reclaim_idle_workspace())
    assert await asyncio.to_thread(delete_started.wait, 1)
    assert feature.get_ui_contributions() is None
    deleting_snapshot = feature.runtime_telemetry_snapshot()
    assert deleting_snapshot.environment_bytes is None
    assert deleting_snapshot.private_writable_bytes is None
    assert deleting_snapshot.downloaded_bytes is None
    assert deleting_snapshot.disk_telemetry_status == "unavailable"
    wake = asyncio.create_task(
        feature.call_isolated_tool("ping", {"message": "after reclaim"})
    )
    await asyncio.sleep(0.05)
    assert wake.done() is False

    release_delete.set()
    assert await reclaim is isolated_runtime.RuntimeNamespaceCleanupOutcome.REMOVED
    assert (await wake)["success"] is True
    assert runtime_dir.is_dir()
    assert not (runtime_dir / "data" / "reclaim-me").exists()
    assert len(clients) == 2
    await feature.shutdown()


@pytest.mark.asyncio
async def test_cancelled_idle_reclaim_keeps_wake_serialized(monkeypatch, tmp_path):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    runtime_dir = feature._feature_runtime_dir()
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )

    delete_started = threading.Event()
    release_delete = threading.Event()
    original_remove = isolated_runtime._remove_isolated_feature_runtime

    def slow_remove(*args):
        delete_started.set()
        assert release_delete.wait(timeout=2)
        return original_remove(*args)

    monkeypatch.setattr(isolated_runtime, "_remove_isolated_feature_runtime", slow_remove)
    reclaim = asyncio.create_task(feature.reclaim_idle_workspace())
    assert await asyncio.to_thread(delete_started.wait, 1)
    reclaim.cancel("host request ended")
    wake = asyncio.create_task(
        feature.call_isolated_tool("ping", {"message": "after cancelled reclaim"})
    )
    await asyncio.sleep(0.05)

    assert reclaim.done() is False
    assert wake.done() is False
    assert feature._reload_lock.locked() is True
    release_delete.set()
    with pytest.raises(asyncio.CancelledError, match="host request ended"):
        await reclaim
    assert (await wake)["success"] is True
    assert runtime_dir.is_dir()
    await feature.shutdown()


@pytest.mark.asyncio
async def test_failed_final_reclaim_rmdir_republishes_durable_repair_intent(
    monkeypatch, tmp_path
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    runtime_dir = feature._feature_runtime_dir()
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )

    real_rmdir = os.rmdir

    def fail_feature_commit(path, *args, **kwargs):
        if path == feature._runtime_directory_name and kwargs.get("dir_fd") is not None:
            raise OSError(errno.EBUSY, "synthetic final reclaim commit failure")
        return real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(isolated_runtime.os, "rmdir", fail_feature_commit)
    with pytest.raises(
        IsolatedRuntimePreparationError,
        match="cleanup could not complete",
    ):
        await feature.reclaim_idle_workspace()

    marker = runtime_dir / isolated_runtime._VENV_RELOCATION_REPAIR_MARKER
    assert marker.read_bytes() == isolated_runtime._VENV_RECLAIM_REPAIR_PAYLOAD
    assert feature._venv_repair_reason() == "reclaim"
    await feature.shutdown()


@pytest.mark.skipif(
    os.name != "posix" or not Path("/dev/fd").is_dir(),
    reason="descriptor-count regression requires POSIX /dev/fd",
)
def test_remove_walker_closes_ascend_parent_fd_when_rmdir_fails(
    monkeypatch, tmp_path
):
    root = tmp_path / "remove-root"
    (root / "child").mkdir(parents=True)
    root_fd = os.open(root, isolated_runtime._directory_open_flags())
    real_rmdir = os.rmdir

    def fail_child_rmdir(path, *args, **kwargs):
        if path == "child" and kwargs.get("dir_fd") is not None:
            raise OSError(errno.ENOTEMPTY, "synthetic concurrent child mutation")
        return real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(isolated_runtime.os, "rmdir", fail_child_rmdir)
    try:
        baseline = len(os.listdir("/dev/fd"))
        for _ in range(5):
            with pytest.raises(OSError, match="concurrent child mutation"):
                isolated_runtime._remove_directory_contents_at(
                    root_fd,
                    allow_owner_marker=False,
                )
            assert len(os.listdir("/dev/fd")) == baseline
    finally:
        os.close(root_fd)


@pytest.mark.skipif(
    os.name != "posix" or not Path("/dev/fd").is_dir(),
    reason="descriptor-count regression requires POSIX /dev/fd",
)
def test_nested_owner_walker_closes_ascend_parent_fd_when_stat_fails(
    monkeypatch, tmp_path
):
    root = tmp_path / "owner-root"
    (root / "child").mkdir(parents=True)
    root_fd = os.open(root, isolated_runtime._directory_open_flags())
    real_open_parent = isolated_runtime._open_cleanup_parent_at
    real_stat = os.stat
    ascending = False

    def mark_ascend(*args, **kwargs):
        nonlocal ascending
        descriptor = real_open_parent(*args, **kwargs)
        ascending = True
        return descriptor

    def fail_ascend_stat(path, *args, **kwargs):
        nonlocal ascending
        if ascending:
            ascending = False
            raise FileNotFoundError(errno.ENOENT, "synthetic ascend race", path)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(isolated_runtime, "_open_cleanup_parent_at", mark_ascend)
    monkeypatch.setattr(isolated_runtime.os, "stat", fail_ascend_stat)
    try:
        baseline = len(os.listdir("/dev/fd"))
        for _ in range(5):
            ascending = False
            with pytest.raises(FileNotFoundError, match="synthetic ascend race"):
                isolated_runtime._assert_no_nested_runtime_owners_at(
                    root_fd,
                    allow_owner_marker=False,
                )
            assert len(os.listdir("/dev/fd")) == baseline
    finally:
        os.close(root_fd)


@pytest.mark.skipif(
    os.name != "posix" or not Path("/dev/fd").is_dir(),
    reason="descriptor-count regression requires POSIX /dev/fd",
)
@pytest.mark.parametrize(
    "walker",
    (
        isolated_runtime._assert_no_nested_runtime_owners_at,
        isolated_runtime._remove_directory_contents_at,
    ),
)
def test_cleanup_walkers_close_root_fd_when_initial_custody_check_fails(
    tmp_path, walker
):
    root = tmp_path / walker.__name__
    root.mkdir()
    (root / isolated_runtime._RUNTIME_OWNER_MARKER).write_text("nested-owner")
    root_fd = os.open(root, isolated_runtime._directory_open_flags())
    try:
        baseline = len(os.listdir("/dev/fd"))
        for _ in range(5):
            with pytest.raises(
                IsolatedRuntimeNamespaceError,
                match="nested ownership marker",
            ):
                walker(root_fd, allow_owner_marker=False)
            assert len(os.listdir("/dev/fd")) == baseline
    finally:
        os.close(root_fd)


@pytest.mark.asyncio
async def test_failed_idle_reclaim_preserves_durable_reprovision_intent(
    monkeypatch, tmp_path
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    runtime = InstalledFeatureRuntime(
        class_name="TestFeature",
        entry_point="test_pkg.feature:TestFeature",
        distribution="test-pkg",
        runtime="isolated-venv",
        service="test_pkg.service:main",
        description="Test callable proxy",
    )
    feature = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    await feature.initialize()
    runtime_dir = feature._feature_runtime_dir()
    venv = runtime_dir / ".venv"
    (venv / "lib").mkdir(parents=True)
    (venv / "lib" / "partial").write_text("remove first")
    (venv / "bin").mkdir()
    (venv / "bin" / "python").write_text("")
    feature._venv_path = venv
    feature._write_provision_manifest_payload({"install_target": "test-pkg"})
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )

    original_remove = isolated_runtime._remove_directory_contents_at

    def fail_after_partial_sweep(directory_fd, **kwargs):
        assert isolated_runtime._read_venv_relocation_repair_marker_at(directory_fd)
        venv_fd = os.open(
            ".venv",
            isolated_runtime._directory_open_flags(),
            dir_fd=directory_fd,
        )
        try:
            lib_fd = os.open(
                "lib",
                isolated_runtime._directory_open_flags(),
                dir_fd=venv_fd,
            )
            try:
                original_remove(lib_fd, allow_owner_marker=False)
            finally:
                os.close(lib_fd)
            os.rmdir("lib", dir_fd=venv_fd)
        finally:
            os.close(venv_fd)
        raise OSError("synthetic partial reclaim failure")

    monkeypatch.setattr(
        isolated_runtime,
        "_remove_directory_contents_at",
        fail_after_partial_sweep,
    )
    with pytest.raises(
        IsolatedRuntimePreparationError,
        match="cleanup could not complete",
    ):
        await feature.reclaim_idle_workspace()

    assert not (venv / "lib").exists()
    assert feature._provision_manifest_path().exists()
    assert feature._venv_relocation_repair_pending() is True
    assert feature._venv_repair_reason() == "reclaim"
    # Callable services have no generated console wrapper to expose partial
    # deletion. The durable marker must independently force reinstall.
    feature._bin_path = None
    feature._service_target = None
    assert feature._provision_status("test-pkg", {}) == (True, True)
    await feature.shutdown()


@pytest.mark.asyncio
async def test_reclaim_fences_coalesced_disk_refresh_and_preserves_bin_accounting(
    monkeypatch, tmp_path
):
    snapshots = []
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=snapshots.append,
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    for _ in range(100):
        if snapshots and not feature._telemetry_emit_tasks:
            break
        await asyncio.sleep(0.01)
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    for _ in range(100):
        if not feature._telemetry_emit_tasks:
            break
        await asyncio.sleep(0.01)

    delete_started = threading.Event()
    release_delete = threading.Event()
    original_remove = isolated_runtime._remove_isolated_feature_runtime

    def slow_remove(*args):
        delete_started.set()
        assert release_delete.wait(timeout=2)
        return original_remove(*args)

    monkeypatch.setattr(isolated_runtime, "_remove_isolated_feature_runtime", slow_remove)
    reclaim = asyncio.create_task(feature.reclaim_idle_workspace())
    assert await asyncio.to_thread(delete_started.wait, 1)
    feature._schedule_runtime_telemetry(force=True, refresh_disk=True)
    release_delete.set()
    assert await reclaim is isolated_runtime.RuntimeNamespaceCleanupOutcome.REMOVED
    for _ in range(100):
        if not feature._telemetry_emit_tasks:
            break
        await asyncio.sleep(0.01)

    snapshot = feature.runtime_telemetry_snapshot()
    assert snapshot.environment_bytes is None
    assert snapshot.private_writable_bytes == 0
    assert snapshot.downloaded_bytes == 0
    assert snapshot.disk_telemetry_status == "complete"
    await feature.shutdown()


@pytest.mark.asyncio
async def test_caller_cancellation_during_idle_wake_does_not_seal_feature(
    monkeypatch, tmp_path
):
    wake_started = asyncio.Event()
    release_wake = asyncio.Event()

    class SlowWakeClient(FakeIsolatedClient):
        async def start(self):
            wake_started.set()
            await release_wake.wait()
            await super().start()

    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = (
            SlowWakeClient(**kwargs)
            if clients
            else FakeIsolatedClient(**kwargs)
        )
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=client_factory)
    await feature.initialize()
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )

    cancelled_call = asyncio.create_task(
        feature.call_isolated_tool("ping", {"message": "cancelled"})
    )
    await wake_started.wait()
    cancelled_call.cancel("caller left")
    release_wake.set()
    with pytest.raises(asyncio.CancelledError, match="caller left"):
        await cancelled_call

    recovered = await feature.call_isolated_tool("ping", {"message": "later"})
    assert recovered["success"] is True
    assert len(clients) == 2
    assert feature._traffic_gate.sealed is False
    assert feature._terminal_lifecycle_latched is False
    await feature.shutdown()


@pytest.mark.asyncio
async def test_monitor_ordering_never_closes_gate_over_admitted_work(
    monkeypatch, tmp_path
):
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingClient(FakeIsolatedClient):
        async def call_tool(self, name, args):
            entered.set()
            await release.wait()
            return await super().call_tool(name, args)

    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=BlockingClient)
    await feature.initialize()
    feature._last_used_monotonic -= 7200
    call = asyncio.create_task(feature.call_isolated_tool("ping", {"message": "long"}))
    await entered.wait()
    monitor_generation = feature._activity_generation
    monitor_last_used = feature._last_used_monotonic

    assert not await feature._retire_idle_generation(
        expected_activity_generation=monitor_generation,
        expected_last_used=monitor_last_used,
    )
    assert feature._traffic_gate.closed is False
    release.set()
    assert (await call)["success"] is True
    assert feature._last_used_monotonic > monitor_last_used
    await feature.shutdown()


@pytest.mark.asyncio
async def test_first_inbound_event_fences_idle_retirement_before_detached_route(
    monkeypatch, tmp_path
):
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = FakeIsolatedClient()
    feature = ProxyFeature(
        agent,
        _idle_test_runtime(),
        client_factory=lambda **_kwargs: client,
    )
    await feature.initialize()
    feature._last_used_monotonic -= 7200
    expected_generation = feature._activity_generation
    expected_last_used = feature._last_used_monotonic
    original_close_if_idle = feature._traffic_gate.close_if_idle

    async def delayed_close_if_idle():
        close_started.set()
        await release_close.wait()
        return await original_close_if_idle()

    monkeypatch.setattr(
        feature._traffic_gate,
        "close_if_idle",
        delayed_close_if_idle,
    )
    retirement = asyncio.create_task(
        feature._retire_idle_generation(
            expected_activity_generation=expected_generation,
            expected_last_used=expected_last_used,
        )
    )
    await close_started.wait()
    await feature._handle_event(
        {"type": "channel.inbound", "payload": {}},
        source_client=client,
    )
    release_close.set()

    assert await retirement is False
    assert feature._client is client
    assert feature._activity_generation == expected_generation + 1
    assert feature._observed_inbound_producer is True
    await feature.shutdown()


@pytest.mark.asyncio
async def test_idle_retirement_refuses_registered_inbound_channel(
    monkeypatch, tmp_path
):
    agent = Mock(
        did=_TEST_AGENT_DID,
        features={"ChannelFeature": FakeChannelFeature()},
    )
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = TelegramChannelClient()
    feature = ProxyFeature(
        agent, _idle_test_runtime(), client_factory=lambda **_kwargs: client
    )
    await feature.initialize()
    feature._last_used_monotonic -= 7200

    assert feature._channel_adapter is not None
    assert not await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    assert client.stopped is False
    assert feature.runtime_telemetry_snapshot().state == "running"
    await feature.shutdown()


@pytest.mark.asyncio
async def test_idle_monitor_retries_after_one_retirement_error(monkeypatch, tmp_path):
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=0.01)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    survived = asyncio.Event()
    attempts = 0

    async def flaky_retirement(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("private synthetic monitor failure")
        survived.set()
        feature._stopping = True
        return False

    monkeypatch.setattr(feature, "_retire_idle_generation", flaky_retirement)
    feature._last_used_monotonic -= 1
    await asyncio.wait_for(survived.wait(), timeout=1)

    assert attempts == 2
    await feature.shutdown()


@pytest.mark.asyncio
async def test_replacement_publication_rearms_monitor_after_baseexception(
    monkeypatch, tmp_path, caplog
):
    class FatalMonitorError(BaseException):
        pass

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=0.01)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    original_retire = feature._retire_idle_generation

    async def fatal_retire(**_kwargs):
        raise FatalMonitorError("synthetic infrastructure failure")

    monkeypatch.setattr(feature, "_retire_idle_generation", fatal_retire)
    await feature.initialize()
    failed_monitor = feature._idle_monitor_task
    assert failed_monitor is not None
    feature._last_used_monotonic -= 1
    for _ in range(200):
        if failed_monitor.done():
            break
        await asyncio.sleep(0.01)
    assert failed_monitor.done()
    with pytest.raises(FatalMonitorError, match="synthetic infrastructure failure"):
        failed_monitor.result()
    assert feature._idle_monitor_task is None
    assert "idle monitor terminated for TestFeature" in caplog.text

    monkeypatch.setattr(feature, "_retire_idle_generation", original_retire)
    await feature.reload()
    assert feature._idle_monitor_task is not None
    assert not feature._idle_monitor_task.done()
    await feature.shutdown()


@pytest.mark.asyncio
async def test_observer_snapshot_sampling_runs_off_event_loop(monkeypatch, tmp_path):
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    snapshots = []
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=snapshots.append,
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    main_thread = threading.get_ident()
    captured_threads = []
    sampled_threads = []
    original_inputs = feature._runtime_telemetry_snapshot_inputs
    original_builder = feature._build_runtime_telemetry_snapshot

    def capture_inputs():
        captured_threads.append(threading.get_ident())
        return original_inputs()

    def observe_thread(*args):
        sampled_threads.append(threading.get_ident())
        return original_builder(*args)

    monkeypatch.setattr(feature, "_runtime_telemetry_snapshot_inputs", capture_inputs)
    monkeypatch.setattr(feature, "_build_runtime_telemetry_snapshot", observe_thread)
    await feature.initialize()
    for _ in range(100):
        if snapshots:
            break
        await asyncio.sleep(0.01)

    assert snapshots
    assert captured_threads == [main_thread]
    assert sampled_threads
    assert all(thread_id != main_thread for thread_id in sampled_threads)

    builder = Mock(side_effect=AssertionError("sync pull must not sample processes"))
    monkeypatch.setattr(feature, "_build_runtime_telemetry_snapshot", builder)
    assert feature.runtime_telemetry_snapshot().state == "running"
    builder.assert_not_called()
    await feature.shutdown()


@pytest.mark.asyncio
async def test_workspace_byte_telemetry_reports_owned_known_directories(
    monkeypatch, tmp_path
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.isolated_runtime_root = tmp_path / "runtimes"
    agent.isolated_runtime_namespace = "tenant/agent"
    agent.isolated_runtime_telemetry_observer = lambda _snapshot: None
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    runtime_dir = feature._prepare_runtime_workspace()
    feature._venv_path = runtime_dir / ".venv"
    feature._venv_path.mkdir()
    (feature._venv_path / "environment.bin").write_bytes(b"environment")
    (runtime_dir / "data" / "private.bin").write_bytes(b"private")
    (runtime_dir / "provisioning_cache" / "download.bin").write_bytes(b"download")

    await feature._refresh_disk_telemetry(refresh_environment=True)
    snapshot = feature.runtime_telemetry_snapshot()

    assert snapshot.environment_bytes == len(b"environment")
    assert snapshot.private_writable_bytes == len(b"private")
    assert snapshot.downloaded_bytes == len(b"download")


@pytest.mark.asyncio
async def test_workspace_byte_telemetry_deduplicates_cross_category_hardlinks(
    monkeypatch, tmp_path
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.isolated_runtime_root = tmp_path / "runtimes"
    agent.isolated_runtime_namespace = "tenant/agent"
    agent.isolated_runtime_telemetry_observer = lambda _snapshot: None
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    runtime_dir = feature._prepare_runtime_workspace()
    feature._venv_path = runtime_dir / ".venv"
    feature._venv_path.mkdir()
    environment_file = feature._venv_path / "shared.bin"
    environment_file.write_bytes(b"shared")
    os.link(
        environment_file,
        runtime_dir / "provisioning_cache" / "shared.bin",
    )

    await feature._refresh_disk_telemetry(refresh_environment=True)
    snapshot = feature.runtime_telemetry_snapshot()

    assert snapshot.environment_bytes == len(b"shared")
    assert snapshot.downloaded_bytes == 0

    measured_paths = []
    original_measure = isolated_runtime._measure_directory_tree_bytes

    def record_measure(path, **kwargs):
        measured_paths.append(path)
        return original_measure(path, **kwargs)

    monkeypatch.setattr(
        isolated_runtime,
        "_measure_directory_tree_bytes",
        record_measure,
    )
    # A state-only refresh preserves the last environment count without
    # spending the shared disk budget on an unchanged managed venv.
    await feature._refresh_disk_telemetry(refresh_environment=False)
    snapshot = feature.runtime_telemetry_snapshot()
    assert snapshot.environment_bytes == len(b"shared")
    assert snapshot.downloaded_bytes == 0
    assert ".venv" not in measured_paths

    measured_paths.clear()
    await feature._refresh_disk_telemetry(refresh_environment=True)
    assert ".venv" in measured_paths


@pytest.mark.asyncio
async def test_persistently_failing_observer_has_bounded_forced_retries(
    monkeypatch, tmp_path, caplog
):
    calls = 0

    def observe(_snapshot):
        nonlocal calls
        calls += 1
        raise RuntimeError("persistent private observer failure")

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=observe,
    )
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_RETRY_BASE_SECONDS", 0.001)
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_RETRY_MAX_SECONDS", 0.001)
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_EMIT_MIN_INTERVAL", 0.001)
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_FORCED_RETRY_LIMIT", 3)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)

    with caplog.at_level("WARNING"):
        await feature.initialize()
        for _ in range(200):
            if (
                calls >= 4
                and feature._telemetry_retry_task is None
                and not feature._telemetry_emit_tasks
                and not feature._telemetry_observer_tasks
            ):
                break
            await asyncio.sleep(0.005)
        settled_calls = calls
        await asyncio.sleep(0.03)

    # Initial delivery plus three forced retries. A final ordinary attempt may
    # run if the emission interval has elapsed, but it cannot reschedule itself.
    assert 4 <= settled_calls <= 5
    assert calls == settled_calls
    assert caplog.messages.count(
        "Hosted isolated runtime telemetry observer failed for TestFeature"
    ) == 1
    await feature.shutdown()


@pytest.mark.asyncio
async def test_bin_runtime_disk_status_treats_unused_venv_as_not_applicable(
    monkeypatch, tmp_path
):
    snapshots = []
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=snapshots.append,
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)

    await feature.initialize()
    for _ in range(100):
        if snapshots:
            break
        await asyncio.sleep(0.01)

    assert snapshots[-1].environment_bytes is None
    assert snapshots[-1].private_writable_bytes == 0
    assert snapshots[-1].downloaded_bytes == 0
    assert snapshots[-1].disk_telemetry_status == "complete"
    await feature.shutdown()


@pytest.mark.asyncio
async def test_disk_refresh_uses_one_shared_deadline(monkeypatch, tmp_path):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.isolated_runtime_root = tmp_path / "runtimes"
    agent.isolated_runtime_namespace = "tenant/agent"
    agent.isolated_runtime_telemetry_observer = lambda _snapshot: None
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    runtime_dir = feature._prepare_runtime_workspace()
    feature._venv_path = runtime_dir / ".venv"
    deadlines = []

    seen_sets = []
    parent_fds = []

    def measure(
        _path, *, parent_fd=None, deadline=None, seen_linked_files=None
    ):
        deadlines.append(deadline)
        seen_sets.append(seen_linked_files)
        parent_fds.append(parent_fd)
        return 0, "complete"

    monkeypatch.setattr(isolated_runtime, "_measure_directory_tree_bytes", measure)
    await feature._refresh_disk_telemetry(refresh_environment=True)

    assert len(deadlines) == 8
    assert deadlines[0] is not None
    assert len(set(deadlines)) == 1
    assert seen_sets[0] is not None
    assert all(seen is seen_sets[0] for seen in seen_sets)
    assert parent_fds[0] is not None
    assert all(parent_fd == parent_fds[0] for parent_fd in parent_fds)


@pytest.mark.asyncio
async def test_disk_budget_exhaustion_is_visible_and_logged_once(
    monkeypatch, tmp_path, caplog
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.isolated_runtime_root = tmp_path / "runtimes"
    agent.isolated_runtime_namespace = "tenant/agent"
    agent.isolated_runtime_telemetry_observer = lambda _snapshot: None
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    runtime_dir = feature._prepare_runtime_workspace()
    feature._venv_path = runtime_dir / ".venv"

    monkeypatch.setattr(
        isolated_runtime,
        "_measure_directory_tree_bytes",
        lambda _path, *, parent_fd=None, deadline=None, seen_linked_files=None: (
            None,
            "budget-exceeded",
        ),
    )
    with caplog.at_level("WARNING"):
        await feature._refresh_disk_telemetry(refresh_environment=True)
        await feature._refresh_disk_telemetry(refresh_environment=True)

    snapshot = feature.runtime_telemetry_snapshot()
    assert snapshot.environment_bytes is None
    assert snapshot.disk_telemetry_status == "budget-exceeded"
    assert caplog.messages.count(
        "Hosted isolated runtime disk telemetry exceeded its shared "
        "measurement budget for TestFeature"
    ) == 1


@pytest.mark.asyncio
async def test_idle_supervisor_waits_for_wake_event_instead_of_polling(
    monkeypatch, tmp_path
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )

    await asyncio.sleep(1.05)
    assert feature._idle_resume_event.is_set() is False
    assert feature._supervision_task is not None
    assert feature._supervision_task.done() is False
    assert len(feature._idle_resume_event._waiters) == 1
    await feature.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutated", [False, True])
async def test_venv_cache_hit_reports_actual_provisioning_mutation(
    monkeypatch, tmp_path, mutated
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    feature._venv_path = tmp_path / "existing" / ".venv"
    feature._venv_path.mkdir(parents=True)
    monkeypatch.setattr(feature, "ensure_venv", Mock(return_value=mutated))

    await feature._ensure_venv_without_blocking_event_loop()

    assert feature._last_cache_hit is (not mutated)
    assert feature._last_provision_seconds is not None


@pytest.mark.asyncio
async def test_terminal_latch_clears_cancelled_monitor_for_immediate_rearm(
    monkeypatch, tmp_path
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    old_monitor = asyncio.create_task(asyncio.Event().wait())
    feature._idle_monitor_task = old_monitor

    feature._latch_terminal_lifecycle()
    assert feature._idle_monitor_task is None
    feature._terminal_lifecycle_latched = False
    feature._stopping = False
    feature._client = FakeIsolatedClient()
    feature._start_idle_monitor()

    assert feature._idle_monitor_task is not None
    assert feature._idle_monitor_task is not old_monitor
    await asyncio.sleep(0)
    assert old_monitor.cancelled()
    feature._latch_terminal_lifecycle()
    await asyncio.sleep(0)


def test_lifecycle_seam_rejects_unscoped_or_post_discovery_configuration(
    monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="explicit hosted scope"):
        configure_hosted_isolated_runtime_lifecycle(
            SimpleNamespace(did="did:test:unscoped"),
            idle_timeout_seconds=60,
        )

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.isolated_runtime_root = tmp_path / "runtimes"
    agent.isolated_runtime_namespace = "tenant/agent"
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)

    with pytest.raises(RuntimeError, match="before feature discovery"):
        configure_hosted_isolated_runtime_lifecycle(
            agent,
            idle_timeout_seconds=60,
        )


def test_agent_warns_for_unmatched_idle_timeout_override(caplog):
    agent = object.__new__(KestrelAgent)
    agent.isolated_runtime_idle_timeouts = {
        "KnownFeature": 60.0,
        "TypoFeature": 60.0,
    }

    with caplog.at_level("WARNING"):
        agent._warn_unmatched_isolated_runtime_idle_timeouts(
            [SimpleNamespace(name="KnownFeature")]
        )

    assert "undiscovered feature TypoFeature" in caplog.text
    assert "undiscovered feature KnownFeature" not in caplog.text


@pytest.mark.asyncio
async def test_refused_idle_monitor_has_bounded_retry_rate(monkeypatch, tmp_path):
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=0.01)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    attempts = 0

    async def refuse(**_kwargs):
        nonlocal attempts
        attempts += 1
        return False

    monkeypatch.setattr(feature, "_retire_idle_generation", refuse)
    await feature.initialize()
    feature._last_used_monotonic -= 1
    await asyncio.sleep(0.08)

    assert 1 <= attempts <= 10
    await feature.shutdown()


@pytest.mark.asyncio
async def test_idle_monitor_uses_agent_background_task_registry(monkeypatch, tmp_path):
    background_tasks = set()
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}

    def track(coro, *, name):
        task = asyncio.create_task(coro, name=name)
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
        return task

    agent._track_background_task = track
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)

    await feature.initialize()

    assert feature._idle_monitor_task in background_tasks
    assert feature._idle_monitor_task.get_name() == "isolated-feature-idle:TestFeature"
    await feature.shutdown()


@pytest.mark.asyncio
async def test_incomplete_advertised_channel_remains_resident(monkeypatch, tmp_path):
    class InboundOnlyClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {"channel": {"channel_type": "inbound-only"}}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = InboundOnlyClient()
    feature = ProxyFeature(
        agent, _idle_test_runtime(), client_factory=lambda **_kwargs: client
    )
    await feature.initialize()
    feature._last_used_monotonic -= 7200

    assert feature._channel_adapter is None
    assert feature._owns_inbound_producer() is True
    assert not await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    assert client.stopped is False
    await feature.shutdown()


@pytest.mark.asyncio
async def test_unproven_legacy_inbound_producer_fails_resident_before_first_event(
    monkeypatch, tmp_path
):
    class MetadataPoorProducer(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {"tools": {}}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = MetadataPoorProducer()
    feature = ProxyFeature(
        agent, _idle_test_runtime(), client_factory=lambda **_kwargs: client
    )
    await feature.initialize()
    feature._last_used_monotonic -= 7200

    assert feature._observed_inbound_producer is False
    assert feature._owns_inbound_producer() is True
    assert not await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    assert client.stopped is False
    await feature.shutdown()


@pytest.mark.asyncio
async def test_named_idle_override_explicitly_allows_ambiguous_utility_feature(
    monkeypatch, tmp_path
):
    class MetadataPoorUtility(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {"tools": {}}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeouts={"TestFeature": 3600},
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = MetadataPoorUtility()
    feature = ProxyFeature(
        agent, _idle_test_runtime(), client_factory=lambda **_kwargs: client
    )
    await feature.initialize()
    feature._last_used_monotonic -= 7200

    assert feature._owns_inbound_producer() is False
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    assert client.stopped is True
    await feature.shutdown()


@pytest.mark.asyncio
async def test_named_idle_override_cannot_retire_declared_inbound_producer(
    monkeypatch, tmp_path
):
    class DeclaredProducer(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {"tools": {}, "inbound_producer": True}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeouts={"TestFeature": 3600},
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = DeclaredProducer()
    feature = ProxyFeature(
        agent, _idle_test_runtime(), client_factory=lambda **_kwargs: client
    )
    await feature.initialize()
    feature._last_used_monotonic -= 7200

    assert feature._owns_inbound_producer() is True
    assert feature._idle_monitor_task is None
    assert not await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    assert client.stopped is False
    await feature.shutdown()


@pytest.mark.asyncio
async def test_observed_legacy_inbound_producer_latches_resident(monkeypatch, tmp_path):
    class MetadataPoorProducer(FakeIsolatedClient):
        @property
        def capabilities(self):
            # Model a legacy/misdeclared child whose observed behavior must
            # override its negative producer declaration. Ambiguous metadata
            # would remain resident even if the observation latch regressed.
            return {"tools": {}, "inbound_producer": False}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(agent, tmp_path, idle_timeout_seconds=3600)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = MetadataPoorProducer()
    feature = ProxyFeature(
        agent, _idle_test_runtime(), client_factory=lambda **_kwargs: client
    )
    await feature.initialize()

    await feature._handle_event(
        {"type": "message.inbound", "payload": {}},
        source_client=client,
    )
    feature._last_used_monotonic -= 7200

    assert feature._observed_inbound_producer is True
    assert feature._owns_inbound_producer() is True
    assert not await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    assert client.stopped is False
    await feature.shutdown()


def test_disk_telemetry_rejects_root_symlink_and_entry_overflow(
    monkeypatch, tmp_path
):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.bin").write_bytes(b"outside")
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    assert isolated_runtime._measure_directory_tree_bytes(linked) == (
        None,
        "unavailable",
    )

    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / "local.bin").write_bytes(b"local")
    (owned / "nested-link").symlink_to(outside, target_is_directory=True)
    assert isolated_runtime._measure_directory_tree_bytes(owned) == (
        len(b"local"),
        "complete",
    )

    bounded = tmp_path / "bounded"
    bounded.mkdir()
    (bounded / "one").write_bytes(b"1")
    (bounded / "two").write_bytes(b"2")
    monkeypatch.setattr(isolated_runtime, "_DISK_TELEMETRY_ENTRY_BUDGET", 1)
    assert isolated_runtime._measure_directory_tree_bytes(bounded) == (
        None,
        "budget-exceeded",
    )


def test_disk_telemetry_enforces_depth_budget(monkeypatch, tmp_path):
    bounded = tmp_path / "bounded-depth"
    (bounded / "level-one" / "level-two").mkdir(parents=True)
    (bounded / "level-one" / "level-two" / "payload").write_bytes(b"owned")

    monkeypatch.setattr(isolated_runtime, "_DISK_TELEMETRY_DEPTH_BUDGET", 2)

    assert isolated_runtime._measure_directory_tree_bytes(bounded) == (
        None,
        "budget-exceeded",
    )


@pytest.mark.asyncio
async def test_process_telemetry_rejects_reused_pid_identity(monkeypatch):
    pid = 4242
    observed = Mock()
    observed.create_time.return_value = 222.0
    observed.children.return_value = []
    observed.memory_info.return_value = SimpleNamespace(rss=1024)
    observed.cpu_times.return_value = SimpleNamespace(user=1.0, system=2.0)
    observed.num_fds.return_value = 3
    monkeypatch.setattr(isolated_runtime.psutil, "Process", lambda _pid: observed)

    agent = Mock(did=_TEST_AGENT_DID, features={})
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    feature._client = SimpleNamespace(
        process=SimpleNamespace(pid=pid, returncode=None)
    )
    feature._process_identity = (pid, 111.0)

    snapshot = await feature.sample_runtime_telemetry()

    observed.create_time.assert_called_once_with()
    assert snapshot.rss_bytes is None
    assert snapshot.cpu_seconds is None
    assert snapshot.open_fds is None
    assert snapshot.process_count is None


@pytest.mark.asyncio
async def test_initialize_skips_disk_walk_without_observer(monkeypatch, tmp_path):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    measured = Mock(side_effect=AssertionError("disk telemetry should be disabled"))
    monkeypatch.setattr(isolated_runtime, "_measure_directory_tree_bytes", measured)
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)

    await feature.initialize()

    measured.assert_not_called()
    await feature.shutdown()


@pytest.mark.asyncio
async def test_pull_snapshot_can_refresh_disk_without_observer(monkeypatch, tmp_path):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.isolated_runtime_root = tmp_path / "runtimes"
    agent.isolated_runtime_namespace = "tenant/agent"
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    runtime_dir = feature._prepare_runtime_workspace()
    feature._venv_path = runtime_dir / ".venv"
    feature._venv_path.mkdir()
    (feature._venv_path / "environment.bin").write_bytes(b"environment")
    (runtime_dir / "data" / "private.bin").write_bytes(b"private")
    (runtime_dir / "provisioning_cache" / "download.bin").write_bytes(b"download")

    snapshot = await feature.sample_runtime_telemetry(refresh_disk=True)

    assert snapshot.environment_bytes == len(b"environment")
    assert snapshot.private_writable_bytes == len(b"private")
    assert snapshot.downloaded_bytes == len(b"download")
    assert snapshot.disk_telemetry_status == "complete"


@pytest.mark.asyncio
async def test_standalone_disk_telemetry_reports_absent_provisioning_cache_as_zero(
    tmp_path,
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    runtime_dir = feature._prepare_runtime_workspace()
    feature._venv_path = runtime_dir / ".venv"
    feature._venv_path.mkdir()
    (feature._venv_path / "environment.bin").write_bytes(b"environment")
    (runtime_dir / "data" / "private.bin").write_bytes(b"private")
    assert not (runtime_dir / "provisioning_cache").exists()

    snapshot = await feature.sample_runtime_telemetry(refresh_disk=True)

    assert snapshot.environment_bytes == len(b"environment")
    assert snapshot.private_writable_bytes == len(b"private")
    assert snapshot.downloaded_bytes == 0
    assert snapshot.disk_telemetry_status == "complete"


@pytest.mark.asyncio
async def test_hosted_disk_telemetry_rejects_symlinked_feature_parent(
    monkeypatch, tmp_path
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.isolated_runtime_root = tmp_path / "runtimes"
    agent.isolated_runtime_namespace = "tenant/agent"
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    runtime_dir = feature._prepare_runtime_workspace()
    feature._venv_path = runtime_dir / ".venv"
    feature._venv_path.mkdir()
    (runtime_dir / "data" / "private.bin").write_bytes(b"owned")

    feature_parent = runtime_dir.parent
    retained_parent = feature_parent.with_name("retained-feature-venvs")
    feature_parent.rename(retained_parent)
    external_parent = tmp_path / "external-feature-venvs"
    external_runtime = external_parent / feature._runtime_directory_name
    for component in (
        ".venv",
        "work",
        "home",
        "tmp",
        "config",
        "data",
        "cache",
        "provisioning_cache",
    ):
        (external_runtime / component).mkdir(parents=True, exist_ok=True)
    (external_runtime / "data" / "other-tenant.bin").write_bytes(b"other-tenant")
    feature_parent.symlink_to(external_parent, target_is_directory=True)

    snapshot = await feature.sample_runtime_telemetry(refresh_disk=True)

    assert snapshot.environment_bytes is None
    assert snapshot.private_writable_bytes is None
    assert snapshot.downloaded_bytes is None
    assert snapshot.disk_telemetry_status == "unavailable"


@pytest.mark.asyncio
async def test_pull_snapshot_preserves_authoritative_post_reclaim_zeroes(
    monkeypatch, tmp_path
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.isolated_runtime_root = tmp_path / "runtimes"
    agent.isolated_runtime_namespace = "tenant/agent"
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    feature._workspace_reclaim_generation = 1
    feature._workspace_reclaimed = True
    feature._environment_bytes = 0
    feature._private_writable_bytes = 0
    feature._downloaded_bytes = 0
    feature._disk_telemetry_status = "complete"
    measured = Mock(side_effect=AssertionError("reclaimed workspace must not be walked"))
    monkeypatch.setattr(
        isolated_runtime,
        "_measure_directory_tree_bytes",
        measured,
    )

    snapshot = await feature.sample_runtime_telemetry(refresh_disk=True)

    measured.assert_not_called()
    assert snapshot.environment_bytes == 0
    assert snapshot.private_writable_bytes == 0
    assert snapshot.downloaded_bytes == 0
    assert snapshot.disk_telemetry_status == "complete"


@pytest.mark.asyncio
async def test_shutdown_cancels_owned_telemetry_tasks(monkeypatch, tmp_path):
    observer_started = asyncio.Event()
    observer_cancelled = asyncio.Event()

    async def observe(_snapshot):
        try:
            observer_started.set()
            await asyncio.Event().wait()
        finally:
            observer_cancelled.set()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=observe,
    )
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_OBSERVER_TIMEOUT", 0.01)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    await asyncio.wait_for(observer_started.wait(), timeout=1)
    assert feature._telemetry_observer_tasks

    await feature.shutdown()
    await asyncio.wait_for(observer_cancelled.wait(), timeout=1)
    await asyncio.sleep(0)

    assert not feature._telemetry_observer_tasks
    assert not feature._telemetry_emit_tasks


@pytest.mark.asyncio
async def test_cancelled_rate_limited_observer_retries_coalesced_idle_snapshot(
    monkeypatch, tmp_path
):
    snapshots = []
    ordinary_started = asyncio.Event()
    cancel_ordinary = asyncio.get_running_loop().create_future()

    async def observe(snapshot):
        snapshots.append(snapshot)
        if len(snapshots) == 2:
            ordinary_started.set()
            await cancel_ordinary

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=observe,
    )
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_OBSERVER_TIMEOUT", 0.01)
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_RETRY_BASE_SECONDS", 0.01)
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_RETRY_MAX_SECONDS", 0.02)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    for _ in range(100):
        if snapshots and not feature._telemetry_emit_tasks:
            break
        await asyncio.sleep(0.01)

    feature._last_telemetry_emit_monotonic = None
    feature._schedule_runtime_telemetry()
    await asyncio.wait_for(ordinary_started.wait(), timeout=1)
    for _ in range(100):
        if not feature._telemetry_emit_tasks:
            break
        await asyncio.sleep(0.01)

    feature._last_used_monotonic -= 7200
    assert await feature._retire_idle_generation(
        expected_activity_generation=feature._activity_generation,
        expected_last_used=feature._last_used_monotonic,
    )
    assert feature._telemetry_observer_emit_pending is True
    assert feature._telemetry_observer_force_pending is True

    cancel_ordinary.cancel()
    for _ in range(200):
        if snapshots and snapshots[-1].cleanup_eligible:
            break
        await asyncio.sleep(0.01)

    assert snapshots[-1].state == "idle"
    assert snapshots[-1].cleanup_eligible is True
    assert feature._telemetry_observer_emit_pending is False
    assert feature._telemetry_observer_force_pending is False
    await feature.shutdown()


@pytest.mark.asyncio
async def test_cancelled_telemetry_callbacks_do_not_reschedule(monkeypatch, tmp_path):
    observer_started = asyncio.Event()

    async def observe(_snapshot):
        observer_started.set()
        await asyncio.Event().wait()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=observe,
    )
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_OBSERVER_TIMEOUT", 0.01)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    await asyncio.wait_for(observer_started.wait(), timeout=1)
    for _ in range(100):
        if not feature._telemetry_emit_tasks:
            break
        await asyncio.sleep(0.01)

    feature._schedule_runtime_telemetry(force=True)
    assert feature._telemetry_observer_emit_pending is True
    observer_task = next(iter(feature._telemetry_observer_tasks))
    observer_task.cancel()
    await asyncio.sleep(0.05)
    assert not feature._telemetry_emit_tasks

    refresh_started = asyncio.Event()

    async def blocked_refresh(**_kwargs):
        refresh_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(feature, "_refresh_disk_telemetry", blocked_refresh)
    feature._schedule_runtime_telemetry(force=True, refresh_disk=True)
    await asyncio.wait_for(refresh_started.wait(), timeout=1)
    feature._schedule_runtime_telemetry(force=True)
    emit_task = next(iter(feature._telemetry_emit_tasks))
    emit_task.cancel()
    await asyncio.sleep(0.05)

    assert not feature._telemetry_emit_tasks
    await feature.shutdown()


@pytest.mark.asyncio
async def test_disk_refresh_failure_still_emits_forced_idle_snapshot(
    monkeypatch, tmp_path, caplog
):
    snapshots = []
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=snapshots.append,
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    for _ in range(100):
        if snapshots and not feature._telemetry_emit_tasks:
            break
        await asyncio.sleep(0.01)

    async def fail_refresh(**_kwargs):
        raise RuntimeError("private disk failure")

    feature._environment_linked_file_identities = frozenset({(123, 456)})
    monkeypatch.setattr(feature, "_refresh_disk_telemetry", fail_refresh)
    feature._last_used_monotonic -= 7200
    with caplog.at_level("WARNING"):
        assert await feature._retire_idle_generation(
            expected_activity_generation=feature._activity_generation,
            expected_last_used=feature._last_used_monotonic,
        )
        for _ in range(100):
            if snapshots[-1].cleanup_eligible:
                break
            await asyncio.sleep(0.01)

    assert snapshots[-1].cleanup_eligible is True
    assert snapshots[-1].disk_telemetry_status == "unavailable"
    assert snapshots[-1].environment_bytes is None
    assert snapshots[-1].private_writable_bytes is None
    assert snapshots[-1].downloaded_bytes is None
    assert feature._environment_linked_file_identities is None
    assert any("disk telemetry refresh failed" in line for line in caplog.messages)
    await feature.shutdown()


@pytest.mark.asyncio
async def test_hot_path_telemetry_never_retains_traffic_admission(monkeypatch, tmp_path):
    hot_observer_started = asyncio.Event()
    release_observer = asyncio.Event()
    observations = 0
    observed_active_counts = []
    feature = None

    async def observe(_snapshot):
        nonlocal observations
        observations += 1
        if observations > 1:
            assert feature is not None
            observed_active_counts.append(feature._traffic_gate._active)
            hot_observer_started.set()
            await release_observer.wait()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=observe,
    )
    monkeypatch.setattr(isolated_runtime, "_TELEMETRY_EMIT_MIN_INTERVAL", 0)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()

    result = await feature.call_isolated_tool("ping", {"message": "fast"})
    await asyncio.wait_for(hot_observer_started.wait(), timeout=1)

    assert result["success"] is True
    assert observed_active_counts == [0]
    assert feature._traffic_gate._active == 0
    assert feature._telemetry_emit_tasks
    release_observer.set()
    await feature.shutdown()


@pytest.mark.asyncio
async def test_concurrent_hot_path_telemetry_preserves_rate_limit(monkeypatch, tmp_path):
    snapshots = []
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    _configure_idle_lifecycle(
        agent,
        tmp_path,
        idle_timeout_seconds=3600,
        telemetry_observer=snapshots.append,
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _idle_test_runtime(), client_factory=FakeIsolatedClient)
    await feature.initialize()
    for _ in range(100):
        if snapshots and not feature._telemetry_emit_tasks:
            break
        await asyncio.sleep(0.01)

    await asyncio.gather(
        *(
            feature.call_isolated_tool("ping", {"message": str(index)})
            for index in range(200)
        )
    )
    for _ in range(100):
        if not feature._telemetry_emit_tasks:
            break
        await asyncio.sleep(0.01)

    assert len(snapshots) == 1
    await feature.shutdown()


def test_hosted_lifecycle_policy_requires_and_binds_explicit_agent_scope(tmp_path):
    observer = Mock()

    with pytest.raises(ValueError, match="explicit hosted scope"):
        KestrelAgent(
            did="did:test:unscoped-idle",
            storage_path=str(tmp_path / "unscoped" / "agent.db"),
            isolated_runtime_idle_timeout_seconds=60,
        )

    agent = KestrelAgent(
        did="did:test:scoped-idle",
        storage_path=str(tmp_path / "scoped" / "agent.db"),
        isolated_runtime_root=tmp_path / "runtimes",
        isolated_runtime_namespace="tenant/agent",
        isolated_runtime_idle_timeout_seconds=60,
        isolated_runtime_idle_timeouts={"TelegramFeature": None},
        isolated_runtime_telemetry_observer=observer,
    )

    assert agent.isolated_runtime_idle_timeout_seconds == 60.0
    assert agent.isolated_runtime_idle_timeouts["TelegramFeature"] is None
    assert agent.isolated_runtime_telemetry_observer is observer

    with pytest.raises(TypeError):
        agent.isolated_runtime_idle_timeouts["OtherFeature"] = 10


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), "60"])
def test_hosted_lifecycle_policy_rejects_invalid_idle_timeout(value):
    agent = SimpleNamespace(
        did="did:test:timeout-validation",
        isolated_runtime_root=Path("/tmp/kestrel-timeout-validation"),
        isolated_runtime_namespace="tenant/agent",
    )
    with pytest.raises((TypeError, ValueError)):
        configure_hosted_isolated_runtime_lifecycle(
            agent, idle_timeout_seconds=value
        )


@pytest.mark.asyncio
async def test_proxy_feature_mirrors_tools_and_forwards_calls(monkeypatch, tmp_path):
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}

    runtime = InstalledFeatureRuntime(
        class_name="TestFeature",
        entry_point="test_pkg.feature:TestFeature",
        distribution="test-pkg",
        runtime="isolated-venv",
        service="test_service",
        description="Test proxy",
    )
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, runtime, client_factory=client_factory)
    await feature.initialize()

    tools = feature.get_tools()
    assert feature.name == "TestFeature"
    assert feature.tool_description == "Test proxy"
    assert len(tools) == 1
    assert tools[0].name == "ping"
    assert tools[0].schema.command_prefix == "!ping"
    # F004: the advertised input_schema is converted into real ToolParameters,
    # so the tool reaches the LLM with usable arguments (not an empty list).
    params = tools[0].schema.parameters
    assert [p.name for p in params] == ["message"]
    assert params[0].type == "string"
    assert params[0].required is True
    assert params[0].description == "text to echo"
    # And the schema survives OpenAI conversion (crashes if params are dicts).
    openai = tools[0].schema.to_openai_format()
    props = openai["function"]["parameters"]["properties"]
    assert props["message"]["type"] == "string"

    result = await tools[0].execute(message="hello")
    assert result["success"] is True
    assert result["result"] == {"echo": {"message": "hello"}}
    assert clients[0].calls == [("ping", {"message": "hello"})]

    await feature.shutdown()
    assert clients[0].stopped is True


@pytest.mark.asyncio
async def test_scheduled_isolated_call_fails_before_dispatch_without_context_capability(
    tmp_path,
):
    """A legacy isolated service must not receive an unkeyed scheduled effect."""

    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)
    client = FakeIsolatedClient()
    feature._client = client
    execution = SchedulerExecution(
        id="execution-1",
        schedule_id="schedule-1",
        agent_id="agent-1",
        task_name="ping",
        args={"message": "hello"},
        scheduled_for="2026-07-25T15:00:00+00:00",
        idempotency_key="stable-effect-key",
        attempt=1,
        owner="runner-1",
    )

    token = _current_execution.set(_SchedulerExecutionScope(execution))
    try:
        with pytest.raises(SchedulerExecutionContextUnavailable, match="advertises"):
            await feature.call_isolated_tool("ping", {"message": "hello"})
    finally:
        _current_execution.reset(token)

    # No RPC is attempted, so a legacy child can never perform a duplicate
    # effect without the scheduler's stable idempotency key.
    assert client.calls == []


@pytest.mark.asyncio
async def test_scheduler_records_terminal_isolated_admission_as_failed(tmp_path, monkeypatch):
    """A sealed proxy raises for scheduler delivery, so the durable row fails."""

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)

    backend = SQLiteBackend(str(tmp_path / "scheduler.db"))
    await backend.connect()
    db = AsyncDatabase(backend)

    async def dispatch_terminal_isolated_call(_task_name, _args):
        return await feature.call_isolated_tool("ping", {})

    runner = SchedulerRunner(
        db,
        _TEST_AGENT_DID,
        dispatch_terminal_isolated_call,
        owner_id="terminal-admission-runner",
    )
    try:
        await feature.initialize()
        client = feature._client
        await feature.shutdown()
        assert feature._traffic_gate.sealed is True

        await runner._ensure_tables()
        due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        await db.execute(
            """
            INSERT INTO scheduled_tasks
                (id, agent_id, task_name, cron_expression, args_json, enabled,
                 next_run_at, created_at, schedule_kind, timezone_name,
                 misfire_policy, idempotency_key, scheduler_protocol_version)
            VALUES (?, ?, 'isolated_ping', '* * * * *', '{}', 1, ?, ?,
                    'cron', 'UTC', 'skip', 'terminal-admission', ?)
            """,
            ("terminal-admission-task", _TEST_AGENT_DID, due, due, SCHEDULER_PROTOCOL_VERSION),
        )

        await runner._tick()

        status, result_text = await db.fetchone(
            "SELECT status, result_text FROM task_execution_log WHERE task_id = ?",
            ("terminal-admission-task",),
        )
        assert status == "failed"
        assert result_text == (
            "SchedulerTerminalAdmissionError: isolated feature traffic is unavailable"
        )
        assert client.calls == []
    finally:
        if not feature._stopping:
            await feature.shutdown()

        await db.close()


@pytest.mark.asyncio
async def test_scheduler_revokes_context_for_detached_core_child_before_late_isolated_call(
    tmp_path,
):
    """A child that outlives dispatch cannot reuse a completed occurrence.

    ``asyncio.create_task`` copies the scheduler ContextVar.  Exercise the
    production runner and proxy together: the child inherits the context
    during dispatch, waits for the runner to clear the occurrence, then calls
    the isolated tool.  That late call must be a normal untrusted call, never
    an RPC bearing the stale scheduler idempotency identity.
    """

    class ContextAwareClient(FakeIsolatedClient):
        supports_tool_execution_context = True

        async def call_tool(self, name, args, *, context=None):
            self.calls.append((name, args, context))
            return {"echo": args}

    class NoStorageRunner(SchedulerRunner):
        # This test isolates context revocation rather than persistence. The
        # real runner proves its durable token before and at effect admission;
        # retain that precondition without giving this deliberately storageless
        # test double a database implementation.
        async def _renew_lease_once(self, task):
            return True

        async def _claim_token_is_live(self, task):
            return True

        async def _renew_lease(self, task):
            await asyncio.Future()

        async def _finalize(self, *args, **kwargs):
            return None

    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)
    client = ContextAwareClient()
    feature._client = client

    child_started = asyncio.Event()
    release_child = asyncio.Event()
    child: asyncio.Task[None] | None = None

    async def late_isolated_call() -> None:
        child_started.set()
        await release_child.wait()
        await feature.call_isolated_tool("ping", {"message": "late"})

    async def executor(name, args):
        nonlocal child
        assert get_current_scheduler_execution() is not None
        child = asyncio.create_task(late_isolated_call())
        await child_started.wait()
        return "dispatched"

    now = datetime.now(timezone.utc).isoformat()
    task = ScheduledTask(
        id="schedule-1",
        agent_id="agent-1",
        task_name="ping",
        cron_expression="* * * * *",
        args_json='{"message": "scheduled"}',
        enabled=True,
        last_run_at=None,
        next_run_at=now,
        created_at=now,
        idempotency_key="stable-effect-key",
        claim_token="claim-token",
        claim_execution_id="execution-1",
        claim_scheduled_for=now,
        attempt_count=1,
    )
    runner = NoStorageRunner(object(), "agent-1", executor, owner_id="runner-1")

    await runner._execute_claim(task)
    assert get_current_scheduler_execution() is None

    release_child.set()
    assert child is not None
    await child

    # A stale scope would give this call a ToolExecutionContext with the
    # completed occurrence ID and stable idempotency key.  It must be absent.
    assert client.calls == [("ping", {"message": "late"}, None)]


def test_service_command_console_script(tmp_path):
    """Verification and launch resolve the same canonical console wrapper."""
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    runtime = InstalledFeatureRuntime(
        class_name="WhatsAppWebFeature",
        entry_point="wa.feature:WhatsAppWebFeature",
        distribution="kestrel-channel-whatsapp",
        runtime="isolated-venv",
        service="kestrel-whatsapp-web",
        project="service",
    )
    feature = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    wrapper = isolated_runtime._console_script_path(
        feature._venv_path,
        runtime.service,
    )
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        f"#!{isolated_runtime._venv_python(feature._venv_path)}\nexit 0\n"
    )
    cmd = feature._service_command()
    assert cmd == [str(wrapper)]
    assert feature._console_script_location_state() == "current"
    feature._verify_launch_artifact()
    # the install target is `project`, never the `service` runnable
    assert (runtime.project or runtime.distribution) == "service"


def test_windows_console_service_uses_and_verifies_exe_launcher(
    monkeypatch,
    tmp_path,
):
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    runtime = InstalledFeatureRuntime(
        class_name="WindowsServiceFeature",
        entry_point="svc.feature:WindowsServiceFeature",
        distribution="windows-service",
        runtime="isolated-venv",
        service="windows-service",
    )
    feature = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    monkeypatch.setattr(isolated_runtime.os, "name", "nt")
    launcher = feature._venv_path / "Scripts" / "windows-service.exe"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(b"MZ\x00binary-console-launcher")

    assert feature._service_command() == [str(launcher)]
    assert feature._console_script_location_state() == "current"
    feature._verify_launch_artifact()


@pytest.mark.parametrize(
    "service",
    (
        "svc_pkg.service:main",
        "_private.pkg_2:_main",
    ),
)
def test_service_command_module_callable(service, tmp_path):
    """Safe module callables run through an isolated venv interpreter."""

    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    runtime = InstalledFeatureRuntime(
        class_name="SvcFeature",
        entry_point="svc.feature:SvcFeature",
        distribution="svc-pkg",
        runtime="isolated-venv",
        service=service,
    )
    feature = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    module, callable_name = service.split(":", 1)

    command = feature._service_command()

    assert command[0] == str(isolated_runtime._venv_python(feature._venv_path))
    assert command[1] == "-P"
    assert command[2] == "-B"
    assert command[3] == "-c"
    assert command[4] == (
        f"from {module} import {callable_name}; {callable_name}()"
    )
    assert "-m" not in command
    assert feature._console_script_location_state() == "not-applicable"


def test_hosted_callable_safe_path_blocks_writable_cwd_module_shadowing(
    tmp_path,
):
    """The real child interpreter imports its venv target, never hosted cwd."""

    runtime_root = tmp_path / "hosted-runtime"
    runtime_root.mkdir(mode=0o700)
    runtime = InstalledFeatureRuntime(
        class_name="SafePathFeature",
        entry_point="safe_path.feature:SafePathFeature",
        distribution="safe-path-fixture",
        runtime="isolated-venv",
        service="trusted_service.runner:main",
    )
    feature = ProxyFeature(
        _hosted_postgres_agent(runtime_root, "agent-safe-path"),
        runtime,
        client_factory=FakeIsolatedClient,
    )
    runtime_dir = feature._prepare_runtime_workspace()
    test_venv = tmp_path / "trusted-venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(test_venv)],
        check=True,
        capture_output=True,
        text=True,
    )
    python = isolated_runtime._venv_python(test_venv)
    purelib = Path(
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    trusted_package = purelib / "trusted_service"
    trusted_package.mkdir()
    (trusted_package / "__init__.py").write_text("")
    (trusted_package / "runner.py").write_text(
        "def main():\n"
        "    print('trusted-target')\n"
    )

    hostile_cwd = runtime_dir / "work"
    hostile_package = hostile_cwd / "trusted_service"
    hostile_package.mkdir()
    shadow_marker = hostile_cwd / "shadow-imported"
    (hostile_package / "__init__.py").write_text(
        "open('shadow-imported', 'w').write('cwd won')\n"
    )
    (hostile_package / "runner.py").write_text(
        "def main():\n"
        "    print('malicious-cwd-target')\n"
    )
    feature._venv_path = test_venv
    feature._bin_path = None
    command = feature._service_command()
    env = isolated_runtime._isolated_child_env(
        test_venv,
        runtime_dir=runtime_dir,
        hosted=True,
        feature_name=feature.name,
        feature_distribution=runtime.distribution,
    )

    completed = subprocess.run(
        command,
        cwd=hostile_cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "trusted-target"
    assert completed.stderr == ""
    assert not shadow_marker.exists()


def test_hosted_callable_preserves_python_stdio_encoding_contract(
    monkeypatch,
    tmp_path,
):
    """Safe-path launch must still honor the hosted JSON-RPC encoding knobs."""

    monkeypatch.setenv("PYTHONIOENCODING", "latin-1")
    monkeypatch.setenv("PYTHONUTF8", "0")
    runtime = InstalledFeatureRuntime(
        class_name="EncodingFeature",
        entry_point="encoding.feature:EncodingFeature",
        distribution="encoding-fixture",
        runtime="isolated-venv",
        service="encoding_service:main",
    )
    feature = ProxyFeature(
        _hosted_postgres_agent(tmp_path / "runtime", "agent-encoding"),
        runtime,
        client_factory=FakeIsolatedClient,
    )
    runtime_dir = feature._prepare_runtime_workspace()
    test_venv = tmp_path / "encoding-venv"
    _write_real_venv_module(
        test_venv,
        "encoding_service",
        "import sys\ndef main():\n    sys.stdout.write('é')\n",
    )
    feature._venv_path = test_venv
    feature._bin_path = None
    command = feature._service_command()
    env = isolated_runtime._isolated_child_env(
        test_venv,
        runtime_dir=runtime_dir,
        hosted=True,
        feature_name=feature.name,
        feature_distribution=runtime.distribution,
    )

    completed = subprocess.run(
        command,
        cwd=runtime_dir / "work",
        env=env,
        check=True,
        capture_output=True,
    )

    assert command[1:3] == ["-P", "-B"]
    assert completed.stdout == b"\xe9"
    assert completed.stderr == b""


def test_hosted_callable_preserves_python_utf8_mode(
    monkeypatch,
    tmp_path,
):
    """``-P`` must not imply ``-E`` and suppress hosted ``PYTHONUTF8``."""

    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    monkeypatch.setenv("PYTHONUTF8", "1")
    venv = Path(sys.executable).parent.parent
    env = isolated_runtime._isolated_child_env(
        venv,
        runtime_dir=tmp_path,
        hosted=True,
        feature_name="EncodingFeature",
        feature_distribution="encoding-fixture",
    )
    completed = subprocess.run(
        isolated_runtime._isolated_python_command(
            Path(sys.executable),
            "import sys; print(sys.flags.utf8_mode)",
        ),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "1"


@pytest.mark.skipif(os.name != "posix", reason="executable shim is POSIX-only")
def test_callable_launch_uses_sdk_python_safe_path_contract(tmp_path):
    """The SDK's Python 3.11 contract launches with ``-P`` safe-path mode."""

    runtime = InstalledFeatureRuntime(
        class_name="LegacyPythonFeature",
        entry_point="legacy.feature:LegacyPythonFeature",
        distribution="legacy-python-feature",
        runtime="isolated-venv",
        service="json.tool:main",
    )
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    feature = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    shim = isolated_runtime._venv_python(feature._venv_path)
    shim.parent.mkdir(parents=True)
    shim.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "if sys.argv[1] == '-I':\n"
        "    raise SystemExit(91)\n"
        "if sys.argv[1] != '-P':\n"
        "    raise SystemExit(92)\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
    )
    shim.chmod(0o700)

    command = feature._service_command()
    completed = subprocess.run(
        command,
        input='{"legacy": true}',
        check=True,
        capture_output=True,
        text=True,
    )

    assert command[1] == "-P"
    assert command[2] == "-B"
    assert '"legacy": true' in completed.stdout


@pytest.mark.asyncio
async def test_venv_preparation_does_not_block_event_loop(tmp_path):
    """Fresh callable verification runs in an owned worker, not the host loop."""

    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    runtime = InstalledFeatureRuntime(
        class_name="WorkerFeature",
        entry_point="worker.feature:WorkerFeature",
        distribution="worker-feature",
        runtime="isolated-venv",
        service="safe.module:main",
    )
    feature = ProxyFeature(
        agent,
        runtime,
        client_factory=FakeIsolatedClient,
    )
    started = threading.Event()
    release = threading.Event()
    worker_thread = []

    def blocking_prepare():
        worker_thread.append(threading.get_ident())
        started.set()
        assert release.wait(timeout=2)

    feature.ensure_venv = blocking_prepare
    preparation = asyncio.create_task(
        feature._ensure_venv_without_blocking_event_loop()
    )
    try:
        assert await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=1)
        heartbeat = asyncio.Event()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        await asyncio.wait_for(heartbeat.wait(), timeout=0.2)
        assert not preparation.done()
        assert worker_thread != [threading.get_ident()]
    finally:
        release.set()
        await preparation


@pytest.mark.parametrize("hosted", (False, True), ids=("standalone", "hosted"))
@pytest.mark.parametrize(
    "service",
    (None, "../../escape", "pkg.service:main()"),
    ids=("missing", "path", "expression"),
)
def test_bin_override_is_authoritative_over_unused_service_metadata(
    monkeypatch,
    tmp_path,
    hosted,
    service,
):
    executable = tmp_path / "operator-service"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv("KESTREL_FEATURE_BINONLYFEATURE_BIN", str(executable))
    runtime = InstalledFeatureRuntime(
        class_name="BinOnlyFeature",
        entry_point="bin_only.feature:BinOnlyFeature",
        distribution="bin-only-feature",
        runtime="isolated-venv",
        service=service,
    )
    if hosted:
        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir(mode=0o700)
        agent = _hosted_postgres_agent(runtime_root, "agent-bin-only")
    else:
        agent = Mock(
            did=_TEST_AGENT_DID,
            storage_path=str(tmp_path / "agent" / "kestrel_prime.db"),
        )
    captured = {}

    def client_factory(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace()

    feature = ProxyFeature(agent, runtime, client_factory=client_factory)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()

    assert feature._service_command() == [str(executable.resolve())]
    feature._build_client()
    assert captured["command"] == [str(executable.resolve())]
    assert captured["kwargs"]["executable"] == str(executable.resolve())
    assert "service" not in captured["kwargs"]


def test_removed_bin_override_revalidates_service_before_path_resolution(
    monkeypatch,
    tmp_path,
):
    executable = tmp_path / "operator-service"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    key = "KESTREL_FEATURE_BINONLYFEATURE_BIN"
    monkeypatch.setenv(key, str(executable))
    runtime = InstalledFeatureRuntime(
        class_name="BinOnlyFeature",
        entry_point="bin_only.feature:BinOnlyFeature",
        distribution="bin-only-feature",
        runtime="isolated-venv",
        service="../../escape",
    )
    feature = ProxyFeature(
        Mock(
            did=_TEST_AGENT_DID,
            storage_path=str(tmp_path / "agent" / "kestrel_prime.db"),
        ),
        runtime,
        client_factory=FakeIsolatedClient,
    )

    monkeypatch.delenv(key)

    with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
        feature.resolve_runtime_paths()
    assert raised.value.safe_diagnostic() == (
        "isolated feature service must be a bare portable console-script "
        "executable name or a safe Python module:callable target"
    )


def test_hosted_bin_without_service_still_requires_immutable_executable(
    monkeypatch,
    tmp_path,
):
    key = "KESTREL_FEATURE_BINONLYFEATURE_BIN"
    monkeypatch.setenv(key, str(tmp_path / "missing-operator-service"))
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(mode=0o700)
    runtime = InstalledFeatureRuntime(
        class_name="BinOnlyFeature",
        entry_point="bin_only.feature:BinOnlyFeature",
        distribution="bin-only-feature",
        runtime="isolated-venv",
        service=None,
    )

    with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
        ProxyFeature(
            _hosted_postgres_agent(runtime_root, "agent-unsafe-bin"),
            runtime,
            client_factory=FakeIsolatedClient,
        )

    diagnostic = raised.value.safe_diagnostic()
    assert key in diagnostic
    assert str(tmp_path) not in diagnostic


@pytest.mark.parametrize(
    "service",
    (
        None,
        "",
        " service",
        "service ",
        ".hidden",
        "_hidden",
        "-option",
        "service.",
        "sub/service",
        r"sub\service",
        "../service",
        "../../../../bin/sh",
        "/bin/sh",
        "C:service",
        "pkg/service:main",
        r"pkg\service:main",
        "../pkg.service:main",
        ".pkg.service:main",
        "pkg..service:main",
        "pkg.service.:main",
        "pkg-service:main",
        "pkg.service:",
        ":main",
        "pkg.service:main.extra",
        "pkg.service:main:again",
        "pkg.service:main()",
        "pkg.service:main;raise_error",
        "class.service:main",
        "pkg.service:class",
        "con",
        "NUL.exe",
    ),
)
def test_isolated_service_requires_safe_console_or_callable_target(
    service,
    tmp_path,
):
    """Unsafe service metadata is quarantinable and never reflected."""

    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    runtime = InstalledFeatureRuntime(
        class_name="SvcFeature",
        entry_point="svc.feature:SvcFeature",
        distribution="svc-pkg",
        runtime="isolated-venv",
        service=service,
    )

    with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
        ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)

    diagnostic = raised.value.safe_diagnostic()
    assert diagnostic == (
        "isolated feature service must be a bare portable console-script "
        "executable name or a safe Python module:callable target"
    )


@pytest.mark.asyncio
async def test_supervision_registered_and_child_stopped_on_cancel(tmp_path):
    """Leak guard: supervision task registers with the agent's background-task
    lifecycle, and cancelling it (agent shutdown path) stops the child."""
    import asyncio

    tracked = []

    class FakeAgent:
        storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
        features: dict = {}

        def _track_background_task(self, coro, *, name):
            task = asyncio.create_task(coro, name=name)
            tracked.append(task)
            return task

    runtime = InstalledFeatureRuntime(
        class_name="SvcFeature",
        entry_point="svc.feature:SvcFeature",
        distribution="svc-pkg",
        runtime="isolated-venv",
        service="svc",
    )
    feature = ProxyFeature(FakeAgent(), runtime, client_factory=FakeIsolatedClient)
    feature._client_factory = lambda **kw: FakeIsolatedClient(**kw)
    monkey_bin = tmp_path  # avoid real venv work
    import os
    os.environ["KESTREL_FEATURE_SVCFEATURE_BIN"] = str(monkey_bin / "svc-bin")
    try:
        await feature.initialize()
        # registered through the agent's tracker, not a bare task
        assert feature._supervision_task in tracked
        client = feature._client
        await asyncio.sleep(0.05)  # let the supervision loop enter its body
        # simulate agent shutdown cancelling tracked background tasks
        feature._supervision_task.cancel()
        try:
            await feature._supervision_task
        except asyncio.CancelledError:
            pass
        assert client.stopped is True  # child torn down despite no shutdown() call
    finally:
        os.environ.pop("KESTREL_FEATURE_SVCFEATURE_BIN", None)


class FakeChannelRegistry:
    def __init__(self):
        self.adapters = {}

    def register(self, adapter):
        self.adapters[adapter.channel_type] = adapter

    def get(self, channel_type):
        return self.adapters.get(channel_type)

    def unregister(self, channel_type):
        return self.adapters.pop(channel_type, None)


class FakeChannelFeature:
    def __init__(self):
        self.registry = FakeChannelRegistry()
        self.inbound = []

    async def handle_inbound(self, message):
        self.inbound.append(message)
        return SimpleNamespace(durably_admitted=True)


def _isolated_runtime():
    return InstalledFeatureRuntime(
        class_name="WhatsAppFeature",
        entry_point="wa.feature:WhatsAppFeature",
        distribution="kestrel-channel-whatsapp",
        runtime="isolated-venv",
        service="wa-service",
    )


def _write_console_script_fixture_wheel(directory: Path) -> Path:
    """Build a dependency-free wheel with a real console entry point."""

    wheel = directory / "kestrel_console_migration_fixture-1.0.0-py3-none-any.whl"
    dist_info = "kestrel_console_migration_fixture-1.0.0.dist-info"
    files = {
        "migration_console/__init__.py": "",
        "migration_console/cli.py": (
            "def main():\n"
            "    print('console-path-ok')\n"
        ),
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            "Name: kestrel-console-migration-fixture\n"
            "Version: 1.0.0\n"
        ),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: kestrel-test\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
        f"{dist_info}/entry_points.txt": (
            "[console_scripts]\n"
            "kestrel-whatsapp-service = migration_console.cli:main\n"
        ),
    }
    record = "".join(f"{name},,\n" for name in files)
    record_name = f"{dist_info}/RECORD"
    files[record_name] = record + f"{record_name},,\n"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return wheel


@pytest.mark.asyncio
async def test_route_link_qr_uses_cached_scope_and_offloads_png_write(
    monkeypatch, tmp_path
):
    """A channel.link_qr event is written to the agent data dir as the latest
    PNG (served by the endpoint the persisted channel_link card fetches). It is
    NO LONGER pushed as a live SSE bubble / sticky event (#2081) — that path
    orphaned on refresh."""
    import base64

    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    emitted = []

    async def emit_event(event_type, data):
        emitted.append((event_type, data))

    agent.emit_event = emit_event

    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)
    feature._prepare_runtime_workspace()
    offloaded = []
    real_to_thread = asyncio.to_thread

    async def observed_to_thread(function, *args, **kwargs):
        offloaded.append(function)
        return await real_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", observed_to_thread)
    monkeypatch.setattr(
        isolated_runtime,
        "agent_runtime_dir",
        lambda _agent: (_ for _ in ()).throw(AssertionError("scope re-prepared")),
    )

    png = b"\x89PNG\r\n\x1a\n" + b"fake-qr-bytes"
    await feature._route_link_qr(
        {
            "channel_type": "WhatsApp",  # mixed case → normalized to lowercase
            "png_b64": base64.b64encode(png).decode("ascii"),
            "caption": "Scan me",
        }
    )

    out = tmp_path / "agent" / "channel_link_artifacts" / "whatsapp_link_qr.png"
    assert out.exists()
    assert out.read_bytes() == png
    assert offloaded == [isolated_runtime._write_private_artifact]

    # No SSE emit and no sticky replay — the card is a persisted typed part now.
    assert emitted == []
    agent.set_sticky_event.assert_not_called()


@pytest.mark.asyncio
async def test_route_link_qr_accepts_only_ascii_whitespace_wrapped_base64(tmp_path):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    feature = ProxyFeature(
        agent, _isolated_runtime(), client_factory=FakeIsolatedClient
    )
    feature._prepare_runtime_workspace()
    png = b"\x89PNG\r\n\x1a\nwrapped-standard-base64"
    encoded = base64.b64encode(png).decode("ascii")
    wrapped = "".join(
        (
            encoded[:4],
            " ",
            encoded[4:8],
            "\t",
            encoded[8:12],
            "\n",
            encoded[12:16],
            "\r",
            encoded[16:20],
            "\v",
            encoded[20:24],
            "\f",
            encoded[24:],
        )
    )

    await feature._route_link_qr(
        {"channel_type": "whatsapp", "png_b64": wrapped}
    )

    artifact = tmp_path / "agent" / "channel_link_artifacts" / "whatsapp_link_qr.png"
    assert artifact.read_bytes() == png


@pytest.mark.asyncio
async def test_route_link_qr_rejects_junk_and_urlsafe_alphabet_without_leaking(
    tmp_path, caplog
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    feature = ProxyFeature(
        agent, _isolated_runtime(), client_factory=FakeIsolatedClient
    )
    feature._prepare_runtime_workspace()
    secret_junk = "private-child-token@"
    standard = base64.b64encode(b"\xfb\xff").decode("ascii")
    assert "+" in standard and "/" in standard

    with caplog.at_level("WARNING", logger=isolated_runtime.__name__):
        await feature._route_link_qr(
            {
                "channel_type": "whatsapp",
                "png_b64": standard[:2] + secret_junk + standard[2:],
            }
        )
        await feature._route_link_qr(
            {
                "channel_type": "whatsapp",
                "png_b64": standard.replace("+", "-").replace("/", "_"),
            }
        )
        await feature._route_link_qr(
            {"channel_type": "whatsapp", "png_b64": " \t\r\n\v\f"}
        )
        await feature._route_link_qr(
            {"channel_type": "whatsapp", "png_b64": standard + (" " * 17)}
        )

    artifact = tmp_path / "agent" / "channel_link_artifacts" / "whatsapp_link_qr.png"
    assert not artifact.exists()
    assert secret_junk not in caplog.text
    assert standard not in caplog.text


@pytest.mark.asyncio
async def test_route_link_qr_rejects_oversized_decoded_png_after_normalization(
    monkeypatch, tmp_path
):
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    feature = ProxyFeature(
        agent, _isolated_runtime(), client_factory=FakeIsolatedClient
    )
    feature._prepare_runtime_workspace()
    monkeypatch.setattr(isolated_runtime, "_MAX_PRIVATE_ARTIFACT_BYTES", 8)

    await feature._route_link_qr(
        {
            "channel_type": "whatsapp",
            "png_b64": "MTIz\r\nNDU2 \tNzg5",
        }
    )

    artifact = tmp_path / "agent" / "channel_link_artifacts" / "whatsapp_link_qr.png"
    assert not artifact.exists()


def test_private_artifact_write_enforces_bound(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    monkeypatch.setattr(isolated_runtime, "_MAX_PRIVATE_ARTIFACT_BYTES", 8)

    with pytest.raises(IsolatedRuntimeNamespaceError, match="write limit"):
        isolated_runtime._write_private_artifact(artifact_dir / "qr.png", b"123456789")


@pytest.mark.skipif(
    os.name != "posix", reason="descriptor-relative write is POSIX-only"
)
def test_private_artifact_cleanup_error_still_closes_directory_fd(
    monkeypatch, tmp_path
):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    original_open = os.open
    original_close = os.close
    original_fstat = os.fstat
    original_unlink = os.unlink
    opened_directories = []
    closed = []

    def tracked_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        if Path(path) == artifact_dir:
            opened_directories.append(descriptor)
        return descriptor

    def tracked_close(descriptor):
        closed.append(descriptor)
        return original_close(descriptor)

    def fail_replace(*_args, **_kwargs):
        raise OSError(errno.EIO, "synthetic replace failure")

    def fail_temp_unlink(path, *args, **kwargs):
        if str(path).startswith(".qr.png.tmp-"):
            raise PermissionError(errno.EACCES, "synthetic cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(isolated_runtime.os, "open", tracked_open)
    monkeypatch.setattr(isolated_runtime.os, "close", tracked_close)
    monkeypatch.setattr(isolated_runtime.os, "replace", fail_replace)
    monkeypatch.setattr(isolated_runtime.os, "unlink", fail_temp_unlink)
    monkeypatch.setattr(isolated_runtime, "_secure_dirfd_supported", lambda: True)

    with pytest.raises(PermissionError, match="synthetic cleanup failure"):
        isolated_runtime._write_private_artifact(artifact_dir / "qr.png", b"bounded")

    assert len(opened_directories) == 1
    directory_fd = opened_directories[0]
    assert directory_fd in closed
    with pytest.raises(OSError) as closed_descriptor:
        original_fstat(directory_fd)
    assert closed_descriptor.value.errno == errno.EBADF


@pytest.mark.asyncio
async def test_route_link_cleared_removes_png(tmp_path):
    """Linking clears the persisted PNG so the channel_link card resolves to
    'expired or already linked'. No sticky/SSE state to retract (#2081)."""
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    emitted = []

    async def emit_event(event_type, data):
        emitted.append((event_type, data))

    agent.emit_event = emit_event
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)

    art = tmp_path / "agent" / "channel_link_artifacts"
    art.mkdir(parents=True)
    (art / "whatsapp_link_qr.png").write_bytes(b"\x89PNG\r\n\x1a\nqr")

    await feature._route_link_cleared({"channel_type": "whatsapp"})

    assert not (art / "whatsapp_link_qr.png").exists()
    assert emitted == []
    agent.clear_sticky_event.assert_not_called()


@pytest.mark.asyncio
async def test_route_link_qr_rejects_malformed_payloads(tmp_path):
    """Bad channel_type / missing PNG / traversal attempts are dropped, no emit."""
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}
    emitted = []

    async def emit_event(event_type, data):
        emitted.append((event_type, data))

    agent.emit_event = emit_event
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)

    # missing png
    await feature._route_link_qr({"channel_type": "whatsapp"})
    # path-traversal / invalid channel type
    await feature._route_link_qr({"channel_type": "../etc", "png_b64": "AAAA"})
    # non-dict
    await feature._route_link_qr(None)

    assert emitted == []
    assert not (tmp_path / "agent" / "channel_link_artifacts").exists()


@pytest.mark.asyncio
async def test_channel_link_tool_emits_persisted_part(monkeypatch, tmp_path):
    """When the bridged channel's pairing tool runs on the streaming turn, the
    host emits a persisted ``channel_link`` typed part (a reference, not the QR
    bytes) so the pairing card rides the conversation that asked for it (#2081)."""
    from kestrel_sovereign.agent import parts as parts_mod

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {"ChannelFeature": channel_feature}

    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_BIN", "/bin/wa-service")

    class ChannelClient(FakeIsolatedClient):
        capabilities = {
            "channel": {"channel_type": "whatsapp", "send_tool": "whatsapp_send"}
        }

        async def list_tools(self):
            return [
                {"name": "whatsapp_link", "description": "Link WhatsApp", "category": "utility"},
                {"name": "whatsapp_send", "description": "Send", "category": "utility"},
            ]

    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=ChannelClient)
    await feature.initialize()

    # Convention fallback: <channel_type>_link is the pairing tool.
    assert feature._link_tool == "whatsapp_link"

    # The pairing tool, run inside a streaming turn's part collector, emits the
    # persisted channel_link part.
    with parts_mod.part_collector():
        await feature.call_isolated_tool("whatsapp_link", {})
        drained = parts_mod.drain_parts()
    assert drained == [{"type": "channel_link", "data": {"channel_type": "whatsapp"}}]

    # A non-link tool (e.g. send) emits nothing.
    with parts_mod.part_collector():
        await feature.call_isolated_tool("whatsapp_send", {"to": "x", "message": "y"})
        assert parts_mod.drain_parts() == []

    await feature.shutdown()


@pytest.mark.asyncio
async def test_proxy_forwards_host_config_into_client(monkeypatch, tmp_path):
    """Persisted host config is loaded and handed to the client (-> initialize handshake)."""
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {}

    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_BIN", "/bin/wa-service")
    captured = {}

    def client_factory(**kwargs):
        captured.update(kwargs)
        return FakeIsolatedClient(**kwargs)

    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=client_factory)

    async def fake_load(**_kwargs):
        return {"provider": "web", "allowed_senders": ["+13035551234"]}

    feature.load_persisted_config = fake_load  # type: ignore[assignment]
    await feature.initialize()

    assert captured["config"] == {
        "provider": "web",
        "allowed_senders": ["+13035551234"],
    }
    await feature.shutdown()


@pytest.mark.asyncio
async def test_transient_startup_config_read_failure_recovers_without_losing_secret(
    monkeypatch, tmp_path
):
    """A failed durable read must not boot an authoritative empty config.

    The feature starts only after storage recovers, at which point a
    write-only-secret-preserving PATCH receives the actual durable value rather
    than replacing it with the empty config used by the old best-effort path.
    """

    stored_config = {"enabled": True, "token": "stored-secret-not-for-logs"}

    class TransientReadStorage(_FakeStorage):
        def __init__(self):
            super().__init__()
            self.fail_reads = False

        async def get_node(self, node_id):
            if self.fail_reads:
                raise OSError("storage temporarily unavailable")
            return await super().get_node(node_id)

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = TransientReadStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    await feature.persist_config(stored_config)
    agent.storage.fail_reads = True

    with pytest.raises(RuntimeError, match="failed to load persisted config"):
        await feature.initialize()
    assert clients == []
    assert feature._host_config_loaded is False

    agent.storage.fail_reads = False
    await feature.initialize()
    assert clients[0].kwargs["config"] == stored_config

    # This mirrors the endpoint's write-only-secret preservation: the request
    # omits ``token``, so the recovered current config supplies it before save.
    partial_patch = {"enabled": False}
    current = await feature.get_config()
    partial_patch["token"] = current["token"]
    await feature.set_config(partial_patch)

    assert agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties["config"] == {
        "enabled": False,
        "token": stored_config["token"],
    }
    await feature.shutdown()


@pytest.mark.asyncio
async def test_proxy_bridges_channel_capability_into_registry(monkeypatch, tmp_path):
    """A service advertising a channel capability is registered as a forwarding adapter,
    and channels_send-style routing reaches the service tool."""
    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {"ChannelFeature": channel_feature}

    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_BIN", "/bin/wa-service")

    class ChannelClient(FakeIsolatedClient):
        capabilities = {
            "channel": {
                "channel_type": "whatsapp",
                "send_tool": "whatsapp_send",
                "status_tool": "whatsapp_status",
            }
        }

        async def call_tool(self, name, args):
            self.calls.append((name, args))
            return {"ok": True, "data": {"message_id": "WAMID.1"}, "message": "sent"}

    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=lambda **kw: ChannelClient(**kw))
    await feature.initialize()

    adapter = channel_feature.registry.adapters.get("whatsapp")
    assert adapter is not None
    assert adapter.is_connected is True

    receipt = await adapter.send_message(to="+13035551234", content="hi")
    assert receipt.status.value == "success"
    assert receipt.message_id == "WAMID.1"
    assert feature._client.calls == [
        ("whatsapp_send", {"to": "+13035551234", "message": "hi"})
    ]

    await feature.shutdown()
    assert "whatsapp" not in channel_feature.registry.adapters


@pytest.mark.asyncio
async def test_replacement_bridge_uses_effective_config_for_outbound_and_inbound_authorization(
    monkeypatch, tmp_path
):
    """A replacement bridge must not inherit stale sender/enabled policy."""

    from kestrel_sovereign.features.channels.feature import ChannelFeature
    from kestrel_sovereign.features.channels.models import (
        ChannelMessage,
        MessageDirection,
    )

    old_config = {
        "agent_id": "old-agent",
        "enabled": False,
        "allowed_senders": ["old-sender"],
    }
    target_config = {
        "agent_id": "target-agent",
        "enabled": True,
        "allowed_senders": ["target-sender"],
    }
    routed_senders: list[str] = []

    async def route_inbound(message):
        routed_senders.append(message.sender)

    channel_agent = SimpleNamespace(
        did=_TEST_AGENT_DID,
        storage=SimpleNamespace(agent_id=_TEST_AGENT_DID),
        dispatcher=None,
        signal_registry=SourceRegistry(),
        features={},
    )
    channel_feature = ChannelFeature(channel_agent)
    await channel_feature.initialize()
    channel_feature.registry.set_inbound_router(route_inbound)

    class ChannelClient(FakeIsolatedClient):
        capabilities = {
            "channel": {
                "channel_type": "whatsapp",
                "send_tool": "whatsapp_send",
            }
        }

        async def call_tool(self, name, args):
            self.calls.append((name, args))
            return {"ok": True, "data": {"message_id": "WAMID.target"}}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_BIN", "/bin/wa-service")
    clients: list[ChannelClient] = []

    def client_factory(**kwargs):
        client = ChannelClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=client_factory)
    feature._host_config = dict(old_config)
    feature._host_config_loaded = True
    try:
        await feature._connect_client(old_config)
        # ``_replace_client(target_config)`` models forced reconciliation: the
        # target child starts before the caller updates ``_host_config``.
        await feature._replace_client(target_config)

        adapter = channel_feature.registry.get("whatsapp")
        assert adapter is not None
        assert feature._host_config == old_config
        assert adapter.config.enabled is True
        assert adapter.config.allowed_senders == ["target-sender"]
        assert adapter.config.agent_id == "target-agent"

        # Outbound authorization reads ``enabled`` from the bridge config.
        outbound = await channel_feature.channels_send("whatsapp", "+1", "hello")
        assert outbound.status is ToolResultStatus.OK
        assert clients[1].calls == [
            ("whatsapp_send", {"to": "+1", "message": "hello"})
        ]

        # Inbound authorization reads ``allowed_senders`` from that same
        # bridge config, so old policy neither admits old senders nor blocks
        # the target sender.
        await channel_feature.handle_inbound(
            ChannelMessage(
                channel_type="whatsapp",
                direction=MessageDirection.INBOUND,
                sender="old-sender",
                recipient="bot",
                content="stale policy",
            )
        )
        await channel_feature.handle_inbound(
            ChannelMessage(
                channel_type="whatsapp",
                direction=MessageDirection.INBOUND,
                sender="target-sender",
                recipient="bot",
                content="target policy",
            )
        )
        assert routed_senders == ["target-sender"]
    finally:
        await feature.shutdown()
        await channel_feature.shutdown()


@pytest.mark.asyncio
async def test_shutdown_does_not_evict_replacement_adapter(monkeypatch, tmp_path):
    """If another adapter replaced our channel_type, shutdown must not remove it."""
    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {"ChannelFeature": channel_feature}
    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_BIN", "/bin/wa-service")

    class ChannelClient(FakeIsolatedClient):
        capabilities = {"channel": {"channel_type": "whatsapp", "send_tool": "whatsapp_send"}}

    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=lambda **kw: ChannelClient(**kw))
    await feature.initialize()
    assert channel_feature.registry.get("whatsapp") is feature._channel_adapter

    # A native/replacement adapter takes over the same channel_type.
    replacement = object.__new__(type(feature._channel_adapter))
    replacement._channel_type = "whatsapp"  # type: ignore[attr-defined]
    channel_feature.registry.adapters["whatsapp"] = replacement

    await feature.shutdown()
    assert channel_feature.registry.get("whatsapp") is replacement


@pytest.mark.asyncio
async def test_proxy_send_maps_failure_receipt(monkeypatch, tmp_path):
    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    agent.features = {"ChannelFeature": channel_feature}
    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_BIN", "/bin/wa-service")

    class FailingChannelClient(FakeIsolatedClient):
        capabilities = {
            "channel": {"channel_type": "whatsapp", "send_tool": "whatsapp_send"}
        }

        async def call_tool(self, name, args):
            return {"ok": False, "error": "not linked"}

    feature = ProxyFeature(
        agent, _isolated_runtime(), client_factory=lambda **kw: FailingChannelClient(**kw)
    )
    await feature.initialize()
    adapter = channel_feature.registry.adapters["whatsapp"]
    receipt = await adapter.send_message(to="+1", content="x")
    assert receipt.status.value == "failure"
    assert "not linked" in (receipt.error or "")
    await feature.shutdown()


@pytest.mark.asyncio
async def test_proxy_send_maps_toolresult_envelopes(monkeypatch, tmp_path):
    """ToolResult wire shapes (status=error/partial) must not read as success."""
    from kestrel_sovereign.features.isolated_runtime import (
        _delivery_receipt_from_result,
    )

    # status=error wrapped as a successful transport call must be a FAILURE
    err = _delivery_receipt_from_result(
        "whatsapp", {"success": True, "result": {"status": "error", "error": "not linked"}}
    )
    assert err.status.value == "failure"
    assert "not linked" in (err.error or "")

    # status=partial -> PENDING (honesty: not yet confirmed)
    part = _delivery_receipt_from_result(
        "whatsapp",
        {"success": True, "result": {"status": "partial", "data": {"receipt": {"message_id": "M2"}}}},
    )
    assert part.status.value == "pending"
    assert part.message_id == "M2"

    # status=ok -> SUCCESS
    ok = _delivery_receipt_from_result(
        "whatsapp", {"success": True, "result": {"status": "ok", "data": {"message_id": "M3"}}}
    )
    assert ok.status.value == "success"
    assert ok.message_id == "M3"

    # top-level message_id on a plain {"ok": True} envelope is preserved
    top = _delivery_receipt_from_result(
        "whatsapp", {"success": True, "result": {"ok": True, "message_id": "WAMID.top"}}
    )
    assert top.status.value == "success"
    assert top.message_id == "WAMID.top"


def test_proxy_feature_resolves_default_per_agent_venv(tmp_path):
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    runtime = InstalledFeatureRuntime(
        class_name="VoiceFeature",
        entry_point="voice.feature:VoiceFeature",
        distribution="kestrel-feature-voice",
        runtime="isolated-venv",
        service="kestrel-voice-service",
    )

    feature = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    venv, bin_path = feature.resolve_runtime_paths()

    assert (
        venv
        == Path(agent.storage_path).parent / "feature_venvs" / "VoiceFeature" / ".venv"
    )
    assert bin_path is None


def test_sqlite_whatsapp_default_state_path_is_not_relocated(tmp_path):
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)
    venv, _ = feature.resolve_runtime_paths()
    runtime_dir = feature._prepare_runtime_workspace()
    env = isolated_runtime._isolated_child_env(
        venv,
        runtime_dir=runtime_dir,
        hosted=False,
        feature_name=feature.name,
        feature_distribution=feature.runtime.distribution,
    )

    old_fallback = venv.parent / "whatsapp_service"
    assert "KESTREL_ISOLATED_FEATURE_DATA_DIR" not in env
    assert "KESTREL_ISOLATED_RUNTIME_DIR" not in env
    assert old_fallback == (
        tmp_path / "agent" / "feature_venvs" / "WhatsAppFeature" / "whatsapp_service"
    )


def test_standalone_prebuilt_whatsapp_state_keeps_legacy_venv_parent_custody(
    monkeypatch, tmp_path
):
    prebuilt = tmp_path / "operator" / ".venv"
    legacy = prebuilt.parent / "whatsapp_service"
    legacy.mkdir(parents=True)
    (legacy / "session.sqlite3").write_bytes(b"linked-device-credential")
    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_VENV", str(prebuilt))
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")

    feature = ProxyFeature(
        agent, _isolated_runtime(), client_factory=FakeIsolatedClient
    )
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    client = feature._build_client()

    assert (legacy / "session.sqlite3").read_bytes() == b"linked-device-credential"
    assert "KESTREL_ISOLATED_FEATURE_DATA_DIR" not in client.kwargs["env"]
    assert "cwd" not in client.kwargs


@pytest.mark.parametrize(
    "legacy_env_key",
    ("KESTREL_FEATURE_DATA_DIR", "KESTREL_DATA_DIR"),
)
def test_standalone_legacy_whatsapp_data_dir_preserves_child_precedence(
    monkeypatch, tmp_path, legacy_env_key
):
    legacy_root = tmp_path / f"legacy-{legacy_env_key.casefold()}"
    credential = legacy_root / "whatsapp_service" / "session.sqlite3"
    credential.parent.mkdir(parents=True)
    credential.write_bytes(b"linked-device-credential")
    monkeypatch.setenv(legacy_env_key, str(legacy_root))
    for agent_name in ("agent-a", "agent-b"):
        agent = Mock(did=f"did:test:{agent_name}")
        agent.storage_path = str(tmp_path / agent_name / "kestrel_prime.db")

        feature = ProxyFeature(
            agent, _isolated_runtime(), client_factory=FakeIsolatedClient
        )
        feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
        client = feature._build_client()
        assert client.kwargs["env"][legacy_env_key] == str(legacy_root)
        assert "KESTREL_ISOLATED_FEATURE_DATA_DIR" not in client.kwargs["env"]
        assert "cwd" not in client.kwargs

    assert credential.read_bytes() == b"linked-device-credential"


def test_feature_data_dir_is_the_active_legacy_whatsapp_precedence(
    monkeypatch, tmp_path
):
    feature_root = tmp_path / "feature-data"
    generic_root = tmp_path / "generic-data"
    feature_credential = feature_root / "whatsapp_service" / "session.sqlite3"
    generic_credential = generic_root / "whatsapp_service" / "session.sqlite3"
    feature_credential.parent.mkdir(parents=True)
    generic_credential.parent.mkdir(parents=True)
    feature_credential.write_bytes(b"active-feature-credential")
    generic_credential.write_bytes(b"shadowed-generic-credential")
    monkeypatch.setenv("KESTREL_FEATURE_DATA_DIR", str(feature_root))
    monkeypatch.setenv("KESTREL_DATA_DIR", str(generic_root))
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")

    feature = ProxyFeature(
        agent, _isolated_runtime(), client_factory=FakeIsolatedClient
    )
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    env = feature._build_client().kwargs["env"]

    assert env["KESTREL_FEATURE_DATA_DIR"] == str(feature_root)
    assert env["KESTREL_DATA_DIR"] == str(generic_root)
    assert "KESTREL_ISOLATED_FEATURE_DATA_DIR" not in env
    assert feature_credential.read_bytes() == b"active-feature-credential"
    assert generic_credential.read_bytes() == b"shadowed-generic-credential"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission mode contract")
def test_standalone_runtime_workspace_parents_are_private(tmp_path):
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    feature = ProxyFeature(
        agent, _isolated_runtime(), client_factory=FakeIsolatedClient
    )

    runtime_dir = feature._prepare_runtime_workspace()

    for directory in (
        tmp_path / "agent",
        tmp_path / "agent" / "feature_venvs",
        runtime_dir,
        runtime_dir / "work",
        runtime_dir / "config",
        tmp_path / "agent" / "channel_link_artifacts",
    ):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission mode contract")
def test_standalone_runtime_preserves_preexisting_storage_parent_mode(tmp_path):
    agent_dir = tmp_path / "operator-agent-data"
    agent_dir.mkdir(mode=0o755)
    agent_dir.chmod(0o755)
    unrelated = agent_dir / "operator-owned.txt"
    unrelated.write_text("retain")
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(agent_dir / "kestrel_prime.db")
    feature = ProxyFeature(
        agent, _isolated_runtime(), client_factory=FakeIsolatedClient
    )

    runtime_dir = feature._prepare_runtime_workspace()

    assert stat.S_IMODE(agent_dir.stat().st_mode) == 0o755
    assert unrelated.read_text() == "retain"
    for directory in (
        agent_dir / "feature_venvs",
        runtime_dir,
        runtime_dir / "work",
        runtime_dir / "config",
        agent_dir / "channel_link_artifacts",
    ):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def test_standalone_runtime_rejects_preexisting_feature_workspace_symlink(tmp_path):
    """Standalone custody also refuses a planted mutable-workspace symlink."""

    agent_dir = tmp_path / "agent"
    feature_parent = agent_dir / "feature_venvs"
    feature_parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "operator-owned.txt"
    sentinel.write_text("retain")
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(agent_dir / "kestrel_prime.db")
    feature = ProxyFeature(
        agent,
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )
    feature._feature_runtime_dir().symlink_to(outside, target_is_directory=True)

    with pytest.raises(IsolatedRuntimeNamespaceError, match="symlinks"):
        feature._prepare_runtime_workspace()

    assert sentinel.read_text() == "retain"
    assert list(outside.iterdir()) == [sentinel]


def _hosted_postgres_agent(
    runtime_root: Path,
    namespace: str,
    *,
    did: str | None = None,
) -> KestrelAgent:
    """Construct only: Postgres-backed hosted agents intentionally lack SQLite paths."""
    llm_service = Mock()
    llm_service.providers = []
    return KestrelAgent(
        did=did or f"did:test:{namespace.replace('/', ':')}",
        storage_path=None,
        llm_service=llm_service,
        database_url="postgresql://hosted.example/kestrel",
        db_backend="postgres",
        isolated_runtime_root=runtime_root,
        isolated_runtime_namespace=namespace,
        isolated_runtime_hosted=True,
    )


def test_postgres_agents_get_distinct_explicit_runtime_namespaces(tmp_path):
    """Postgres storage must never collapse isolated venvs into CWD/default."""
    runtime_root = tmp_path / "hosted-runtime"
    agent_a = _hosted_postgres_agent(runtime_root, "tenant-a/agent-a")
    agent_b = _hosted_postgres_agent(runtime_root, "tenant-b/agent-b")
    runtime = _isolated_runtime()

    feature_a = ProxyFeature(agent_a, runtime, client_factory=FakeIsolatedClient)
    feature_b = ProxyFeature(agent_b, runtime, client_factory=FakeIsolatedClient)
    venv_a, _ = feature_a.resolve_runtime_paths()
    venv_b, _ = feature_b.resolve_runtime_paths()

    assert agent_a.storage_path is None
    assert agent_b.storage_path is None
    assert venv_a != venv_b
    assert venv_a == (
        runtime_root.resolve()
        / "tenant-a"
        / "agent-a"
        / "feature_venvs"
        / feature_a._runtime_directory_name
        / ".venv"
    )
    assert venv_b == (
        runtime_root.resolve()
        / "tenant-b"
        / "agent-b"
        / "feature_venvs"
        / feature_b._runtime_directory_name
        / ".venv"
    )
    assert feature_a._runtime_directory_name == feature_b._runtime_directory_name
    assert feature_a._runtime_directory_name.startswith("feature-")


@pytest.mark.parametrize("suffix", ("BIN", "VENV"))
def test_hosted_process_override_must_already_exist(
    monkeypatch,
    tmp_path,
    suffix,
):
    key = f"KESTREL_FEATURE_WHATSAPPFEATURE_{suffix}"
    monkeypatch.setenv(key, str(tmp_path / f"missing-{suffix.casefold()}"))

    for namespace in ("agent-a", "agent-b"):
        agent = _hosted_postgres_agent(tmp_path / "runtime", namespace)
        with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
            ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)
        diagnostic = raised.value.safe_diagnostic()
        assert key in diagnostic
        assert str(tmp_path) not in diagnostic


def _write_prebuilt_venv_shape(
    venv: Path,
    *,
    console_service: str | None = None,
) -> Path:
    python = isolated_runtime._venv_python(venv)
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o700)
    (venv / "pyvenv.cfg").write_text("home = /operator/python\n")
    if console_service is not None:
        wrapper = isolated_runtime._console_script_path(venv, console_service)
        wrapper.write_text(f"#!{python}\nexit 0\n")
        wrapper.chmod(0o700)
    return python


def _stat_result_with_uid(metadata: os.stat_result, uid: int) -> os.stat_result:
    """Clone metadata with a deterministic foreign owner for custody tests."""

    fields = list(metadata)
    fields[stat.ST_UID] = uid
    return os.stat_result(fields)


def _write_real_venv_module(
    venv: Path,
    module: str,
    source: str,
) -> Path:
    """Create a pip-free real venv and install one importable source module."""

    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)],
        check=True,
        capture_output=True,
        text=True,
    )
    python = isolated_runtime._venv_python(venv)
    purelib = Path(
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    components = module.split(".")
    package = purelib
    for component in components[:-1]:
        package /= component
        package.mkdir(exist_ok=True)
        (package / "__init__.py").touch()
    (package / f"{components[-1]}.py").write_text(source)
    return python


def _materialize_fake_provisioned_venv(feature: ProxyFeature) -> Path:
    """Create the launch artifacts a successful uv install must provide."""

    assert feature._venv_path is not None
    python = isolated_runtime._venv_python(feature._venv_path)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o700)
    service = feature.runtime.service
    if isinstance(service, str) and service and ":" not in service:
        console = isolated_runtime._console_script_path(
            feature._venv_path,
            service,
        )
        console.write_text(f"#!{python}\nexit 0\n")
        console.chmod(0o700)
    return python


def _stamp_current_fake_venv(feature: ProxyFeature, monkeypatch) -> Path:
    """Materialize and stamp one venv whose manifest is fully current."""

    python = _materialize_fake_provisioned_venv(feature)
    feature._probe_sdk_version = Mock(return_value="current-host-sdk")
    feature._probe_feature_distribution = Mock(
        return_value=_child_distribution_probe("1.0.0")
    )
    monkeypatch.setattr(
        isolated_runtime,
        "_host_sdk_version",
        lambda: "current-host-sdk",
    )
    monkeypatch.setattr(
        isolated_runtime,
        "_feature_distribution_version",
        lambda _distribution, _target: "1.0.0",
    )
    feature._write_provision_manifest(
        feature.runtime.project or feature.runtime.distribution,
        "current-host-sdk",
        "current-host-sdk",
        "1.0.0",
        _child_distribution_probe("1.0.0"),
    )
    return python


@pytest.mark.skipif(os.name != "posix", reason="repair marker uses POSIX dirfds")
def test_relocation_marker_forces_reinstall_when_other_evidence_looks_current(
    monkeypatch, tmp_path
):
    agent = _hosted_postgres_agent(
        tmp_path / "runtime",
        "agent-marker-only-relocation",
        did="did:test:marker-only-relocation",
    )
    feature = ProxyFeature(
        agent,
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )
    feature._prepare_runtime_workspace()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    _stamp_current_fake_venv(feature, monkeypatch)
    console = isolated_runtime._console_script_path(
        feature._venv_path,
        feature.runtime.service,
    )
    console.write_text("#!/usr/bin/env python\n")
    runtime_fd = os.open(
        feature._feature_runtime_dir(),
        isolated_runtime._directory_open_flags(),
    )
    try:
        isolated_runtime._ensure_venv_relocation_repair_marker_at(runtime_fd)
    finally:
        os.close(runtime_fd)

    manifest = feature._read_provision_manifest()
    assert feature._console_script_location_state() == "current"
    assert manifest["venv_path"] == str(feature._venv_path.resolve())
    assert feature._location_requires_forced_reinstall(manifest) is True


def _runtime_with_declared_venv(venv: str) -> InstalledFeatureRuntime:
    runtime = _isolated_runtime()
    return InstalledFeatureRuntime(
        class_name=runtime.class_name,
        entry_point=runtime.entry_point,
        distribution=runtime.distribution,
        runtime=runtime.runtime,
        service=runtime.service,
        project=runtime.project,
        venv=venv,
    )


def test_hosted_declared_runtime_venv_rejects_relative_and_missing_paths(tmp_path):
    for declared in ("service", str(tmp_path / "missing-prebuilt")):
        for namespace in ("agent-a", "agent-b"):
            with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
                ProxyFeature(
                    _hosted_postgres_agent(tmp_path / "runtime", namespace),
                    _runtime_with_declared_venv(declared),
                    client_factory=FakeIsolatedClient,
                )
            diagnostic = raised.value.safe_diagnostic()
            assert "runtime.venv" in diagnostic
            assert declared not in diagnostic


def test_hosted_declared_runtime_venv_rejects_core_managed_shared_path(tmp_path):
    shared = tmp_path / "core-managed-shared"
    _write_prebuilt_venv_shape(shared)
    manifest = shared / ".kestrel_provision.json"
    manifest.write_text('{"venv_path": "private-host-path"}')

    for namespace in ("agent-a", "agent-b"):
        with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
            ProxyFeature(
                _hosted_postgres_agent(tmp_path / "runtime", namespace),
                _runtime_with_declared_venv(str(shared)),
                client_factory=FakeIsolatedClient,
            )
        assert "runtime.venv" in raised.value.safe_diagnostic()
        assert str(shared) not in raised.value.safe_diagnostic()
    assert manifest.read_text() == '{"venv_path": "private-host-path"}'


def test_hosted_process_venv_override_rejects_core_managed_manifest(
    monkeypatch,
    tmp_path,
):
    prebuilt = tmp_path / "shared-prebuilt"
    _write_prebuilt_venv_shape(prebuilt)
    manifest = prebuilt / ".kestrel_provision.json"
    manifest.write_text('{"install_target": "old-core-state"}')
    key = "KESTREL_FEATURE_WHATSAPPFEATURE_VENV"
    monkeypatch.setenv(key, str(prebuilt))

    with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
        ProxyFeature(
            _hosted_postgres_agent(tmp_path / "runtime", "agent-core-managed"),
            _isolated_runtime(),
            client_factory=FakeIsolatedClient,
        )

    assert key in raised.value.safe_diagnostic()
    assert manifest.read_text() == '{"install_target": "old-core-state"}'


@pytest.mark.skipif(os.name != "posix", reason="POSIX immutable custody contract")
@pytest.mark.parametrize("component", ("root", "config", "bin", "python"))
def test_hosted_process_venv_override_rejects_mutable_components(
    monkeypatch,
    tmp_path,
    component,
):
    prebuilt = tmp_path / "mutable-prebuilt"
    python = _write_prebuilt_venv_shape(prebuilt)
    components = {
        "root": prebuilt,
        "config": prebuilt / "pyvenv.cfg",
        "bin": isolated_runtime._venv_bin_dir(prebuilt),
        "python": python,
    }
    selected = components[component]
    selected.chmod(stat.S_IMODE(selected.stat().st_mode) | 0o020)
    key = "KESTREL_FEATURE_WHATSAPPFEATURE_VENV"
    monkeypatch.setenv(key, str(prebuilt))

    with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
        ProxyFeature(
            _hosted_postgres_agent(tmp_path / "runtime", f"mutable-{component}"),
            _isolated_runtime(),
            client_factory=FakeIsolatedClient,
        )

    diagnostic = raised.value.safe_diagnostic()
    assert key in diagnostic
    assert str(prebuilt) not in diagnostic


@pytest.mark.parametrize("component", ("root", "config", "bin", "python"))
def test_hosted_process_venv_override_rejects_wrong_component_types(
    monkeypatch,
    tmp_path,
    component,
):
    prebuilt = tmp_path / "wrong-type-prebuilt"
    if component == "root":
        prebuilt.write_text("not a venv directory")
    else:
        python = _write_prebuilt_venv_shape(prebuilt)
        selected = {
            "config": prebuilt / "pyvenv.cfg",
            "bin": isolated_runtime._venv_bin_dir(prebuilt),
            "python": python,
        }[component]
        if selected.is_dir():
            python.unlink()
            selected.rmdir()
            selected.write_text("not a directory")
        else:
            selected.unlink()
            selected.mkdir()
    key = "KESTREL_FEATURE_WHATSAPPFEATURE_VENV"
    monkeypatch.setenv(key, str(prebuilt))

    with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
        ProxyFeature(
            _hosted_postgres_agent(tmp_path / "runtime", f"wrong-{component}"),
            _isolated_runtime(),
            client_factory=FakeIsolatedClient,
        )

    assert key in raised.value.safe_diagnostic()
    assert str(prebuilt) not in raised.value.safe_diagnostic()


@pytest.mark.skipif(os.name != "posix", reason="POSIX immutable custody contract")
@pytest.mark.parametrize("component", ("root", "config", "bin", "python"))
def test_hosted_process_venv_override_rejects_foreign_owned_components(
    monkeypatch,
    tmp_path,
    component,
):
    prebuilt = tmp_path / "foreign-prebuilt"
    python = _write_prebuilt_venv_shape(prebuilt)
    components = {
        "root": prebuilt.resolve(),
        "config": (prebuilt / "pyvenv.cfg").resolve(),
        "bin": isolated_runtime._venv_bin_dir(prebuilt).resolve(),
        "python": python.resolve(),
    }
    selected = components[component]
    real_stat = Path.stat
    foreign_uid = next(uid for uid in range(1, 4) if uid != os.geteuid())

    def foreign_component_stat(path, *, follow_symlinks=True):
        metadata = real_stat(path, follow_symlinks=follow_symlinks)
        if path == selected:
            return _stat_result_with_uid(metadata, foreign_uid)
        return metadata

    monkeypatch.setattr(Path, "stat", foreign_component_stat)
    key = "KESTREL_FEATURE_WHATSAPPFEATURE_VENV"
    monkeypatch.setenv(key, str(prebuilt))

    with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
        ProxyFeature(
            _hosted_postgres_agent(tmp_path / "runtime", f"foreign-{component}"),
            _isolated_runtime(),
            client_factory=FakeIsolatedClient,
        )

    assert key in raised.value.safe_diagnostic()
    assert str(prebuilt) not in raised.value.safe_diagnostic()


@pytest.mark.skipif(os.name != "posix", reason="POSIX operator symlink contract")
def test_hosted_process_venv_accepts_secure_operator_symlink_chains(
    monkeypatch,
    tmp_path,
):
    actual_venv = tmp_path / "operator-venv-v1"
    original_python = _write_prebuilt_venv_shape(actual_venv)
    original_python.unlink()
    interpreter_target = tmp_path / "operator-python"
    interpreter_target.write_text("#!/bin/sh\nexit 0\n")
    interpreter_target.chmod(0o500)
    original_python.symlink_to(interpreter_target)
    original_config = actual_venv / "pyvenv.cfg"
    original_config.unlink()
    config_target = tmp_path / "operator-pyvenv.cfg"
    config_target.write_text("home = /operator/python\n")
    config_target.chmod(0o400)
    original_config.symlink_to(config_target)
    public_venv = tmp_path / "current-operator-venv"
    public_venv.symlink_to(actual_venv, target_is_directory=True)
    key = "KESTREL_FEATURE_WHATSAPPFEATURE_VENV"
    monkeypatch.setenv(key, str(public_venv))

    feature = ProxyFeature(
        _hosted_postgres_agent(tmp_path / "runtime", "secure-symlink-venv"),
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )
    venv_path, bin_path = feature.resolve_runtime_paths()

    assert venv_path == actual_venv.resolve()
    assert bin_path is None
    assert isolated_runtime._venv_python(venv_path).resolve() == (
        interpreter_target.resolve()
    )


def _assert_mutated_hosted_venv_selection_fails_before_probe(
    feature: ProxyFeature,
    *,
    expected_setting: str,
) -> None:
    feature._verify_launch_artifact = Mock(
        side_effect=AssertionError("stale launch artifact was inspected")
    )
    feature._verify_prebuilt_feature_distribution = Mock(
        side_effect=AssertionError("stale distribution was probed")
    )
    feature._warn_on_sdk_mismatch = Mock(
        side_effect=AssertionError("stale SDK was probed")
    )
    feature._run = Mock(side_effect=AssertionError("stale venv was provisioned"))

    with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
        feature.ensure_venv()

    diagnostic = raised.value.safe_diagnostic()
    assert expected_setting in diagnostic
    assert str(feature._venv_path) not in diagnostic
    feature._verify_launch_artifact.assert_not_called()
    feature._verify_prebuilt_feature_distribution.assert_not_called()
    feature._warn_on_sdk_mismatch.assert_not_called()
    feature._run.assert_not_called()
    assert feature._validated_hosted_console_path is None


@pytest.mark.skipif(os.name != "posix", reason="POSIX release symlink contract")
def test_hosted_venv_release_symlink_flip_fails_before_stale_tree_probe(
    monkeypatch,
    tmp_path,
):
    releases = tmp_path / "releases"
    release_1 = releases / "v1"
    release_2 = releases / "v2"
    for release in (release_1, release_2):
        _write_prebuilt_venv_shape(
            release,
            console_service=_isolated_runtime().service,
        )
    selected = tmp_path / "current-venv"
    selected.symlink_to(release_1, target_is_directory=True)
    key = "KESTREL_FEATURE_WHATSAPPFEATURE_VENV"
    monkeypatch.setenv(key, str(selected))
    client_factory = Mock(side_effect=AssertionError("stale child was launched"))
    feature = ProxyFeature(
        _hosted_postgres_agent(tmp_path / "runtime", "release-link-flip"),
        _isolated_runtime(),
        client_factory=client_factory,
    )
    feature._prepare_runtime_workspace()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    assert feature._venv_path == release_1.resolve()

    selected.unlink()
    selected.symlink_to(release_2, target_is_directory=True)

    _assert_mutated_hosted_venv_selection_fails_before_probe(
        feature,
        expected_setting=key,
    )
    client_factory.assert_not_called()


def test_hosted_runtime_venv_to_process_override_flip_fails_before_stale_probe(
    monkeypatch,
    tmp_path,
):
    declared = tmp_path / "declared-venv"
    process_override = tmp_path / "process-venv"
    for prebuilt in (declared, process_override):
        _write_prebuilt_venv_shape(
            prebuilt,
            console_service=_isolated_runtime().service,
        )
    key = "KESTREL_FEATURE_WHATSAPPFEATURE_VENV"
    client_factory = Mock(side_effect=AssertionError("stale child was launched"))
    feature = ProxyFeature(
        _hosted_postgres_agent(tmp_path / "runtime", "selection-flip"),
        _runtime_with_declared_venv(str(declared)),
        client_factory=client_factory,
    )
    feature._prepare_runtime_workspace()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    assert feature._venv_path == declared.resolve()

    monkeypatch.setenv(key, str(process_override))

    _assert_mutated_hosted_venv_selection_fails_before_probe(
        feature,
        expected_setting=key,
    )
    client_factory.assert_not_called()


def test_hosted_process_override_removal_selects_runtime_venv_and_fails_closed(
    monkeypatch,
    tmp_path,
):
    declared = tmp_path / "declared-venv"
    process_override = tmp_path / "process-venv"
    for prebuilt in (declared, process_override):
        _write_prebuilt_venv_shape(
            prebuilt,
            console_service=_isolated_runtime().service,
        )
    key = "KESTREL_FEATURE_WHATSAPPFEATURE_VENV"
    monkeypatch.setenv(key, str(process_override))
    client_factory = Mock(side_effect=AssertionError("stale child was launched"))
    feature = ProxyFeature(
        _hosted_postgres_agent(tmp_path / "runtime", "selection-removal"),
        _runtime_with_declared_venv(str(declared)),
        client_factory=client_factory,
    )
    feature._prepare_runtime_workspace()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    assert feature._venv_path == process_override.resolve()

    monkeypatch.delenv(key)

    _assert_mutated_hosted_venv_selection_fails_before_probe(
        feature,
        expected_setting="runtime.venv",
    )
    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_hosted_agents_share_only_immutable_prebuilt_venv_without_writes(
    monkeypatch,
    tmp_path,
):
    prebuilt = tmp_path / "operator-prebuilt"
    python = _write_prebuilt_venv_shape(
        prebuilt,
        console_service=_isolated_runtime().service,
    )
    before = {
        path.relative_to(prebuilt): (path.read_bytes(), path.stat().st_mode)
        for path in prebuilt.rglob("*")
        if path.is_file()
    }
    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_VENV", str(prebuilt))
    features = [
        ProxyFeature(
            _hosted_postgres_agent(tmp_path / "runtime", namespace),
            _isolated_runtime(),
            client_factory=FakeIsolatedClient,
        )
        for namespace in ("agent-a", "agent-b")
    ]
    for feature in features:
        feature._prepare_runtime_workspace()
        feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
        feature._verify_prebuilt_feature_distribution = Mock()
        feature._warn_on_sdk_mismatch = Mock()
        feature._run = Mock(side_effect=AssertionError("prebuilt venv was mutated"))

    await asyncio.gather(
        *(asyncio.to_thread(feature.ensure_venv) for feature in features)
    )

    assert all(feature._venv_path == prebuilt for feature in features)
    assert python.is_file()
    assert not (prebuilt / ".kestrel_provision.json").exists()
    after = {
        path.relative_to(prebuilt): (path.read_bytes(), path.stat().st_mode)
        for path in prebuilt.rglob("*")
        if path.is_file()
    }
    assert after == before
    for feature in features:
        feature._run.assert_not_called()


@pytest.mark.asyncio
async def test_hosted_agents_share_declared_immutable_venv_without_writes(
    tmp_path,
):
    prebuilt = tmp_path / "declared-operator-prebuilt"
    python = _write_prebuilt_venv_shape(
        prebuilt,
        console_service=_isolated_runtime().service,
    )
    runtime = _runtime_with_declared_venv(str(prebuilt))
    before = {
        path.relative_to(prebuilt): (path.read_bytes(), path.stat().st_mode)
        for path in prebuilt.rglob("*")
        if path.is_file()
    }
    features = [
        ProxyFeature(
            _hosted_postgres_agent(tmp_path / "runtime", namespace),
            runtime,
            client_factory=FakeIsolatedClient,
        )
        for namespace in ("agent-a", "agent-b")
    ]
    for feature in features:
        feature._prepare_runtime_workspace()
        feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
        feature._verify_prebuilt_feature_distribution = Mock()
        feature._warn_on_sdk_mismatch = Mock()
        feature._run = Mock(side_effect=AssertionError("declared venv was mutated"))

    await asyncio.gather(
        *(asyncio.to_thread(feature.ensure_venv) for feature in features)
    )

    assert all(feature._venv_path == prebuilt for feature in features)
    assert python.is_file()
    assert not (prebuilt / ".kestrel_provision.json").exists()
    after = {
        path.relative_to(prebuilt): (path.read_bytes(), path.stat().st_mode)
        for path in prebuilt.rglob("*")
        if path.is_file()
    }
    assert after == before
    for feature in features:
        feature._run.assert_not_called()


def test_hosted_declared_venv_late_manifest_never_enters_provisioning(tmp_path):
    prebuilt = tmp_path / "declared-late-core-managed"
    _write_prebuilt_venv_shape(prebuilt)
    feature = ProxyFeature(
        _hosted_postgres_agent(tmp_path / "runtime", "agent-late-declared"),
        _runtime_with_declared_venv(str(prebuilt)),
        client_factory=FakeIsolatedClient,
    )
    feature._prepare_runtime_workspace()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    feature._run = Mock(side_effect=AssertionError("declared provisioning ran"))
    manifest = prebuilt / ".kestrel_provision.json"
    manifest.write_text('{"venv_path": "late-private-path"}')

    with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
        feature.ensure_venv()

    assert "runtime.venv" in raised.value.safe_diagnostic()
    assert str(prebuilt) not in raised.value.safe_diagnostic()
    assert manifest.read_text() == '{"venv_path": "late-private-path"}'
    feature._run.assert_not_called()


def test_hosted_process_venv_late_manifest_never_enters_provisioning(
    monkeypatch,
    tmp_path,
):
    prebuilt = tmp_path / "late-core-managed"
    _write_prebuilt_venv_shape(prebuilt)
    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_VENV", str(prebuilt))
    feature = ProxyFeature(
        _hosted_postgres_agent(tmp_path / "runtime", "agent-late-manifest"),
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )
    feature._prepare_runtime_workspace()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    feature._run = Mock(side_effect=AssertionError("override provisioning ran"))
    manifest = prebuilt / ".kestrel_provision.json"
    manifest.write_text('{"install_target": "racing-core-state"}')

    with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
        feature.ensure_venv()

    assert "KESTREL_FEATURE_WHATSAPPFEATURE_VENV" in raised.value.safe_diagnostic()
    assert manifest.read_text() == '{"install_target": "racing-core-state"}'
    feature._run.assert_not_called()


def test_hosted_agents_accept_existing_regular_prebuilt_bin_without_venv(
    monkeypatch,
    tmp_path,
):
    executable = tmp_path / "operator-bin" / "whatsapp-service"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_BIN", str(executable))

    features = [
        ProxyFeature(
            _hosted_postgres_agent(tmp_path / "runtime", namespace),
            _isolated_runtime(),
            client_factory=FakeIsolatedClient,
        )
        for namespace in ("agent-bin-a", "agent-bin-b")
    ]

    for feature in features:
        venv, bin_path = feature.resolve_runtime_paths()
        assert bin_path == executable
        assert not venv.exists()
    assert executable.read_text() == "#!/bin/sh\nexit 0\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink custody contract")
def test_hosted_prebuilt_bin_symlink_is_pinned_to_validated_target(
    monkeypatch,
    tmp_path,
):
    operator_bin = tmp_path / "operator-bin"
    operator_bin.mkdir()
    first_target = operator_bin / "whatsapp-service-v1"
    second_target = operator_bin / "whatsapp-service-v2"
    for target in (first_target, second_target):
        target.write_text("#!/bin/sh\nexit 0\n")
        target.chmod(0o700)
    public_link = operator_bin / "whatsapp-service"
    public_link.symlink_to(first_target.name)
    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_BIN", str(public_link))
    feature = ProxyFeature(
        _hosted_postgres_agent(tmp_path / "runtime", "agent-bin-symlink"),
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )

    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    assert feature._bin_path == first_target.resolve()
    public_link.unlink()
    public_link.symlink_to(second_target.name)

    assert feature._service_command() == [str(first_target.resolve())]


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink custody contract")
def test_hosted_prebuilt_bin_symlink_rejects_mutable_target(
    monkeypatch,
    tmp_path,
):
    target = tmp_path / "operator-bin" / "mutable-service"
    target.parent.mkdir()
    target.write_text("#!/bin/sh\nexit 0\n")
    target.chmod(0o777)
    public_link = target.with_name("whatsapp-service")
    public_link.symlink_to(target.name)
    key = "KESTREL_FEATURE_WHATSAPPFEATURE_BIN"
    monkeypatch.setenv(key, str(public_link))

    with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
        ProxyFeature(
            _hosted_postgres_agent(tmp_path / "runtime", "agent-unsafe-bin"),
            _isolated_runtime(),
            client_factory=FakeIsolatedClient,
        )

    diagnostic = raised.value.safe_diagnostic()
    assert key in diagnostic
    assert str(tmp_path) not in diagnostic


def test_hosted_feature_runtime_identity_ignores_module_and_service_refactors():
    original = InstalledFeatureRuntime(
        class_name="WhatsAppFeature",
        entry_point="old_package.feature:WhatsAppFeature",
        distribution="Kestrel_Channel.WhatsApp",
        runtime="isolated-venv",
        service="old-service",
    )
    refactored = InstalledFeatureRuntime(
        class_name="WhatsAppFeature",
        entry_point="new_package.channel:WhatsAppFeature",
        distribution="kestrel-channel-whatsapp",
        runtime="isolated-venv",
        service="new-channel-service",
    )
    distinct = InstalledFeatureRuntime(
        class_name="TelegramFeature",
        entry_point="new_package.channel:TelegramFeature",
        distribution="kestrel-channel-whatsapp",
        runtime="isolated-venv",
        service="new-channel-service",
    )

    original_component = isolated_runtime._hosted_feature_runtime_component(original)
    assert original_component == isolated_runtime._hosted_feature_runtime_component(
        refactored
    )
    assert original_component != isolated_runtime._hosted_feature_runtime_component(
        distinct
    )


@pytest.mark.skipif(os.name != "posix", reason="migration uses secure POSIX dirfds")
def test_hosted_feature_runtime_adopts_released_class_named_whatsapp_tree(tmp_path):
    """The deployed PG layout moves intact, including venv and credentials."""

    agent_dir = tmp_path / "agent_data" / "Hosted"
    legacy_root = agent_dir / "feature_venvs"
    legacy_feature = legacy_root / "WhatsAppFeature"
    credential = legacy_feature / "whatsapp_service" / "session.sqlite3"
    venv_marker = legacy_feature / ".venv" / "pyvenv.cfg"
    credential.parent.mkdir(parents=True, mode=0o700)
    venv_marker.parent.mkdir(mode=0o700)
    legacy_root.chmod(0o700)
    legacy_feature.chmod(0o700)
    credential.write_bytes(b"released-linked-device-credential")
    venv_marker.write_text("home = /operator/python\n")
    runtime_root = tmp_path / "isolated_feature_runtime"
    agent = KestrelAgent(
        did="did:test:released-hosted-layout",
        storage_path=str(agent_dir / "kestrel_prime.db"),
        llm_service=Mock(providers=[]),
        database_url="postgresql://hosted.example/kestrel",
        db_backend="postgres",
        isolated_runtime_root=runtime_root,
        isolated_runtime_namespace="agent-released",
        isolated_runtime_legacy_root=legacy_root,
        isolated_runtime_hosted=True,
    )
    feature = ProxyFeature(
        agent,
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )

    stable = feature._prepare_runtime_workspace()

    assert (stable / "whatsapp_service" / "session.sqlite3").read_bytes() == (
        b"released-linked-device-credential"
    )
    assert (stable / ".venv" / "pyvenv.cfg").read_text() == (
        "home = /operator/python\n"
    )
    assert not legacy_feature.exists()
    assert legacy_root.is_dir()


@pytest.mark.skipif(os.name != "posix", reason="console shebangs are POSIX paths")
@pytest.mark.parametrize(
    ("migration_kind", "manifest_kind"),
    (("released", "missing"), ("pre-stable", "old-path")),
    ids=("released-layout", "pre-stable-layout"),
)
def test_hosted_migration_repairs_real_console_script_once(
    tmp_path,
    migration_kind,
    manifest_kind,
):
    """A moved real venv gets one forced reinstall, preserving adjacent state."""

    wheel = _write_console_script_fixture_wheel(tmp_path)
    runtime = InstalledFeatureRuntime(
        class_name="WhatsAppFeature",
        entry_point="wa.feature:WhatsAppFeature",
        distribution="kestrel-console-migration-fixture",
        runtime="isolated-venv",
        service="kestrel-whatsapp-service",
        project=str(wheel),
    )
    runtime_root = tmp_path / "hosted-runtime"
    if migration_kind == "released":
        agent_dir = tmp_path / "agent_data" / "Hosted"
        legacy_root = agent_dir / "feature_venvs"
        source_feature = legacy_root / "WhatsAppFeature"
        source_feature.mkdir(parents=True, mode=0o700)
        legacy_root.chmod(0o700)
        agent = KestrelAgent(
            did="did:test:real-console-released",
            storage_path=str(agent_dir / "kestrel_prime.db"),
            llm_service=Mock(providers=[]),
            database_url="postgresql://hosted.example/kestrel",
            db_backend="postgres",
            isolated_runtime_root=runtime_root,
            isolated_runtime_namespace="agent-real-console-released",
            isolated_runtime_legacy_root=legacy_root,
            isolated_runtime_hosted=True,
        )
        feature = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    else:
        agent = _hosted_postgres_agent(
            runtime_root,
            "agent-real-console-pre-stable",
            did="did:test:real-console-pre-stable",
        )
        feature = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
        legacy_component = feature._legacy_runtime_directory_name
        assert legacy_component is not None
        isolated_runtime.prepare_isolated_runtime_namespace(
            feature._isolated_runtime_scope,
            agent.did,
            relative_directories=(("feature_venvs", legacy_component),),
        )
        source_feature = (
            feature._agent_runtime_dir / "feature_venvs" / legacy_component
        )

    source_venv = source_feature / ".venv"
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(source_venv)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(isolated_runtime._venv_python(source_venv)),
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    credential = source_feature / "data" / "session.sqlite3"
    credential.parent.mkdir(mode=0o700)
    credential.write_bytes(b"linked-device-custody")
    if manifest_kind == "old-path":
        (source_venv / ".kestrel_provision.json").write_text(
            json.dumps(
                {
                    "install_target": str(wheel),
                    "venv_path": str(source_venv.resolve()),
                }
            )
        )

    old_interpreter = str(isolated_runtime._venv_python(source_venv))
    feature._prepare_runtime_workspace()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    target_venv = feature._venv_path
    console = target_venv / "bin" / "kestrel-whatsapp-service"
    stale_console = console.read_text()
    assert old_interpreter in stale_console
    assert str(isolated_runtime._venv_python(target_venv)) not in stale_console
    assert not source_feature.exists()

    feature.ensure_venv()

    target_interpreter = str(isolated_runtime._venv_python(target_venv))
    repaired_console = console.read_text()
    assert target_interpreter in repaired_console
    assert old_interpreter not in repaired_console
    launched = subprocess.run(
        [str(console)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert launched.stdout.strip() == "console-path-ok"
    assert (feature._feature_runtime_dir() / "data" / "session.sqlite3").read_bytes() == (
        b"linked-device-custody"
    )
    manifest = json.loads((target_venv / ".kestrel_provision.json").read_text())
    assert manifest["venv_path"] == str(target_venv.resolve())
    assert stat.S_IMODE((target_venv / ".kestrel_provision.json").stat().st_mode) == 0o600

    restarted = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    restarted._prepare_runtime_workspace()
    restarted._venv_path, restarted._bin_path = restarted.resolve_runtime_paths()
    restarted._run = Mock(side_effect=AssertionError("same-path venv reinstalled"))
    restarted.ensure_venv()
    restarted._run.assert_not_called()


def test_unmoved_legacy_manifest_backfills_venv_path_without_index_access(
    monkeypatch,
    tmp_path,
):
    """A pre-upgrade location-less stamp is not evidence of relocation."""

    runtime = InstalledFeatureRuntime(
        class_name="WhatsAppFeature",
        entry_point="wa.feature:WhatsAppFeature",
        distribution="offline-feature",
        runtime="isolated-venv",
        service="offline-service",
        project="offline-feature",
    )
    feature = ProxyFeature(
        _hosted_postgres_agent(
            tmp_path / "runtime",
            "agent-unmoved-manifest",
            did="did:test:unmoved-manifest",
        ),
        runtime,
        client_factory=FakeIsolatedClient,
    )
    feature._prepare_runtime_workspace()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    python = isolated_runtime._venv_python(feature._venv_path)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o700)
    console = isolated_runtime._venv_bin_dir(feature._venv_path) / runtime.service
    console.write_text(f"#!{python.resolve()}\nprint('usable')\n")
    console.chmod(0o700)
    manifest_path = feature._provision_manifest_path()
    manifest_path.write_text(
        json.dumps(
            {
                "install_target": runtime.project,
                "provisioned_against_host_sdk": "1.2.3",
                "child_sdk_version": "1.2.3",
                "feature_distribution_version": "4.5.6",
                "child_feature_distribution_state": "versioned",
                "child_feature_distribution_version": "4.5.6",
            }
        )
    )
    feature._run = Mock(side_effect=OSError("index must remain offline"))
    feature._probe_feature_distribution = Mock(
        return_value=_child_distribution_probe("4.5.6")
    )
    monkeypatch.setattr(isolated_runtime, "_host_sdk_version", lambda: "1.2.3")
    monkeypatch.setattr(
        isolated_runtime,
        "_feature_distribution_version",
        lambda _distribution, _target: "4.5.6",
    )

    feature.ensure_venv()

    feature._run.assert_not_called()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["venv_path"] == str(feature._venv_path.resolve())
    assert manifest["child_feature_distribution_version"] == "4.5.6"
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600


@pytest.mark.parametrize("staleness", ("host-sdk", "target", "feature-version"))
def test_locationless_manifest_adoption_refuses_non_location_staleness(
    monkeypatch,
    tmp_path,
    staleness,
):
    """A missing path stamp cannot bless unrelated stale child contents."""

    runtime = InstalledFeatureRuntime(
        class_name="WhatsAppFeature",
        entry_point="wa.feature:WhatsAppFeature",
        distribution="offline-feature",
        runtime="isolated-venv",
        service="offline-service",
        project="offline-feature",
    )
    feature = ProxyFeature(
        _hosted_postgres_agent(
            tmp_path / "runtime",
            f"agent-locationless-{staleness}",
            did=f"did:test:locationless-{staleness}",
        ),
        runtime,
        client_factory=FakeIsolatedClient,
    )
    feature._prepare_runtime_workspace()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    python = isolated_runtime._venv_python(feature._venv_path)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o700)
    console = isolated_runtime._venv_bin_dir(feature._venv_path) / runtime.service
    console.write_text(f"#!{python}\nprint('usable')\n")
    console.chmod(0o700)
    manifest = {
        "install_target": runtime.project,
        "provisioned_against_host_sdk": "2.0.0",
        "child_sdk_version": "2.0.0",
        "feature_distribution_version": "4.5.6",
        "child_feature_distribution_state": "versioned",
        "child_feature_distribution_version": "4.5.6",
    }
    if staleness == "host-sdk":
        manifest["provisioned_against_host_sdk"] = "1.0.0"
    elif staleness == "target":
        manifest["install_target"] = "different-feature"
    else:
        manifest["feature_distribution_version"] = "4.5.5"
    manifest_path = feature._provision_manifest_path()
    manifest_path.write_text(json.dumps(manifest))
    commands = []
    feature._run = lambda command: commands.append(command)
    feature._probe_sdk_version = Mock(return_value="2.0.0")
    feature._probe_feature_distribution = Mock(
        return_value=_child_distribution_probe("4.5.6")
    )
    monkeypatch.setattr(isolated_runtime, "_host_sdk_version", lambda: "2.0.0")
    monkeypatch.setattr(
        isolated_runtime,
        "_feature_distribution_version",
        lambda _distribution, _target: "4.5.6",
    )

    feature.ensure_venv()

    installs = [command for command in commands if "install" in command]
    assert len(installs) == 1
    assert "--upgrade" in installs[0]
    assert "--reinstall" not in installs[0]
    refreshed = json.loads(manifest_path.read_text())
    assert refreshed["install_target"] == runtime.project
    assert refreshed["provisioned_against_host_sdk"] == "2.0.0"
    assert refreshed["feature_distribution_version"] == "4.5.6"


def test_migrated_callable_runtime_is_adopted_offline_without_reinstall(
    monkeypatch,
    tmp_path,
):
    """A moved callable venv has no console wrapper to rewrite."""

    runtime = InstalledFeatureRuntime(
        class_name="ModuleFeature",
        entry_point="module_feature.feature:ModuleFeature",
        distribution="module-feature",
        runtime="isolated-venv",
        service="module_feature.service:main",
        project="module-feature",
    )
    agent = _hosted_postgres_agent(
        tmp_path / "runtime",
        "agent-module-migration",
        did="did:test:module-migration",
    )
    feature = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    legacy_component = feature._legacy_runtime_directory_name
    assert legacy_component is not None
    isolated_runtime.prepare_isolated_runtime_namespace(
        feature._isolated_runtime_scope,
        agent.did,
        relative_directories=(("feature_venvs", legacy_component),),
    )
    source_venv = (
        feature._agent_runtime_dir
        / "feature_venvs"
        / legacy_component
        / ".venv"
    )
    _write_real_venv_module(
        source_venv,
        "module_feature.service",
        "def main():\n    return None\n",
    )
    (source_venv / ".kestrel_provision.json").write_text(
        json.dumps(
            {
                "install_target": runtime.project,
                "provisioned_against_host_sdk": "1.2.3",
                "child_sdk_version": "1.2.3",
                "feature_distribution_version": "7.8.9",
                "child_feature_distribution_state": "versioned",
                "child_feature_distribution_version": "7.8.9",
            }
        )
    )

    feature._prepare_runtime_workspace()
    assert feature._venv_relocated_this_startup is True
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    feature._run = Mock(side_effect=OSError("offline package index"))
    feature._probe_feature_distribution = Mock(
        return_value=_child_distribution_probe("7.8.9")
    )
    feature._probe_sdk_version = Mock(return_value="1.2.3")
    monkeypatch.setattr(isolated_runtime, "_host_sdk_version", lambda: "1.2.3")
    monkeypatch.setattr(
        isolated_runtime,
        "_feature_distribution_version",
        lambda _distribution, _target: "7.8.9",
    )

    feature.ensure_venv()

    feature._run.assert_not_called()
    assert feature._console_script_location_state() == "not-applicable"
    manifest = json.loads(feature._provision_manifest_path().read_text())
    assert manifest["venv_path"] == str(feature._venv_path.resolve())
    assert manifest["feature_distribution_version"] == "7.8.9"
    assert not (
        feature._feature_runtime_dir()
        / isolated_runtime._VENV_RELOCATION_REPAIR_MARKER
    ).exists()


@pytest.mark.parametrize(
    ("service", "installed_module", "source", "diagnostic_fragment"),
    (
        (
            "missing_callable_module:main",
            "different_module",
            "def main():\n    return None\n",
            "callable module is absent",
        ),
        (
            "callable_fixture:main",
            "callable_fixture",
            "def other():\n    return None\n",
            "callable attribute is absent",
        ),
        (
            "callable_fixture:main",
            "callable_fixture",
            "main = object()\n",
            "resolves to a non-callable object",
        ),
        (
            "callable_fixture:main",
            "callable_fixture",
            "import dependency_that_is_not_installed\n"
            "def main():\n    return None\n",
            "host could not complete isolated feature callable verification",
        ),
    ),
    ids=(
        "missing-module",
        "missing-attribute",
        "non-callable-attribute",
        "transitive-import-failure",
    ),
)
def test_callable_target_must_resolve_before_fresh_manifest_stamp(
    monkeypatch,
    tmp_path,
    service,
    installed_module,
    source,
    diagnostic_fragment,
):
    """A successful resolver cannot stamp an unlaunchable callable target."""

    runtime = InstalledFeatureRuntime(
        class_name="CallableFixtureFeature",
        entry_point="callable_fixture.feature:CallableFixtureFeature",
        distribution="callable-fixture",
        runtime="isolated-venv",
        service=service,
        project="callable-fixture",
    )
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    feature = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    _write_real_venv_module(feature._venv_path, installed_module, source)
    feature._run = Mock()
    feature._probe_sdk_version = Mock(return_value="1.2.3")
    feature._probe_feature_distribution = Mock(
        return_value=_child_distribution_probe("4.5.6")
    )
    monkeypatch.setattr(isolated_runtime, "_host_sdk_version", lambda: "1.2.3")
    monkeypatch.setattr(
        isolated_runtime,
        "_feature_distribution_version",
        lambda _distribution, _target: "4.5.6",
    )

    with pytest.raises(IsolatedRuntimePreparationError) as raised:
        feature.ensure_venv()

    diagnostic = isolated_runtime.safe_isolated_runtime_preparation_diagnostic(
        raised.value
    )
    assert diagnostic_fragment in diagnostic
    assert service not in str(raised.value)
    assert not feature._provision_manifest_path().exists()


def test_current_callable_manifest_reverifies_removed_target_without_reinstall(
    monkeypatch,
    tmp_path,
):
    """A matching fresh stamp cannot hide a callable removed after stamping."""

    runtime = InstalledFeatureRuntime(
        class_name="FreshCallableFeature",
        entry_point="fresh_callable.feature:FreshCallableFeature",
        distribution="fresh-callable",
        runtime="isolated-venv",
        service="fresh_callable:main",
        project="fresh-callable",
    )
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    feature = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    _write_real_venv_module(
        feature._venv_path,
        "fresh_callable",
        "def main():\n    return None\n",
    )
    child_distribution = _child_distribution_probe("4.5.6")
    feature._probe_sdk_version = Mock(return_value="1.2.3")
    feature._probe_feature_distribution = Mock(return_value=child_distribution)
    monkeypatch.setattr(isolated_runtime, "_host_sdk_version", lambda: "1.2.3")
    monkeypatch.setattr(
        isolated_runtime,
        "_feature_distribution_version",
        lambda _distribution, _target: "4.5.6",
    )
    feature._write_provision_manifest(
        runtime.project,
        "1.2.3",
        "1.2.3",
        "4.5.6",
        child_distribution,
    )
    manifest_before = feature._provision_manifest_path().read_bytes()
    module_path = next(feature._venv_path.rglob("fresh_callable.py"))
    module_path.unlink()
    feature._run = Mock(side_effect=AssertionError("fresh venv reinstalled"))

    with pytest.raises(IsolatedRuntimePreparationError) as raised:
        feature.ensure_venv()

    diagnostic = isolated_runtime.safe_isolated_runtime_preparation_diagnostic(
        raised.value
    )
    assert "callable module is absent" in diagnostic
    assert feature._provision_manifest_path().read_bytes() == manifest_before
    feature._run.assert_not_called()


def test_hosted_prebuilt_callable_target_failure_is_actionable_and_read_only(
    tmp_path,
):
    """An immutable venv stays untouched while callable failure is quarantinable."""

    prebuilt = tmp_path / "operator-callable-venv"
    _write_real_venv_module(
        prebuilt,
        "callable_fixture",
        "def other():\n    return None\n",
    )
    before = {
        path.relative_to(prebuilt): path.read_bytes()
        for path in prebuilt.rglob("*")
        if path.is_file()
    }
    runtime = InstalledFeatureRuntime(
        class_name="CallableFixtureFeature",
        entry_point="callable_fixture.feature:CallableFixtureFeature",
        distribution="callable-fixture",
        runtime="isolated-venv",
        service="callable_fixture:main",
        project="callable-fixture",
        venv=str(prebuilt),
    )
    feature = ProxyFeature(
        _hosted_postgres_agent(tmp_path / "runtime", "agent-prebuilt-callable"),
        runtime,
        client_factory=FakeIsolatedClient,
    )
    feature._prepare_runtime_workspace()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    feature._verify_prebuilt_feature_distribution = Mock()
    feature._warn_on_sdk_mismatch = Mock()
    feature._run = Mock(side_effect=AssertionError("prebuilt venv was mutated"))

    with pytest.raises(IsolatedRuntimePreparationError) as raised:
        feature.ensure_venv()

    assert "callable attribute is absent" in (
        isolated_runtime.safe_isolated_runtime_preparation_diagnostic(raised.value)
    )
    assert feature._run.call_count == 0
    assert not feature._provision_manifest_path().exists()
    after = {
        path.relative_to(prebuilt): path.read_bytes()
        for path in prebuilt.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_hosted_prebuilt_missing_console_names_selecting_venv_setting(
    monkeypatch,
    tmp_path,
):
    prebuilt = tmp_path / "operator-console-venv"
    _write_prebuilt_venv_shape(prebuilt)
    key = "KESTREL_FEATURE_WHATSAPPFEATURE_VENV"
    monkeypatch.setenv(key, str(prebuilt))
    feature = ProxyFeature(
        _hosted_postgres_agent(tmp_path / "runtime", "missing-console"),
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )
    feature._prepare_runtime_workspace()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    feature._run = Mock(side_effect=AssertionError("prebuilt venv was mutated"))

    with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
        feature.ensure_venv()

    diagnostic = raised.value.safe_diagnostic()
    assert key in diagnostic
    assert str(prebuilt) not in diagnostic
    assert "filesystem health" not in diagnostic
    feature._run.assert_not_called()


@pytest.mark.skipif(os.name != "posix", reason="POSIX console custody contract")
@pytest.mark.parametrize("case", ("group-writable", "non-executable", "escape"))
def test_hosted_prebuilt_console_rejects_unsafe_or_escaping_target(
    monkeypatch,
    tmp_path,
    case,
):
    prebuilt = tmp_path / f"operator-console-{case}"
    _write_prebuilt_venv_shape(
        prebuilt,
        console_service=_isolated_runtime().service,
    )
    wrapper = isolated_runtime._console_script_path(
        prebuilt,
        _isolated_runtime().service,
    )
    outside_target = None
    if case == "group-writable":
        wrapper.chmod(0o720)
    elif case == "non-executable":
        wrapper.chmod(0o600)
    else:
        outside_parent = tmp_path / "world-writable-shared-bin"
        outside_parent.mkdir(mode=0o700)
        outside_parent.chmod(0o777)
        outside_target = outside_parent / "secure-looking-service"
        outside_target.write_bytes(wrapper.read_bytes())
        outside_target.chmod(0o700)
        wrapper.unlink()
        wrapper.symlink_to(outside_target)
    key = "KESTREL_FEATURE_WHATSAPPFEATURE_VENV"
    monkeypatch.setenv(key, str(prebuilt))
    client_factory = Mock(side_effect=AssertionError("unsafe child was launched"))
    feature = ProxyFeature(
        _hosted_postgres_agent(tmp_path / "runtime", f"console-{case}"),
        _isolated_runtime(),
        client_factory=client_factory,
    )
    feature._prepare_runtime_workspace()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    feature._verify_launch_artifact = Mock(
        side_effect=AssertionError("unsafe console reached artifact probe")
    )
    feature._verify_prebuilt_feature_distribution = Mock(
        side_effect=AssertionError("unsafe console reached distribution probe")
    )
    feature._warn_on_sdk_mismatch = Mock(
        side_effect=AssertionError("unsafe console reached SDK probe")
    )
    feature._run = Mock(side_effect=AssertionError("unsafe console was provisioned"))

    with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
        feature.ensure_venv()

    diagnostic = raised.value.safe_diagnostic()
    assert key in diagnostic
    assert str(prebuilt) not in diagnostic
    assert "world-writable-shared-bin" not in diagnostic
    feature._verify_launch_artifact.assert_not_called()
    feature._verify_prebuilt_feature_distribution.assert_not_called()
    feature._warn_on_sdk_mismatch.assert_not_called()
    feature._run.assert_not_called()
    client_factory.assert_not_called()
    assert feature._validated_hosted_console_path is None
    if outside_target is not None:
        assert outside_target.read_bytes().startswith(b"#!")


@pytest.mark.skipif(os.name != "posix", reason="POSIX console custody contract")
def test_hosted_prebuilt_console_rejects_foreign_owned_target(
    monkeypatch,
    tmp_path,
):
    prebuilt = tmp_path / "operator-console-foreign"
    _write_prebuilt_venv_shape(
        prebuilt,
        console_service=_isolated_runtime().service,
    )
    wrapper = isolated_runtime._console_script_path(
        prebuilt,
        _isolated_runtime().service,
    ).resolve()
    key = "KESTREL_FEATURE_WHATSAPPFEATURE_VENV"
    monkeypatch.setenv(key, str(prebuilt))
    feature = ProxyFeature(
        _hosted_postgres_agent(tmp_path / "runtime", "console-foreign"),
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )
    feature._prepare_runtime_workspace()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    real_stat = Path.stat
    foreign_uid = next(uid for uid in range(1, 4) if uid != os.geteuid())

    def foreign_console_stat(path, *, follow_symlinks=True):
        metadata = real_stat(path, follow_symlinks=follow_symlinks)
        if path == wrapper:
            return _stat_result_with_uid(metadata, foreign_uid)
        return metadata

    monkeypatch.setattr(Path, "stat", foreign_console_stat)
    feature._verify_prebuilt_feature_distribution = Mock(
        side_effect=AssertionError("foreign console reached distribution probe")
    )

    with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
        feature.ensure_venv()

    assert key in raised.value.safe_diagnostic()
    assert str(prebuilt) not in raised.value.safe_diagnostic()
    feature._verify_prebuilt_feature_distribution.assert_not_called()
    assert feature._validated_hosted_console_path is None


@pytest.mark.skipif(os.name != "posix", reason="POSIX console symlink contract")
def test_hosted_prebuilt_console_symlink_launches_validated_pinned_target(
    monkeypatch,
    tmp_path,
):
    prebuilt = tmp_path / "operator-console-symlink"
    _write_prebuilt_venv_shape(
        prebuilt,
        console_service=_isolated_runtime().service,
    )
    public_wrapper = isolated_runtime._console_script_path(
        prebuilt,
        _isolated_runtime().service,
    )
    first_target = public_wrapper.with_name(f"{public_wrapper.name}-v1")
    second_target = public_wrapper.with_name(f"{public_wrapper.name}-v2")
    public_wrapper.rename(first_target)
    second_target.write_bytes(first_target.read_bytes())
    second_target.chmod(0o700)
    public_wrapper.symlink_to(first_target.name)
    key = "KESTREL_FEATURE_WHATSAPPFEATURE_VENV"
    monkeypatch.setenv(key, str(prebuilt))
    feature = ProxyFeature(
        _hosted_postgres_agent(tmp_path / "runtime", "console-symlink"),
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )
    feature._prepare_runtime_workspace()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    feature._verify_prebuilt_feature_distribution = Mock()
    feature._warn_on_sdk_mismatch = Mock()

    feature.ensure_venv()

    assert feature._validated_hosted_console_path == first_target.resolve()
    public_wrapper.unlink()
    public_wrapper.symlink_to(second_target.name)
    assert feature._service_command() == [str(first_target.resolve())]
    feature._verify_prebuilt_feature_distribution.assert_called_once()


def _callable_verification_feature(tmp_path: Path) -> ProxyFeature:
    runtime = InstalledFeatureRuntime(
        class_name="CallableVerificationFeature",
        entry_point="callable_verification.feature:CallableVerificationFeature",
        distribution="callable-verification",
        runtime="isolated-venv",
        service="callable_verification:main",
    )
    agent = Mock(did=_TEST_AGENT_DID)
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    feature = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    return feature


def test_callable_target_verification_timeout_is_bounded_infrastructure_failure(
    monkeypatch,
    tmp_path,
):
    feature = _callable_verification_feature(tmp_path)
    captured = {}

    def timeout(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(isolated_runtime.subprocess, "run", timeout)

    with pytest.raises(IsolatedRuntimePreparationError) as raised:
        feature._verify_launch_artifact()

    diagnostic = isolated_runtime.safe_isolated_runtime_preparation_diagnostic(
        raised.value
    )
    assert captured["timeout"] == (
        isolated_runtime._CALLABLE_TARGET_VERIFICATION_TIMEOUT_S
    )
    assert 0 < captured["timeout"] <= 30
    assert captured["stdout"] == subprocess.DEVNULL
    assert captured["stderr"] == subprocess.DEVNULL
    assert "bounded startup timeout" in diagnostic
    assert "absent" not in diagnostic
    assert "callable_verification" not in diagnostic


def test_callable_target_verification_spawn_failure_preserves_host_diagnostic(
    monkeypatch,
    tmp_path,
):
    feature = _callable_verification_feature(tmp_path)

    def fail_spawn(*_args, **_kwargs):
        raise OSError(errno.EMFILE, "/private/tenant/secret-python")

    monkeypatch.setattr(isolated_runtime.subprocess, "run", fail_spawn)

    with pytest.raises(IsolatedRuntimePreparationError) as raised:
        feature._verify_launch_artifact()

    diagnostic = isolated_runtime.safe_isolated_runtime_preparation_diagnostic(
        raised.value
    )
    assert "file-descriptor capacity" in diagnostic
    assert "absent" not in diagnostic
    assert "secret-python" not in diagnostic


def test_callable_target_requires_supported_safe_path_interpreter(
    monkeypatch,
    tmp_path,
):
    feature = _callable_verification_feature(tmp_path)
    monkeypatch.setattr(
        isolated_runtime.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 2),
    )

    with pytest.raises(IsolatedRuntimeConfigurationError) as raised:
        feature._verify_launch_artifact()

    assert raised.value.safe_diagnostic() == (
        "isolated Python callable services require a feature interpreter that "
        "supports the SDK's Python 3.11 safe-path contract"
    )


@pytest.mark.skipif(os.name != "posix", reason="console shebangs are POSIX paths")
@pytest.mark.parametrize("migration_kind", ("released", "pre-stable"))
def test_relocation_repair_intent_survives_failed_boot_and_forces_restart_repair(
    tmp_path,
    migration_kind,
):
    """A failed first repair cannot lose the evidence required by boot two."""

    wheel = _write_console_script_fixture_wheel(tmp_path)
    runtime = InstalledFeatureRuntime(
        class_name="WhatsAppFeature",
        entry_point="wa.feature:WhatsAppFeature",
        distribution="kestrel-console-migration-fixture",
        runtime="isolated-venv",
        service="kestrel-whatsapp-service",
        project=str(wheel),
    )
    runtime_root = tmp_path / "hosted-runtime"
    if migration_kind == "released":
        agent_dir = tmp_path / "agent_data" / "Hosted"
        legacy_root = agent_dir / "feature_venvs"
        source_feature = legacy_root / "WhatsAppFeature"
        source_feature.mkdir(parents=True, mode=0o700)
        legacy_root.chmod(0o700)
        agent = KestrelAgent(
            did="did:test:repair-restart-released",
            storage_path=str(agent_dir / "kestrel_prime.db"),
            llm_service=Mock(providers=[]),
            database_url="postgresql://hosted.example/kestrel",
            db_backend="postgres",
            isolated_runtime_root=runtime_root,
            isolated_runtime_namespace="agent-repair-restart-released",
            isolated_runtime_legacy_root=legacy_root,
            isolated_runtime_hosted=True,
        )
        first = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    else:
        agent = _hosted_postgres_agent(
            runtime_root,
            "agent-repair-restart-pre-stable",
            did="did:test:repair-restart-pre-stable",
        )
        first = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
        legacy_component = first._legacy_runtime_directory_name
        assert legacy_component is not None
        isolated_runtime.prepare_isolated_runtime_namespace(
            first._isolated_runtime_scope,
            agent.did,
            relative_directories=(("feature_venvs", legacy_component),),
        )
        source_feature = (
            first._agent_runtime_dir / "feature_venvs" / legacy_component
        )

    source_venv = source_feature / ".venv"
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(source_venv)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(isolated_runtime._venv_python(source_venv)),
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    old_python = str(isolated_runtime._venv_python(source_venv))
    first._prepare_runtime_workspace()
    first._venv_path, first._bin_path = first.resolve_runtime_paths()
    marker = (
        first._feature_runtime_dir()
        / isolated_runtime._VENV_RELOCATION_REPAIR_MARKER
    )
    assert marker.read_bytes() == isolated_runtime._VENV_RELOCATION_REPAIR_PAYLOAD
    failed_commands = []

    def fail_install(command):
        failed_commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    first._run = fail_install
    with pytest.raises(IsolatedRuntimePreparationError):
        first.ensure_venv()
    assert "--reinstall" in failed_commands[-1]
    assert marker.is_file()
    assert old_python in (
        first._venv_path / "bin" / "kestrel-whatsapp-service"
    ).read_text()

    unverified = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    unverified._prepare_runtime_workspace()
    unverified._venv_path, unverified._bin_path = unverified.resolve_runtime_paths()
    unverified._run = Mock(return_value=None)
    with pytest.raises(
        IsolatedRuntimePreparationError,
        match="launch artifact could not be verified",
    ):
        unverified.ensure_venv()
    unverified._run.assert_called_once()
    assert "--reinstall" in unverified._run.call_args.args[0]
    assert marker.is_file()
    assert not unverified._provision_manifest_path().exists()

    restarted = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    restarted._prepare_runtime_workspace()
    assert restarted._venv_relocated_this_startup is False
    restarted._venv_path, restarted._bin_path = restarted.resolve_runtime_paths()
    restarted.ensure_venv()

    console = restarted._venv_path / "bin" / "kestrel-whatsapp-service"
    assert str(isolated_runtime._venv_python(restarted._venv_path)) in console.read_text()
    assert old_python not in console.read_text()
    assert subprocess.run(
        [str(console)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == "console-path-ok"
    assert not marker.exists()

    final_restart = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    final_restart._prepare_runtime_workspace()
    final_restart._venv_path, final_restart._bin_path = (
        final_restart.resolve_runtime_paths()
    )
    final_restart._run = Mock(side_effect=AssertionError("repair repeated"))
    final_restart.ensure_venv()
    final_restart._run.assert_not_called()


@pytest.mark.skipif(os.name != "posix", reason="console shebangs are POSIX paths")
@pytest.mark.asyncio
async def test_relocated_console_index_failure_quarantines_before_child_start(
    tmp_path,
):
    """An offline repair never launches a wrapper bound to the old venv path."""

    wheel = _write_console_script_fixture_wheel(tmp_path)
    runtime = InstalledFeatureRuntime(
        class_name="WhatsAppFeature",
        entry_point="wa.feature:WhatsAppFeature",
        distribution="kestrel-console-migration-fixture",
        runtime="isolated-venv",
        service="kestrel-whatsapp-service",
        project=str(wheel),
    )
    agent = _hosted_postgres_agent(
        tmp_path / "runtime",
        "agent-offline-console-migration",
        did="did:test:offline-console-migration",
    )
    factory = Mock(side_effect=AssertionError("stale child must not be built"))
    feature = ProxyFeature(agent, runtime, client_factory=factory)
    legacy_component = feature._legacy_runtime_directory_name
    assert legacy_component is not None
    isolated_runtime.prepare_isolated_runtime_namespace(
        feature._isolated_runtime_scope,
        agent.did,
        relative_directories=(("feature_venvs", legacy_component),),
    )
    source_venv = (
        feature._agent_runtime_dir
        / "feature_venvs"
        / legacy_component
        / ".venv"
    )
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(source_venv)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(isolated_runtime._venv_python(source_venv)),
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    old_python = str(isolated_runtime._venv_python(source_venv))
    feature._run = Mock(
        side_effect=subprocess.CalledProcessError(1, ["uv", "pip", "install"])
    )

    with pytest.raises(IsolatedRuntimePreparationError):
        await feature.initialize()

    target_console = (
        feature._default_venv_path() / "bin" / "kestrel-whatsapp-service"
    )
    assert old_python in target_console.read_text()
    assert not feature._provision_manifest_path().exists()
    factory.assert_not_called()
    install = feature._run.call_args.args[0]
    assert "--reinstall" in install


def test_relocated_venv_failed_repair_retains_stale_stamp_for_retry(
    monkeypatch,
    tmp_path,
):
    agent = _hosted_postgres_agent(
        tmp_path / "runtime",
        "agent-repair-retry",
        did="did:test:repair-retry",
    )
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)
    feature._prepare_runtime_workspace()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    python = isolated_runtime._venv_python(feature._venv_path)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o700)
    stale_manifest = feature._venv_path / ".kestrel_provision.json"
    stale_payload = json.dumps(
        {
            "install_target": feature.runtime.distribution,
            "venv_path": "/old/private/tenant/.venv",
        }
    )
    stale_manifest.write_text(stale_payload)
    feature._probe_sdk_version = Mock(return_value="unknown")
    feature._probe_feature_distribution = Mock(
        return_value=isolated_runtime._FeatureDistributionProbe.present_unversioned()
    )
    monkeypatch.setattr(
        isolated_runtime,
        "_feature_distribution_version",
        lambda _distribution, _target: "unknown",
    )
    failed_runs = []

    def fail_repair(command):
        failed_runs.append(command)
        raise OSError(errno.ENOSPC, "private tenant path")

    feature._run = fail_repair
    with pytest.raises(IsolatedRuntimePreparationError):
        feature.ensure_venv()

    assert "--reinstall" in failed_runs[-1]
    assert stale_manifest.read_text() == stale_payload

    successful_runs = []

    def complete_repair(command):
        successful_runs.append(command)
        console = (
            isolated_runtime._venv_bin_dir(feature._venv_path)
            / feature.runtime.service
        )
        console.write_text(
            f"#!{isolated_runtime._venv_python(feature._venv_path)}\n"
            "print('repaired')\n"
        )
        console.chmod(0o700)

    feature._run = complete_repair
    with monkeypatch.context() as manifest_failure:
        manifest_failure.setattr(
            isolated_runtime.os,
            "replace",
            Mock(side_effect=OSError(errno.ENOSPC, "private manifest path")),
        )
        with pytest.raises(IsolatedRuntimePreparationError):
            feature.ensure_venv()
    assert "--reinstall" in successful_runs[-1]
    assert stale_manifest.read_text() == stale_payload
    assert list(feature._venv_path.glob(".kestrel_provision.json.tmp-*")) == []

    successful_runs.clear()
    feature.ensure_venv()
    # The old-path stamp remains durable evidence of relocation after the
    # prior manifest publish failed. A restart must keep forcing console-script
    # repair until the new canonical stamp is atomically durable.
    assert "--reinstall" in successful_runs[-1]
    assert "--upgrade" not in successful_runs[-1]
    repaired = json.loads(stale_manifest.read_text())
    assert repaired["venv_path"] == str(feature._venv_path.resolve())

    feature._run = Mock(side_effect=AssertionError("repair repeated after stamp"))
    feature.ensure_venv()
    feature._run.assert_not_called()


@pytest.mark.skipif(os.name != "posix", reason="POSIX custody mode contract")
@pytest.mark.asyncio
async def test_hosted_released_empty_permissive_root_is_a_noop(
    monkeypatch,
    tmp_path,
):
    """An already-migrated 0775 release root must not wedge every restart."""

    agent_dir = tmp_path / "agent_data" / "Hosted"
    legacy_root = agent_dir / "feature_venvs"
    legacy_root.mkdir(parents=True, mode=0o775)
    legacy_root.chmod(0o775)
    agent = KestrelAgent(
        did="did:test:released-empty-root",
        storage_path=str(agent_dir / "kestrel_prime.db"),
        llm_service=Mock(providers=[]),
        database_url="postgresql://hosted.example/kestrel",
        db_backend="postgres",
        isolated_runtime_root=tmp_path / "isolated_feature_runtime",
        isolated_runtime_namespace="agent-empty-root",
        isolated_runtime_legacy_root=legacy_root,
        isolated_runtime_hosted=True,
    )
    executable = tmp_path / "operator" / "whatsapp-service"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_BIN", str(executable))
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)

    try:
        await feature.initialize()
        runtime_dir = feature._feature_runtime_dir()

        assert feature._client is not None
        assert runtime_dir.is_dir()
        assert stat.S_IMODE(legacy_root.stat().st_mode) == 0o775
        assert not (legacy_root / "WhatsAppFeature").exists()
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX custody mode contract")
async def test_hosted_released_populated_permissive_root_quarantines_feature(
    tmp_path,
):
    """Unsafe released custody is optional-feature failure, never a move."""

    agent_dir = tmp_path / "agent_data" / "Hosted"
    legacy_root = agent_dir / "feature_venvs"
    credential = (
        legacy_root / "WhatsAppFeature" / "whatsapp_service" / "session.sqlite3"
    )
    credential.parent.mkdir(parents=True, mode=0o700)
    credential.write_bytes(b"permissive-root-credential")
    legacy_root.chmod(0o775)
    agent = KestrelAgent(
        did="did:test:released-populated-root",
        storage_path=str(agent_dir / "kestrel_prime.db"),
        llm_service=Mock(providers=[]),
        database_url="postgresql://hosted.example/kestrel",
        db_backend="postgres",
        isolated_runtime_root=tmp_path / "isolated_feature_runtime",
        isolated_runtime_namespace="agent-populated-root",
        isolated_runtime_legacy_root=legacy_root,
        isolated_runtime_hosted=True,
    )
    client_started = False

    def client_factory(**kwargs):
        nonlocal client_started
        client_started = True
        return FakeIsolatedClient(**kwargs)

    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=client_factory)

    available = await agent._register_startup_feature(
        feature,
        prepared_contributions=Mock(),
    )

    assert available is False
    assert client_started is False
    assert feature.name not in agent.features
    assert credential.read_bytes() == b"permissive-root-credential"
    assert not feature._feature_runtime_dir().exists()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX custody mode contract")
async def test_hosted_released_writable_feature_child_quarantines_only_optional(
    tmp_path,
):
    """Ambiguous child custody retains state without aborting mandatory boot."""

    agent_dir = tmp_path / "agent_data" / "Hosted"
    legacy_root = agent_dir / "feature_venvs"
    legacy_feature = legacy_root / "WhatsAppFeature"
    credential = legacy_feature / "whatsapp_service" / "session.sqlite3"
    credential.parent.mkdir(parents=True, mode=0o700)
    credential.write_bytes(b"writable-child-credential")
    legacy_root.chmod(0o755)
    legacy_feature.chmod(0o775)
    agent = KestrelAgent(
        did="did:test:released-writable-child",
        storage_path=str(agent_dir / "kestrel_prime.db"),
        llm_service=Mock(providers=[]),
        database_url="postgresql://hosted.example/kestrel",
        db_backend="postgres",
        isolated_runtime_root=tmp_path / "isolated_feature_runtime",
        isolated_runtime_namespace="agent-writable-child",
        isolated_runtime_legacy_root=legacy_root,
        isolated_runtime_hosted=True,
    )
    mandatory = Mock(name="mandatory-feature")
    agent.features["IdentityFeature"] = mandatory
    client_factory = Mock(side_effect=AssertionError("child must not start"))
    feature = ProxyFeature(
        agent,
        _isolated_runtime(),
        client_factory=client_factory,
    )

    available = await agent._register_startup_feature(
        feature,
        prepared_contributions=Mock(),
    )

    assert available is False
    assert agent.features == {"IdentityFeature": mandatory}
    assert feature._client is None
    client_factory.assert_not_called()
    assert credential.read_bytes() == b"writable-child-credential"
    assert legacy_feature.is_dir()
    assert not feature._feature_runtime_dir().exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_errno", "safe_phrase"),
    (
        (errno.EXDEV, "cannot be adopted across filesystems"),
        (errno.ENOSPC, "insufficient free space or quota"),
        (errno.EACCES, "configured mount ownership and write policy"),
    ),
)
async def test_optional_preparation_log_is_actionable_with_sanitized_traceback(
    caplog,
    failure_errno,
    tmp_path,
    safe_phrase,
):
    agent = _hosted_postgres_agent(
        tmp_path / "runtime",
        "agent-sanitized-preparation",
        did="did:test:sanitized-preparation",
    )
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)
    secret_path = tmp_path / "tenant-secret" / "credentials.sqlite3"
    secret_value = "private-feature-cause-value"

    async def fail_initialize():
        try:
            raise OSError(failure_errno, secret_value, secret_path)
        except OSError as cause:
            raise IsolatedRuntimePreparationError(
                f"third-party text {secret_value} at {secret_path}"
            ) from cause

    feature.initialize = fail_initialize

    with caplog.at_level("ERROR"):
        available = await agent._register_startup_feature(
            feature,
            prepared_contributions=Mock(),
        )

    assert available is False
    assert safe_phrase in caplog.text
    assert "other agent features will continue" in caplog.text
    assert secret_value not in caplog.text
    assert str(secret_path) not in caplog.text
    records = [record for record in caplog.records if record.exc_info is not None]
    assert len(records) == 1
    assert isinstance(records[0].exc_info[1], IsolatedRuntimePreparationError)
    assert records[0].exc_info[1].__cause__ is None


def test_preparation_traceback_filter_rejects_forged_core_module_name(tmp_path):
    """A dependency cannot retain its source frame by forging ``__name__``."""

    secret = "dependency-source-secret"
    dependency_path = tmp_path / "forged_dependency.py"
    source = (
        "def fail():\n"
        f"    raise IsolatedRuntimePreparationError({secret!r})\n"
    )
    dependency_path.write_text(source)
    namespace = {
        "__name__": "kestrel_sovereign.forged_dependency",
        "IsolatedRuntimePreparationError": IsolatedRuntimePreparationError,
    }
    exec(compile(source, str(dependency_path), "exec"), namespace)
    try:
        namespace["fail"]()
    except IsolatedRuntimePreparationError as error:
        exc_info = (
            isolated_runtime.sanitized_isolated_runtime_preparation_exc_info(
                error
            )
        )
    else:  # pragma: no cover - mutation guard
        pytest.fail("forged dependency did not raise")

    rendered = "".join(traceback.format_exception(*exc_info))
    assert "agent-scoped runtime could not be prepared" in rendered
    assert secret not in rendered
    assert str(dependency_path) not in rendered
    assert "raise IsolatedRuntimePreparationError" not in rendered


@pytest.mark.skipif(os.name != "posix", reason="POSIX custody mode contract")
@pytest.mark.parametrize("populated", (False, True))
def test_portable_released_migration_checks_component_before_root_mode(
    tmp_path,
    populated,
):
    """Mutation guard for the non-dirfd implementation's validation order."""

    legacy_root = tmp_path / "released_feature_venvs"
    legacy_root.mkdir(mode=0o775)
    legacy_root.chmod(0o775)
    if populated:
        (legacy_root / "WhatsAppFeature").mkdir(mode=0o700)
    scope = resolve_isolated_runtime_namespace(
        tmp_path / "runtime",
        "agent-portable-released",
    )
    isolated_runtime.prepare_isolated_runtime_namespace(
        scope,
        "did:test:portable-released",
        relative_directories=(("feature_venvs",),),
    )

    if populated:
        with pytest.raises(IsolatedRuntimePreparationError, match="custody"):
            isolated_runtime._migrate_released_runtime_directory_portable(
                legacy_root,
                scope,
                "WhatsAppFeature",
                "feature-stable",
            )
        assert (legacy_root / "WhatsAppFeature").is_dir()
        assert not (scope.path / "feature_venvs" / "feature-stable").exists()
    else:
        isolated_runtime._migrate_released_runtime_directory_portable(
            legacy_root,
            scope,
            "WhatsAppFeature",
            "feature-stable",
        )
        assert stat.S_IMODE(legacy_root.stat().st_mode) == 0o775


def test_portable_released_migration_detects_root_symlink_substitution(
    monkeypatch,
    tmp_path,
):
    legacy_root = tmp_path / "released_feature_venvs"
    legacy_root.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside_credential = (
        outside / "WhatsAppFeature" / "whatsapp_service" / "session.sqlite3"
    )
    outside_credential.parent.mkdir(parents=True)
    outside_credential.write_bytes(b"foreign-tenant-credential")
    moved_root = tmp_path / "original-released-root"
    scope = resolve_isolated_runtime_namespace(
        tmp_path / "runtime",
        "agent-portable-race",
    )
    isolated_runtime.prepare_isolated_runtime_namespace(
        scope,
        "did:test:portable-race",
        relative_directories=(("feature_venvs",),),
    )
    real_stat = Path.stat
    swapped = False

    def substitute_root_during_component_stat(path, *args, **kwargs):
        nonlocal swapped
        if path == legacy_root / "WhatsAppFeature" and not swapped:
            legacy_root.rename(moved_root)
            legacy_root.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", substitute_root_during_component_stat)

    with pytest.raises(IsolatedRuntimeNamespaceError, match="root changed"):
        isolated_runtime._migrate_released_runtime_directory_portable(
            legacy_root,
            scope,
            "WhatsAppFeature",
            "feature-stable",
        )

    assert swapped is True
    assert outside_credential.read_bytes() == b"foreign-tenant-credential"
    assert (moved_root).is_dir()
    assert not (scope.path / "feature_venvs" / "feature-stable").exists()


@pytest.mark.skipif(os.name != "posix", reason="migration uses secure POSIX dirfds")
def test_hosted_released_runtime_symlink_is_rejected_without_touching_target(tmp_path):
    agent_dir = tmp_path / "agent_data" / "Hosted"
    legacy_root = agent_dir / "feature_venvs"
    legacy_root.mkdir(parents=True, mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "session.sqlite3"
    secret.write_bytes(b"other-tenant-secret")
    (legacy_root / "WhatsAppFeature").symlink_to(outside, target_is_directory=True)
    agent = KestrelAgent(
        did="did:test:released-hosted-symlink",
        storage_path=str(agent_dir / "kestrel_prime.db"),
        llm_service=Mock(providers=[]),
        database_url="postgresql://hosted.example/kestrel",
        db_backend="postgres",
        isolated_runtime_root=tmp_path / "isolated_feature_runtime",
        isolated_runtime_namespace="agent-symlink",
        isolated_runtime_legacy_root=legacy_root,
        isolated_runtime_hosted=True,
    )
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)

    with pytest.raises(IsolatedRuntimeNamespaceError, match="unsafe"):
        feature._prepare_runtime_workspace()

    assert secret.read_bytes() == b"other-tenant-secret"
    assert not feature._feature_runtime_dir().exists()


@pytest.mark.skipif(os.name != "posix", reason="migration uses secure POSIX dirfds")
@pytest.mark.asyncio
async def test_hosted_released_runtime_collision_retains_both_credential_trees(
    caplog,
    tmp_path,
):
    agent_dir = tmp_path / "agent_data" / "Hosted"
    legacy_root = agent_dir / "feature_venvs"
    legacy_credential = (
        legacy_root / "WhatsAppFeature" / "whatsapp_service" / "session.sqlite3"
    )
    legacy_credential.parent.mkdir(parents=True, mode=0o700)
    legacy_root.chmod(0o700)
    (legacy_root / "WhatsAppFeature").chmod(0o700)
    legacy_credential.write_bytes(b"released-custody")
    agent = KestrelAgent(
        did="did:test:released-hosted-collision",
        storage_path=str(agent_dir / "kestrel_prime.db"),
        llm_service=Mock(providers=[]),
        database_url="postgresql://hosted.example/kestrel",
        db_backend="postgres",
        isolated_runtime_root=tmp_path / "isolated_feature_runtime",
        isolated_runtime_namespace="agent-collision",
        isolated_runtime_legacy_root=legacy_root,
        isolated_runtime_hosted=True,
    )
    client_started = False

    def client_factory(**kwargs):
        nonlocal client_started
        client_started = True
        return FakeIsolatedClient(**kwargs)

    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=client_factory)
    isolated_runtime.prepare_isolated_runtime_namespace(
        feature._isolated_runtime_scope,
        agent.did,
        relative_directories=(
            (
                "feature_venvs",
                feature._runtime_directory_name,
                "whatsapp_service",
            ),
        ),
    )
    stable_credential = (
        feature._feature_runtime_dir() / "whatsapp_service" / "session.sqlite3"
    )
    stable_credential.write_bytes(b"stable-custody")

    with caplog.at_level("ERROR"):
        available = await agent._register_startup_feature(
            feature,
            prepared_contributions=Mock(),
        )

    assert available is False
    assert feature.name not in agent.features
    assert client_started is False
    assert feature._client is None
    assert legacy_credential.read_bytes() == b"released-custody"
    assert stable_credential.read_bytes() == b"stable-custody"
    assert "released_feature_venvs/WhatsAppFeature" in caplog.text
    assert f"feature_venvs/{feature._runtime_directory_name}" in caplog.text
    assert str(agent_dir) not in caplog.text
    assert "released-custody" not in caplog.text


@pytest.mark.skipif(os.name != "posix", reason="migration uses secure POSIX dirfds")
@pytest.mark.parametrize("collision_errno", (errno.EEXIST, errno.ENOTEMPTY))
def test_hosted_released_runtime_rename_race_never_overwrites_new_custody(
    caplog,
    collision_errno,
    monkeypatch,
    tmp_path,
):
    agent_dir = tmp_path / "agent_data" / "Hosted"
    legacy_root = agent_dir / "feature_venvs"
    legacy_credential = (
        legacy_root / "WhatsAppFeature" / "whatsapp_service" / "session.sqlite3"
    )
    legacy_credential.parent.mkdir(parents=True, mode=0o700)
    legacy_root.chmod(0o700)
    (legacy_root / "WhatsAppFeature").chmod(0o700)
    legacy_credential.write_bytes(b"released-race-custody")
    agent = KestrelAgent(
        did="did:test:released-hosted-race",
        storage_path=str(agent_dir / "kestrel_prime.db"),
        llm_service=Mock(providers=[]),
        database_url="postgresql://hosted.example/kestrel",
        db_backend="postgres",
        isolated_runtime_root=tmp_path / "isolated_feature_runtime",
        isolated_runtime_namespace="agent-race",
        isolated_runtime_legacy_root=legacy_root,
        isolated_runtime_hosted=True,
    )
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)
    stable_credential = (
        feature._feature_runtime_dir() / "whatsapp_service" / "session.sqlite3"
    )
    real_rename = isolated_runtime._rename_directory_noreplace_at
    rename_calls = 0

    def collide(source_fd, source, target_fd, target):
        nonlocal rename_calls
        if source != "WhatsAppFeature":
            return real_rename(source_fd, source, target_fd, target)
        rename_calls += 1
        stable_credential.parent.mkdir(parents=True)
        stable_credential.write_bytes(b"stable-race-custody")
        raise OSError(collision_errno, "synthetic released-layout collision")

    monkeypatch.setattr(
        isolated_runtime,
        "_rename_directory_noreplace_at",
        collide,
    )

    with caplog.at_level("ERROR"), pytest.raises(
        IsolatedRuntimePreparationError,
        match="custody reconciliation",
    ):
        feature._prepare_runtime_workspace()

    assert rename_calls == 1
    assert legacy_credential.read_bytes() == b"released-race-custody"
    assert stable_credential.read_bytes() == b"stable-race-custody"
    assert "released_feature_venvs/WhatsAppFeature" in caplog.text
    assert str(agent_dir) not in caplog.text


@pytest.mark.skipif(os.name != "posix", reason="exclusive rename is POSIX-specific")
def test_runtime_migration_atomic_rename_refuses_even_empty_target(tmp_path):
    """Mutation guard: plain rename would silently replace this empty target."""

    (tmp_path / "legacy").mkdir()
    (tmp_path / "stable").mkdir()
    parent_fd = os.open(tmp_path, isolated_runtime._directory_open_flags())
    try:
        with pytest.raises(OSError) as raised:
            isolated_runtime._rename_directory_noreplace_at(
                parent_fd,
                "legacy",
                parent_fd,
                "stable",
            )
    finally:
        os.close(parent_fd)

    assert raised.value.errno in {errno.EEXIST, errno.ENOTEMPTY}
    assert (tmp_path / "legacy").is_dir()
    assert (tmp_path / "stable").is_dir()


@pytest.mark.skipif(os.name != "posix", reason="ownership policy is POSIX-specific")
def test_hosted_released_runtime_refuses_writable_source_without_moving_state(
    tmp_path,
):
    agent_dir = tmp_path / "agent_data" / "Hosted"
    legacy_root = agent_dir / "feature_venvs"
    legacy_feature = legacy_root / "WhatsAppFeature"
    credential = legacy_feature / "whatsapp_service" / "session.sqlite3"
    credential.parent.mkdir(parents=True, mode=0o700)
    credential.write_bytes(b"untrusted-writable-custody")
    legacy_root.chmod(0o777)
    agent = KestrelAgent(
        did="did:test:released-hosted-writable",
        storage_path=str(agent_dir / "kestrel_prime.db"),
        llm_service=Mock(providers=[]),
        database_url="postgresql://hosted.example/kestrel",
        db_backend="postgres",
        isolated_runtime_root=tmp_path / "isolated_feature_runtime",
        isolated_runtime_namespace="agent-writable",
        isolated_runtime_legacy_root=legacy_root,
        isolated_runtime_hosted=True,
    )
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)

    with pytest.raises(IsolatedRuntimePreparationError, match="custody"):
        feature._prepare_runtime_workspace()

    assert credential.read_bytes() == b"untrusted-writable-custody"
    assert not feature._feature_runtime_dir().exists()


@pytest.mark.skipif(os.name != "posix", reason="migration uses secure POSIX dirfds")
def test_hosted_feature_runtime_adopts_legacy_hash_before_refactor(tmp_path):
    runtime_root = tmp_path / "hosted-runtime"
    agent = _hosted_postgres_agent(runtime_root, "tenant/agent")
    original = _isolated_runtime()
    feature = ProxyFeature(agent, original, client_factory=FakeIsolatedClient)
    legacy_component = isolated_runtime._legacy_hosted_feature_runtime_component(
        original
    )
    assert legacy_component != feature._runtime_directory_name
    isolated_runtime.prepare_isolated_runtime_namespace(
        feature._isolated_runtime_scope,
        agent.did,
        relative_directories=(("feature_venvs", legacy_component, "data"),),
    )
    legacy_credential = (
        feature._agent_runtime_dir
        / "feature_venvs"
        / legacy_component
        / "data"
        / "credentials.db"
    )
    legacy_credential.write_text("linked-device-state")

    stable_runtime = feature._prepare_runtime_workspace()

    stable_credential = stable_runtime / "data" / "credentials.db"
    assert stable_credential.read_text() == "linked-device-state"
    assert not legacy_credential.parent.parent.exists()

    refactored = InstalledFeatureRuntime(
        class_name=original.class_name,
        entry_point="refactored.channel:WhatsAppFeature",
        distribution=original.distribution,
        runtime=original.runtime,
        service="refactored-service",
    )
    restarted = ProxyFeature(agent, refactored, client_factory=FakeIsolatedClient)
    assert restarted._runtime_directory_name == feature._runtime_directory_name
    assert (
        restarted._prepare_runtime_workspace() / "data" / "credentials.db"
    ).read_text() == "linked-device-state"


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="migration uses secure POSIX dirfds")
async def test_hosted_feature_runtime_collision_retains_both_trees_and_quarantines(
    caplog, tmp_path
):
    runtime_root = tmp_path / "hosted-runtime"
    agent = _hosted_postgres_agent(runtime_root, "tenant/agent")
    runtime = _isolated_runtime()
    client_started = False

    def client_factory(**_kwargs):
        nonlocal client_started
        client_started = True
        return FakeIsolatedClient(**_kwargs)

    feature = ProxyFeature(agent, runtime, client_factory=client_factory)
    legacy_component = feature._legacy_runtime_directory_name
    stable_component = feature._runtime_directory_name
    assert legacy_component is not None
    isolated_runtime.prepare_isolated_runtime_namespace(
        feature._isolated_runtime_scope,
        agent.did,
        relative_directories=(
            ("feature_venvs", legacy_component, "data"),
            ("feature_venvs", stable_component, "data"),
        ),
    )
    legacy_credential = (
        feature._agent_runtime_dir
        / "feature_venvs"
        / legacy_component
        / "data"
        / "credentials.db"
    )
    stable_credential = (
        feature._agent_runtime_dir
        / "feature_venvs"
        / stable_component
        / "data"
        / "credentials.db"
    )
    legacy_credential.write_text("legacy-custody")
    stable_credential.write_text("stable-custody")

    with caplog.at_level("ERROR"):
        available = await agent._register_startup_feature(
            feature,
            prepared_contributions=Mock(),
        )

    assert available is False
    assert feature.name not in agent.features
    assert client_started is False
    assert feature._client is None
    assert feature._supervision_task is None
    assert legacy_credential.read_text() == "legacy-custody"
    assert stable_credential.read_text() == "stable-custody"
    assert f"feature_venvs/{legacy_component}" in caplog.text
    assert f"feature_venvs/{stable_component}" in caplog.text


@pytest.mark.asyncio
async def test_portable_runtime_collision_logs_opaque_custody_and_quarantines(
    caplog,
    monkeypatch,
    tmp_path,
):
    """The non-dirfd path reports the same retained-tree operator evidence."""

    runtime_root = tmp_path / "portable-hosted-runtime"
    agent = _hosted_postgres_agent(runtime_root, "tenant/agent")
    client_started = False

    def client_factory(**kwargs):
        nonlocal client_started
        client_started = True
        return FakeIsolatedClient(**kwargs)

    feature = ProxyFeature(
        agent,
        _isolated_runtime(),
        client_factory=client_factory,
    )
    legacy_component = feature._legacy_runtime_directory_name
    stable_component = feature._runtime_directory_name
    assert legacy_component is not None
    isolated_runtime.prepare_isolated_runtime_namespace(
        feature._isolated_runtime_scope,
        agent.did,
        relative_directories=(
            ("feature_venvs", legacy_component, "data"),
            ("feature_venvs", stable_component, "data"),
        ),
    )
    legacy_credential = (
        feature._agent_runtime_dir
        / "feature_venvs"
        / legacy_component
        / "data"
        / "credentials.db"
    )
    stable_credential = (
        feature._agent_runtime_dir
        / "feature_venvs"
        / stable_component
        / "data"
        / "credentials.db"
    )
    legacy_secret = "portable-legacy-credential"
    stable_secret = "portable-stable-credential"
    legacy_credential.write_text(legacy_secret)
    stable_credential.write_text(stable_secret)
    monkeypatch.setattr(isolated_runtime, "_secure_dirfd_supported", lambda: False)

    with caplog.at_level("ERROR"):
        available = await agent._register_startup_feature(
            feature,
            prepared_contributions=Mock(),
        )

    assert available is False
    assert feature.name not in agent.features
    assert client_started is False
    assert feature._client is None
    assert legacy_credential.read_text() == legacy_secret
    assert stable_credential.read_text() == stable_secret
    assert f"feature_venvs/{legacy_component}" in caplog.text
    assert f"feature_venvs/{stable_component}" in caplog.text
    assert str(runtime_root) not in caplog.text
    assert legacy_secret not in caplog.text
    assert stable_secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="migration uses secure POSIX dirfds")
@pytest.mark.parametrize(
    "collision_errno",
    (errno.ENOTEMPTY, errno.EEXIST),
    ids=("nonempty-directory", "existing-directory"),
)
async def test_hosted_feature_runtime_rename_race_retains_both_custody_trees(
    caplog,
    collision_errno,
    monkeypatch,
    tmp_path,
):
    """A target appearing after the pre-check is still a custody collision."""

    runtime_root = tmp_path / "hosted-runtime"
    agent = _hosted_postgres_agent(runtime_root, "tenant/agent")
    runtime = _isolated_runtime()
    client_started = False

    def client_factory(**_kwargs):
        nonlocal client_started
        client_started = True
        return FakeIsolatedClient(**_kwargs)

    feature = ProxyFeature(agent, runtime, client_factory=client_factory)
    legacy_component = feature._legacy_runtime_directory_name
    stable_component = feature._runtime_directory_name
    assert legacy_component is not None
    isolated_runtime.prepare_isolated_runtime_namespace(
        feature._isolated_runtime_scope,
        agent.did,
        relative_directories=(("feature_venvs", legacy_component, "data"),),
    )
    legacy_credential = (
        feature._agent_runtime_dir
        / "feature_venvs"
        / legacy_component
        / "data"
        / "credentials.db"
    )
    stable_credential = (
        feature._agent_runtime_dir
        / "feature_venvs"
        / stable_component
        / "data"
        / "credentials.db"
    )
    legacy_credential.write_text("legacy-race-custody")
    rename_calls = 0

    def collide_after_precheck(source_fd, source, target_fd, target):
        nonlocal rename_calls
        rename_calls += 1
        assert source == legacy_component
        assert target == stable_component
        assert source_fd == target_fd
        stable_credential.parent.mkdir(parents=True)
        stable_credential.write_text("stable-race-custody")
        raise OSError(collision_errno, "synthetic directory collision")

    monkeypatch.setattr(
        isolated_runtime,
        "_rename_directory_noreplace_at",
        collide_after_precheck,
    )

    with caplog.at_level("ERROR"):
        available = await agent._register_startup_feature(
            feature,
            prepared_contributions=Mock(),
        )

    assert available is False
    assert rename_calls == 1
    assert feature.name not in agent.features
    assert client_started is False
    assert feature._client is None
    assert legacy_credential.read_text() == "legacy-race-custody"
    assert stable_credential.read_text() == "stable-race-custody"
    assert f"feature_venvs/{legacy_component}" in caplog.text
    assert f"feature_venvs/{stable_component}" in caplog.text
    assert str(runtime_root) not in caplog.text
    assert "legacy-race-custody" not in caplog.text
    assert "stable-race-custody" not in caplog.text


@pytest.mark.skipif(os.name != "posix", reason="migration uses secure POSIX dirfds")
def test_hosted_feature_runtime_rename_unrelated_oserror_is_not_collision(
    caplog,
    monkeypatch,
    tmp_path,
):
    runtime_root = tmp_path / "hosted-runtime"
    agent = _hosted_postgres_agent(runtime_root, "tenant/agent")
    feature = ProxyFeature(
        agent,
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )
    legacy_component = feature._legacy_runtime_directory_name
    stable_component = feature._runtime_directory_name
    assert legacy_component is not None
    isolated_runtime.prepare_isolated_runtime_namespace(
        feature._isolated_runtime_scope,
        agent.did,
        relative_directories=(("feature_venvs", legacy_component, "data"),),
    )
    legacy_credential = (
        feature._agent_runtime_dir
        / "feature_venvs"
        / legacy_component
        / "data"
        / "credentials.db"
    )
    stable_runtime = (
        feature._agent_runtime_dir / "feature_venvs" / stable_component
    )
    legacy_credential.write_text("legacy-retained")

    def fail_rename(*_args, **_kwargs):
        raise OSError(errno.EIO, "synthetic unrelated I/O failure")

    monkeypatch.setattr(
        isolated_runtime,
        "_rename_directory_noreplace_at",
        fail_rename,
    )

    with caplog.at_level("ERROR"), pytest.raises(
        IsolatedRuntimePreparationError,
        match="path could not be prepared",
    ) as raised:
        feature._prepare_runtime_workspace()

    assert isinstance(raised.value.__cause__, OSError)
    assert raised.value.__cause__.errno == errno.EIO
    assert legacy_credential.read_text() == "legacy-retained"
    assert not stable_runtime.exists()
    assert "migration collision" not in caplog.text
    assert f"feature_venvs/{legacy_component}" not in caplog.text
    assert f"feature_venvs/{stable_component}" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="symlink policy is POSIX-specific")
async def test_startup_preparation_quarantine_does_not_downgrade_namespace_attack(
    tmp_path,
):
    agent = _hosted_postgres_agent(tmp_path / "hosted-runtime", "tenant/agent")
    feature = ProxyFeature(
        agent,
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )
    isolated_runtime.prepare_isolated_runtime_namespace(
        feature._isolated_runtime_scope,
        agent.did,
        relative_directories=(("feature_venvs",),),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (
        feature._agent_runtime_dir
        / "feature_venvs"
        / feature._runtime_directory_name
    ).symlink_to(outside, target_is_directory=True)

    with pytest.raises(IsolatedRuntimeNamespaceError, match="unsafe"):
        await agent._register_startup_feature(
            feature,
            prepared_contributions=Mock(),
        )

    assert feature.name not in agent.features
    assert list(outside.iterdir()) == []


def test_hosted_agent_without_runtime_namespace_fails_before_feature_starts(tmp_path):
    agent = SimpleNamespace(
        storage_path=None,
        isolated_runtime_hosted=True,
        isolated_runtime_root=tmp_path / "hosted-runtime",
        isolated_runtime_namespace=None,
        features={},
    )

    with pytest.raises(IsolatedRuntimeNamespaceError, match="runtime namespace"):
        ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)


def test_postgres_agent_without_filesystem_scope_fails_closed_before_feature_starts():
    """Legacy hosted factories cannot silently regain the shared CWD fallback."""
    llm_service = Mock()
    llm_service.providers = []
    agent = KestrelAgent(
        did="did:test:legacy-postgres-host",
        storage_path=None,
        llm_service=llm_service,
        database_url="postgresql://hosted.example/kestrel",
        db_backend="postgres",
    )

    assert agent.isolated_runtime_hosted is True
    with pytest.raises(IsolatedRuntimeNamespaceError, match="no explicit runtime"):
        ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)


def test_hosted_agent_rejects_legacy_full_runtime_path(tmp_path):
    """A host-supplied full path is not an explicit root/namespace boundary."""
    agent = SimpleNamespace(
        did="did:test:legacy-full-path",
        storage_path=None,
        isolated_runtime_hosted=True,
        isolated_feature_data_dir=tmp_path / "legacy-host-path",
        features={},
    )

    with pytest.raises(
        IsolatedRuntimeNamespaceError,
        match="isolated_runtime_root and isolated_runtime_namespace",
    ):
        ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)


@pytest.mark.parametrize(
    "namespace",
    [
        "../other-agent",
        "tenant/../other",
        "/outside",
        "tenant//agent",
        "Tenant/agent",
        "tenant\\agent",
        "tenant/con",
        "tenant/con.txt",
    ],
)
def test_hosted_runtime_namespace_rejects_traversal_and_absolute_paths(tmp_path, namespace):
    with pytest.raises(IsolatedRuntimeNamespaceError, match="canonical|lowercase"):
        resolve_isolated_runtime_namespace(tmp_path / "runtime-root", namespace)


def test_hosted_runtime_namespace_rejects_aliases_that_can_cause_collisions(tmp_path):
    """The accepted namespace has one unambiguous, root-contained spelling."""
    scope = resolve_isolated_runtime_namespace(
        tmp_path / "runtime-root", "tenant-a/agent-a"
    )
    assert scope.path == (tmp_path / "runtime-root" / "tenant-a" / "agent-a").resolve()

    with pytest.raises(IsolatedRuntimeNamespaceError, match="lowercase"):
        resolve_isolated_runtime_namespace(
            tmp_path / "runtime-root", "tenant-a/agent-a/../agent-b"
        )
    with pytest.raises(IsolatedRuntimeNamespaceError, match="lowercase"):
        resolve_isolated_runtime_namespace(
            tmp_path / "runtime-root", "tenant-a//agent-a"
        )


def test_hosted_runtime_root_refuses_missing_operator_parent(tmp_path):
    missing_parent = tmp_path / "missing-volume" / "runtime"

    with pytest.raises(IsolatedRuntimeNamespaceError, match="parent must already exist"):
        resolve_isolated_runtime_namespace(missing_parent, "agent-safe")

    assert not (tmp_path / "missing-volume").exists()


def test_hosted_runtime_namespace_derivation_is_tuple_safe_and_path_inert(tmp_path):
    first = derive_isolated_runtime_namespace("ab", "c")
    second = derive_isolated_runtime_namespace("a", "bc")

    assert first != second
    assert first == derive_isolated_runtime_namespace("ab", "c")
    assert first.startswith("agent-")
    assert len(first) == 64
    scope = resolve_isolated_runtime_namespace(tmp_path / "runtime-root", first)
    assert str(scope.namespace) == first


def test_hosted_runtime_namespace_rejects_cross_agent_collision(tmp_path):
    runtime_root = tmp_path / "hosted-runtime"
    first = _hosted_postgres_agent(
        runtime_root,
        "tenant/agent",
        did="did:test:first-agent",
    )
    second = _hosted_postgres_agent(
        runtime_root,
        "tenant/agent",
        did="did:test:second-agent",
    )
    ProxyFeature(first, _isolated_runtime(), client_factory=FakeIsolatedClient)

    with pytest.raises(IsolatedRuntimeNamespaceError, match="different agent"):
        ProxyFeature(second, _isolated_runtime(), client_factory=FakeIsolatedClient)


@pytest.mark.skipif(os.name != "posix", reason="secure cleanup is POSIX-only")
def test_cleanup_refuses_nested_owned_namespace_without_deleting_either(tmp_path):
    runtime_root = tmp_path / "hosted-runtime"
    outer = resolve_isolated_runtime_namespace(runtime_root, "tenant")
    inner = resolve_isolated_runtime_namespace(runtime_root, "tenant/agent")
    isolated_runtime.prepare_isolated_runtime_namespace(
        outer,
        "did:test:outer",
        relative_directories=(("outer-state",),),
    )
    isolated_runtime.prepare_isolated_runtime_namespace(
        inner,
        "did:test:inner",
        relative_directories=(("inner-state",),),
    )
    outer_state = outer.path / "outer-state" / "keep"
    inner_state = inner.path / "inner-state" / "credential"
    outer_state.write_text("outer")
    inner_state.write_text("inner")

    with pytest.raises(IsolatedRuntimeNamespaceError, match="nested ownership marker"):
        isolated_runtime.remove_isolated_runtime_namespace(
            outer,
            "did:test:outer",
        )

    assert (outer.path / ".kestrel-runtime-owner").is_file()
    assert (inner.path / ".kestrel-runtime-owner").is_file()
    assert outer_state.read_text() == "outer"
    assert inner_state.read_text() == "inner"
    assert isolated_runtime.remove_isolated_runtime_namespace(
        inner,
        "did:test:inner",
    ) is isolated_runtime.RuntimeNamespaceCleanupOutcome.REMOVED
    assert isolated_runtime.remove_isolated_runtime_namespace(
        outer,
        "did:test:outer",
    ) is isolated_runtime.RuntimeNamespaceCleanupOutcome.REMOVED


@pytest.mark.skipif(os.name != "posix", reason="secure cleanup is POSIX-only")
def test_cleanup_handles_tree_deeper_than_python_recursion_limit(
    monkeypatch, tmp_path
):
    scope = resolve_isolated_runtime_namespace(tmp_path / "runtime", "tenant")
    owner = "did:test:deep-cleanup"
    isolated_runtime.prepare_isolated_runtime_namespace(
        scope,
        owner,
        relative_directories=(("state",),),
    )
    current_fd = os.open(
        scope.path / "state",
        isolated_runtime._directory_open_flags(),
    )
    stack_depth = len(traceback.extract_stack())
    recursion_limit = max(stack_depth + 40, 80)
    chain_depth = recursion_limit + 20
    try:
        for _index in range(chain_depth):
            os.mkdir("nested", mode=0o700, dir_fd=current_fd)
            child_fd = os.open(
                "nested",
                isolated_runtime._directory_open_flags(),
                dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = child_fd
        payload_fd = os.open(
            "payload",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=current_fd,
        )
        os.close(payload_fd)
    finally:
        os.close(current_fd)

    original_limit = sys.getrecursionlimit()
    original_open_child = isolated_runtime._open_cleanup_child_at
    original_open_parent = isolated_runtime._open_cleanup_parent_at
    traversal_calls = 0

    def count_open_child(*args, **kwargs):
        nonlocal traversal_calls
        traversal_calls += 1
        return original_open_child(*args, **kwargs)

    def count_open_parent(*args, **kwargs):
        nonlocal traversal_calls
        traversal_calls += 1
        return original_open_parent(*args, **kwargs)

    monkeypatch.setattr(isolated_runtime, "_open_cleanup_child_at", count_open_child)
    monkeypatch.setattr(isolated_runtime, "_open_cleanup_parent_at", count_open_parent)
    sys.setrecursionlimit(recursion_limit)
    try:
        assert (
            isolated_runtime.remove_isolated_runtime_namespace(scope, owner)
            is isolated_runtime.RuntimeNamespaceCleanupOutcome.REMOVED
        )
    finally:
        sys.setrecursionlimit(original_limit)

    assert not scope.path.exists()
    assert traversal_calls < chain_depth * 6


def test_runtime_cleanup_primitive_reports_exact_absent_and_unhosted_custody(
    tmp_path,
):
    scope = resolve_isolated_runtime_namespace(tmp_path / "runtime", "tenant")

    assert (
        isolated_runtime.remove_isolated_runtime_namespace(scope, "did:test:absent")
        is isolated_runtime.RuntimeNamespaceCleanupOutcome.ALREADY_ABSENT
    )
    assert (
        isolated_runtime.remove_agent_runtime_namespace(
            SimpleNamespace(
                did="did:test:storage-backed",
                storage_path=str(tmp_path / "agent" / "kestrel_prime.db"),
            )
        )
        is isolated_runtime.RuntimeNamespaceCleanupOutcome.NOT_HOSTED
    )


@pytest.mark.skipif(os.name != "posix", reason="released cleanup uses POSIX dirfds")
def test_released_cleanup_refuses_nested_runtime_owner_without_mutation(tmp_path):
    legacy_root = tmp_path / "agent" / "feature_venvs"
    nested = legacy_root / "LegacyFeature" / "nested-agent"
    nested.mkdir(parents=True, mode=0o700)
    legacy_root.chmod(0o700)
    (legacy_root / "LegacyFeature").chmod(0o700)
    credential = legacy_root / "LegacyFeature" / "credential"
    credential.write_text("retain-me")
    (nested / isolated_runtime._RUNTIME_OWNER_MARKER).write_text("foreign-owner")

    with pytest.raises(IsolatedRuntimeNamespaceError, match="nested ownership marker"):
        isolated_runtime.remove_released_legacy_runtime_root(legacy_root)

    assert credential.read_text() == "retain-me"
    assert (nested / isolated_runtime._RUNTIME_OWNER_MARKER).is_file()


@pytest.mark.skipif(os.name != "posix", reason="repair marker uses POSIX dirfds")
def test_relocation_repair_marker_recovers_only_its_interrupted_temp_link(tmp_path):
    runtime_dir = tmp_path / "feature-runtime"
    runtime_dir.mkdir(mode=0o700)
    directory_fd = os.open(runtime_dir, isolated_runtime._directory_open_flags())
    try:
        assert (
            isolated_runtime._read_venv_relocation_repair_marker_at(directory_fd)
            is None
        )
        isolated_runtime._ensure_venv_relocation_repair_marker_at(directory_fd)
        marker = runtime_dir / isolated_runtime._VENV_RELOCATION_REPAIR_MARKER
        interrupted = runtime_dir / (
            f"{isolated_runtime._VENV_RELOCATION_REPAIR_TEMP_PREFIX}interrupted"
        )
        os.link(marker, interrupted)

        assert isolated_runtime._read_venv_relocation_repair_marker_at(directory_fd)
        assert not interrupted.exists()
        assert marker.stat().st_nlink == 1

        external = tmp_path / "unrelated-hard-link"
        os.link(marker, external)
        with pytest.raises(IsolatedRuntimeNamespaceError, match="external hard link"):
            isolated_runtime._read_venv_relocation_repair_marker_at(directory_fd)
        assert external.is_file()
    finally:
        os.close(directory_fd)


@pytest.mark.skipif(os.name != "posix", reason="repair marker uses POSIX dirfds")
def test_relocation_marker_cannot_downgrade_reclaim_repair_intent(tmp_path):
    dirfd_runtime = tmp_path / "dirfd-runtime"
    dirfd_runtime.mkdir(mode=0o700)
    directory_fd = os.open(dirfd_runtime, isolated_runtime._directory_open_flags())
    try:
        isolated_runtime._ensure_venv_relocation_repair_marker_at(
            directory_fd,
            payload=isolated_runtime._VENV_RECLAIM_REPAIR_PAYLOAD,
        )
        isolated_runtime._ensure_venv_relocation_repair_marker_at(directory_fd)
        assert (
            isolated_runtime._read_venv_relocation_repair_marker_at(directory_fd)
            == "reclaim"
        )
    finally:
        os.close(directory_fd)

    portable_runtime = tmp_path / "portable-runtime"
    portable_runtime.mkdir(mode=0o700)
    isolated_runtime._ensure_venv_relocation_repair_marker_portable(
        portable_runtime,
        payload=isolated_runtime._VENV_RECLAIM_REPAIR_PAYLOAD,
    )
    isolated_runtime._ensure_venv_relocation_repair_marker_portable(portable_runtime)
    assert (
        isolated_runtime._read_venv_relocation_repair_marker_portable(
            portable_runtime
        )
        == "reclaim"
    )


def test_venv_repair_pending_tracks_real_marker_transition(tmp_path):
    agent = _hosted_postgres_agent(
        tmp_path / "hosted-runtime",
        "tenant/agent",
    )
    feature = ProxyFeature(
        agent,
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )
    runtime_dir = feature._prepare_runtime_workspace()

    assert feature._venv_repair_reason() is None
    assert feature._venv_relocation_repair_pending() is False

    if isolated_runtime._secure_dirfd_supported():
        directory_fd = os.open(
            runtime_dir,
            isolated_runtime._directory_open_flags(),
        )
        try:
            isolated_runtime._ensure_venv_relocation_repair_marker_at(directory_fd)
        finally:
            os.close(directory_fd)
    else:  # pragma: no cover - portable fallback
        marker = runtime_dir / isolated_runtime._VENV_RELOCATION_REPAIR_MARKER
        marker.write_bytes(isolated_runtime._VENV_RELOCATION_REPAIR_PAYLOAD)

    assert feature._venv_repair_reason() == "relocation"
    assert feature._venv_relocation_repair_pending() is True


@pytest.mark.skipif(os.name != "posix", reason="repair marker uses POSIX dirfds")
def test_relocation_marker_publisher_refuses_missing_post_link_witness(
    monkeypatch,
    tmp_path,
):
    runtime_dir = tmp_path / "feature-runtime"
    runtime_dir.mkdir(mode=0o700)
    directory_fd = os.open(runtime_dir, isolated_runtime._directory_open_flags())
    reads = iter((None, None))
    links = []
    real_link = os.link

    def record_link(*args, **kwargs):
        links.append((args, kwargs))
        return real_link(*args, **kwargs)

    monkeypatch.setattr(
        isolated_runtime,
        "_read_venv_relocation_repair_marker_at",
        lambda _fd: next(reads),
    )
    monkeypatch.setattr(isolated_runtime.os, "link", record_link)
    try:
        with pytest.raises(
            IsolatedRuntimePreparationError,
            match="could not be recorded",
        ):
            isolated_runtime._ensure_venv_relocation_repair_marker_at(directory_fd)
        assert len(links) == 1
    finally:
        os.close(directory_fd)


@pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX")
def test_portable_relocation_marker_fsyncs_directory_before_return(
    monkeypatch,
    tmp_path,
):
    runtime_dir = tmp_path / "feature-runtime"
    runtime_dir.mkdir(mode=0o700)
    real_fsync = os.fsync
    fsynced_directory = False

    def observe_fsync(descriptor):
        nonlocal fsynced_directory
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            fsynced_directory = True
        return real_fsync(descriptor)

    monkeypatch.setattr(isolated_runtime.os, "fsync", observe_fsync)

    isolated_runtime._ensure_venv_relocation_repair_marker_portable(runtime_dir)

    assert fsynced_directory is True
    assert isolated_runtime._read_venv_relocation_repair_marker_portable(
        runtime_dir
    )


def test_windows_portable_metadata_accepts_synthesized_permission_modes(
    monkeypatch,
    tmp_path,
):
    """Windows' synthetic 0777/0666 bits are not POSIX custody evidence."""

    runtime_dir = tmp_path / "feature-runtime"
    runtime_dir.mkdir()
    assert (
        isolated_runtime._read_venv_relocation_repair_marker_portable(runtime_dir)
        is None
    )
    marker = runtime_dir / isolated_runtime._VENV_RELOCATION_REPAIR_MARKER
    marker.write_bytes(isolated_runtime._VENV_RELOCATION_REPAIR_PAYLOAD)
    real_path_stat = Path.stat

    def windows_shaped_stat(path, *args, **kwargs):
        metadata = real_path_stat(path, *args, **kwargs)
        if path == marker:
            return SimpleNamespace(
                st_mode=stat.S_IFREG | 0o666,
                st_uid=-1,
                st_nlink=1,
                st_dev=metadata.st_dev,
                st_ino=metadata.st_ino,
            )
        return metadata

    # Construct paths before changing the platform seam: pathlib selects its
    # concrete path class from os.name at construction time.
    monkeypatch.setattr(Path, "stat", windows_shaped_stat)
    monkeypatch.setattr(isolated_runtime.os, "name", "nt")
    directory_metadata = SimpleNamespace(
        st_mode=stat.S_IFDIR | 0o777,
        st_uid=-1,
    )

    isolated_runtime._validate_operator_root_metadata(directory_metadata)
    isolated_runtime._validate_released_legacy_directory_metadata(
        directory_metadata
    )
    assert isolated_runtime._read_venv_relocation_repair_marker_portable(
        runtime_dir
    )


@pytest.mark.skipif(os.name != "posix", reason="secure cleanup is POSIX-only")
def test_partial_cleanup_preserves_owner_marker_and_retry_succeeds(
    monkeypatch, tmp_path
):
    scope = resolve_isolated_runtime_namespace(tmp_path / "runtime", "tenant")
    owner = "did:test:partial-cleanup"
    isolated_runtime.prepare_isolated_runtime_namespace(
        scope,
        owner,
        relative_directories=(("state",),),
    )
    blocked = scope.path / "state" / "blocked"
    blocked.write_text("retain-until-retry")
    original_unlink = isolated_runtime.os.unlink

    def fail_blocked(name, *args, **kwargs):
        if name == "blocked":
            raise PermissionError(errno.EACCES, "synthetic cleanup failure")
        return original_unlink(name, *args, **kwargs)

    monkeypatch.setattr(isolated_runtime.os, "unlink", fail_blocked)
    with pytest.raises(IsolatedRuntimePreparationError, match="state was retained"):
        isolated_runtime.remove_isolated_runtime_namespace(scope, owner)

    assert (scope.path / ".kestrel-runtime-owner").is_file()
    assert blocked.read_text() == "retain-until-retry"
    monkeypatch.setattr(isolated_runtime.os, "unlink", original_unlink)
    assert (
        isolated_runtime.remove_isolated_runtime_namespace(scope, owner)
        is isolated_runtime.RuntimeNamespaceCleanupOutcome.REMOVED
    )
    assert not scope.path.exists()


@pytest.mark.skipif(os.name != "posix", reason="secure cleanup is POSIX-only")
def test_final_rmdir_failure_restores_marker_for_retry(monkeypatch, tmp_path):
    scope = resolve_isolated_runtime_namespace(tmp_path / "runtime", "tenant")
    owner = "did:test:rmdir-retry"
    isolated_runtime.prepare_isolated_runtime_namespace(scope, owner)
    original_rmdir = isolated_runtime.os.rmdir
    failed = False

    def fail_once(name, *args, **kwargs):
        nonlocal failed
        if name == "tenant" and not failed:
            failed = True
            raise OSError(errno.ENOTEMPTY, "synthetic final race")
        return original_rmdir(name, *args, **kwargs)

    monkeypatch.setattr(isolated_runtime.os, "rmdir", fail_once)
    with pytest.raises(IsolatedRuntimePreparationError, match="state was retained"):
        isolated_runtime.remove_isolated_runtime_namespace(scope, owner)

    assert (scope.path / ".kestrel-runtime-owner").is_file()
    assert (
        isolated_runtime.remove_isolated_runtime_namespace(scope, owner)
        is isolated_runtime.RuntimeNamespaceCleanupOutcome.REMOVED
    )


@pytest.mark.skipif(os.name != "posix", reason="secure cleanup is POSIX-only")
def test_final_rmdir_concurrent_disappearance_is_success(monkeypatch, tmp_path):
    scope = resolve_isolated_runtime_namespace(tmp_path / "runtime", "tenant")
    owner = "did:test:rmdir-concurrent-success"
    isolated_runtime.prepare_isolated_runtime_namespace(scope, owner)
    original_rmdir = isolated_runtime.os.rmdir

    def remove_then_report_missing(name, *args, **kwargs):
        if name == "tenant" and kwargs.get("dir_fd") is not None:
            original_rmdir(name, *args, **kwargs)
            raise FileNotFoundError(errno.ENOENT, "synthetic concurrent cleanup")
        return original_rmdir(name, *args, **kwargs)

    monkeypatch.setattr(isolated_runtime.os, "rmdir", remove_then_report_missing)

    assert (
        isolated_runtime.remove_isolated_runtime_namespace(scope, owner)
        is isolated_runtime.RuntimeNamespaceCleanupOutcome.REMOVED
    )
    assert not scope.path.exists()


@pytest.mark.skipif(os.name != "posix", reason="secure cleanup is POSIX-only")
def test_final_rmdir_marker_restore_failure_preserves_both_diagnoses(
    monkeypatch, tmp_path
):
    scope = resolve_isolated_runtime_namespace(tmp_path / "runtime", "tenant")
    owner = "did:test:rmdir-compensation-failure"
    isolated_runtime.prepare_isolated_runtime_namespace(scope, owner)
    original_rmdir = isolated_runtime.os.rmdir

    def fail_final_rmdir(name, *args, **kwargs):
        if name == "tenant" and kwargs.get("dir_fd") is not None:
            raise OSError(errno.ENOTEMPTY, "synthetic final removal failure")
        return original_rmdir(name, *args, **kwargs)

    def fail_marker_restore(*_args, **_kwargs):
        raise PermissionError(errno.EACCES, "synthetic marker restore failure")

    monkeypatch.setattr(isolated_runtime.os, "rmdir", fail_final_rmdir)
    monkeypatch.setattr(
        isolated_runtime,
        "_read_or_create_runtime_owner",
        fail_marker_restore,
    )

    with pytest.raises(
        IsolatedRuntimePreparationError,
        match="marker could not be restored",
    ) as raised:
        isolated_runtime.remove_isolated_runtime_namespace(scope, owner)

    causes = raised.value.__cause__
    assert isinstance(causes, ExceptionGroup)
    assert any(
        isinstance(error, OSError)
        and "final removal failure" in str(error)
        for error in causes.exceptions
    )
    assert any(
        isinstance(error, PermissionError)
        and "marker restore failure" in str(error)
        for error in causes.exceptions
    )
    assert scope.path.is_dir()
    # The compensation itself was forced to fail, so remove the now-empty
    # test directory without weakening the production missing-marker policy.
    original_rmdir(scope.path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission mode contract")
def test_hosted_runtime_namespace_and_marker_are_private(tmp_path):
    agent = _hosted_postgres_agent(
        tmp_path / "hosted-runtime",
        "tenant/agent",
    )
    feature = ProxyFeature(
        agent, _isolated_runtime(), client_factory=FakeIsolatedClient
    )
    workspace = feature._prepare_runtime_workspace()
    marker = feature._agent_runtime_dir / ".kestrel-runtime-owner"

    assert stat.S_IMODE(feature._agent_runtime_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(workspace.stat().st_mode) == 0o700
    assert stat.S_IMODE((workspace / "config").stat().st_mode) == 0o700
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="same-UID boundary is POSIX-specific")
def test_hosted_private_modes_do_not_claim_same_uid_process_isolation(tmp_path):
    agent = _hosted_postgres_agent(tmp_path / "hosted-runtime", "tenant/agent")
    feature = ProxyFeature(
        agent, _isolated_runtime(), client_factory=FakeIsolatedClient
    )
    workspace = feature._prepare_runtime_workspace()

    assert workspace.stat().st_uid == os.geteuid()
    design = (
        Path(__file__).parents[2] / "docs" / "design" / "ISOLATED_FEATURE_RUNTIME.md"
    ).read_text()
    assert "not an operating-system sandbox" in design
    assert "cannot prevent a malicious" in design
    assert "same-UID child from traversing a sibling namespace" in design


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission mode contract")
def test_existing_operator_root_permissions_are_verified_not_mutated(tmp_path):
    runtime_root = tmp_path / "hosted-runtime"
    runtime_root.mkdir(mode=0o755)
    runtime_root.chmod(0o755)

    ProxyFeature(
        _hosted_postgres_agent(runtime_root, "agent-safe"),
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )

    assert stat.S_IMODE(runtime_root.stat().st_mode) == 0o755


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission mode contract")
def test_group_writable_operator_root_fails_without_chmod(tmp_path):
    runtime_root = tmp_path / "hosted-runtime"
    runtime_root.mkdir(mode=0o770)
    runtime_root.chmod(0o770)

    with pytest.raises(IsolatedRuntimeNamespaceError, match="group- or world-writable"):
        ProxyFeature(
            _hosted_postgres_agent(runtime_root, "agent-safe"),
            _isolated_runtime(),
            client_factory=FakeIsolatedClient,
        )

    assert stat.S_IMODE(runtime_root.stat().st_mode) == 0o770


def test_corrupt_owner_marker_has_explicit_operator_recovery_error(tmp_path):
    agent = _hosted_postgres_agent(tmp_path / "hosted-runtime", "agent-safe")
    first = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)
    marker = first._agent_runtime_dir / ".kestrel-runtime-owner"
    marker.write_bytes(b"")

    with pytest.raises(IsolatedRuntimeNamespaceError, match="marker is corrupt"):
        ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)


def test_owner_marker_write_failure_never_publishes_partial_marker(
    monkeypatch, tmp_path
):
    scope = resolve_isolated_runtime_namespace(
        tmp_path / "hosted-runtime", "agent-safe"
    )

    def fail_write(_descriptor, _payload):
        raise OSError(errno.ENOSPC, "synthetic full disk")

    monkeypatch.setattr(isolated_runtime, "_write_all", fail_write)
    with pytest.raises(IsolatedRuntimePreparationError, match="could not be prepared"):
        isolated_runtime.prepare_isolated_runtime_namespace(
            scope, "did:test:atomic-marker"
        )

    assert not (scope.path / ".kestrel-runtime-owner").exists()
    assert list(scope.path.glob(".kestrel-runtime-owner.tmp-*")) == []


@pytest.mark.skipif(os.name != "posix", reason="hard-link recovery is POSIX-only")
def test_owner_marker_recovers_crash_between_link_and_temp_unlink(tmp_path):
    agent = _hosted_postgres_agent(tmp_path / "hosted-runtime", "agent-safe")
    first = ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)
    marker = first._agent_runtime_dir / ".kestrel-runtime-owner"
    interrupted_temp = first._agent_runtime_dir / (
        ".kestrel-runtime-owner.tmp-crashed-install"
    )
    os.link(marker, interrupted_temp)
    assert marker.stat().st_nlink == 2

    ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)

    assert marker.stat().st_nlink == 1
    assert not interrupted_temp.exists()


@pytest.mark.skipif(os.name != "posix", reason="hard-link recovery is POSIX-only")
def test_owner_marker_creator_tolerates_concurrent_temp_recovery(monkeypatch, tmp_path):
    """A second preparer may unlink the creator's temp after publication."""

    scope = resolve_isolated_runtime_namespace(
        tmp_path / "hosted-runtime", "agent-safe"
    )
    original_link = isolated_runtime.os.link

    def link_then_recover(source, destination, **kwargs):
        original_link(source, destination, **kwargs)
        source_dir_fd = kwargs["src_dir_fd"]
        marker_fd = os.open(
            isolated_runtime._RUNTIME_OWNER_MARKER,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=source_dir_fd,
        )
        try:
            isolated_runtime._recover_runtime_owner_install_link(
                source_dir_fd, marker_fd
            )
        finally:
            os.close(marker_fd)

    monkeypatch.setattr(isolated_runtime.os, "link", link_then_recover)

    path = isolated_runtime.prepare_isolated_runtime_namespace(
        scope, "did:test:concurrent-marker-recovery"
    )

    marker = path / isolated_runtime._RUNTIME_OWNER_MARKER
    assert marker.is_file()
    assert marker.stat().st_nlink == 1
    assert list(path.glob(f"{isolated_runtime._RUNTIME_OWNER_TEMP_PREFIX}*")) == []


@pytest.mark.skipif(os.name != "posix", reason="hard-link recovery is POSIX-only")
def test_owner_marker_prepare_tolerates_temp_disappearing_before_stat(
    monkeypatch, tmp_path
):
    scope = resolve_isolated_runtime_namespace(
        tmp_path / "hosted-runtime", "agent-safe"
    )
    owner = "did:test:prepare-marker-race"
    path = isolated_runtime.prepare_isolated_runtime_namespace(scope, owner)
    marker = path / isolated_runtime._RUNTIME_OWNER_MARKER
    interrupted = path / (f"{isolated_runtime._RUNTIME_OWNER_TEMP_PREFIX}prepare-race")
    os.link(marker, interrupted)
    original_listdir = os.listdir
    original_unlink = os.unlink
    raced = False

    def list_then_remove(directory_fd):
        nonlocal raced
        names = original_listdir(directory_fd)
        if interrupted.name in names and not raced:
            raced = True
            original_unlink(interrupted.name, dir_fd=directory_fd)
        return names

    monkeypatch.setattr(isolated_runtime.os, "listdir", list_then_remove)

    isolated_runtime.prepare_isolated_runtime_namespace(scope, owner)

    assert raced
    assert marker.stat().st_nlink == 1


@pytest.mark.skipif(os.name != "posix", reason="hard-link recovery is POSIX-only")
def test_owner_marker_cleanup_tolerates_temp_disappearing_before_unlink(
    monkeypatch, tmp_path
):
    scope = resolve_isolated_runtime_namespace(
        tmp_path / "hosted-runtime", "agent-safe"
    )
    owner = "did:test:cleanup-marker-race"
    path = isolated_runtime.prepare_isolated_runtime_namespace(scope, owner)
    marker = path / isolated_runtime._RUNTIME_OWNER_MARKER
    interrupted = path / (f"{isolated_runtime._RUNTIME_OWNER_TEMP_PREFIX}cleanup-race")
    os.link(marker, interrupted)
    original_stat = os.stat
    original_unlink = os.unlink
    raced = False

    def stat_then_remove(candidate, *args, **kwargs):
        nonlocal raced
        metadata = original_stat(candidate, *args, **kwargs)
        if (
            candidate == interrupted.name
            and kwargs.get("dir_fd") is not None
            and not raced
        ):
            raced = True
            original_unlink(candidate, dir_fd=kwargs["dir_fd"])
        return metadata

    monkeypatch.setattr(isolated_runtime.os, "stat", stat_then_remove)
    monkeypatch.setattr(isolated_runtime, "_secure_dirfd_supported", lambda: True)

    assert (
        isolated_runtime.remove_isolated_runtime_namespace(scope, owner)
        is isolated_runtime.RuntimeNamespaceCleanupOutcome.REMOVED
    )
    assert raced
    assert not path.exists()


def test_hosted_runtime_namespace_rejects_symlink_escape(tmp_path):
    runtime_root = tmp_path / "hosted-runtime"
    outside = tmp_path / "outside"
    runtime_root.mkdir()
    outside.mkdir()
    (runtime_root / "tenant").symlink_to(outside, target_is_directory=True)
    agent = _hosted_postgres_agent(runtime_root, "tenant/agent")

    with pytest.raises(IsolatedRuntimeNamespaceError, match="unsafe"):
        ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)
    assert list(outside.iterdir()) == []


def test_hosted_runtime_namespace_detects_path_swap_after_secure_open(
    monkeypatch, tmp_path
):
    """The final inode binding closes the validate/open pathname race."""

    runtime_root = tmp_path / "hosted-runtime"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_open = isolated_runtime._open_or_create_directory_at
    swapped = False

    def swap_after_open(parent_fd, component, **kwargs):
        nonlocal swapped
        descriptor = original_open(parent_fd, component, **kwargs)
        if component == "agent" and not swapped:
            swapped = True
            namespace = runtime_root / "tenant" / "agent"
            namespace.rename(runtime_root / "tenant" / "opened-agent")
            namespace.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(
        isolated_runtime,
        "_open_or_create_directory_at",
        swap_after_open,
    )
    agent = _hosted_postgres_agent(runtime_root, "tenant/agent")

    with pytest.raises(IsolatedRuntimeNamespaceError, match="changed"):
        ProxyFeature(agent, _isolated_runtime(), client_factory=FakeIsolatedClient)
    assert list(outside.iterdir()) == []


def test_hosted_runtime_workspace_rejects_noncanonical_relative_components(tmp_path):
    scope = resolve_isolated_runtime_namespace(
        tmp_path / "hosted-runtime",
        "tenant/agent",
    )

    with pytest.raises(IsolatedRuntimeNamespaceError, match="relative components"):
        isolated_runtime.prepare_isolated_runtime_namespace(
            scope,
            "did:test:tenant-agent",
            relative_directories=(("feature_venvs", "..", "outside"),),
        )

    assert not scope.root.exists()


def test_hosted_runtime_root_os_error_is_optional_feature_preparation_failure(
    monkeypatch, tmp_path
):
    scope = resolve_isolated_runtime_namespace(
        tmp_path / "hosted-runtime",
        "tenant/agent",
    )

    def fail_root_open(_path):
        raise OSError("synthetic root race")

    monkeypatch.setattr(
        isolated_runtime,
        "_open_secure_absolute_directory",
        fail_root_open,
    )
    with pytest.raises(IsolatedRuntimePreparationError, match="could not be prepared"):
        isolated_runtime.prepare_isolated_runtime_namespace(
            scope,
            "did:test:tenant-agent",
        )


def test_hosted_runtime_workspace_rejects_feature_directory_symlink(tmp_path):
    """Workspace creation does not follow a pre-planted feature path outside root."""
    runtime_root = tmp_path / "hosted-runtime"
    outside = tmp_path / "outside"
    outside.mkdir()
    agent = _hosted_postgres_agent(runtime_root, "tenant/agent")
    feature = ProxyFeature(
        agent,
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )
    feature_parent = feature._agent_runtime_dir / "feature_venvs"
    feature_parent.mkdir(mode=0o700)
    (feature_parent / feature._runtime_directory_name).symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(IsolatedRuntimeNamespaceError, match="unsafe"):
        feature._prepare_runtime_workspace()
    assert list(outside.iterdir()) == []


def test_portable_runtime_workspace_rejects_parent_symlink(monkeypatch, tmp_path):
    """The non-dirfd fallback validates each parent, not only the final leaf."""
    runtime_root = tmp_path / "hosted-runtime"
    outside = tmp_path / "outside"
    outside.mkdir()
    agent = _hosted_postgres_agent(runtime_root, "tenant/agent")
    feature = ProxyFeature(
        agent,
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )
    feature_venvs = feature._agent_runtime_dir / "feature_venvs"
    feature_venvs.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(isolated_runtime, "_secure_dirfd_supported", lambda: False)

    with pytest.raises(IsolatedRuntimeNamespaceError, match="symlinks"):
        feature._prepare_runtime_workspace()
    assert list(outside.iterdir()) == []


def test_hosted_child_launch_uses_agent_scoped_workspace_config_and_environment(
    monkeypatch, tmp_path
):
    """Same feature on two hosted agents gets isolated state and launch inputs."""
    monkeypatch.setenv("OPENAI_API_KEY", "another-tenants-key")
    monkeypatch.setenv("HTTPS_PROXY", "http://egress-proxy.internal:8443")
    monkeypatch.setenv("NO_PROXY", "metadata.internal")
    runtime_root = tmp_path / "hosted-runtime"
    shared_prebuilt_venv = runtime_root / "prebuilt" / ".venv"
    captured = []

    def client_factory(**kwargs):
        captured.append(kwargs)
        return FakeIsolatedClient(**kwargs)

    for namespace, token in (("tenant-a/agent-a", "a-secret"), ("tenant-b/agent-b", "b-secret")):
        agent = _hosted_postgres_agent(runtime_root, namespace)
        feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
        # A host may deliberately share an immutable prebuilt venv.  Its
        # mutable WhatsApp state must still follow the agent-scoped data-dir
        # contract rather than resolve beside this venv.
        feature._venv_path = shared_prebuilt_venv
        feature._host_config = {"token": token}
        workspace = feature._prepare_runtime_workspace()
        feature._build_client()

        launch = captured[-1]
        assert launch["cwd"] == str(workspace / "work")
        assert launch["config"] == {"token": token}
        assert launch["env"]["KESTREL_ISOLATED_RUNTIME_DIR"] == str(workspace)
        assert launch["env"]["KESTREL_ISOLATED_FEATURE_DATA_DIR"] == str(workspace)
        assert launch["env"]["HOME"] == str(workspace / "home")
        assert launch["env"]["TMPDIR"] == str(workspace / "tmp")
        assert launch["env"]["XDG_CACHE_HOME"] == str(workspace / "cache")
        provisioning_cache = workspace / "provisioning_cache"
        assert provisioning_cache.is_dir()
        assert provisioning_cache != Path(launch["env"]["XDG_CACHE_HOME"])
        assert "UV_CACHE_DIR" not in launch["env"]
        assert str(provisioning_cache) not in launch["env"].values()
        if os.name == "posix":
            provisioning_metadata = provisioning_cache.stat()
            assert provisioning_metadata.st_uid == os.geteuid()
            assert stat.S_IMODE(provisioning_metadata.st_mode) == 0o700
        assert "OPENAI_API_KEY" not in launch["env"]
        assert launch["env"]["HTTPS_PROXY"] == "http://egress-proxy.internal:8443"
        assert launch["env"]["NO_PROXY"] == "metadata.internal"

    assert captured[0]["cwd"] != captured[1]["cwd"]
    assert captured[0]["config"]["token"] != captured[1]["config"]["token"]
    assert captured[0]["venv_path"] == captured[1]["venv_path"]
    assert (
        captured[0]["env"]["KESTREL_ISOLATED_FEATURE_DATA_DIR"]
        != captured[1]["env"]["KESTREL_ISOLATED_FEATURE_DATA_DIR"]
    )

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, os, tempfile; "
                "print(json.dumps({'cwd': os.getcwd(), 'home': os.environ['HOME'], "
                "'tmp': tempfile.gettempdir(), "
                "'secret': os.environ.get('OPENAI_API_KEY')}))"
            ),
        ],
        cwd=captured[0]["cwd"],
        env=captured[0]["env"],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(probe.stdout)
    assert observed == {
        "cwd": captured[0]["cwd"],
        "home": captured[0]["env"]["HOME"],
        "tmp": captured[0]["env"]["TMPDIR"],
        "secret": None,
    }


@pytest.mark.parametrize(
    "legacy_key",
    [
        "KESTREL_FEATURE_TESTFEATURE_TOKEN",
        "KESTREL_TELEGRAM_TOKEN",
        "TELEGRAM_BOT_TOKEN",
        "KESTREL_WHATSAPP_PROVIDER",
        "KESTREL_WHATSAPP_SESSION_DB",
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_WHATSAPP_FROM",
    ],
)
def test_hosted_child_fails_loudly_for_unscoped_legacy_environment(
    monkeypatch, tmp_path, legacy_key
):
    secret = "must-not-appear-in-error"
    monkeypatch.setenv(legacy_key, secret)
    if "TELEGRAM" in legacy_key:
        runtime = InstalledFeatureRuntime(
            class_name="TelegramFeature",
            entry_point="telegram.feature:TelegramFeature",
            distribution="kestrel-channel-telegram",
            runtime="isolated-venv",
            service="telegram-service",
        )
    elif "WHATSAPP" in legacy_key or "TWILIO" in legacy_key:
        runtime = _isolated_runtime()
    else:
        runtime = _cfg_runtime()
    with pytest.raises(IsolatedRuntimeConfigurationError) as failure:
        ProxyFeature(
            _hosted_postgres_agent(tmp_path / "hosted-runtime", "agent-safe"),
            runtime,
            client_factory=FakeIsolatedClient,
        )

    assert legacy_key in str(failure.value)
    assert secret not in str(failure.value)


def test_hosted_child_keeps_host_side_bin_override_out_of_child_env(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/operator/test-service")
    feature = ProxyFeature(
        _hosted_postgres_agent(tmp_path / "hosted-runtime", "agent-safe"),
        _cfg_runtime(),
        client_factory=FakeIsolatedClient,
    )
    feature._venv_path = tmp_path / "prebuilt" / ".venv"
    feature._prepare_runtime_workspace()

    client = feature._build_client()

    assert "KESTREL_FEATURE_TESTFEATURE_BIN" not in client.kwargs["env"]


@pytest.mark.asyncio
async def test_hosted_workspace_survives_failed_start_and_restarts_in_same_scope(
    monkeypatch, tmp_path
):
    """Cleanup retires the failed child without deleting durable agent state."""

    executable = tmp_path / "operator" / "wa-service"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv(
        "KESTREL_FEATURE_WHATSAPPFEATURE_BIN",
        str(executable),
    )
    agent = _hosted_postgres_agent(
        tmp_path / "hosted-runtime",
        "tenant/agent",
    )
    failed_clients = []

    class FailingStartClient(FakeIsolatedClient):
        async def start(self):
            raise RuntimeError("synthetic start failure")

    def failing_factory(**kwargs):
        client = FailingStartClient(**kwargs)
        failed_clients.append(client)
        return client

    failed = ProxyFeature(agent, _isolated_runtime(), client_factory=failing_factory)
    workspace = failed._feature_runtime_dir()
    with pytest.raises(IsolatedRuntimePreparationError) as failure:
        await failed.initialize()

    assert "synthetic start failure" not in str(failure.value)
    assert isinstance(failure.value.__cause__, RuntimeError)
    assert failed._client is None
    assert failed_clients[0].stopped is True
    assert workspace.is_dir()

    restarted_clients = []

    def restart_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        restarted_clients.append(client)
        return client

    restarted = ProxyFeature(agent, _isolated_runtime(), client_factory=restart_factory)
    await restarted.initialize()
    try:
        assert restarted._feature_runtime_dir() == workspace
        assert restarted_clients[0].kwargs["cwd"] == str(workspace / "work")
    finally:
        await restarted.shutdown()

    assert restarted_clients[0].stopped is True
    assert workspace.is_dir()


@pytest.mark.asyncio
async def test_supervision_restarts_child_on_wedged_health_probe(tmp_path):
    """F013: a health() that never returns must not wedge supervision forever —
    the probe is bounded and a timeout drops through to stop()/start()."""
    import asyncio

    import kestrel_sovereign.features.isolated_runtime as ir

    class WedgedThenHealthyClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.health_calls = 0
            self.starts = 0

        async def start(self):
            self.starts += 1
            await super().start()

        async def health(self):
            self.health_calls += 1
            if self.health_calls == 1:
                # First probe hangs past the bound → treated as wedged.
                await asyncio.sleep(3600)
            return True

    runtime = InstalledFeatureRuntime(
        class_name="WedgeFeature",
        entry_point="w.feature:WedgeFeature",
        distribution="w-pkg",
        runtime="isolated-venv",
        service="w",
    )
    client_holder = {}

    def factory(**kw):
        client_holder["c"] = WedgedThenHealthyClient(**kw)
        return client_holder["c"]

    feature = ProxyFeature(Mock(storage_path=str(tmp_path / "a" / "db.db"), features={}),
                           runtime, client_factory=factory)
    import os
    os.environ["KESTREL_FEATURE_WEDGEFEATURE_BIN"] = str(tmp_path / "w-bin")
    # Shrink the probe bound and backoff so the test is fast.
    monkey = pytest.MonkeyPatch()
    monkey.setattr(ir, "_HEALTH_PROBE_TIMEOUT", 0.05)
    try:
        await feature.initialize()
        client = client_holder["c"]
        # Wait for: first (hanging) probe to time out, then the restart path.
        snapshot = feature.runtime_telemetry_snapshot()
        for _ in range(200):
            await asyncio.sleep(0.02)
            snapshot = feature.runtime_telemetry_snapshot()
            if client.starts >= 2 and snapshot.restart_count == 1:
                break
        assert client.stopped is True
        assert client.starts >= 2  # child was restarted after the wedged probe
        assert snapshot.restart_count == 1
        assert snapshot.idle_wake_count == 0
    finally:
        feature._stopping = True
        if feature._supervision_task:
            feature._supervision_task.cancel()
            try:
                await feature._supervision_task
            except asyncio.CancelledError:
                pass
        monkey.undo()
        os.environ.pop("KESTREL_FEATURE_WEDGEFEATURE_BIN", None)


def test_ensure_venv_reprovisions_when_host_sdk_upgrades(tmp_path, monkeypatch):
    """F019: a provisioned venv whose stamped host SDK version no longer matches
    the running host is reinstalled (--upgrade), not silently reused."""
    import json

    import kestrel_sovereign.features.isolated_runtime as ir

    runtime = InstalledFeatureRuntime(
        class_name="StampFeature",
        entry_point="s.feature:StampFeature",
        distribution="stamp-pkg",
        runtime="isolated-venv",
        service="s",
        project="stamp-pkg",
    )
    feature = ProxyFeature(Mock(storage_path=str(tmp_path / "a" / "db.db")),
                           runtime, client_factory=FakeIsolatedClient)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()

    runs = []

    def fake_run(cmd):
        runs.append(cmd)
        _materialize_fake_provisioned_venv(feature)

    monkeypatch.setattr(feature, "_run", fake_run)
    # Child venv python is a stub (empty file) — report a concrete SDK version
    # rather than shelling out to it.
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _p: "0.28.0")
    monkeypatch.setattr(
        ir, "_feature_distribution_version", lambda _distribution, _target: "1.0.0"
    )
    monkeypatch.setattr(
        ir, "_venv_feature_distribution_probe", lambda _path, _distribution: _child_distribution_probe("1.0.0")
    )

    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.28.0")
    feature.ensure_venv()  # fresh: uv venv + uv pip install (no --upgrade)
    assert any(c[:3] == ["uv", "venv"] or c[0] == "uv" and "venv" in c for c in runs)
    install_cmds = [c for c in runs if "pip" in c and "install" in c]
    assert install_cmds and "--upgrade" not in install_cmds[-1]
    manifest = json.loads((feature._venv_path / ".kestrel_provision.json").read_text())
    assert manifest["provisioned_against_host_sdk"] == "0.28.0"
    assert manifest["child_sdk_version"] == "0.28.0"
    assert manifest["feature_distribution_version"] == "1.0.0"
    assert manifest["child_feature_distribution_state"] == "versioned"
    assert manifest["child_feature_distribution_version"] == "1.0.0"

    runs.clear()
    # Same host SDK → no reprovision.
    feature.ensure_venv()
    assert runs == []

    # Host upgraded → reprovision with --upgrade, stamped against the new host.
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.29.0")
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _p: "0.29.0")
    feature.ensure_venv()
    upgrades = [c for c in runs if "pip" in c and "install" in c]
    assert upgrades and "--upgrade" in upgrades[-1]
    manifest = json.loads((feature._venv_path / ".kestrel_provision.json").read_text())
    assert manifest["provisioned_against_host_sdk"] == "0.29.0"

    runs.clear()
    # A feature that pins an OLD sdk installs "successfully" but the child stays
    # behind: we still stamp against the new host so we don't reinstall every
    # startup, but the mismatch is recorded (and warned) rather than hidden.
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.30.0")
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _p: "0.28.0")
    feature.ensure_venv()
    manifest = json.loads((feature._venv_path / ".kestrel_provision.json").read_text())
    assert manifest["provisioned_against_host_sdk"] == "0.30.0"
    assert manifest["child_sdk_version"] == "0.28.0"
    runs.clear()
    feature.ensure_venv()  # not stale now (host unchanged) → no thrash
    assert runs == []


@pytest.mark.parametrize("existing", (False, True), ids=("fresh", "upgrade"))
def test_every_provision_refuses_to_stamp_missing_console_service(
    tmp_path,
    monkeypatch,
    existing,
):
    runtime = InstalledFeatureRuntime(
        class_name="MissingConsoleFeature",
        entry_point="missing.feature:MissingConsoleFeature",
        distribution="missing-console-package",
        runtime="isolated-venv",
        service="missing-console-service",
        project="missing-console-package",
    )
    feature = ProxyFeature(
        Mock(storage_path=str(tmp_path / "agent" / "kestrel_prime.db")),
        runtime,
        client_factory=FakeIsolatedClient,
    )
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    manifest_path = feature._provision_manifest_path()
    if existing:
        python = isolated_runtime._venv_python(feature._venv_path)
        python.parent.mkdir(parents=True)
        python.touch()
        feature._write_provision_manifest(
            runtime.project,
            "old-host-sdk",
            "old-host-sdk",
            "1.0.0",
            _child_distribution_probe("1.0.0"),
        )
        original_manifest = manifest_path.read_bytes()
    else:
        original_manifest = None

    def provision_without_console(_command):
        python = isolated_runtime._venv_python(feature._venv_path)
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()

    feature._run = provision_without_console
    feature._probe_sdk_version = Mock(return_value="current-host-sdk")
    feature._probe_feature_distribution = Mock(
        return_value=_child_distribution_probe("1.0.0")
    )
    monkeypatch.setattr(
        isolated_runtime,
        "_host_sdk_version",
        lambda: "current-host-sdk",
    )
    monkeypatch.setattr(
        isolated_runtime,
        "_feature_distribution_version",
        lambda _distribution, _target: "1.0.0",
    )

    with pytest.raises(
        IsolatedRuntimePreparationError,
        match="launch artifact could not be verified",
    ):
        feature.ensure_venv()

    if original_manifest is None:
        assert not manifest_path.exists()
    else:
        assert manifest_path.read_bytes() == original_manifest


def test_current_manifest_missing_console_forces_verified_reinstall(
    tmp_path,
    monkeypatch,
):
    """A deleted wrapper cannot make a manifest-current venv look fresh."""

    runtime = InstalledFeatureRuntime(
        class_name="CurrentMissingConsoleFeature",
        entry_point="missing.feature:CurrentMissingConsoleFeature",
        distribution="missing-console-package",
        runtime="isolated-venv",
        service="missing-console-service",
        project="missing-console-package",
    )
    feature = ProxyFeature(
        Mock(storage_path=str(tmp_path / "agent" / "kestrel_prime.db")),
        runtime,
        client_factory=FakeIsolatedClient,
    )
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    _stamp_current_fake_venv(feature, monkeypatch)
    wrapper = isolated_runtime._console_script_path(
        feature._venv_path,
        runtime.service,
    )
    wrapper.unlink()
    runs = []

    def repair_console(command):
        runs.append(command)
        _materialize_fake_provisioned_venv(feature)

    feature._run = repair_console
    feature.ensure_venv()

    assert len(runs) == 1
    assert "--reinstall" in runs[0]
    assert feature._console_script_location_state() == "current"
    feature._verify_launch_artifact()

    runs.clear()
    feature.ensure_venv()
    assert runs == []


def test_current_manifest_missing_console_failed_repair_is_not_reblessed(
    tmp_path,
    monkeypatch,
):
    """Failed forced repair preserves the last truthful manifest."""

    runtime = InstalledFeatureRuntime(
        class_name="FailedConsoleRepairFeature",
        entry_point="missing.feature:FailedConsoleRepairFeature",
        distribution="missing-console-package",
        runtime="isolated-venv",
        service="missing-console-service",
        project="missing-console-package",
    )
    feature = ProxyFeature(
        Mock(storage_path=str(tmp_path / "agent" / "kestrel_prime.db")),
        runtime,
        client_factory=FakeIsolatedClient,
    )
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    _stamp_current_fake_venv(feature, monkeypatch)
    manifest_path = feature._provision_manifest_path()
    original_manifest = manifest_path.read_bytes()
    isolated_runtime._console_script_path(
        feature._venv_path,
        runtime.service,
    ).unlink()
    runs = []
    feature._run = lambda command: runs.append(command)

    with pytest.raises(
        IsolatedRuntimePreparationError,
        match="launch artifact could not be verified",
    ):
        feature.ensure_venv()

    assert len(runs) == 1
    assert "--reinstall" in runs[0]
    assert manifest_path.read_bytes() == original_manifest


def test_degenerate_console_shebang_is_typed_repair_failure(
    tmp_path,
    monkeypatch,
):
    """An empty interpreter never escapes as IndexError or gets stamped."""

    runtime = InstalledFeatureRuntime(
        class_name="DegenerateShebangFeature",
        entry_point="degenerate.feature:DegenerateShebangFeature",
        distribution="degenerate-console-package",
        runtime="isolated-venv",
        service="degenerate-console-service",
        project="degenerate-console-package",
    )
    feature = ProxyFeature(
        Mock(storage_path=str(tmp_path / "agent" / "kestrel_prime.db")),
        runtime,
        client_factory=FakeIsolatedClient,
    )
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    _stamp_current_fake_venv(feature, monkeypatch)
    manifest_path = feature._provision_manifest_path()
    original_manifest = manifest_path.read_bytes()
    wrapper = isolated_runtime._console_script_path(
        feature._venv_path,
        runtime.service,
    )
    wrapper.write_bytes(b"#! \t\nprint('never launched')\n")
    runs = []
    feature._run = lambda command: runs.append(command)

    assert feature._console_script_location_state() == "missing"
    with pytest.raises(
        IsolatedRuntimePreparationError,
        match="launch artifact could not be verified",
    ):
        feature.ensure_venv()

    assert len(runs) == 1
    assert "--reinstall" in runs[0]
    assert manifest_path.read_bytes() == original_manifest


def test_ensure_venv_reprovisions_when_telegram_distribution_upgrades(tmp_path, monkeypatch):
    """An unversioned Telegram service target still follows its distribution release."""
    import kestrel_sovereign.features.isolated_runtime as ir

    runtime = InstalledFeatureRuntime(
        class_name="TelegramFeature",
        entry_point="kestrel_channel_telegram.feature:TelegramFeature",
        distribution="kestrel-channel-telegram",
        runtime="isolated-venv",
        service="kestrel-telegram-service",
        project="kestrel-channel-telegram[service]",
    )
    feature = ProxyFeature(
        Mock(storage_path=str(tmp_path / "agent" / "kestrel_prime.db")),
        runtime,
        client_factory=FakeIsolatedClient,
    )
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    runs = []

    def fake_run(cmd):
        runs.append(cmd)
        _materialize_fake_provisioned_venv(feature)

    version = {"value": "0.1.1"}
    monkeypatch.setattr(feature, "_run", fake_run)
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.35.1")
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _path: "0.35.1")
    monkeypatch.setattr(
        ir,
        "_feature_distribution_version",
        lambda distribution, target: version["value"],
    )
    monkeypatch.setattr(
        ir,
        "_venv_feature_distribution_probe",
        lambda _path, _distribution: _child_distribution_probe(version["value"]),
    )

    feature.ensure_venv()
    runs.clear()
    feature.ensure_venv()
    assert runs == [], "an unchanged Telegram release must remain fresh"

    version["value"] = "0.1.2"
    feature.ensure_venv()
    installs = [cmd for cmd in runs if "pip" in cmd and "install" in cmd]
    assert installs and "--upgrade" in installs[-1]

    runs.clear()
    feature.ensure_venv()
    assert runs == [], "the new Telegram release stamp must converge after one upgrade"


@pytest.mark.parametrize("current_child_version", ("0.1.1", "missing"))
def test_ensure_venv_reprovisions_when_installed_child_no_longer_matches_manifest(
    tmp_path, monkeypatch, current_child_version
):
    """Freshness probes the child, not just the recorded successful install."""

    import kestrel_sovereign.features.isolated_runtime as ir

    runtime = InstalledFeatureRuntime(
        class_name="TelegramFeature",
        entry_point="kestrel_channel_telegram.feature:TelegramFeature",
        distribution="kestrel-channel-telegram",
        runtime="isolated-venv",
        service="kestrel-telegram-service",
        project="kestrel-channel-telegram[service]",
    )
    feature = ProxyFeature(
        Mock(storage_path=str(tmp_path / "agent" / "kestrel_prime.db")),
        runtime,
        client_factory=FakeIsolatedClient,
    )
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    _materialize_fake_provisioned_venv(feature)
    feature._write_provision_manifest(
        runtime.project,
        "0.35.1",
        "0.35.1",
        "0.1.2",
        _child_distribution_probe("0.1.2"),
    )
    child_version = {"value": current_child_version}
    runs = []

    def fake_run(cmd):
        runs.append(cmd)
        if "pip" in cmd and "install" in cmd:
            child_version["value"] = "0.1.2"
            _materialize_fake_provisioned_venv(feature)

    monkeypatch.setattr(feature, "_run", fake_run)
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.35.1")
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _path: "0.35.1")
    monkeypatch.setattr(ir, "_feature_distribution_version", lambda _d, _t: "0.1.2")
    monkeypatch.setattr(
        ir,
        "_venv_feature_distribution_probe",
        lambda _path, _distribution: _child_distribution_probe(child_version["value"]),
    )

    feature.ensure_venv()

    installs = [cmd for cmd in runs if "pip" in cmd and "install" in cmd]
    assert installs and "--upgrade" in installs[-1]
    assert child_version["value"] == "0.1.2"

    runs.clear()
    feature.ensure_venv()
    assert runs == []


def test_feature_distribution_version_reads_local_editable_project_metadata(
    tmp_path, monkeypatch
):
    """Editable source metadata wins over its stale installed dist-info stamp."""
    import kestrel_sovereign.features.isolated_runtime as ir

    project = tmp_path / "telegram"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        "[project]\nname = 'kestrel-channel-telegram'\nversion = '0.1.2'\n"
    )
    monkeypatch.setattr(ir.importlib_metadata, "version", lambda _name: "0.1.1")

    assert (
        ir._feature_distribution_version(
            "kestrel-channel-telegram", f"-e {project}[service]"
        )
        == "0.1.2"
    )


def test_ensure_venv_local_editable_version_stamp_converges(tmp_path, monkeypatch):
    """A local editable source upgrade reprovisions once, then stays fresh."""
    import kestrel_sovereign.features.isolated_runtime as ir

    project = tmp_path / "telegram"
    project.mkdir()
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        "[project]\nname = 'kestrel-channel-telegram'\nversion = '0.1.1'\n"
    )
    runtime = InstalledFeatureRuntime(
        class_name="TelegramFeature",
        entry_point="kestrel_channel_telegram.feature:TelegramFeature",
        distribution="kestrel-channel-telegram",
        runtime="isolated-venv",
        service="kestrel-telegram-service",
        project=f"-e {project}[service]",
    )
    feature = ProxyFeature(
        Mock(storage_path=str(tmp_path / "agent" / "kestrel_prime.db")),
        runtime,
        client_factory=FakeIsolatedClient,
    )
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    runs = []

    import tomllib

    child_version = {"value": "0.1.1"}

    def fake_run(cmd):
        runs.append(cmd)
        _materialize_fake_provisioned_venv(feature)
        if "pip" in cmd and "install" in cmd:
            child_version["value"] = tomllib.loads(pyproject.read_text())["project"][
                "version"
            ]

    monkeypatch.setattr(feature, "_run", fake_run)
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.35.1")
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _path: "0.35.1")
    # A real editable installation can retain this old dist-info version until
    # it is reinstalled; the local source stamp must still see the upgrade.
    monkeypatch.setattr(ir.importlib_metadata, "version", lambda _name: "0.1.1")
    monkeypatch.setattr(
        ir,
        "_venv_feature_distribution_probe",
        lambda _path, _distribution: _child_distribution_probe(child_version["value"]),
    )

    feature.ensure_venv()
    runs.clear()
    feature.ensure_venv()
    assert runs == []

    pyproject.write_text(
        "[project]\nname = 'kestrel-channel-telegram'\nversion = '0.1.2'\n"
    )
    feature.ensure_venv()
    installs = [cmd for cmd in runs if "pip" in cmd and "install" in cmd]
    assert installs and "--upgrade" in installs[-1]

    runs.clear()
    feature.ensure_venv()
    assert runs == []


def test_ensure_venv_does_not_mutate_operator_override_venv(tmp_path, monkeypatch):
    """#2125 regression: an operator-supplied (override) venv that already exists
    must NOT be `uv pip install --upgrade`'d — that rewrites a prebuilt/pinned
    environment the operator provided (and hard-fails the feature at startup if
    the index is unreachable). ensure_venv verifies + warns but leaves it
    untouched and stamps no manifest we don't own."""
    import kestrel_sovereign.features.isolated_runtime as ir

    override_venv = tmp_path / "prebuilt" / ".venv"
    py = override_venv / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.touch()  # a venv that ALREADY exists (operator built it)
    wrapper = override_venv / "bin" / "o"
    wrapper.write_text(f"#!{py}\nexit 0\n")
    wrapper.chmod(0o700)

    runtime = InstalledFeatureRuntime(
        class_name="OverrideFeature",
        entry_point="o.feature:OverrideFeature",
        distribution="override-pkg",
        runtime="isolated-venv",
        service="o",
        project="override-pkg",
        venv=str(override_venv),  # pyproject `venv =` override
    )
    feature = ProxyFeature(Mock(storage_path=str(tmp_path / "a" / "db.db")),
                           runtime, client_factory=FakeIsolatedClient)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    assert feature._venv_path == override_venv.resolve()

    runs = []
    monkeypatch.setattr(feature, "_run", lambda cmd: runs.append(cmd))
    # SDK mismatch present — must warn, but still NOT mutate.
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _p: "0.28.0")
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.29.0")
    monkeypatch.setattr(ir, "_feature_distribution_version", lambda _d, _t: "1.2.0")
    monkeypatch.setattr(
        ir, "_venv_feature_distribution_probe", lambda _p, _d: _child_distribution_probe("1.2.0")
    )

    feature.ensure_venv()

    assert runs == [], f"override venv must not be touched, ran: {runs}"
    assert not (override_venv / ".kestrel_provision.json").exists()


@pytest.mark.parametrize(
    ("desired_version", "child_version"),
    (
        ("1.2.0", "1.1.9"),
        ("1.2.0", "missing"),
        ("unknown", "missing"),
        ("unknown", "probe-failed"),
    ),
    ids=("stale", "missing", "unknown-desired-missing", "probe-failed"),
)
def test_prebuilt_override_refuses_stale_missing_or_unverifiable_distribution(
    tmp_path, monkeypatch, desired_version, child_version
):
    """An immutable override must prove that its child distribution is usable."""

    import kestrel_sovereign.features.isolated_runtime as ir

    override_venv = tmp_path / "prebuilt" / ".venv"
    python = override_venv / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    wrapper = override_venv / "bin" / "o"
    wrapper.write_text(f"#!{python}\nexit 0\n")
    wrapper.chmod(0o700)
    runtime = InstalledFeatureRuntime(
        class_name="OverrideFeature",
        entry_point="o.feature:OverrideFeature",
        distribution="override-pkg",
        runtime="isolated-venv",
        service="o",
        project="override-pkg",
        venv=str(override_venv),
    )
    feature = ProxyFeature(
        Mock(storage_path=str(tmp_path / "a" / "db.db")),
        runtime,
        client_factory=FakeIsolatedClient,
    )
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    runs = []
    monkeypatch.setattr(feature, "_run", lambda cmd: runs.append(cmd))
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.35.1")
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _p: "0.35.1")
    monkeypatch.setattr(
        ir, "_feature_distribution_version", lambda _d, _t: desired_version
    )
    monkeypatch.setattr(
        ir, "_venv_feature_distribution_probe", lambda _p, _d: _child_distribution_probe(child_version)
    )

    with pytest.raises(RuntimeError, match="refusing to run an unverifiable override venv"):
        feature.ensure_venv()

    assert runs == []
    assert not (override_venv / ".kestrel_provision.json").exists()


def test_prebuilt_editable_override_probes_source_release_without_mutation(tmp_path, monkeypatch):
    """A local editable desired release is checked against the child metadata."""

    import kestrel_sovereign.features.isolated_runtime as ir

    project = tmp_path / "telegram"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        "[project]\nname = 'override-pkg'\nversion = '1.2.0'\n"
    )
    override_venv = tmp_path / "prebuilt" / ".venv"
    python = override_venv / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    wrapper = override_venv / "bin" / "o"
    wrapper.write_text(f"#!{python}\nexit 0\n")
    wrapper.chmod(0o700)
    runtime = InstalledFeatureRuntime(
        class_name="OverrideFeature",
        entry_point="o.feature:OverrideFeature",
        distribution="override-pkg",
        runtime="isolated-venv",
        service="o",
        project=f"-e {project}[service]",
        venv=str(override_venv),
    )
    feature = ProxyFeature(
        Mock(storage_path=str(tmp_path / "a" / "db.db")),
        runtime,
        client_factory=FakeIsolatedClient,
    )
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    runs = []
    monkeypatch.setattr(feature, "_run", lambda cmd: runs.append(cmd))
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.35.1")
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _p: "0.35.1")
    monkeypatch.setattr(
        ir, "_venv_feature_distribution_probe", lambda _p, _d: _child_distribution_probe("1.2.0")
    )

    feature.ensure_venv()

    assert runs == []
    assert not (override_venv / ".kestrel_provision.json").exists()


def test_prebuilt_override_accepts_positively_present_versionless_child_for_unknown_desired(
    tmp_path, monkeypatch
):
    """A genuine editable/versionless child is usable when desired is unknown."""

    import kestrel_sovereign.features.isolated_runtime as ir

    override_venv = tmp_path / "prebuilt" / ".venv"
    python = override_venv / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    wrapper = override_venv / "bin" / "o"
    wrapper.write_text(f"#!{python}\nexit 0\n")
    wrapper.chmod(0o700)
    runtime = InstalledFeatureRuntime(
        class_name="OverrideFeature",
        entry_point="o.feature:OverrideFeature",
        distribution="override-pkg",
        runtime="isolated-venv",
        service="o",
        project="override-pkg",
        venv=str(override_venv),
    )
    feature = ProxyFeature(
        Mock(storage_path=str(tmp_path / "a" / "db.db")),
        runtime,
        client_factory=FakeIsolatedClient,
    )
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    runs = []
    monkeypatch.setattr(feature, "_run", lambda cmd: runs.append(cmd))
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.35.1")
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _p: "0.35.1")
    monkeypatch.setattr(ir, "_feature_distribution_version", lambda _d, _t: "unknown")
    monkeypatch.setattr(
        ir, "_venv_feature_distribution_probe", lambda _p, _d: _child_distribution_probe("unknown")
    )

    feature.ensure_venv()
    feature.ensure_venv()

    assert runs == []
    assert not (override_venv / ".kestrel_provision.json").exists()


def test_ensure_venv_reprovisions_host_created_override_venv(tmp_path, monkeypatch):
    """An override venv that KESTREL created earlier (carries our manifest) keeps
    the reprovision lifecycle — a host SDK upgrade still triggers --upgrade. Only
    prebuilt override venvs (no manifest) are left untouched (#2125)."""
    import kestrel_sovereign.features.isolated_runtime as ir

    override_venv = tmp_path / "hostbuilt" / ".venv"
    runtime = InstalledFeatureRuntime(
        class_name="OverrideFeature", entry_point="o.feature:OverrideFeature",
        distribution="override-pkg", runtime="isolated-venv", service="o",
        project="override-pkg", venv=str(override_venv),
    )
    feature = ProxyFeature(Mock(storage_path=str(tmp_path / "a" / "db.db")),
                           runtime, client_factory=FakeIsolatedClient)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()

    runs = []

    def fake_run(cmd):
        runs.append(cmd)
        _materialize_fake_provisioned_venv(feature)

    monkeypatch.setattr(feature, "_run", fake_run)
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _p: "0.28.0")
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.28.0")
    monkeypatch.setattr(ir, "_feature_distribution_version", lambda _d, _t: "unknown")
    monkeypatch.setattr(
        ir, "_venv_feature_distribution_probe", lambda _p, _d: _child_distribution_probe("unknown")
    )

    # First startup: override path missing → bootstrap (create + install + stamp).
    feature.ensure_venv()
    assert any("install" in c for c in runs)
    assert (override_venv / ".kestrel_provision.json").exists()

    runs.clear()
    feature.ensure_venv()  # unchanged host → no thrash even though it's an override
    assert runs == []

    # Host SDK upgrade → OUR override venv reprovisions (--upgrade).
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.29.0")
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _p: "0.29.0")
    feature.ensure_venv()
    upgrades = [c for c in runs if "pip" in c and "install" in c]
    assert upgrades and "--upgrade" in upgrades[-1]


def test_ensure_venv_refuses_to_stamp_an_older_child_feature_distribution(
    tmp_path, monkeypatch
):
    """A successful resolver run cannot mark an obsolete child package fresh."""

    import kestrel_sovereign.features.isolated_runtime as ir

    runtime = InstalledFeatureRuntime(
        class_name="TelegramFeature",
        entry_point="kestrel_channel_telegram.feature:TelegramFeature",
        distribution="kestrel-channel-telegram",
        runtime="isolated-venv",
        service="kestrel-telegram-service",
        project="kestrel-channel-telegram[service]",
    )
    feature = ProxyFeature(
        Mock(storage_path=str(tmp_path / "agent" / "kestrel_prime.db")),
        runtime,
        client_factory=FakeIsolatedClient,
    )
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    runs = []

    def fake_run(cmd):
        runs.append(cmd)
        _materialize_fake_provisioned_venv(feature)

    monkeypatch.setattr(feature, "_run", fake_run)
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.35.1")
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _path: "0.35.1")
    monkeypatch.setattr(ir, "_feature_distribution_version", lambda _d, _t: "0.2.0")
    monkeypatch.setattr(
        ir, "_venv_feature_distribution_probe", lambda _path, _distribution: _child_distribution_probe("0.1.9")
    )

    with pytest.raises(
        IsolatedRuntimePreparationError,
        match="did not match the host release",
    ):
        feature.ensure_venv()

    assert not (feature._venv_path / ".kestrel_provision.json").exists()
    installs = [cmd for cmd in runs if "pip" in cmd and "install" in cmd]
    assert len(installs) == 1 and "--upgrade" not in installs[0]


def test_venv_feature_distribution_probe_executes_in_target_interpreter():
    """The generated isolated probe contains executable newlines."""

    assert isolated_runtime._venv_feature_distribution_probe(
        Path(sys.executable), "pytest"
    ) == isolated_runtime._FeatureDistributionProbe.versioned(
        importlib_metadata.version("pytest")
    )


def test_venv_freshness_probes_ignore_hostile_cwd_modules(
    monkeypatch,
    tmp_path,
):
    """Cwd modules cannot forge feature or SDK freshness observations."""

    hostile_cwd = tmp_path / "hostile-cwd"
    hostile_cwd.mkdir()
    json_marker = hostile_cwd / "json-shadow-imported"
    sdk_marker = hostile_cwd / "sdk-shadow-imported"
    (hostile_cwd / "json.py").write_text(
        f"from pathlib import Path\nPath({str(json_marker)!r}).write_text('bad')\n"
        "raise RuntimeError('cwd json shadow')\n"
    )
    (hostile_cwd / "kestrel_sdk.py").write_text(
        f"from pathlib import Path\nPath({str(sdk_marker)!r}).write_text('bad')\n"
        "__version__ = 'forged-sdk-version'\n"
    )
    monkeypatch.chdir(hostile_cwd)

    feature_probe = isolated_runtime._venv_feature_distribution_probe(
        Path(sys.executable),
        "pytest",
    )
    sdk_version = isolated_runtime._venv_sdk_version(Path(sys.executable))

    assert feature_probe == isolated_runtime._FeatureDistributionProbe.versioned(
        importlib_metadata.version("pytest")
    )
    assert sdk_version == isolated_runtime._host_sdk_version()
    assert not json_marker.exists()
    assert not sdk_marker.exists()


@pytest.mark.parametrize(
    ("stdout", "expected"),
    (
        ('{"state": "missing"}', isolated_runtime._FeatureDistributionProbe.missing()),
        (
            '{"state": "present-unversioned"}',
            isolated_runtime._FeatureDistributionProbe.present_unversioned(),
        ),
        (
            '{"state": "versioned", "version": "1.2.3"}',
            isolated_runtime._FeatureDistributionProbe.versioned("1.2.3"),
        ),
    ),
)
def test_venv_feature_distribution_probe_preserves_distinct_positive_states(
    monkeypatch, tmp_path, stdout, expected
):
    """The child probe never reduces missing and versionless to ``unknown``."""

    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess([], 0, stdout=stdout)

    monkeypatch.setattr(isolated_runtime.subprocess, "run", fake_run)
    assert (
        isolated_runtime._venv_feature_distribution_probe(
            tmp_path / "venv" / "bin" / "python", "example-pkg"
        )
        == expected
    )
    assert captured["command"][1:3] == ["-P", "-B"]


def test_venv_feature_distribution_probe_marks_execution_failure_unverifiable(
    monkeypatch, tmp_path
):
    """A failed child interpreter probe is not evidence of an editable install."""

    def fail_run(*_args, **_kwargs):
        raise OSError("child unavailable")

    monkeypatch.setattr(isolated_runtime.subprocess, "run", fail_run)
    assert (
        isolated_runtime._venv_feature_distribution_probe(
            tmp_path / "venv" / "bin" / "python", "example-pkg"
        )
        == isolated_runtime._FeatureDistributionProbe.failed()
    )


@pytest.mark.parametrize("probe_name", ("distribution", "sdk"))
def test_venv_freshness_probes_timeout_to_fail_closed_outcomes(
    monkeypatch,
    tmp_path,
    probe_name,
):
    captured = {}

    def time_out(command, **kwargs):
        captured.update(kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(isolated_runtime.subprocess, "run", time_out)
    python = tmp_path / "venv" / "bin" / "python"
    if probe_name == "distribution":
        outcome = isolated_runtime._venv_feature_distribution_probe(
            python,
            "example-pkg",
        )
        assert outcome == isolated_runtime._FeatureDistributionProbe.failed()
    else:
        assert isolated_runtime._venv_sdk_version(python) == "unknown"

    assert captured["timeout"] == isolated_runtime._FRESHNESS_PROBE_TIMEOUT_S
    assert 0 < captured["timeout"] <= 30


def test_ensure_venv_unknown_feature_versions_stamp_once_without_reinstall_loop(
    tmp_path, monkeypatch
):
    """Unobservable host/child metadata is stable rather than permanently stale."""

    import kestrel_sovereign.features.isolated_runtime as ir

    runtime = InstalledFeatureRuntime(
        class_name="UnknownFeature",
        entry_point="unknown.feature:UnknownFeature",
        distribution="unknown-package",
        runtime="isolated-venv",
        service="unknown-service",
        project="unknown-package",
    )
    feature = ProxyFeature(
        Mock(storage_path=str(tmp_path / "agent" / "kestrel_prime.db")),
        runtime,
        client_factory=FakeIsolatedClient,
    )
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    runs = []

    def fake_run(cmd):
        runs.append(cmd)
        _materialize_fake_provisioned_venv(feature)

    monkeypatch.setattr(feature, "_run", fake_run)
    monkeypatch.setattr(ir, "_host_sdk_version", lambda: "0.35.1")
    monkeypatch.setattr(ir, "_venv_sdk_version", lambda _path: "0.35.1")
    monkeypatch.setattr(ir, "_feature_distribution_version", lambda _d, _t: "unknown")
    monkeypatch.setattr(
        ir, "_venv_feature_distribution_probe", lambda _path, _distribution: _child_distribution_probe("unknown")
    )

    feature.ensure_venv()
    manifest = json.loads((feature._venv_path / ".kestrel_provision.json").read_text())
    assert manifest["feature_distribution_version"] == "unknown"
    assert manifest["child_feature_distribution_state"] == "present-unversioned"
    assert manifest["child_feature_distribution_version"] is None
    runs.clear()

    feature.ensure_venv()
    assert runs == []


# --- F023: isolated service launch env must not inherit interpreter shadowing --


def test_isolated_child_env_strips_shadowing_vars(monkeypatch, tmp_path):
    """The launch env must drop host PYTHONPATH/PYTHONHOME/PYTHONSTARTUP/
    VIRTUAL_ENV so the host interpreter can't shadow the isolated venv (F023)."""
    from kestrel_sovereign.features.isolated_runtime import (
        _isolated_child_env,
        _venv_bin_dir,
    )

    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/host/python")
    monkeypatch.setenv("PYTHONSTARTUP", "/host/startup.py")
    monkeypatch.setenv("VIRTUAL_ENV", "/host/venv")
    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_TOKEN", "keep-me")

    venv = tmp_path / "svc-venv"
    env = _isolated_child_env(venv)

    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert "PYTHONSTARTUP" not in env
    # Feature config/secrets still pass through (documented config channel).
    assert env["KESTREL_FEATURE_WHATSAPPFEATURE_TOKEN"] == "keep-me"
    # VIRTUAL_ENV re-points at the isolated venv and its bin leads PATH.
    assert env["VIRTUAL_ENV"] == str(venv)
    assert env["PATH"].split(os.pathsep)[0] == str(_venv_bin_dir(venv))


def test_build_client_passes_stripped_env(monkeypatch, tmp_path):
    """``_build_client`` must hand the stripped env to the client factory."""
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    captured = {}

    def client_factory(**kwargs):
        captured.update(kwargs)
        return FakeIsolatedClient(**kwargs)

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    feature = ProxyFeature(agent, _isolated_runtime(), client_factory=client_factory)
    feature._venv_path = tmp_path / "svc-venv"
    feature._bin_path = None

    feature._build_client()

    assert "env" in captured
    assert "PYTHONPATH" not in captured["env"]


@pytest.mark.parametrize(
    "runtime",
    (
        InstalledFeatureRuntime(
            class_name="GeneralFeature",
            entry_point="general.feature:GeneralFeature",
            distribution="kestrel-feature-general",
            runtime="isolated-venv",
            service="general-service",
        ),
        InstalledFeatureRuntime(
            class_name="TelegramFeature",
            entry_point="telegram.feature:TelegramFeature",
            distribution="kestrel-channel-telegram",
            runtime="isolated-venv",
            service="telegram-service",
        ),
    ),
)
def test_standalone_launch_preserves_legacy_home_temp_xdg_and_cwd(
    monkeypatch, tmp_path, runtime
):
    """Dependency isolation must not change standalone process semantics."""

    legacy_home = tmp_path / "legacy-home"
    legacy_tmp = tmp_path / "legacy-tmp"
    legacy_xdg = tmp_path / "legacy-xdg"
    monkeypatch.setenv("HOME", str(legacy_home))
    monkeypatch.setenv("TMPDIR", str(legacy_tmp))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(legacy_xdg))
    monkeypatch.setenv("KESTREL_FEATURE_DATA_DIR", str(tmp_path / "legacy-data"))
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    feature = ProxyFeature(agent, runtime, client_factory=FakeIsolatedClient)
    feature._venv_path = tmp_path / "prebuilt" / ".venv"

    client = feature._build_client()

    assert "cwd" not in client.kwargs
    assert client.kwargs["env"]["HOME"] == str(legacy_home)
    assert client.kwargs["env"]["TMPDIR"] == str(legacy_tmp)
    assert client.kwargs["env"]["XDG_CONFIG_HOME"] == str(legacy_xdg)
    assert client.kwargs["env"]["KESTREL_FEATURE_DATA_DIR"] == str(
        tmp_path / "legacy-data"
    )
    assert "KESTREL_ISOLATED_RUNTIME_DIR" not in client.kwargs["env"]


def test_venv_sdk_version_uses_canonical_isolated_child_env(monkeypatch, tmp_path):
    """The version probe cannot resolve the host SDK through hostile env vars."""
    import kestrel_sovereign.features.isolated_runtime as ir

    venv = tmp_path / "feature-venv"
    python = venv / "bin" / "python"
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/host/python")
    monkeypatch.setenv("PYTHONSTARTUP", "/host/startup.py")
    monkeypatch.setenv("VIRTUAL_ENV", "/host/venv")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return ir.subprocess.CompletedProcess([], 0, stdout="0.35.1\n")

    monkeypatch.setattr(ir.subprocess, "run", fake_run)

    assert ir._venv_sdk_version(python) == "0.35.1"
    assert captured["command"][1:3] == ["-P", "-B"]
    env = captured["env"]
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert "PYTHONSTARTUP" not in env
    assert env["VIRTUAL_ENV"] == str(venv)
    assert env["PATH"].split(os.pathsep)[0] == str(venv / "bin")


def test_hosted_venv_probes_receive_no_host_or_package_secrets(monkeypatch, tmp_path):
    import kestrel_sovereign.features.isolated_runtime as ir

    venv = tmp_path / "hosted-feature" / ".venv"
    python = venv / "bin" / "python"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "host-api-secret")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "host-channel-secret")
    monkeypatch.setenv("KESTREL_DB_PATH", "/host/tenant.db")
    monkeypatch.setenv("KESTREL_FEATURE_PROBEFEATURE_TOKEN", "host-feature-secret")
    monkeypatch.setenv("PIP_INDEX_URL", "https://user:password@index.example/simple")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((cmd, kwargs["env"]))
        stdout = (
            '{"state": "versioned", "version": "1.2.3"}\n'
            if "distribution =" in cmd[-1]
            else "0.35.1\n"
        )
        return ir.subprocess.CompletedProcess(cmd, 0, stdout=stdout)

    monkeypatch.setattr(ir.subprocess, "run", fake_run)

    assert ir._venv_sdk_version(python, hosted=True) == "0.35.1"
    assert ir._venv_feature_distribution_probe(
        python,
        "probe-package",
        hosted=True,
    ) == ir._FeatureDistributionProbe.versioned("1.2.3")
    assert len(captured) == 2
    for command, env in captured:
        assert command[1:3] == ["-P", "-B"]
        assert env["HTTP_PROXY"] == "http://proxy.example:8080"
        assert env["VIRTUAL_ENV"] == str(venv)
        assert env["PATH"].split(os.pathsep)[0] == str(venv / "bin")
        for secret in (
            "ANTHROPIC_API_KEY",
            "TELEGRAM_BOT_TOKEN",
            "KESTREL_DB_PATH",
            "KESTREL_FEATURE_PROBEFEATURE_TOKEN",
            "PIP_INDEX_URL",
        ):
            assert secret not in env


def test_hosted_uv_provisioning_inherits_only_explicit_package_authority(
    monkeypatch, tmp_path
):
    import kestrel_sovereign.features.isolated_runtime as ir

    feature = ProxyFeature(
        _hosted_postgres_agent(tmp_path / "runtime", "tenant/agent"),
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )
    feature._venv_path = feature._default_venv_path()
    feature._prepare_runtime_workspace()
    hostile_uv = feature._venv_path / "bin" / "uv"
    hostile_uv.parent.mkdir(parents=True)
    hostile_uv.write_text("#!/bin/sh\nexit 99\n")
    hostile_uv.chmod(0o700)
    trusted_bin = tmp_path / "operator-bin"
    trusted_bin.mkdir()
    trusted_uv = trusted_bin / "uv"
    trusted_uv.write_text("#!/bin/sh\nexit 0\n")
    trusted_uv.chmod(0o700)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join((str(hostile_uv.parent), str(trusted_bin))),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "host-api-secret")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "host-channel-secret")
    monkeypatch.setenv("KESTREL_DB_PATH", "/host/tenant.db")
    monkeypatch.setenv("KESTREL_FEATURE_WHATSAPPFEATURE_TOKEN", "host-feature-secret")
    monkeypatch.setenv("PIP_INDEX_URL", "https://index.example/simple")
    monkeypatch.setenv("UV_INDEX_PRIVATE_USERNAME", "index-user")
    monkeypatch.setenv("UV_INDEX_PRIVATE_PASSWORD", "index-password")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "host-cache"))
    # Model a service UID with no inherited/passwd-derived HOME. uv must still
    # receive an explicit cache without consulting an operator home directory.
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "host-config"))
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return ir.subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ir.subprocess, "run", fake_run)

    command = ["uv", "pip", "install", "--python", "/scoped/python", "pkg"]
    feature._run(command)

    assert captured["cmd"] == [str(trusted_uv.resolve()), *command[1:]]
    assert captured["check"] is True
    env = captured["env"]
    assert env["PATH"] == str(trusted_bin.resolve())
    assert str(hostile_uv.parent) not in env["PATH"]
    assert "VIRTUAL_ENV" not in env
    assert env["PIP_INDEX_URL"] == "https://index.example/simple"
    assert env["UV_INDEX_PRIVATE_USERNAME"] == "index-user"
    assert env["UV_INDEX_PRIVATE_PASSWORD"] == "index-password"
    assert env["UV_NO_CONFIG"] == "1"
    assert env["UV_CACHE_DIR"] == str(
        feature._feature_runtime_dir() / "provisioning_cache"
    )
    cache_metadata = Path(env["UV_CACHE_DIR"]).stat()
    assert stat.S_IMODE(cache_metadata.st_mode) == 0o700
    if os.name == "posix":
        assert cache_metadata.st_uid == os.geteuid()
    for secret in (
        "ANTHROPIC_API_KEY",
        "TWILIO_AUTH_TOKEN",
        "KESTREL_DB_PATH",
        "KESTREL_FEATURE_WHATSAPPFEATURE_TOKEN",
        "HOME",
        "XDG_CONFIG_HOME",
    ):
        assert secret not in env


@pytest.mark.skipif(os.name != "posix", reason="POSIX cache ownership contract")
@pytest.mark.parametrize(
    ("mutation", "message"),
    (("mode", "not private"), ("owner", "foreign owner")),
)
def test_hosted_provisioning_cache_revalidates_private_owner(
    message,
    monkeypatch,
    mutation,
    tmp_path,
):
    feature = ProxyFeature(
        _hosted_postgres_agent(tmp_path / "runtime", "tenant/agent"),
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )
    feature._prepare_runtime_workspace()
    cache_dir = feature._feature_runtime_dir() / "provisioning_cache"

    if mutation == "mode":
        cache_dir.chmod(0o755)
    else:
        real_stat = Path.stat

        def foreign_cache_stat(path, *args, **kwargs):
            metadata = real_stat(path, *args, **kwargs)
            if path == cache_dir:
                return SimpleNamespace(
                    st_mode=metadata.st_mode,
                    st_uid=os.geteuid() + 1,
                )
            return metadata

        monkeypatch.setattr(Path, "stat", foreign_cache_stat)

    with pytest.raises(IsolatedRuntimeNamespaceError, match=message):
        feature._hosted_provisioning_cache_dir()


def test_hosted_uv_provisioning_refuses_feature_venv_as_only_executable(
    monkeypatch, tmp_path
):
    feature = ProxyFeature(
        _hosted_postgres_agent(tmp_path / "runtime", "tenant/agent"),
        _isolated_runtime(),
        client_factory=FakeIsolatedClient,
    )
    feature._venv_path = feature._default_venv_path()
    feature._prepare_runtime_workspace()
    hostile_uv = feature._venv_path / "bin" / "uv"
    hostile_uv.parent.mkdir(parents=True)
    hostile_uv.write_text("#!/bin/sh\nexit 0\n")
    hostile_uv.chmod(0o700)
    monkeypatch.setenv("PATH", str(hostile_uv.parent))
    run = Mock(side_effect=AssertionError("hostile uv must not execute"))
    monkeypatch.setattr(isolated_runtime.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="Required executable not found: uv"):
        feature._run(["uv", "venv", str(feature._venv_path)])

    run.assert_not_called()


def test_build_client_preserves_explicit_empty_config(tmp_path):
    """An effective empty config is sent as ``{}``, not omitted as ``None``."""

    captured = {}

    def client_factory(**kwargs):
        captured.update(kwargs)
        return FakeIsolatedClient(**kwargs)

    feature = ProxyFeature(
        Mock(features={}),
        _isolated_runtime(),
        client_factory=client_factory,
    )
    feature._venv_path = tmp_path / "svc-venv"
    feature._bin_path = tmp_path / "test-service"
    feature._host_config = {}

    feature._build_client()

    assert captured["config"] == {}


@pytest.mark.asyncio
async def test_hosted_factory_missing_env_and_cwd_is_quarantined_without_call(
    caplog,
    monkeypatch,
    tmp_path,
):
    """A legacy factory cannot silently launch with inherited host scope."""

    monkeypatch.setenv(
        "KESTREL_FEATURE_TESTFEATURE_BIN", str(Path(sys.executable).resolve())
    )
    host_secret = "host-secret-must-not-reach-child-or-log"
    monkeypatch.setenv("UNSCOPED_HOST_SECRET", host_secret)
    calls = []

    def legacy_factory(command):
        calls.append(command)
        raise AssertionError("incompatible hosted factory must not be called")

    agent = _hosted_postgres_agent(tmp_path / "runtime", "tenant/agent")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=legacy_factory)

    with caplog.at_level("ERROR"):
        available = await agent._register_startup_feature(
            feature,
            prepared_contributions=Mock(),
        )

    assert available is False
    assert calls == []
    assert feature._client is None
    assert feature._supervision_task is None
    assert feature.name not in agent.features
    assert "tenant-scoped env and cwd delivery" in caplog.text
    assert host_secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_type", (TypeError, ValueError))
async def test_hosted_factory_constructor_failure_never_retries_without_scope(
    caplog,
    failure_type,
    monkeypatch,
    tmp_path,
):
    """A constructor failure gets one scoped call and no legacy fallback."""

    monkeypatch.setenv(
        "KESTREL_FEATURE_TESTFEATURE_BIN", str(Path(sys.executable).resolve())
    )
    host_secret = "host-secret-must-not-be-inherited"
    constructor_secret = "constructor-secret-must-not-be-logged"
    monkeypatch.setenv("UNSCOPED_HOST_SECRET", host_secret)
    calls = []

    def failing_factory(command, *, env, cwd):
        calls.append({"command": command, "env": env, "cwd": cwd})
        assert host_secret not in env.values()
        raise failure_type(constructor_secret)

    agent = _hosted_postgres_agent(tmp_path / "runtime", "tenant/agent")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=failing_factory)

    with caplog.at_level("ERROR"):
        available = await agent._register_startup_feature(
            feature,
            prepared_contributions=Mock(),
        )

    assert available is False
    assert len(calls) == 1
    assert calls[0]["cwd"] == str(feature._feature_runtime_dir() / "work")
    assert feature._client is None
    assert feature._supervision_task is None
    assert feature.name not in agent.features
    assert "tenant-scoped env and cwd delivery" in caplog.text
    assert constructor_secret not in caplog.text
    assert host_secret not in caplog.text


def test_pinned_sdk_hosted_client_receives_required_env_and_cwd(
    monkeypatch,
    tmp_path,
):
    """The installed SDK constructor satisfies Core's hosted launch contract."""

    from kestrel_sdk.isolated_feature import SubprocessIsolatedFeatureClient

    executable = str(Path(sys.executable).resolve())
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", executable)
    host_secret = "host-secret-must-not-be-inherited"
    monkeypatch.setenv("UNSCOPED_HOST_SECRET", host_secret)
    agent = _hosted_postgres_agent(tmp_path / "runtime", "tenant/agent")
    feature = ProxyFeature(agent, _cfg_runtime())
    workspace = feature._prepare_runtime_workspace()
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()

    client = feature._build_client()

    assert isinstance(client, SubprocessIsolatedFeatureClient)
    assert list(client.command) == [executable]
    assert client.cwd == str(workspace / "work")
    assert client.env is not None
    assert client.env["XDG_CACHE_HOME"] == str(workspace / "cache")
    assert "UV_CACHE_DIR" not in client.env
    assert host_secret not in client.env.values()


def test_build_client_injects_telegram_acknowledged_ingress_capability(tmp_path):
    """The Telegram child receives Core's non-persisted protocol contract."""

    captured = {}

    def client_factory(**kwargs):
        captured.update(kwargs)
        return FakeIsolatedClient(**kwargs)

    runtime = InstalledFeatureRuntime(
        class_name="TelegramFeature",
        entry_point="kestrel_channel_telegram.feature:TelegramFeature",
        distribution="Kestrel_Channel_Telegram",
        runtime="isolated-venv",
        service="kestrel-telegram-service",
    )
    feature = ProxyFeature(Mock(features={}), runtime, client_factory=client_factory)
    feature._venv_path = tmp_path / "svc-venv"
    feature._bin_path = tmp_path / "test-service"
    feature._host_config = {
        "enabled": True,
        "_kestrel_host_runtime_capabilities": ["untrusted-user-value"],
    }

    feature._build_client()

    assert captured["config"] == {
        "enabled": True,
        "_kestrel_host_runtime_capabilities": [
            "channel-inbound-acknowledgement-v1"
        ],
    }
    assert feature._host_config["_kestrel_host_runtime_capabilities"] == [
        "untrusted-user-value"
    ]


@pytest.mark.asyncio
async def test_build_client_injects_durably_claimed_telegram_startup_fence(tmp_path):
    """The launch-only fence requires a durable route claim, never config."""

    captured = {}

    def client_factory(**kwargs):
        captured.update(kwargs)
        return FakeIsolatedClient(**kwargs)

    runtime = InstalledFeatureRuntime(
        class_name="TelegramFeature",
        entry_point="kestrel_channel_telegram.feature:TelegramFeature",
        distribution="kestrel-channel-telegram",
        runtime="isolated-venv",
        service="kestrel-telegram-service",
    )
    backend = SQLiteBackend(str(tmp_path / "route-ownership.db"))
    await backend.connect()
    feature = ProxyFeature(
        Mock(did=_TEST_AGENT_DID, features={}), runtime, client_factory=client_factory
    )
    feature._venv_path = tmp_path / "svc-venv"
    feature._bin_path = tmp_path / "test-service"
    feature._host_config = {"enabled": True}

    try:
        assert await feature.reconcile_hosted_telegram_route_claim(
            ownership_store=ChannelRouteOwnershipStore(backend),
            bot_id="123456",
        ) is True
        feature._build_client()

        assert captured["config"] == {
            "enabled": True,
            "_kestrel_host_runtime_capabilities": [
                "channel-inbound-acknowledgement-v1",
                "telegram-hosted-ingress-owner-v1",
            ],
        }
        assert feature._host_config == {"enabled": True}
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_hosted_telegram_attestation_requires_exclusive_claim_and_reconciles(
    tmp_path,
):
    """A loser never gets the child capability until the winner releases it."""
    runtime = InstalledFeatureRuntime(
        class_name="TelegramFeature",
        entry_point="kestrel_channel_telegram.feature:TelegramFeature",
        distribution="kestrel-channel-telegram",
        runtime="isolated-venv",
        service="kestrel-telegram-service",
    )
    backend = SQLiteBackend(str(tmp_path / "shared-routes.db"))
    await backend.connect()
    store = ChannelRouteOwnershipStore(backend)
    first_capture: dict = {}
    second_capture: dict = {}

    def first_factory(**kwargs):
        first_capture.update(kwargs)
        return FakeIsolatedClient(**kwargs)

    def second_factory(**kwargs):
        second_capture.update(kwargs)
        return FakeIsolatedClient(**kwargs)

    first = ProxyFeature(
        Mock(did="did:test:telegram-first", features={}),
        runtime,
        client_factory=first_factory,
    )
    second = ProxyFeature(
        Mock(did="did:test:telegram-second", features={}),
        runtime,
        client_factory=second_factory,
    )
    for feature in (first, second):
        feature._venv_path = tmp_path / "svc-venv"
        feature._bin_path = tmp_path / "test-service"
        feature._host_config = {"enabled": True}

    try:
        assert await first.reconcile_hosted_telegram_route_claim(
            ownership_store=store,
            bot_id="123456",
        ) is True
        assert await second.reconcile_hosted_telegram_route_claim(
            ownership_store=store,
            bot_id="123456",
        ) is False
        first._build_client()
        second._build_client()
        assert "telegram-hosted-ingress-owner-v1" in first_capture["config"][
            "_kestrel_host_runtime_capabilities"
        ]
        assert "telegram-hosted-ingress-owner-v1" not in second_capture["config"][
            "_kestrel_host_runtime_capabilities"
        ]

        assert await first.release_hosted_telegram_route_claim(
            ownership_store=store,
            bot_id="123456",
        ) is True
        assert await second.reconcile_hosted_telegram_route_claim(
            ownership_store=store,
            bot_id="123456",
        ) is True
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_hosted_telegram_reassertion_cannot_release_replacement_claim(tmp_path):
    """An older proxy instance cannot ABA-delete its successor's same-DID claim."""

    runtime = InstalledFeatureRuntime(
        class_name="TelegramFeature",
        entry_point="kestrel_channel_telegram.feature:TelegramFeature",
        distribution="kestrel-channel-telegram",
        runtime="isolated-venv",
        service="kestrel-telegram-service",
    )
    backend = SQLiteBackend(str(tmp_path / "same-agent-route-aba.db"))
    await backend.connect()
    store = ChannelRouteOwnershipStore(backend)
    first = ProxyFeature(Mock(did="did:test:telegram", features={}), runtime)
    replacement = ProxyFeature(Mock(did="did:test:telegram", features={}), runtime)
    try:
        assert await first.reconcile_hosted_telegram_route_claim(
            ownership_store=store, bot_id="000123"
        )
        with pytest.raises(RuntimeError, match="release it first"):
            await first.reconcile_hosted_telegram_route_claim(
                ownership_store=store, bot_id="456"
            )
        assert await replacement.reconcile_hosted_telegram_route_claim(
            ownership_store=store, bot_id="123"
        )
        assert await first.release_hosted_telegram_route_claim(
            ownership_store=store, bot_id="123"
        ) is False
        assert await store.is_claimed_by(
            channel_type="telegram",
            canonical_route_identity="telegram-bot:123",
            agent_id="did:test:telegram",
        )
        assert await replacement.release_hosted_telegram_route_claim(
            ownership_store=store, bot_id="123"
        )
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_normal_telegram_initialize_resolves_route_fence_before_child_start(
    monkeypatch, tmp_path
):
    """The boot seam claims hosted ingress before the child can start polling."""

    runtime = InstalledFeatureRuntime(
        class_name="TelegramFeature",
        entry_point="kestrel_channel_telegram.feature:TelegramFeature",
        distribution="kestrel-channel-telegram",
        runtime="isolated-venv",
        service="kestrel-telegram-service",
    )
    backend = SQLiteBackend(str(tmp_path / "boot-route-fence.db"))
    await backend.connect()
    captured: dict = {}

    def client_factory(**kwargs):
        captured.update(kwargs)
        return FakeIsolatedClient(**kwargs)

    agent = SimpleNamespace(
        did="did:test:telegram-boot",
        features={},
        storage_path=str(tmp_path / "agent" / "kestrel_prime.db"),
    )
    set_hosted_telegram_route_attestation_resolver(
        agent,
        lambda _proxy: HostedTelegramRouteAttestation(
            ownership_store=ChannelRouteOwnershipStore(backend), bot_id="000123"
        ),
    )
    feature = ProxyFeature(agent, runtime, client_factory=client_factory)

    async def load_host_config():
        return {"enabled": True}

    feature._load_host_config = load_host_config  # type: ignore[method-assign]
    monkeypatch.setenv("KESTREL_FEATURE_TELEGRAMFEATURE_BIN", "/bin/test-service")
    try:
        await feature.initialize()
        assert captured["config"]["_kestrel_host_runtime_capabilities"] == [
            "channel-inbound-acknowledgement-v1",
            "telegram-hosted-ingress-owner-v1",
        ]
        assert await ChannelRouteOwnershipStore(backend).is_claimed_by(
            channel_type="telegram",
            canonical_route_identity="telegram-bot:123",
            agent_id=agent.did,
        )
    finally:
        await feature.shutdown()
        await backend.close()


@pytest.mark.asyncio
async def test_normal_telegram_initialize_refuses_existing_hosted_route_before_child_start(
    monkeypatch, tmp_path
):
    """A resolver conflict fails before the child factory/start handshake runs."""

    runtime = InstalledFeatureRuntime(
        class_name="TelegramFeature",
        entry_point="kestrel_channel_telegram.feature:TelegramFeature",
        distribution="kestrel-channel-telegram",
        runtime="isolated-venv",
        service="kestrel-telegram-service",
    )
    backend = SQLiteBackend(str(tmp_path / "boot-route-conflict.db"))
    await backend.connect()
    store = ChannelRouteOwnershipStore(backend)
    assert await store.claim(
        channel_type="telegram",
        canonical_route_identity="telegram-bot:123",
        agent_id="did:test:existing-host",
    )
    built: list[object] = []

    def client_factory(**kwargs):
        built.append(kwargs)
        return FakeIsolatedClient(**kwargs)

    agent = SimpleNamespace(
        did="did:test:blocked-host",
        features={},
        storage_path=str(tmp_path / "agent" / "kestrel_prime.db"),
        hosted_telegram_route_attestation_resolver=(
            lambda _proxy: HostedTelegramRouteAttestation(store, "123")
        ),
    )
    feature = ProxyFeature(agent, runtime, client_factory=client_factory)

    async def load_host_config():
        return {"enabled": True}

    feature._load_host_config = load_host_config  # type: ignore[method-assign]
    monkeypatch.setenv("KESTREL_FEATURE_TELEGRAMFEATURE_BIN", "/bin/test-service")
    try:
        with pytest.raises(RuntimeError, match="already owned"):
            await feature.initialize()
        assert built == []
    finally:
        await feature.shutdown()
        await backend.close()


def test_telegram_route_identity_accepts_only_canonicalized_decimal_bot_ids():
    """Provider prefixes cannot forge a second Telegram ownership key."""

    assert canonical_telegram_bot_id("000123") == "123"
    with pytest.raises(ValueError, match="positive decimal"):
        canonical_telegram_bot_id("telegram-bot:123")
    with pytest.raises(ValueError, match="positive"):
        canonical_telegram_bot_id("000")


class _FakeStorage:
    """Minimal graph store double with the production CAS contract."""

    def __init__(self):
        self.nodes = {}
        # Isolated config migration is only permitted through a graph-store
        # capability bound to the same DID as the proxy.
        self.agent_id = _TEST_AGENT_DID

    async def add_node(self, node):
        self.nodes[node.node_id] = node

    async def get_node(self, node_id):
        return self.nodes.get(node_id)

    async def compare_and_swap_node(self, node_id, expected, new_node):
        current = self.nodes.get(node_id)
        current_properties = None if current is None else current.properties
        if current_properties != expected:
            return "predicate_failed"
        self.nodes[node_id] = new_node
        return "swapped"


class _CASStorage(_FakeStorage):
    """Named CAS double retained for tests that model hosted contention."""


def _cfg_runtime():
    return InstalledFeatureRuntime(
        class_name="TestFeature",
        entry_point="test_pkg.feature:TestFeature",
        distribution="test-pkg",
        runtime="isolated-venv",
        service="test_service",
        description="Test proxy",
    )


@pytest.mark.asyncio
async def test_set_config_persists_node_reflects_in_get_config_and_reloads(monkeypatch, tmp_path):
    """#2214: set_config must persist to the feature_config:<name> graph node,
    get_config must reflect it (not an empty client passthrough), and the running
    service must be reloaded so the new config actually takes effect."""
    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = _FakeStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    clients = []

    def client_factory(**kwargs):
        c = FakeIsolatedClient(**kwargs)
        clients.append(c)
        return c

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    await feature.initialize()
    assert len(clients) == 1

    new_cfg = {"enabled": True, "allowed_senders": ["8825903191"], "token": "12345:abc"}
    await feature.set_config(new_cfg)

    # 1) Persisted to the graph node the isolated runtime reads at startup.
    node = agent.storage.nodes.get(_TEST_CONFIG_NODE_ID)
    assert node is not None, "set_config did not persist the feature_config node"
    assert node.properties["config"]["allowed_senders"] == ["8825903191"]
    assert "pending_config" not in node.properties
    assert "_isolated_pending_generation" not in node.properties
    assert "_isolated_pending_owner" not in node.properties
    assert "_isolated_pending_lease_expires_at" not in node.properties

    # 2) get_config reflects it (previously always returned {}), incl. the secret
    #    so the endpoint's write-only-secret preservation works.
    got = await feature.get_config()
    assert got["allowed_senders"] == ["8825903191"]
    assert got["token"] == "12345:abc"

    # 3) Reloaded: old client stopped, a NEW client built with the new config
    #    forwarded through the initialize handshake (config only flows at init).
    assert clients[0].stopped is True
    assert len(clients) == 2
    assert clients[1].kwargs.get("config") == new_cfg


@pytest.mark.asyncio
async def test_post_promotion_start_failure_rebuilds_active_child_before_traffic_reopens(
    monkeypatch, tmp_path
):
    """A failed promoted candidate is repaired before finite traffic reopens.

    ``_replace_client`` must unpublish the old child before stopping it, so a
    startup failure after promotion leaves no client published.  The cleanup
    path must rebuild from the now-durable active config rather than reopening
    an empty proxy.
    """

    old_config = {"enabled": True, "revision": "old"}
    next_config = {"enabled": False, "revision": "next"}

    class PromotionCandidate(FakeIsolatedClient):
        async def start(self):
            self.started = True
            raise RuntimeError("promoted candidate could not start")

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = (
            FakeIsolatedClient(**kwargs)
            if len(clients) != 1
            else PromotionCandidate(**kwargs)
        )
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    reopen_clients = []
    original_reopen = feature._reopen_traffic_gate

    async def assert_reconciled_then_reopen():
        # This is the finite gate's only reopen point for the transition.  The
        # recovery child must already be reachable and configured from the
        # durable promotion before any traffic can be admitted again.
        assert feature._traffic_gate.closed is True
        assert feature._client is clients[2]
        assert feature._client.kwargs["config"] == next_config
        assert feature._host_config == next_config
        reopen_clients.append(feature._client)
        await original_reopen()

    monkeypatch.setattr(feature, "_reopen_traffic_gate", assert_reconciled_then_reopen)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(IsolatedRuntimePreparationError) as failure:
            await feature.set_config(next_config)
        assert "promoted candidate could not start" not in str(failure.value)
        assert isinstance(failure.value.__cause__, RuntimeError)

        assert clients[0].stopped is True
        assert clients[1].stopped is True
        assert clients[2].started is True
        assert feature._client is clients[2]
        assert feature.get_tools()
        assert feature._traffic_gate.closed is False
        assert feature._traffic_gate.sealed is False
        assert reopen_clients == [clients[2]]
        assert (
            await feature.call_isolated_tool("ping", {"message": "recovered"})
        ) == {
            "success": True,
            "result": {"echo": {"message": "recovered"}},
            "tool": "ping",
        }
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == next_config
        assert "pending_config" not in properties
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_post_promotion_recovery_start_failure_seals_traffic(monkeypatch, tmp_path):
    """A failed forced rebuild leaves the promoted state terminally unreachable."""

    old_config = {"enabled": True, "revision": "old"}
    next_config = {"enabled": False, "revision": "next"}

    class StartFailingClient(FakeIsolatedClient):
        async def start(self):
            self.started = True
            raise RuntimeError("replacement child could not start")

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs) if not clients else StartFailingClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)

    async def fail_if_traffic_reopens():
        raise AssertionError("failed recovery must not reopen traffic")

    monkeypatch.setattr(feature, "_reopen_traffic_gate", fail_if_traffic_reopens)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(IsolatedRuntimePreparationError) as failure:
            await feature.set_config(next_config)
        assert "replacement child could not start" not in str(failure.value)
        assert isinstance(failure.value.__cause__, RuntimeError)

        # The first failed child was the post-promotion candidate and the
        # second was the forced durable recovery. Neither is publishable.
        assert clients[0].stopped is True
        assert clients[1].stopped is True
        assert clients[2].stopped is True
        assert clients[2].kwargs["config"] == next_config
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._stopping is True
        assert feature._traffic_gate.sealed is True
        assert await feature.call_isolated_tool("ping", {"message": "blocked"}) == {
            "status": "error",
            "error": "isolated feature traffic is unavailable",
            "tool": "ping",
            "success": False,
        }
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == next_config
        assert "pending_config" not in properties
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "next_config",
    [
        {
            "enabled": False,
            "token": "12345678:old-token",
            "transport": "webhook",
            "webhook_url": "https://ingress.example.test/telegram",
            "webhook_secret": "old-webhook-secret",
        },
        {
            "enabled": True,
            "token": "87654321:new-token",
            "transport": "webhook",
            "webhook_url": "https://ingress.example.test/telegram",
            "webhook_secret": "new-webhook-secret",
        },
    ],
    ids=["disable", "credential-rotation"],
)
async def test_supported_transition_cleans_up_before_replacing_old_service(
    monkeypatch, tmp_path, caplog, next_config
):
    """A webhook-shaped service prepares both disable and token-rotation first.

    The test deliberately records only ordering and public config shape. It
    never logs the credentials it carries, which proves the host can honor the
    generic SDK lifecycle without introducing Telegram-specific dispatch.
    """

    old_config = {
        "enabled": True,
        "token": "12345678:old-token",
        "transport": "webhook",
        "webhook_url": "https://ingress.example.test/telegram",
        "webhook_secret": "old-webhook-secret",
    }
    events = []

    class WebhookCleanupClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            events.append(("cleanup", dict(config)))
            return ConfigTransitionResult.restart_required()

        async def stop(self):
            events.append(("stop", None))
            await super().stop()

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = _FakeStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    clients = []

    def client_factory(**kwargs):
        client = (
            WebhookCleanupClient(**kwargs)
            if not clients
            else FakeIsolatedClient(**kwargs)
        )
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    await feature.persist_config(old_config)
    await feature.initialize()

    await feature.set_config(next_config)

    # The lifecycle RPC receives the complete effective next config and its
    # ordered resource cleanup completes before the old process is stopped.
    assert events == [("cleanup", next_config), ("stop", None)]
    assert clients[0].kwargs["config"] == old_config
    assert clients[1].kwargs["config"] == next_config
    assert clients[0].stopped is True
    for secret in (
        old_config["token"],
        old_config["webhook_secret"],
        next_config["token"],
        next_config["webhook_secret"],
    ):
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_supported_transition_failure_keeps_old_config_and_service(monkeypatch, tmp_path):
    """A failed negotiated cleanup cannot masquerade as a successful apply."""

    old_config = {"enabled": True, "token": "old-token"}
    next_config = {"enabled": True, "token": "next-token"}

    class FailingTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == next_config
            raise ConfigTransitionError("config transition failed")

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = _FakeStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(
        agent,
        _cfg_runtime(),
        client_factory=FailingTransitionClient,
    )
    await feature.persist_config(old_config)
    await feature.initialize()
    old_client = feature._client

    with pytest.raises(ConfigTransitionError, match="config transition failed"):
        await feature.set_config(next_config)

    # The normal SDK failure contract preserves the known-safe old child. The
    # host also restores durable/in-memory config so a later restart cannot
    # launch the candidate without the cleanup that failed above.
    assert feature._client is old_client
    assert old_client.stopped is False
    assert feature._host_config == old_config
    assert (await feature.get_config()) == old_config
    assert agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties["config"] == old_config


@pytest.mark.asyncio
async def test_external_ingress_quiesces_callback_while_admission_is_open_then_drains(
    monkeypatch, tmp_path
):
    """A real pending producer callback cannot deadlock Core's transition."""

    old_config = {"enabled": True, "revision": "old"}
    next_config = {"enabled": True, "revision": "next"}
    delivered = []
    clients = []
    callback_started = asyncio.Event()
    callback_finished = asyncio.Event()
    release_callback = asyncio.Event()

    class QuiescingIngressClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.quiesced = False
            self.pending_events = []
            self.ingress_calls = []
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=(
                    "telegram-polling-ack",
                    "external-ingress-quiesce",
                    "external-ingress-resume",
                )
            )

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if name == "telegram-polling-ack":
                callback_started.set()
                await release_callback.wait()
                callback_finished.set()
                return {"status": "ok", "http_status": 200, "state": "acknowledged"}
            if name == "external-ingress-quiesce":
                # The provider quiesce waits for its actual callback to
                # finish. Core must leave admission open for that wait.
                assert feature._traffic_gate.closed is False
                await callback_finished.wait()
                self.quiesced = True
                return {"status": "ok", "http_status": 200, "state": "quiesced"}
            assert name == "external-ingress-resume"
            # Resume is an owned finalizer: deferred work cannot replay until
            # this exact source has resumed and Core reopens admission.
            assert feature._traffic_gate.closed is True
            self.quiesced = False
            for event in self.pending_events:
                await self.event_handler(event)
            self.pending_events.clear()
            return {"status": "ok", "http_status": 200, "state": "resumed"}

        async def emit_external_update(self, event):
            if self.quiesced:
                self.pending_events.append(event)
                return
            await self.event_handler(event)

        async def prepare_config_transition(self, config):
            assert config == next_config
            # Quiesce has now completed, so Core closes admission before the
            # hook runs. New producer events are retained until resume.
            assert feature._traffic_gate.closed is True
            await self.emit_external_update(
                {
                    "type": "channel.inbound",
                    "payload": {
                        "message": {
                            "id": "late-update",
                            "metadata": {"dedupe_key": "late-update"},
                        },
                        "_host_ingress_ack": {
                            "name": "telegram-polling-ack",
                            "payload": {
                                "dedupe_key": "late-update",
                                "attempt_token": _TELEGRAM_ATTEMPT_TOKEN,
                            },
                        },
                    },
                }
            )
            return ConfigTransitionResult.applied()

    def client_factory(**kwargs):
        client = QuiescingIngressClient(**kwargs)
        clients.append(client)
        return client

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)

    async def record(event):
        delivered.append(event)
        return SimpleNamespace(durably_admitted=True)

    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        feature._route_inbound = record  # type: ignore[method-assign]

        await clients[0].event_handler(
            _acknowledged_telegram_event("telegram:v2:bot:42:update:early")
        )
        await asyncio.wait_for(callback_started.wait(), timeout=1)

        transition_task = asyncio.create_task(feature.set_config(next_config))
        await asyncio.sleep(0)
        assert not transition_task.done()
        release_callback.set()
        await asyncio.wait_for(transition_task, timeout=1)

        assert [
            call[0]
            for call in clients[0].ingress_calls
            if call[0].startswith("external-ingress-")
        ] == [
            "external-ingress-quiesce",
            "external-ingress-resume",
        ]
        for _ in range(20):
            if delivered:
                break
            await asyncio.sleep(0)
        assert delivered == [
            {
                "channel_type": "telegram",
                "direction": "inbound",
                "sender": "555",
                "recipient": "42",
                "content": "hello",
                "id": "telegram:v2:bot:42:update:early",
                "metadata": {"dedupe_key": "telegram:v2:bot:42:update:early"},
            },
            {"id": "late-update", "metadata": {"dedupe_key": "late-update"}}
        ]
        assert clients[0].pending_events == []
        assert feature._traffic_gate.closed is False
        assert feature._host_config == next_config
    finally:
        release_callback.set()
        await feature.shutdown()


@pytest.mark.asyncio
async def test_external_ingress_resumes_after_failed_transition_rollback(monkeypatch, tmp_path):
    """A failed staged transition resumes polling before reopening Core admission."""

    old_config = {"enabled": True, "revision": "old"}
    rejected_config = {"enabled": True, "revision": "rejected"}

    class RollbackIngressClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.quiesced = False
            self.ingress_calls = []
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("external-ingress-quiesce", "external-ingress-resume")
            )

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if name == "external-ingress-quiesce":
                self.quiesced = True
                return {"status": "ok", "http_status": 200, "state": "quiesced"}
            assert name == "external-ingress-resume"
            assert feature._traffic_gate.closed is True
            self.quiesced = False
            return {"status": "ok", "http_status": 200, "state": "resumed"}

        async def prepare_config_transition(self, config):
            assert self.quiesced is True
            assert config == rejected_config
            raise ConfigTransitionError("transition rejected")

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=RollbackIngressClient)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        old_client = feature._client

        with pytest.raises(ConfigTransitionError, match="transition rejected"):
            await feature.set_config(rejected_config)

        assert feature._client is old_client
        assert old_client.quiesced is False
        assert [call[0] for call in old_client.ingress_calls] == [
            "external-ingress-quiesce",
            "external-ingress-resume",
        ]
        assert feature._traffic_gate.closed is False
        assert feature._host_config == old_config
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_external_ingress_resumes_after_cancelled_transition_rollback(monkeypatch, tmp_path):
    """Cancellation after admission closes releases the exact producer."""

    old_config = {"enabled": True, "revision": "old"}
    pending_config = {"enabled": True, "revision": "pending"}
    quiesce_started = asyncio.Event()
    release_quiesce = asyncio.Event()

    class CancelledIngressClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.quiesced = False
            self.ingress_calls = []
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("external-ingress-quiesce", "external-ingress-resume")
            )

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if name == "external-ingress-quiesce":
                self.quiesced = True
                quiesce_started.set()
                await release_quiesce.wait()
                return {"status": "ok", "http_status": 200, "state": "quiesced"}
            assert name == "external-ingress-resume"
            assert feature._traffic_gate.closed is True
            self.quiesced = False
            return {"status": "ok", "http_status": 200, "state": "resumed"}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=CancelledIngressClient)
    update = None
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        client = feature._client

        update = asyncio.create_task(feature.set_config(pending_config))
        await asyncio.wait_for(quiesce_started.wait(), timeout=1)
        update.cancel()
        release_quiesce.set()
        with pytest.raises(asyncio.CancelledError):
            await update

        assert feature._client is client
        assert client.quiesced is False
        assert [call[0] for call in client.ingress_calls] == [
            "external-ingress-quiesce",
            "external-ingress-resume",
        ]
        assert feature._traffic_gate.closed is False
        assert feature._host_config == old_config
    finally:
        if update is not None and not update.done():
            update.cancel()
            with pytest.raises(asyncio.CancelledError):
                await update
        await feature.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resume_error", "expected_error"),
    [
        (RuntimeError("resume failed"), HostIngressError),
        (asyncio.CancelledError("resume cancelled"), asyncio.CancelledError),
    ],
    ids=["failure", "cancellation"],
)
async def test_external_ingress_resume_failure_never_reports_config_success(
    monkeypatch, tmp_path, resume_error, expected_error
):
    """A failed resume quarantines and remains the successful body's outcome."""

    old_config = {"enabled": True, "revision": "old"}
    next_config = {"enabled": True, "revision": "next"}

    class ResumeFailingClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("external-ingress-quiesce", "external-ingress-resume")
            )

        async def call_host_ingress(self, name, payload=None):
            if name == "external-ingress-quiesce":
                return {"status": "ok", "http_status": 200, "state": "quiesced"}
            assert name == "external-ingress-resume"
            raise resume_error

        async def prepare_config_transition(self, config):
            assert config == next_config
            return ConfigTransitionResult.applied()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=ResumeFailingClient)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(expected_error):
            await feature.set_config(next_config)

        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._stopping is True
        assert feature._traffic_gate.sealed is True
        assert feature._host_config == next_config
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_external_ingress_resume_failure_preserves_active_transition_error(
    monkeypatch, tmp_path
):
    """Resume cleanup cannot hide the transition error already being unwound."""

    old_config = {"enabled": True, "revision": "old"}
    rejected_config = {"enabled": True, "revision": "rejected"}

    class BodyFailingClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("external-ingress-quiesce", "external-ingress-resume")
            )

        async def call_host_ingress(self, name, payload=None):
            if name == "external-ingress-quiesce":
                return {"status": "ok", "http_status": 200, "state": "quiesced"}
            assert name == "external-ingress-resume"
            raise RuntimeError("resume failed during unwind")

        async def prepare_config_transition(self, config):
            assert config == rejected_config
            raise ConfigTransitionError("transition rejected")

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=BodyFailingClient)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(ConfigTransitionError, match="transition rejected"):
            await feature.set_config(rejected_config)

        assert feature._client is None
        assert feature._traffic_gate.sealed is True
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_transition_failure_clears_pending_config_when_promotion_fails(
    monkeypatch, tmp_path
):
    """A failed hook clears its generation instead of wedging future updates."""

    old_config = {"enabled": True, "token": "old-token"}
    next_config = {"enabled": True, "token": "next-token"}

    class RollbackFailingStorage(_FakeStorage):
        def __init__(self):
            super().__init__()
            self.cas_calls = 0

        async def compare_and_swap_node(self, node_id, expected, new_node):
            self.cas_calls += 1
            is_owned_cleanup = (
                isinstance(expected, dict)
                and "pending_config" in expected
                and "pending_config" not in new_node.properties
            )
            if is_owned_cleanup and self.cas_calls == 3:
                raise OSError("storage offline during promotion")
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class FailingTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == next_config
            raise ConfigTransitionError("config transition failed")

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = RollbackFailingStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FailingTransitionClient)
    await feature.persist_config(old_config)
    await feature.initialize()

    with pytest.raises(ConfigTransitionError, match="config transition failed"):
        await feature.set_config(next_config)

    node = agent.storage.nodes[_TEST_CONFIG_NODE_ID]
    assert node.properties["config"] == old_config
    assert "pending_config" not in node.properties
    assert "_isolated_pending_generation" not in node.properties
    # stage → post-reconciliation lease renewal → failed cleanup → retry
    assert agent.storage.cas_calls == 4

    fresh = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    await fresh.initialize()
    assert fresh._host_config == old_config
    assert (await fresh.get_config()) == old_config


@pytest.mark.asyncio
async def test_hook_failure_clears_generation_and_allows_immediate_retry(
    monkeypatch, tmp_path
):
    """A known hook failure leaves no durable barrier to the next update."""

    old_config = {"enabled": True, "token": "old-token"}
    failed_config = {"enabled": False, "token": "failed-token"}
    retry_config = {"enabled": False, "token": "retry-token"}

    class FlakyTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.preparations = 0

        async def prepare_config_transition(self, config):
            self.preparations += 1
            if self.preparations == 1:
                assert config == failed_config
                raise ConfigTransitionError("hook rejected staged config")
            assert config == retry_config
            return ConfigTransitionResult.restart_required()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = FlakyTransitionClient(**kwargs) if not clients else FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(ConfigTransitionError, match="hook rejected"):
            await feature.set_config(failed_config)

        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == old_config
        assert "pending_config" not in properties
        assert "_isolated_pending_generation" not in properties

        await feature.set_config(retry_config)

        assert clients[0].preparations == 2
        assert feature._host_config == retry_config
        assert agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties["config"] == retry_config
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_cancellation_after_stage_clears_owned_generation(monkeypatch, tmp_path):
    """Cancelling a staged hook aborts that generation and starts no candidate."""

    old_config = {"enabled": True, "token": "old-token"}
    pending_config = {"enabled": False, "token": "pending-token"}
    hook_started = asyncio.Event()

    class BlockingTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == pending_config
            hook_started.set()
            await asyncio.Event().wait()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = BlockingTransitionClient(**kwargs) if not clients else FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    update = None
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        update = asyncio.create_task(feature.set_config(pending_config))
        await asyncio.wait_for(hook_started.wait(), timeout=1)

        staged = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert staged["pending_config"] == pending_config
        assert isinstance(staged["_isolated_pending_owner"], str)
        assert isinstance(staged["_isolated_pending_lease_expires_at"], str)

        update.cancel()
        with pytest.raises(asyncio.CancelledError):
            await update

        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == old_config
        assert "pending_config" not in properties
        assert all(client.kwargs["config"] != pending_config for client in clients)
    finally:
        if update is not None and not update.done():
            update.cancel()
            with pytest.raises(asyncio.CancelledError):
                await update
        await feature.shutdown()


@pytest.mark.asyncio
async def test_abort_commit_then_raise_is_reconciled_before_retry(monkeypatch, tmp_path):
    """An abort connection error after commit does not strand its generation."""

    old_config = {"enabled": True, "token": "old-token"}
    rejected_config = {"enabled": False, "token": "rejected-token"}

    class CommitThenRaiseAbortStorage(_CASStorage):
        def __init__(self):
            super().__init__()
            self.abort_raised = False

        async def compare_and_swap_node(self, node_id, expected, new_node):
            result = await super().compare_and_swap_node(node_id, expected, new_node)
            is_abort = (
                result == "swapped"
                and isinstance(expected, dict)
                and "pending_config" in expected
                and "pending_config" not in new_node.properties
            )
            if is_abort and not self.abort_raised:
                self.abort_raised = True
                raise ConnectionError("connection dropped after abort commit")
            return result

    class FailingTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == rejected_config
            raise ConfigTransitionError("hook rejected staged config")

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = CommitThenRaiseAbortStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FailingTransitionClient)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(ConfigTransitionError, match="hook rejected"):
            await feature.set_config(rejected_config)

        assert agent.storage.abort_raised is True
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == old_config
        assert "pending_config" not in properties
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_fresh_replica_cannot_steal_unexpired_pending_generation(
    monkeypatch, tmp_path
):
    """A new proxy fences itself behind a healthy writer's durable lease."""

    old_config = {"enabled": True, "token": "old-token"}
    first_config = {"enabled": False, "token": "first-token"}
    second_config = {"enabled": False, "token": "second-token"}
    hook_started = asyncio.Event()
    release_hook = asyncio.Event()

    class BlockingTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == first_config
            hook_started.set()
            await release_hook.wait()
            return ConfigTransitionResult.restart_required()

    storage = _CASStorage()
    first_agent = Mock(did=_TEST_AGENT_DID, features={})
    first_agent.storage = storage
    first_agent.storage_path = str(tmp_path / "first" / "kestrel_prime.db")
    second_agent = Mock(did=_TEST_AGENT_DID, features={})
    second_agent.storage = storage
    second_agent.storage_path = str(tmp_path / "second" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    first = ProxyFeature(first_agent, _cfg_runtime(), client_factory=BlockingTransitionClient)
    second_clients = []

    def second_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        second_clients.append(client)
        return client

    second = ProxyFeature(second_agent, _cfg_runtime(), client_factory=second_factory)
    first_update = None
    try:
        await first.persist_config(old_config)
        await first.initialize()
        first_update = asyncio.create_task(first.set_config(first_config))
        await asyncio.wait_for(hook_started.wait(), timeout=1)
        staged = dict(storage.nodes[_TEST_CONFIG_NODE_ID].properties)

        await second.initialize()
        with pytest.raises(RuntimeError, match="already in progress"):
            await second.set_config(second_config)

        assert storage.nodes[_TEST_CONFIG_NODE_ID].properties == staged
        assert second_clients[0].kwargs["config"] == old_config
        assert all(client.kwargs["config"] != first_config for client in second_clients)

        release_hook.set()
        await first_update
    finally:
        release_hook.set()
        if first_update is not None and not first_update.done():
            await first_update
        await second.shutdown()
        await first.shutdown()


@pytest.mark.asyncio
async def test_expired_pending_generation_is_cleared_then_retried_from_active_config(
    monkeypatch, tmp_path
):
    """Takeover removes only expired metadata and never starts its candidate."""

    now = datetime(2026, 7, 26, 12, tzinfo=timezone.utc)
    old_config = {"enabled": True, "token": "old-token"}
    abandoned_config = {"enabled": False, "token": "abandoned-token"}
    next_config = {"enabled": False, "token": "next-token"}
    monkeypatch.setattr(isolated_runtime, "_utc_now", lambda: now)
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    storage = _CASStorage()
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = storage
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    seed = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    await seed.persist_config(old_config)
    storage.nodes[_TEST_CONFIG_NODE_ID].properties.update(
        {
            "pending_config": abandoned_config,
            "_isolated_pending_generation": "abandoned-generation",
            "_isolated_pending_owner": "abandoned-owner",
            "_isolated_pending_lease_expires_at": (
                now - isolated_runtime._PENDING_CONFIG_CLOCK_SKEW - timedelta(seconds=1)
            ).isoformat(),
        }
    )

    clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.initialize()
        await feature.set_config(next_config)

        properties = storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == next_config
        assert "pending_config" not in properties
        assert all(client.kwargs["config"] != abandoned_config for client in clients)
        assert clients[0].kwargs["config"] == old_config
        assert clients[1].kwargs["config"] == next_config
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_malformed_pending_lease_metadata_fails_closed(monkeypatch, tmp_path):
    """A corrupt lease is never treated as expired or cleared speculatively."""

    old_config = {"enabled": True, "token": "old-token"}
    malformed_config = {"enabled": False, "token": "malformed-token"}
    next_config = {"enabled": False, "token": "next-token"}
    storage = _CASStorage()
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = storage
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    seed = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    await seed.persist_config(old_config)
    storage.nodes[_TEST_CONFIG_NODE_ID].properties.update(
        {
            "pending_config": malformed_config,
            "_isolated_pending_generation": "malformed-generation",
            "_isolated_pending_owner": "malformed-owner",
            "_isolated_pending_lease_expires_at": "not-a-timestamp",
        }
    )
    original_properties = dict(storage.nodes[_TEST_CONFIG_NODE_ID].properties)
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    try:
        await feature.initialize()
        with pytest.raises(RuntimeError, match="stored pending config lease is invalid"):
            await feature.set_config(next_config)

        assert storage.nodes[_TEST_CONFIG_NODE_ID].properties == original_properties
        assert feature._host_config == old_config
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_empty_active_config_hides_pending_candidate_from_concurrent_reads(
    monkeypatch, tmp_path
):
    """An initialized empty config is active, not a signal to read persistence."""

    next_config = {"enabled": True, "token": "next-token"}
    preparation_started = asyncio.Event()
    release_preparation = asyncio.Event()

    class SlowFailingTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == next_config
            preparation_started.set()
            await release_preparation.wait()
            raise ConfigTransitionError("config transition failed")

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = _FakeStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(
        agent,
        _cfg_runtime(),
        client_factory=SlowFailingTransitionClient,
    )
    await feature.initialize()
    assert feature._host_config == {}

    update = asyncio.create_task(feature.set_config(next_config))
    await asyncio.wait_for(preparation_started.wait(), timeout=1)

    properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
    assert properties["config"] == {}
    assert properties["pending_config"] == next_config
    assert isinstance(properties["_isolated_pending_generation"], str)
    assert await feature.get_config() == {}

    release_preparation.set()
    with pytest.raises(ConfigTransitionError, match="config transition failed"):
        await update
    assert await feature.get_config() == {}


@pytest.mark.asyncio
async def test_persistence_failure_prevents_transition_hook_from_running(monkeypatch, tmp_path):
    """A non-privacy storage failure is a fail-closed transition boundary."""

    class FailingStorage(_FakeStorage):
        async def compare_and_swap_node(self, node_id, expected, new_node):
            raise OSError("storage unavailable")

    class TransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.preparation_calls = 0

        async def prepare_config_transition(self, config):
            self.preparation_calls += 1
            return ConfigTransitionResult.restart_required()

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = FailingStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=TransitionClient)
    await feature.initialize()
    old_client = feature._client

    with pytest.raises(RuntimeError, match="failed to persist config"):
        await feature.set_config({"enabled": True, "token": "next-token"})

    assert old_client.preparation_calls == 0
    assert old_client.stopped is False
    assert feature._client is old_client
    assert feature._host_config == {}


@pytest.mark.asyncio
async def test_volatile_privacy_noop_allows_transition_without_durable_config(
    monkeypatch, tmp_path
):
    """The explicit privacy-policy no-op is the one allowed write exception."""

    class VolatileStorage(_FakeStorage):
        def allows_persistent_writes(self):
            return False

    class TransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prepared = []

        async def prepare_config_transition(self, config):
            self.prepared.append(dict(config))
            return ConfigTransitionResult.applied()

    next_config = {"enabled": True, "token": "volatile-token"}
    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = VolatileStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=TransitionClient)
    await feature.initialize()
    client = feature._client

    await feature.set_config(next_config)

    assert client.prepared == [next_config]
    assert feature._client is client
    assert feature._host_config == next_config
    assert agent.storage.nodes == {}


@pytest.mark.asyncio
async def test_fenced_transition_failure_replaces_with_next_config(monkeypatch, tmp_path):
    """An unknown lifecycle outcome follows the SDK's required replacement path."""

    old_config = {"enabled": True, "token": "old-token"}
    next_config = {"enabled": True, "token": "next-token"}

    class FencedTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.replacement_required = False

        async def prepare_config_transition(self, config):
            self.replacement_required = True
            raise ConfigTransitionError("config transition failed")

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = _FakeStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = (
            FencedTransitionClient(**kwargs)
            if not clients
            else FakeIsolatedClient(**kwargs)
        )
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    await feature.persist_config(old_config)
    await feature.initialize()

    with pytest.raises(ConfigTransitionError, match="config transition failed"):
        await feature.set_config(next_config)

    # A cancellation or transport break can leave the child partly transitioned;
    # the SDK marks that state explicitly. The apply still fails to its caller,
    # but the host must not keep using that fenced old process.
    assert clients[0].stopped is True
    assert clients[1].kwargs["config"] == next_config
    assert feature._host_config == next_config


@pytest.mark.asyncio
async def test_fenced_transition_promotion_failure_restores_active_config_for_restart(
    monkeypatch, tmp_path
):
    """A fenced candidate is retired when its durable promotion fails.

    A cancelled/transport-broken SDK transition can have applied ``next_config``
    in the old child, so the proxy first replaces it with a candidate child on
    that config.  Fault the *following* promotion write: the live replacement,
    host state, and a fresh proxy must all return to the still-durable old
    config.  The existing supervisor must remain live on that restored child.
    """

    old_config = {"enabled": True, "token": "old-token"}
    next_config = {"enabled": False, "token": "next-token"}

    class PromotionFailingStorage(_FakeStorage):
        def __init__(self):
            super().__init__()
            self.cas_calls = 0

        async def compare_and_swap_node(self, node_id, expected, new_node):
            self.cas_calls += 1
            # Identify promotion by its exact state shape rather than write
            # ordinal: a pre-hook lease renewal is now a separate CAS.
            if (
                "pending_config" not in new_node.properties
                and new_node.properties.get("config") == next_config
            ):
                raise OSError("storage offline during promotion")
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class FencedTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.replacement_required = False

        async def prepare_config_transition(self, config):
            assert config == next_config
            self.replacement_required = True
            raise ConfigTransitionError("config transition failed")

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = PromotionFailingStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = (
            FencedTransitionClient(**kwargs)
            if not clients
            else FakeIsolatedClient(**kwargs)
        )
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    fresh = None
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(ConfigTransitionError, match="config transition failed"):
            await feature.set_config(next_config)

        # The fenced process is retired, but no next-config child is ever
        # started: its promotion failed while next_config was still pending.
        # The replacement and host-facing config therefore agree with the
        # durable active value.
        assert clients[0].stopped is True
        assert clients[1].kwargs["config"] == old_config
        assert feature._client is clients[1]
        assert feature._host_config == old_config
        assert await feature.get_config() == old_config
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == old_config
        assert "pending_config" not in properties
        assert "_isolated_pending_generation" not in properties

        # The recovery stays inside the serialized reload section without
        # disabling supervision. A later fresh proxy (host restart) also sees
        # and launches only the durable old config.
        assert feature._supervision_task is not None
        assert feature._supervision_task.done() is False
        restart_clients = []

        def restart_factory(**kwargs):
            client = FakeIsolatedClient(**kwargs)
            restart_clients.append(client)
            return client

        fresh = ProxyFeature(agent, _cfg_runtime(), client_factory=restart_factory)
        await fresh.initialize()
        assert fresh._host_config == old_config
        assert restart_clients[0].kwargs["config"] == old_config
    finally:
        if fresh is not None:
            await fresh.shutdown()
        await feature.shutdown()


@pytest.mark.asyncio
async def test_fenced_recovery_never_starts_candidate_before_durable_promotion(
    monkeypatch, tmp_path
):
    """A pending fenced candidate is never started, even detached.

    Child startup may allocate external resources. While promotion is blocked
    (and then fails), the only permitted replacement is the old configuration
    that the graph node still names as active.
    """

    old_config = {"enabled": True, "token": "old-token-not-for-logs"}
    next_config = {"enabled": False, "token": "next-token-not-for-logs"}
    promotion_started = asyncio.Event()
    release_promotion = asyncio.Event()

    class BlockingPromotionStorage(_FakeStorage):
        def __init__(self):
            super().__init__()
            self.cas_calls = 0

        async def compare_and_swap_node(self, node_id, expected, new_node):
            self.cas_calls += 1
            if (
                "pending_config" not in new_node.properties
                and new_node.properties.get("config") == next_config
            ):
                promotion_started.set()
                await release_promotion.wait()
                raise OSError("storage unavailable during promotion")
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class FencedTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.replacement_required = False

        async def prepare_config_transition(self, config):
            assert config == next_config
            self.replacement_required = True
            raise ConfigTransitionError("transition outcome unknown")

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = BlockingPromotionStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = (
            FencedTransitionClient(**kwargs)
            if not clients
            else FakeIsolatedClient(**kwargs)
        )
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    update = None

    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        update = asyncio.create_task(feature.set_config(next_config))
        await asyncio.wait_for(promotion_started.wait(), timeout=1)
        await asyncio.sleep(0)

        # The old child is fenced and removed from host traffic, but a pending
        # next-config candidate has not even been constructed or started.
        assert len(clients) == 1
        assert feature._client is None
        assert feature.get_tools() == []

        # Direct traffic is admitted only after recovery: it cannot reach the
        # fenced child while its candidate remains merely pending.
        blocked_call = asyncio.create_task(
            feature.call_isolated_tool("ping", {"message": "blocked"})
        )
        await asyncio.sleep(0)
        assert blocked_call.done() is False

        release_promotion.set()
        with pytest.raises(ConfigTransitionError, match="transition outcome unknown"):
            await update

        result = await blocked_call
        assert result["success"] is True

        # The only recovery child starts from the still-durable old config.
        assert len(clients) == 2
        assert clients[1].kwargs["config"] == old_config
        assert feature._client is clients[1]
        assert feature._host_config == old_config
    finally:
        release_promotion.set()
        if update is not None and not update.done():
            try:
                await update
            except (RuntimeError, ConfigTransitionError):
                pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_promotion_commit_then_raise_reconciles_to_committed_next_config(
    monkeypatch, tmp_path
):
    """A post-commit connection failure is not misreported as a rollback."""

    old_config = {"enabled": True, "token": "old-token-not-for-errors"}
    next_config = {"enabled": False, "token": "next-token-not-for-errors"}

    class CommitThenRaiseStorage(_CASStorage):
        async def compare_and_swap_node(self, node_id, expected, new_node):
            result = await super().compare_and_swap_node(node_id, expected, new_node)
            if (
                result == "swapped"
                and new_node.properties.get("config") == next_config
                and "pending_config" not in new_node.properties
            ):
                raise ConnectionError("connection dropped after commit")
            return result

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = CommitThenRaiseStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(RuntimeError, match="failed to persist config") as exc_info:
            await feature.set_config(next_config)

        # The caller learns that the transport outcome was uncertain, while
        # the local child and the durable source of truth follow the committed
        # next generation rather than a fictional rollback.
        assert old_config["token"] not in str(exc_info.value)
        assert next_config["token"] not in str(exc_info.value)
        assert clients[0].stopped is True
        assert clients[1].kwargs["config"] == next_config
        assert feature._client is clients[1]
        assert feature._host_config == next_config
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == next_config
        assert "pending_config" not in properties
        assert isinstance(properties["_isolated_config_generation"], str)
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_promotion_cancellation_after_commit_reconciles_before_propagating(
    monkeypatch, tmp_path
):
    """Cancellation after a committed CAS leaves no stale child or rollback lie."""

    old_config = {"enabled": True, "token": "old-token-not-for-errors"}
    next_config = {"enabled": False, "token": "next-token-not-for-errors"}
    promotion_committed = asyncio.Event()

    class CommitThenCancelStorage(_CASStorage):
        async def compare_and_swap_node(self, node_id, expected, new_node):
            result = await super().compare_and_swap_node(node_id, expected, new_node)
            if (
                result == "swapped"
                and new_node.properties.get("config") == next_config
                and "pending_config" not in new_node.properties
            ):
                promotion_committed.set()
                await asyncio.Event().wait()
            return result

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = CommitThenCancelStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    update = None
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        update = asyncio.create_task(feature.set_config(next_config))
        await asyncio.wait_for(promotion_committed.wait(), timeout=1)
        update.cancel()
        with pytest.raises(asyncio.CancelledError):
            await update

        assert clients[0].stopped is True
        assert clients[1].kwargs["config"] == next_config
        assert feature._client is clients[1]
        assert feature._host_config == next_config
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == next_config
        assert "pending_config" not in properties
        assert isinstance(properties["_isolated_config_generation"], str)
    finally:
        if update is not None and not update.done():
            update.cancel()
            with pytest.raises(asyncio.CancelledError):
                await update
        await feature.shutdown()


@pytest.mark.asyncio
async def test_stale_replica_cas_conflict_cannot_overwrite_winner_and_reconciles(
    monkeypatch, tmp_path
):
    """A loser with an old read reloads the winner instead of clobbering it."""

    old_config = {"enabled": True, "token": "old-token"}
    winner_config = {"enabled": False, "token": "winner-token"}
    loser_config = {"enabled": False, "token": "loser-token"}
    loser_stage_attempted = asyncio.Event()
    winner_promoted = asyncio.Event()

    class CoordinatedCASStorage(_CASStorage):
        async def compare_and_swap_node(self, node_id, expected, new_node):
            is_loser_stage = (
                new_node.properties.get("pending_config") == loser_config
            )
            if is_loser_stage:
                loser_stage_attempted.set()
                await winner_promoted.wait()
            result = await super().compare_and_swap_node(node_id, expected, new_node)
            if (
                result == "swapped"
                and new_node.properties.get("config") == winner_config
                and "pending_config" not in new_node.properties
            ):
                winner_promoted.set()
            return result

    storage = CoordinatedCASStorage()
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    winner_agent = Mock(did=_TEST_AGENT_DID, features={})
    winner_agent.storage = storage
    winner_agent.storage_path = str(tmp_path / "winner" / "kestrel_prime.db")
    loser_agent = Mock(did=_TEST_AGENT_DID, features={})
    loser_agent.storage = storage
    loser_agent.storage_path = str(tmp_path / "loser" / "kestrel_prime.db")
    winner_clients = []
    loser_clients = []

    def winner_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        winner_clients.append(client)
        return client

    def loser_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        loser_clients.append(client)
        return client

    winner = ProxyFeature(winner_agent, _cfg_runtime(), client_factory=winner_factory)
    loser = ProxyFeature(loser_agent, _cfg_runtime(), client_factory=loser_factory)
    loser_update = None
    try:
        await winner.persist_config(old_config)
        await winner.initialize()
        await loser.initialize()

        # The loser reads ``old_config`` and pauses immediately before its CAS.
        # The winner then stages/promotes from the same former snapshot.
        loser_update = asyncio.create_task(loser.set_config(loser_config))
        await asyncio.wait_for(loser_stage_attempted.wait(), timeout=1)
        await winner.set_config(winner_config)

        with pytest.raises(RuntimeError, match="conflicts with a newer durable state") as exc_info:
            await loser_update

        assert loser_config["token"] not in str(exc_info.value)
        assert winner_config["token"] not in str(exc_info.value)
        properties = storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == winner_config
        assert "pending_config" not in properties
        assert loser_clients[0].stopped is True
        assert loser_clients[1].kwargs["config"] == winner_config
        assert loser._host_config == winner_config
        assert await loser.get_config() == winner_config
    finally:
        if loser_update is not None and not loser_update.done():
            loser_update.cancel()
            with pytest.raises(asyncio.CancelledError):
                await loser_update
        await loser.shutdown()
        await winner.shutdown()


@pytest.mark.asyncio
async def test_promotion_conflict_reconciles_a_live_applied_child_to_durable_winner(
    monkeypatch, tmp_path
):
    """A losing promotion replaces a locally applied child from the re-read node."""

    old_config = {"enabled": True, "token": "old-token"}
    next_config = {"enabled": False, "token": "next-token"}
    winner_config = {"enabled": False, "token": "winner-token"}

    class PromotionConflictStorage(_CASStorage):
        async def compare_and_swap_node(self, node_id, expected, new_node):
            if (
                new_node.properties.get("config") == next_config
                and "pending_config" not in new_node.properties
            ):
                # Another replica has already promoted a different generation
                # after this proxy staged and live-applied its own candidate.
                current = self.nodes[node_id]
                current.properties = {
                    "config": dict(winner_config),
                    "_isolated_config_generation": "other-replica",
                }
                return "predicate_failed"
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class LiveApplyClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == next_config
            return ConfigTransitionResult.applied()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = PromotionConflictStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = LiveApplyClient(**kwargs) if not clients else FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(RuntimeError, match="conflicts with a newer durable state") as exc_info:
            await feature.set_config(next_config)

        assert old_config["token"] not in str(exc_info.value)
        assert next_config["token"] not in str(exc_info.value)
        assert winner_config["token"] not in str(exc_info.value)
        assert clients[0].stopped is True
        assert clients[1].kwargs["config"] == winner_config
        assert feature._client is clients[1]
        assert feature._host_config == winner_config
        assert await feature.get_config() == winner_config
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == winner_config
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_promotion_policy_probe_failure_reconciles_live_child_to_staged_active(
    monkeypatch, tmp_path
):
    """A post-hook policy failure cannot leave an in-process next config live."""

    next_config = {"enabled": False, "token": "next-token-not-for-errors"}

    class FlappingPolicyStorage(_CASStorage):
        def __init__(self):
            super().__init__()
            self.policy_calls = 0

        def allows_persistent_writes(self):
            self.policy_calls += 1
            if self.policy_calls == 2:
                raise OSError("policy backend unavailable")
            return True

    class LiveApplyClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == next_config
            return ConfigTransitionResult.applied()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = FlappingPolicyStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = LiveApplyClient(**kwargs) if not clients else FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.initialize()

        with pytest.raises(RuntimeError, match="failed to persist config") as exc_info:
            await feature.set_config(next_config)

        assert next_config["token"] not in str(exc_info.value)
        assert clients[0].stopped is True
        assert clients[1].kwargs["config"] == {}
        assert feature._client is clients[1]
        assert feature._host_config == {}
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == {}
        assert "pending_config" not in properties
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_fenced_promotion_recovery_quarantines_when_old_child_cannot_restart(
    monkeypatch, tmp_path
):
    """A failed restoration exposes no candidate-config client to the host."""

    old_config = {"enabled": True, "token": "old-token"}
    next_config = {"enabled": False, "token": "next-token"}

    class PromotionFailingStorage(_FakeStorage):
        def __init__(self):
            super().__init__()
            self.cas_calls = 0

        async def compare_and_swap_node(self, node_id, expected, new_node):
            self.cas_calls += 1
            if (
                "pending_config" not in new_node.properties
                and new_node.properties.get("config") == next_config
            ):
                raise OSError("storage offline during promotion")
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class FencedTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.replacement_required = False

        async def prepare_config_transition(self, config):
            self.replacement_required = True
            raise ConfigTransitionError("config transition failed")

    class FailingRestoreClient(FakeIsolatedClient):
        async def start(self):
            raise RuntimeError("old-config child could not start")

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = PromotionFailingStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        if not clients:
            client = FencedTransitionClient(**kwargs)
        else:
            client = FailingRestoreClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()

        with pytest.raises(IsolatedRuntimePreparationError) as failure:
            await feature.set_config(next_config)
        assert "old-config child could not start" not in str(failure.value)
        assert isinstance(failure.value.__cause__, RuntimeError)

        # No pending next-config candidate starts. The sole attempted recovery
        # child uses durable old config, fails, and remains detached; the
        # supervisor latch prevents a later candidate resurrection.
        assert clients[1].kwargs["config"] == old_config
        assert clients[1].stopped is True
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._host_config == old_config
        assert feature._stopping is True
        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == old_config
        assert "pending_config" not in properties
        assert "_isolated_pending_generation" not in properties
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_live_applied_transition_retains_the_initialized_service(monkeypatch, tmp_path):
    """An SDK ``applied`` result keeps the existing process alive."""

    old_config = {"enabled": True, "token": "old-token"}
    next_config = {"enabled": False, "token": "next-token"}

    class LiveApplyClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prepared = []

        async def prepare_config_transition(self, config):
            self.prepared.append(dict(config))
            return ConfigTransitionResult.applied()

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = _FakeStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=LiveApplyClient)
    await feature.persist_config(old_config)
    await feature.initialize()
    client = feature._client

    await feature.set_config(next_config)

    assert feature._client is client
    assert client.prepared == [next_config]
    assert client.stopped is False
    assert feature._host_config == next_config


@pytest.mark.asyncio
@pytest.mark.parametrize("transition_action", ["applied", "restart"])
async def test_failed_promotion_restores_previous_child_and_config(
    monkeypatch, tmp_path, transition_action
):
    """A failed second write cannot leave a successful hook's config live.

    Both negotiated outcomes can mutate or retire the current service before
    durable promotion. The host must therefore replace it from the staged
    active config when that promotion write fails.
    """

    old_config = {"enabled": True, "token": "old-token"}
    next_config = {"enabled": False, "token": "next-token"}

    class PromotionFailingStorage(_FakeStorage):
        def __init__(self):
            super().__init__()
            self.cas_calls = 0

        async def compare_and_swap_node(self, node_id, expected, new_node):
            self.cas_calls += 1
            if (
                "pending_config" not in new_node.properties
                and new_node.properties.get("config") == next_config
            ):
                raise OSError("storage offline during promotion")
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class TransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.transitioned_to = None

        async def prepare_config_transition(self, config):
            self.transitioned_to = dict(config)
            if transition_action == "applied":
                return ConfigTransitionResult.applied()
            return ConfigTransitionResult.restart_required()

    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = PromotionFailingStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = TransitionClient(**kwargs) if not clients else FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    await feature.persist_config(old_config)
    await feature.initialize()

    with pytest.raises(RuntimeError, match="failed to persist config"):
        await feature.set_config(next_config)

    # The child that ran the successful transition is retired. Its replacement,
    # proxy state, and durable active value all agree on the prior config.
    assert clients[0].transitioned_to == next_config
    assert clients[0].stopped is True
    assert clients[1].kwargs["config"] == old_config
    assert feature._client is clients[1]
    assert feature._host_config == old_config
    assert await feature.get_config() == old_config
    properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
    assert properties["config"] == old_config
    assert "pending_config" not in properties
    assert "_isolated_pending_generation" not in properties


@pytest.mark.asyncio
async def test_supervision_restarts_on_sdk_fenced_health_response(tmp_path, monkeypatch):
    """The real SDK's fenced health envelope is not a healthy non-empty dict."""

    import kestrel_sovereign.features.isolated_runtime as ir

    health_checked = asyncio.Event()
    restarted = asyncio.Event()

    class FencedHealthClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.start_calls = 0

        async def health(self):
            health_checked.set()
            if self.start_calls:
                return {"status": "ready", "ready": True}
            # This is the exact return shape from SDK health() after a
            # transition fences a child for replacement.
            return {"status": "restart-required", "ready": False}

        async def stop(self):
            self.stop_calls += 1
            await super().stop()

        async def start(self):
            self.start_calls += 1
            restarted.set()
            await super().start()

    client = FencedHealthClient()
    feature = ProxyFeature(
        Mock(storage_path=str(tmp_path / "agent" / "kestrel_prime.db"), features={}),
        _cfg_runtime(),
        client_factory=FakeIsolatedClient,
    )
    feature._client = client

    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(ir.asyncio, "sleep", immediate_sleep)
    supervisor = asyncio.create_task(feature._supervise())
    try:
        await asyncio.wait_for(health_checked.wait(), timeout=1)
        await asyncio.wait_for(restarted.wait(), timeout=1)
        assert client.stop_calls == 1
        assert client.start_calls == 1
    finally:
        feature._stopping = True
        supervisor.cancel()
        try:
            await supervisor
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_config_transition_serializes_with_a_pending_health_restart(
    monkeypatch, tmp_path
):
    """A stale failed health probe cannot restart through a config transition."""

    import kestrel_sovereign.features.isolated_runtime as ir

    health_started = asyncio.Event()
    release_health = asyncio.Event()
    preparation_started = asyncio.Event()
    release_preparation = asyncio.Event()

    class HealthAndTransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def health(self):
            health_started.set()
            await release_health.wait()
            return False

        async def prepare_config_transition(self, config):
            preparation_started.set()
            await release_preparation.wait()
            return ConfigTransitionResult.restart_required()

        async def stop(self):
            self.stop_calls += 1
            await super().stop()

    old_client = HealthAndTransitionClient()
    replacement_clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        replacement_clients.append(client)
        return client

    agent = Mock(did=_TEST_AGENT_DID, storage_path=str(tmp_path / "agent" / "kestrel_prime.db"), features={})
    agent.storage = _FakeStorage()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    feature._client = old_client
    feature._host_config = {"enabled": True, "token": "old-token"}

    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(ir.asyncio, "sleep", immediate_sleep)
    supervisor = asyncio.create_task(feature._supervise())
    try:
        await asyncio.wait_for(health_started.wait(), timeout=1)
        update = asyncio.create_task(
            feature.set_config({"enabled": False, "token": "next-token"})
        )
        await asyncio.wait_for(preparation_started.wait(), timeout=1)

        # The supervisor has a stale unhealthy result but cannot acquire the
        # shared lifecycle lock until the transition has fully prepared and
        # replaced the child.
        release_health.set()
        release_preparation.set()
        await update
        await real_sleep(0)

        assert old_client.stop_calls == 1
        assert len(replacement_clients) == 1
        assert replacement_clients[0].started is True
        assert replacement_clients[0].stopped is False
    finally:
        feature._stopping = True
        supervisor.cancel()
        try:
            await supervisor
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_persisted_config_survives_restart(monkeypatch, tmp_path):
    """A config set on one ProxyFeature is loaded by a fresh one (restart)."""
    agent = Mock(did=_TEST_AGENT_DID)
    agent.features = {}
    agent.storage = _FakeStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    f1 = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    await f1.initialize()
    await f1.set_config({"allowed_senders": ["777"], "token": "t"})

    # Fresh feature (simulating a host/agent restart) reads the persisted node.
    f2 = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    await f2.initialize()
    assert (await f2.get_config())["allowed_senders"] == ["777"]
    # And the reloaded service was launched with that config.
    assert f2._host_config["allowed_senders"] == ["777"]


@pytest.mark.asyncio
async def test_reenable_restarts_live_health_supervisor(tmp_path):
    """Runtime disable → re-enable on the SAME ProxyFeature instance must leave a
    LIVE health supervisor.

    ``shutdown()`` latches ``_stopping=True`` to unwind the supervisor; the
    canonical re-enable (``_activate_feature_runtime``) re-runs ``initialize()``
    on the same instance. Before the fix the stale ``_stopping`` made the new
    ``_supervise()`` task exit on its first ``while not self._stopping`` check,
    silently leaving the re-enabled service with NO health supervisor
    (kestrel-sovereign#2522 P2). This drives the real initialize/shutdown/
    initialize lifecycle and proves the second supervisor stays alive AND
    actively probes the freshly launched client."""
    import asyncio
    import os

    import kestrel_sovereign.features.isolated_runtime as ir

    clients = []

    class CountingClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.health_calls = 0

        async def health(self):
            self.health_calls += 1
            return True

    def factory(**kw):
        client = CountingClient(**kw)
        clients.append(client)
        return client

    runtime = InstalledFeatureRuntime(
        class_name="ReenableFeature",
        entry_point="r.feature:ReenableFeature",
        distribution="r-pkg",
        runtime="isolated-venv",
        service="r",
    )
    agent = Mock(did=_TEST_AGENT_DID, storage_path=str(tmp_path / "a" / "db.db"), features={})
    feature = ProxyFeature(agent, runtime, client_factory=factory)
    os.environ["KESTREL_FEATURE_REENABLEFEATURE_BIN"] = str(tmp_path / "r-bin")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(ir, "_HEALTH_PROBE_TIMEOUT", 0.05)
    try:
        # First enable → supervisor #1 against client #1.
        await feature.initialize()
        first_task = feature._supervision_task
        assert first_task is not None and not first_task.done()
        assert feature._stopping is False

        # Canonical disable latches _stopping and unwinds supervisor #1.
        await feature.shutdown()
        assert feature._stopping is True
        assert first_task.done()

        # Re-enable: initialize() re-runs on the SAME instance.
        await feature.initialize()
        second_task = feature._supervision_task
        # The lifecycle flag was reset to the fresh-start baseline.
        assert feature._stopping is False
        assert second_task is not first_task
        second_client = clients[-1]

        # The second supervisor is LIVE — it does NOT exit on its first check and
        # it actively probes the newly launched client. Before the fix
        # second_task would already be done() and health_calls would stay 0.
        for _ in range(150):
            if second_client.health_calls >= 1:
                break
            await asyncio.sleep(0.02)
        assert not second_task.done(), (
            "re-enabled health supervisor exited immediately (stale _stopping)"
        )
        assert second_client.health_calls >= 1, (
            "re-enabled supervisor never probed health — supervisor is dead"
        )
    finally:
        feature._stopping = True
        if feature._supervision_task:
            feature._supervision_task.cancel()
            try:
                await feature._supervision_task
            except asyncio.CancelledError:
                pass
        monkey.undo()
        os.environ.pop("KESTREL_FEATURE_REENABLEFEATURE_BIN", None)


@pytest.mark.asyncio
async def test_live_apply_blocks_tools_and_drops_stale_legacy_inbound_until_promotion(
    monkeypatch, tmp_path
):
    """No host-visible effect may observe an applied candidate before its CAS.

    The client intentionally mutates its local mode before reporting ``applied``.
    Promotion is then held at the durable write boundary while a direct tool,
    the generic channel adapter, and an SDK inbound callback all try to enter.
    Tool/channel calls wait for the finite transition. The legacy SDK callback
    has no producer cursor or NACK, so a callback from the old child is stale
    once the live gate closes and must not replay under the promoted config.
    """

    from kestrel_sdk.channels import ChannelMessage, MessageDirection

    old_config = {"mode": "old", "enabled": True}
    next_config = {"mode": "next", "enabled": True}
    promotion_started = asyncio.Event()
    release_promotion = asyncio.Event()

    class BlockingPromotionStorage(_CASStorage):
        async def compare_and_swap_node(self, node_id, expected, new_node):
            if (
                new_node.properties.get("config") == next_config
                and "pending_config" not in new_node.properties
            ):
                promotion_started.set()
                await release_promotion.wait()
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class LiveApplyChannelClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.mode = dict(kwargs["config"])["mode"]
            self.effects = []

        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

        async def prepare_config_transition(self, config):
            self.mode = config["mode"]
            return ConfigTransitionResult.applied()

        async def call_tool(self, name, args):
            self.effects.append((name, self.mode))
            return {"ok": True, "data": {"message_id": "receipt"}}

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = BlockingPromotionStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=LiveApplyChannelClient)
    update = None
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        client = feature._client
        adapter = channel_feature.registry.get("whatsapp")
        assert adapter is not None

        update = asyncio.create_task(feature.set_config(next_config))
        await asyncio.wait_for(promotion_started.wait(), timeout=1)
        assert client.mode == "next"  # hook already applied locally
        assert agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties["config"] == old_config

        tool_call = asyncio.create_task(feature.call_isolated_tool("ping", {}))
        channel_send = asyncio.create_task(adapter.send_message(to="+1", content="hello"))
        inbound = ChannelMessage(
            channel_type="whatsapp",
            direction=MessageDirection.INBOUND,
            sender="sender",
            recipient="agent",
            content="hello",
        )
        inbound_callback = asyncio.create_task(
            feature._handle_event({"type": "channel.inbound", "payload": inbound.to_dict()})
        )
        await asyncio.sleep(0)

        assert not tool_call.done()
        assert not channel_send.done()
        assert inbound_callback.done()
        assert client.effects == []
        assert channel_feature.inbound == []

        release_promotion.set()
        await update
        assert (await tool_call)["success"] is True
        assert (await channel_send).status.value == "success"
        await inbound_callback
        assert client.effects == [("ping", "next"), ("whatsapp_send", "next")]
        await asyncio.sleep(0)
        assert channel_feature.inbound == []
    finally:
        release_promotion.set()
        if update is not None and not update.done():
            await update
        await feature.shutdown()


@pytest.mark.asyncio
async def test_live_apply_republishes_config_dependent_tool_and_channel_inventory(
    monkeypatch, tmp_path
):
    """A successful in-process transition refreshes every host advertisement."""

    old_config = {"mode": "old", "enabled": True}
    next_config = {"mode": "next", "enabled": True}

    class InventoryClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.mode = kwargs["config"]["mode"]

        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": f"{self.mode}_send",
                }
            }

        async def list_tools(self):
            return [{"name": f"{self.mode}_tool", "description": self.mode}]

        async def prepare_config_transition(self, config):
            self.mode = config["mode"]
            return ConfigTransitionResult.applied()

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=InventoryClient)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        old_adapter = channel_feature.registry.get("whatsapp")
        assert [tool.name for tool in feature.get_tools()] == ["old_tool"]
        assert old_adapter._send_tool == "old_send"

        await feature.set_config(next_config)

        new_adapter = channel_feature.registry.get("whatsapp")
        assert feature._client.stopped is False
        assert [tool.name for tool in feature.get_tools()] == ["next_tool"]
        assert new_adapter is not old_adapter
        assert new_adapter._send_tool == "next_send"
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_persistent_transition_requires_compare_and_swap_before_hook(monkeypatch, tmp_path):
    """An upsert-only storage surface fails before an SDK hook can run."""

    class NoCASStorage:
        def __init__(self):
            self.nodes = {}
            self.agent_id = _TEST_AGENT_DID

        async def add_node(self, node):
            self.nodes[node.node_id] = node

        async def get_node(self, node_id):
            return self.nodes.get(node_id)

    class TransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prepare_calls = 0

        async def prepare_config_transition(self, config):
            self.prepare_calls += 1
            return ConfigTransitionResult.restart_required()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = NoCASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=TransitionClient)
    try:
        with pytest.raises(RuntimeError, match="requires compare_and_swap_node"):
            await feature.persist_config({"enabled": True})
        await feature.initialize()
        with pytest.raises(RuntimeError, match="failed to persist config"):
            await feature.set_config({"enabled": False})
        assert feature._client.prepare_calls == 0
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_healthy_long_hook_renews_lease_before_another_replica_can_takeover(
    monkeypatch, tmp_path
):
    """A live hook remains owned beyond its original lease interval."""

    old_config = {"enabled": True}
    first_config = {"enabled": False, "revision": "first"}
    second_config = {"enabled": False, "revision": "second"}
    hook_started = asyncio.Event()
    release_hook = asyncio.Event()
    lease_renewed = asyncio.Event()
    monkeypatch.setattr(isolated_runtime, "_PENDING_CONFIG_LEASE_TTL", timedelta(milliseconds=60))
    monkeypatch.setattr(isolated_runtime, "_PENDING_CONFIG_CLOCK_SKEW", timedelta(0))
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    class LongHookClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            hook_started.set()
            await release_hook.wait()
            return ConfigTransitionResult.restart_required()

    class RenewalObservingStorage(_CASStorage):
        def __init__(self):
            super().__init__()
            self.initial_lease = None

        async def compare_and_swap_node(self, node_id, expected, new_node):
            lease = new_node.properties.get("_isolated_pending_lease_expires_at")
            if isinstance(lease, str):
                if self.initial_lease is None:
                    self.initial_lease = datetime.fromisoformat(lease)
                elif (
                    datetime.fromisoformat(lease)
                    > self.initial_lease + timedelta(milliseconds=5)
                ):
                    # The post-reconciliation fence renewal happens almost
                    # immediately. Wait for the heartbeat's later renewal so
                    # this test still proves long-hook ownership beyond the
                    # original lease deadline.
                    lease_renewed.set()
            return await super().compare_and_swap_node(node_id, expected, new_node)

    storage = RenewalObservingStorage()
    first_agent = Mock(did=_TEST_AGENT_DID, features={})
    first_agent.storage = storage
    first_agent.storage_path = str(tmp_path / "first" / "kestrel_prime.db")
    second_agent = Mock(did=_TEST_AGENT_DID, features={})
    second_agent.storage = storage
    second_agent.storage_path = str(tmp_path / "second" / "kestrel_prime.db")
    first = ProxyFeature(first_agent, _cfg_runtime(), client_factory=LongHookClient)
    second = ProxyFeature(second_agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    first_update = None
    try:
        await first.persist_config(old_config)
        await first.initialize()
        await second.initialize()
        first_update = asyncio.create_task(first.set_config(first_config))
        await asyncio.wait_for(hook_started.wait(), timeout=1)
        await asyncio.wait_for(lease_renewed.wait(), timeout=1)
        assert storage.initial_lease is not None
        # Advance the takeover clock past the original lease.  It remains
        # before the renewed generation-scoped lease, so a healthy hook cannot
        # be reclaimed merely because its first two-minute interval elapsed.
        monkeypatch.setattr(
            isolated_runtime,
            "_utc_now",
            lambda: storage.initial_lease + timedelta(milliseconds=1),
        )
        renewed_expires_at = datetime.fromisoformat(
            storage.nodes[_TEST_CONFIG_NODE_ID].properties[
                "_isolated_pending_lease_expires_at"
            ]
        )
        assert renewed_expires_at > isolated_runtime._utc_now()

        with pytest.raises(RuntimeError, match="already in progress"):
            await second.set_config(second_config)
        assert storage.nodes[_TEST_CONFIG_NODE_ID].properties["pending_config"] == first_config

        release_hook.set()
        await first_update
    finally:
        release_hook.set()
        if first_update is not None and not first_update.done():
            await first_update
        await second.shutdown()
        await first.shutdown()


@pytest.mark.asyncio
async def test_second_cancellation_waits_for_owned_cleanup_before_releasing_reload_lock(
    monkeypatch, tmp_path
):
    """Repeated cancellation cannot leave a cleanup task mutating after unlock."""

    old_config = {"enabled": True}
    next_config = {"enabled": False}
    hook_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    class BlockingCleanupStorage(_CASStorage):
        async def compare_and_swap_node(self, node_id, expected, new_node):
            is_owned_cleanup = (
                isinstance(expected, dict)
                and "pending_config" in expected
                and "pending_config" not in new_node.properties
            )
            if is_owned_cleanup:
                cleanup_started.set()
                await release_cleanup.wait()
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class BlockingClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            hook_started.set()
            await asyncio.Event().wait()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = BlockingCleanupStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=BlockingClient)
    update = None
    second_update = None
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        update = asyncio.create_task(feature.set_config(next_config))
        await asyncio.wait_for(hook_started.wait(), timeout=1)
        update.cancel()
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)

        update.cancel()
        second_update = asyncio.create_task(feature.set_config({"enabled": True, "revision": 2}))
        await asyncio.sleep(0)
        assert not update.done()
        assert not second_update.done()
        assert "pending_config" in agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties

        # It is still waiting on the first transition's reload lock, so this
        # cancellation must prevent a second generation from being staged once
        # cleanup releases the lock.
        second_update.cancel()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await update
        assert "pending_config" not in agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties

        with pytest.raises(asyncio.CancelledError):
            await second_update
    finally:
        release_cleanup.set()
        for task in (update, second_update):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_stale_empty_replica_patch_preserves_concurrent_secret_rotation_at_stage_cas(
    monkeypatch, tmp_path
):
    """An initially empty replica cannot re-stage an old write-only secret."""

    first_secret_config = {"enabled": True, "api_key": "first-secret"}
    rotated_config = {"enabled": True, "api_key": "rotated-secret"}
    stale_patch = {"enabled": False}
    stale_stage_ready = asyncio.Event()
    release_stale_stage = asyncio.Event()

    class CoordinatedStorage(_CASStorage):
        def __init__(self):
            super().__init__()
            self._paused = False

        async def compare_and_swap_node(self, node_id, expected, new_node):
            pending = new_node.properties.get("pending_config")
            if (
                not self._paused
                and pending == {"enabled": False, "api_key": "first-secret"}
            ):
                self._paused = True
                stale_stage_ready.set()
                await release_stale_stage.wait()
            return await super().compare_and_swap_node(node_id, expected, new_node)

    storage = CoordinatedStorage()
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    stale_agent = Mock(did=_TEST_AGENT_DID, features={})
    stale_agent.storage = storage
    stale_agent.storage_path = str(tmp_path / "stale" / "kestrel_prime.db")
    writer_agent = Mock(did=_TEST_AGENT_DID, features={})
    writer_agent.storage = storage
    writer_agent.storage_path = str(tmp_path / "writer" / "kestrel_prime.db")
    stale = ProxyFeature(stale_agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    writer = ProxyFeature(writer_agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    stale_update = None
    try:
        # The stale replica has legitimately loaded an empty config before any
        # credential exists.  Hosted ``get_config`` must subsequently re-read
        # storage, not return this cached empty dict forever.
        await stale.initialize()
        await writer.initialize()
        assert await stale.get_config() == {}

        await writer.set_config(first_secret_config)
        assert await stale.get_config() == first_secret_config

        stale_update = asyncio.create_task(
            stale.set_config_with_secret_preservation(
                stale_patch,
                {"api_key"},
                lambda effective: None,
            )
        )
        await asyncio.wait_for(stale_stage_ready.wait(), timeout=1)

        # Rotate the credential after stale preservation chose the first value
        # but before its stage predicate commits.  The stale CAS must lose,
        # re-read, and preserve this winner instead.
        await writer.set_config(rotated_config)
        release_stale_stage.set()
        await stale_update

        properties = storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == {"enabled": False, "api_key": "rotated-secret"}
        assert "pending_config" not in properties
    finally:
        release_stale_stage.set()
        if stale_update is not None and not stale_update.done():
            await stale_update
        await writer.shutdown()
        await stale.shutdown()


@pytest.mark.asyncio
async def test_replica_get_does_not_mask_stale_child_before_next_patch(
    monkeypatch, tmp_path
):
    """A durable GET cannot make a stale local child skip reconciliation.

    Replica two is still running ``old_config`` when replica one promotes
    ``winner_config``.  Its GET must return the durable winner, but a later
    PATCH must first replace the locally stale child before it invokes the
    negotiated hook.
    """

    old_config = {"enabled": True, "revision": "old"}
    winner_config = {"enabled": True, "revision": "winner"}
    next_config = {"enabled": False, "revision": "next"}
    storage = _CASStorage()
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    winner_agent = Mock(did=_TEST_AGENT_DID, features={})
    winner_agent.storage = storage
    winner_agent.storage_path = str(tmp_path / "winner" / "kestrel_prime.db")
    stale_agent = Mock(did=_TEST_AGENT_DID, features={})
    stale_agent.storage = storage
    stale_agent.storage_path = str(tmp_path / "stale" / "kestrel_prime.db")
    stale_clients = []

    class ConfigAwareClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prepared = []

        async def prepare_config_transition(self, config):
            # This hook owns resources created by the child config supplied at
            # initialize.  It must never receive the next PATCH while those
            # resources still describe the old durable generation.
            assert self.kwargs["config"] == winner_config
            self.prepared.append(dict(config))
            return ConfigTransitionResult.applied()

    def stale_factory(**kwargs):
        client = ConfigAwareClient(**kwargs)
        stale_clients.append(client)
        return client

    winner = ProxyFeature(
        winner_agent, _cfg_runtime(), client_factory=FakeIsolatedClient
    )
    stale = ProxyFeature(stale_agent, _cfg_runtime(), client_factory=stale_factory)
    try:
        await winner.persist_config(old_config)
        await winner.initialize()
        await stale.initialize()
        stale_child = stale_clients[0]

        await winner.set_config(winner_config)
        assert await stale.get_config() == winner_config
        # GET reports durable state without rewriting the identity of the
        # child that is actually still running old_config.
        assert stale._host_config == old_config

        await stale.set_config(next_config)

        assert stale_child.stopped is True
        assert stale._client is stale_clients[1]
        assert stale_clients[1].kwargs["config"] == winner_config
        assert stale_clients[1].prepared == [next_config]
        assert (
            storage.nodes[_TEST_CONFIG_NODE_ID].properties["config"]
            == next_config
        )
    finally:
        await stale.shutdown()
        await winner.shutdown()


@pytest.mark.asyncio
async def test_stale_equal_config_reconciles_only_after_external_ingress_fence(
    monkeypatch, tmp_path
):
    """A stale ``B -> B`` update fences old polling before replacing child A."""

    old_config = {"enabled": True, "revision": "old"}
    winner_config = {"enabled": True, "revision": "winner"}
    storage = _CASStorage()
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    winner_agent = Mock(did=_TEST_AGENT_DID, features={})
    winner_agent.storage = storage
    winner_agent.storage_path = str(tmp_path / "winner" / "kestrel_prime.db")
    stale_agent = Mock(did=_TEST_AGENT_DID, features={})
    stale_agent.storage = storage
    stale_agent.storage_path = str(tmp_path / "stale" / "kestrel_prime.db")
    stale_clients = []

    class FencedStaleClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.quiesced = False
            self.ingress_calls = []
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("external-ingress-quiesce", "external-ingress-resume")
            )

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if name == "external-ingress-quiesce":
                assert stale._traffic_gate.closed is False
                self.quiesced = True
                return {"status": "ok", "http_status": 200, "state": "quiesced"}
            raise AssertionError("a retired child must not be resumed")

        async def stop(self):
            if self is stale_clients[0]:
                assert self.quiesced is True
                assert stale._traffic_gate.closed is True
            await super().stop()

    def stale_factory(**kwargs):
        client = FencedStaleClient(**kwargs)
        stale_clients.append(client)
        return client

    winner = ProxyFeature(
        winner_agent, _cfg_runtime(), client_factory=FakeIsolatedClient
    )
    stale = ProxyFeature(stale_agent, _cfg_runtime(), client_factory=stale_factory)
    try:
        await winner.persist_config(old_config)
        await winner.initialize()
        await stale.initialize()
        stale_child = stale_clients[0]

        await winner.set_config(winner_config)
        # Stale's local applied identity remains A even though its stage reads
        # durable B, which is exactly the prior no-op fast-path hole.
        assert stale._host_config == old_config
        await stale.set_config(winner_config)

        assert stale_child.stopped is True
        assert [call[0] for call in stale_child.ingress_calls] == [
            "external-ingress-quiesce"
        ]
        assert stale._client is stale_clients[1]
        assert stale_clients[1].kwargs["config"] == winner_config
        assert stale._host_config == winner_config
    finally:
        await stale.shutdown()
        await winner.shutdown()


@pytest.mark.asyncio
async def test_ambiguous_promotion_reread_quarantine_cannot_publish_recovery_child(
    monkeypatch, tmp_path
):
    """Storage recovery after quarantine stays terminal until initialize()."""

    old_config = {"enabled": True, "revision": "old"}
    next_config = {"enabled": False, "revision": "next"}

    class PromotionReadFailureStorage(_CASStorage):
        def __init__(self):
            super().__init__()
            self.cas_calls = 0
            self.fail_next_read = False

        async def compare_and_swap_node(self, node_id, expected, new_node):
            self.cas_calls += 1
            # The stage and pre-hook renewal succeed.  The promotion response
            # is ambiguous, and its required durable re-read then fails,
            # latching quarantine.
            if (
                "pending_config" not in new_node.properties
                and new_node.properties.get("config") == next_config
            ):
                self.fail_next_read = True
                raise OSError("promotion transport failed")
            return await super().compare_and_swap_node(node_id, expected, new_node)

        async def get_node(self, node_id):
            if self.fail_next_read:
                self.fail_next_read = False
                raise OSError("promotion reread unavailable")
            return await super().get_node(node_id)

    class LiveApplyClient(FakeIsolatedClient):
        supports_config_transition = True

        async def prepare_config_transition(self, config):
            assert config == next_config
            return ConfigTransitionResult.applied()

    storage = PromotionReadFailureStorage()
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = storage
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = (
            LiveApplyClient(**kwargs) if not clients else FakeIsolatedClient(**kwargs)
        )
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        old_child = clients[0]

        with pytest.raises(RuntimeError, match="could not reconcile durable config"):
            await feature.set_config(next_config)

        # The follow-up cleanup can read storage again and clear its pending
        # generation, but it must not force-start a replacement behind the
        # terminal gate that the ambiguous reread already sealed.
        properties = storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == old_config
        assert "pending_config" not in properties
        assert old_child.stopped is True
        assert clients == [old_child]
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._supervision_task is None
        assert feature._stopping is True
        assert feature._traffic_gate.sealed is True

        # Only an explicit enable-cycle initialization may create a new child.
        await feature.initialize()
        assert len(clients) == 2
        assert clients[1].kwargs["config"] == old_config
        assert feature._client is clients[1]
        assert feature._traffic_gate.sealed is False
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_fenced_cancellation_promotes_generation_before_starting_replacement(
    monkeypatch, tmp_path
):
    """A fenced cancellation publishes only a durably promoted next child."""

    old_config = {"enabled": True, "revision": "old"}
    next_config = {"enabled": False, "revision": "next"}
    hook_started = asyncio.Event()

    class FencedCancellationClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.replacement_required = False

        async def prepare_config_transition(self, config):
            hook_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.replacement_required = True
                raise

    storage = _CASStorage()
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = storage
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = FencedCancellationClient(**kwargs) if not clients else FakeIsolatedClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    update = None
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        update = asyncio.create_task(feature.set_config(next_config))
        await asyncio.wait_for(hook_started.wait(), timeout=1)
        update.cancel()

        async def _wait_for_fence():
            while not clients[0].replacement_required:
                await asyncio.sleep(0)

        await asyncio.wait_for(_wait_for_fence(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await update

        properties = storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == next_config
        assert "pending_config" not in properties
        assert clients[0].stopped is True
        assert clients[1].kwargs["config"] == next_config
        assert feature._client is clients[1]
    finally:
        if update is not None and not update.done():
            update.cancel()
            with pytest.raises(asyncio.CancelledError):
                await update
        await feature.shutdown()


@pytest.mark.asyncio
async def test_fenced_recovery_old_client_stop_error_cleans_owned_pending_and_quarantines(
    monkeypatch, tmp_path
):
    """A failed old-child stop never leaves its generation wedged in storage."""

    old_config = {"enabled": True}
    next_config = {"enabled": False}

    class StopFailingFencedClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.replacement_required = False

        async def prepare_config_transition(self, config):
            self.replacement_required = True
            raise ConfigTransitionError("transition transport failed")

        async def stop(self):
            self.stopped = True
            raise OSError("old child would not stop")

    storage = _CASStorage()
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = storage
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=StopFailingFencedClient)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        with pytest.raises(OSError, match="old child would not stop"):
            await feature.set_config(next_config)

        properties = storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == old_config
        assert "pending_config" not in properties
        assert feature._client is None
        assert feature._stopping is True
        assert feature._terminal_retirement_clients
        assert feature._traffic_gate.sealed is True
    finally:
        with pytest.raises(
            RuntimeError, match="isolated feature terminal retirement is incomplete"
        ):
            await feature.shutdown()


@pytest.mark.asyncio
async def test_set_config_cancellation_while_gate_drains_reopens_before_staging(
    monkeypatch, tmp_path
):
    """Cancelling an admitted-call drain cannot wedge config traffic closed."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class BlockingToolClient(FakeIsolatedClient):
        async def call_tool(self, name, args):
            self.calls.append((name, args))
            active_started.set()
            await release_active.wait()
            return {"echo": args}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=BlockingToolClient)
    active = update = None
    try:
        await feature.persist_config({"enabled": True})
        await feature.initialize()
        active = asyncio.create_task(feature.call_isolated_tool("hold", {}))
        await asyncio.wait_for(active_started.wait(), timeout=1)

        update = asyncio.create_task(feature.set_config({"enabled": False}))
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)
        assert feature._traffic_gate.closed

        update.cancel()
        await asyncio.sleep(0)
        assert not update.done()
        release_active.set()

        with pytest.raises(asyncio.CancelledError):
            await update
        assert (await active)["success"] is True
        assert feature._traffic_gate.closed is False
        assert feature._traffic_gate.sealed is False
        assert feature._reloading is False
        assert agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties["config"] == {
            "enabled": True
        }
    finally:
        release_active.set()
        for task in (update, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_reload_cancellation_while_gate_drains_reopens_without_restart(
    monkeypatch, tmp_path
):
    """Reload owns and reopens its gate even when cancellation wins the drain."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class BlockingToolClient(FakeIsolatedClient):
        async def call_tool(self, name, args):
            self.calls.append((name, args))
            active_started.set()
            await release_active.wait()
            return {"echo": args}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=BlockingToolClient)
    active = reload_task = None
    try:
        await feature.initialize()
        client = feature._client
        active = asyncio.create_task(feature.call_isolated_tool("hold", {}))
        await asyncio.wait_for(active_started.wait(), timeout=1)

        reload_task = asyncio.create_task(feature.reload())
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)
        assert feature._traffic_gate.closed

        reload_task.cancel()
        await asyncio.sleep(0)
        assert not reload_task.done()
        release_active.set()

        with pytest.raises(asyncio.CancelledError):
            await reload_task
        assert (await active)["success"] is True
        assert client.stopped is False
        assert feature._traffic_gate.closed is False
        assert feature._reloading is False
    finally:
        release_active.set()
        for task in (reload_task, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_second_cancellation_during_gate_reopen_waits_for_final_boundary(
    monkeypatch, tmp_path
):
    """A second cancellation cannot strand a reopened finite gate behind reload."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()
    reopen_started = asyncio.Event()
    release_reopen = asyncio.Event()

    class BlockingToolClient(FakeIsolatedClient):
        async def call_tool(self, name, args):
            active_started.set()
            await release_active.wait()
            return {"echo": args}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=BlockingToolClient)
    active = update = None
    try:
        await feature.initialize()
        original_reopen = feature._traffic_gate.reopen

        async def blocking_reopen():
            reopen_started.set()
            await release_reopen.wait()
            await original_reopen()

        monkeypatch.setattr(feature._traffic_gate, "reopen", blocking_reopen)
        active = asyncio.create_task(feature.call_isolated_tool("hold", {}))
        await asyncio.wait_for(active_started.wait(), timeout=1)
        update = asyncio.create_task(feature.set_config({"enabled": False}))
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)

        update.cancel()
        release_active.set()
        await asyncio.wait_for(reopen_started.wait(), timeout=1)
        update.cancel()
        await asyncio.sleep(0)
        assert not update.done()

        release_reopen.set()
        with pytest.raises(asyncio.CancelledError):
            await update
        assert feature._traffic_gate.closed is False
        assert feature._traffic_gate.sealed is False
        assert feature._reloading is False
    finally:
        release_active.set()
        release_reopen.set()
        for task in (update, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_terminal_quarantine_wakes_tool_and_channel_waiters_without_rpc(
    monkeypatch, tmp_path
):
    """Sealing quarantine releases finite waiters with no post-terminal effect."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class BlockingChannelClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

        async def call_tool(self, name, args):
            self.calls.append((name, args))
            if name == "hold":
                active_started.set()
                await release_active.wait()
            return {"ok": True}

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=BlockingChannelClient)
    active = close = tool_waiter = channel_waiter = quarantine = None
    try:
        await feature.initialize()
        client = feature._client
        adapter = channel_feature.registry.get("whatsapp")
        assert adapter is not None
        active = asyncio.create_task(feature.call_isolated_tool("hold", {}))
        await asyncio.wait_for(active_started.wait(), timeout=1)
        close = asyncio.create_task(feature._close_traffic_gate())
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)

        tool_waiter = asyncio.create_task(feature.call_isolated_tool("ping", {}))
        channel_waiter = asyncio.create_task(adapter.send_message(to="+1", content="hello"))
        await asyncio.sleep(0)
        assert not tool_waiter.done()
        assert not channel_waiter.done()

        quarantine = asyncio.create_task(feature._quarantine_unreconciled_client())
        await asyncio.sleep(0)
        tool_result = await tool_waiter
        channel_result = await channel_waiter
        assert tool_result["error"] == "isolated feature traffic is unavailable"
        assert channel_result.status.value == "failure"
        assert channel_result.error == "isolated feature traffic is unavailable"
        assert client.calls == [("hold", {})]
        assert feature._traffic_gate.sealed is True
        assert not quarantine.done()

        release_active.set()
        assert (await active)["success"] is True
        await quarantine
        with pytest.raises(RuntimeError, match="isolated feature traffic is unavailable"):
            await close
    finally:
        release_active.set()
        for task in (quarantine, close, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, RuntimeError):
                    pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_terminal_shutdown_wakes_tool_waiter(monkeypatch, tmp_path):
    """Shutdown seals admission before draining an already admitted RPC."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class BlockingToolClient(FakeIsolatedClient):
        async def call_tool(self, name, args):
            self.calls.append((name, args))
            if name == "hold":
                active_started.set()
                await release_active.wait()
            return {"echo": args}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=BlockingToolClient)
    active = close = waiter = shutdown_task = None
    try:
        await feature.initialize()
        active = asyncio.create_task(feature.call_isolated_tool("hold", {}))
        await asyncio.wait_for(active_started.wait(), timeout=1)
        close = asyncio.create_task(feature._close_traffic_gate())
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)

        waiter = asyncio.create_task(feature.call_isolated_tool("ping", {}))
        await asyncio.sleep(0)
        shutdown_task = asyncio.create_task(feature.shutdown())
        await asyncio.sleep(0)
        result = await waiter
        assert result["error"] == "isolated feature traffic is unavailable"
        assert feature._traffic_gate.sealed is True
        assert not shutdown_task.done()

        execution = SchedulerExecution(
            id="terminal-execution",
            schedule_id="terminal-schedule",
            agent_id="agent-1",
            task_name="ping",
            args={},
            scheduled_for="2026-07-26T00:00:00+00:00",
            idempotency_key="terminal-effect-key",
            attempt=1,
            owner="runner-1",
        )
        token = _current_execution.set(_SchedulerExecutionScope(execution))
        try:
            with pytest.raises(
                SchedulerTerminalAdmissionError,
                match="isolated feature traffic is unavailable",
            ):
                await feature.call_isolated_tool("ping", {})
        finally:
            _current_execution.reset(token)

        release_active.set()
        await active
        await shutdown_task
        with pytest.raises(RuntimeError, match="isolated feature traffic is unavailable"):
            await close
    finally:
        release_active.set()
        for task in (shutdown_task, close, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, RuntimeError):
                    pass
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_repeated_cancellation_during_terminal_seal_finishes_gate_boundary(
    monkeypatch, tmp_path
):
    """Repeated shutdown cancellation cannot leave a terminal waiter asleep."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class BlockingToolClient(FakeIsolatedClient):
        async def call_tool(self, name, args):
            self.calls.append((name, args))
            if name == "hold":
                active_started.set()
                await release_active.wait()
            return {"echo": args}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=BlockingToolClient)
    active = close = waiter = shutdown_task = None
    try:
        await feature.initialize()
        active = asyncio.create_task(feature.call_isolated_tool("hold", {}))
        await asyncio.wait_for(active_started.wait(), timeout=1)
        close = asyncio.create_task(feature._close_traffic_gate())
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)

        waiter = asyncio.create_task(feature.call_isolated_tool("ping", {}))
        shutdown_task = asyncio.create_task(feature.shutdown())
        await asyncio.sleep(0)
        assert (await waiter)["error"] == "isolated feature traffic is unavailable"
        assert feature._traffic_gate.sealed is True

        shutdown_task.cancel("first terminal cleanup cancellation")
        await asyncio.sleep(0)
        assert not shutdown_task.done()
        shutdown_task.cancel("later terminal cleanup cancellation")
        await asyncio.sleep(0)
        assert not shutdown_task.done()
        release_active.set()
        await active
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await shutdown_task
        assert cancelled.value.args == ("first terminal cleanup cancellation",)
        assert feature._traffic_gate.sealed is True
        assert feature._traffic_gate._active == 0
        with pytest.raises(RuntimeError, match="isolated feature traffic is unavailable"):
            await close
    finally:
        release_active.set()
        for task in (shutdown_task, close, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, RuntimeError):
                    pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_terminal_inbound_event_is_dropped_without_callback_error(monkeypatch, tmp_path):
    """A terminal SDK callback is deliberately ignored rather than propagated."""

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    delivered = []

    async def record_delivery(event):
        delivered.append(event)

    try:
        await feature.initialize()
        feature._handle_event_admitted = record_delivery  # type: ignore[method-assign]
        await feature.shutdown()

        await feature._handle_event({"type": "channel.inbound", "payload": {}})
        assert delivered == []
    finally:
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_reinitialize_after_terminal_quarantine_reopens_only_after_child_init(
    monkeypatch, tmp_path
):
    """A durable re-enable keeps terminal admission sealed until its child is ready."""

    second_start = asyncio.Event()
    release_second_start = asyncio.Event()
    clients = []

    class DelayedReinitializeClient(FakeIsolatedClient):
        async def start(self):
            self.started = True
            if len(clients) == 2:
                second_start.set()
                await release_second_start.wait()

    def client_factory(**kwargs):
        client = DelayedReinitializeClient(**kwargs)
        clients.append(client)
        return client

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    initialize_task = None
    try:
        await feature.persist_config({"enabled": True})
        await feature.initialize()
        await feature._quarantine_unreconciled_client()
        assert feature._traffic_gate.sealed is True

        initialize_task = asyncio.create_task(feature.initialize())
        await asyncio.wait_for(second_start.wait(), timeout=1)
        assert feature._traffic_gate.sealed is True
        blocked = await feature.call_isolated_tool("ping", {"message": "blocked"})
        assert blocked["error"] == "isolated feature traffic is unavailable"
        assert clients[1].calls == []

        release_second_start.set()
        await initialize_task
        assert feature._traffic_gate.sealed is False
        result = await feature.call_isolated_tool("ping", {"message": "ready"})
        assert result["success"] is True
        assert clients[1].calls == [("ping", {"message": "ready"})]
    finally:
        release_second_start.set()
        if initialize_task is not None and not initialize_task.done():
            initialize_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await initialize_task
        await feature.shutdown()


@pytest.mark.asyncio
async def test_soft_disabled_terminal_proxy_persists_repaired_config_until_reenable(
    monkeypatch, tmp_path
):
    """A disabled proxy can rotate config without reviving its sealed child."""

    old_config = {"enabled": True, "token": "old-token"}
    repaired_config = {"enabled": True, "token": "rotated-token"}
    clients = []

    class TransitionClient(FakeIsolatedClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prepared = []

        async def prepare_config_transition(self, config):
            self.prepared.append(dict(config))
            return ConfigTransitionResult.applied()

    def client_factory(**kwargs):
        client = TransitionClient(**kwargs)
        clients.append(client)
        return client

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        old_child = clients[0]
        await feature.shutdown()

        assert old_child.stopped is True
        assert feature._client is None
        assert feature._supervision_task is None
        assert feature._stopping is True
        assert feature._traffic_gate.sealed is True

        await feature.set_config(repaired_config)

        properties = agent.storage.nodes[_TEST_CONFIG_NODE_ID].properties
        assert properties["config"] == repaired_config
        assert "pending_config" not in properties
        # A config repair is storage-only while disabled: no transition hook,
        # child, supervision task, or admission reset is permitted.
        assert old_child.prepared == []
        assert clients == [old_child]
        assert feature._client is None
        assert feature._supervision_task is None
        assert feature._stopping is True
        assert feature._traffic_gate.sealed is True

        await feature.initialize()

        assert len(clients) == 2
        assert clients[1].kwargs["config"] == repaired_config
        assert feature._client is clients[1]
        assert feature._supervision_task is not None
        assert feature._stopping is False
        assert feature._traffic_gate.sealed is False
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_shutdown_cancellation_waiting_for_lifecycle_lock_finishes_teardown(
    monkeypatch, tmp_path
):
    """Caller cancellation waits through lock ownership and final retirement."""

    class ChannelClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=ChannelClient)
    shutdown_task = None
    lock_held = False
    try:
        await feature.initialize()
        client = feature._client
        adapter = channel_feature.registry.get("whatsapp")
        assert client is not None
        assert adapter is not None

        await feature._reload_lock.acquire()
        lock_held = True
        shutdown_task = asyncio.create_task(feature.shutdown())
        for _ in range(100):
            if feature._traffic_gate.sealed:
                break
            await asyncio.sleep(0)
        assert feature._traffic_gate.sealed is True
        assert (await feature.call_isolated_tool("ping", {}))["error"] == (
            "isolated feature traffic is unavailable"
        )
        assert not shutdown_task.done()

        shutdown_task.cancel()
        await asyncio.sleep(0)
        assert not shutdown_task.done()
        feature._reload_lock.release()
        lock_held = False

        with pytest.raises(asyncio.CancelledError):
            await shutdown_task
        assert client.stopped is True
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._channel_adapter is None
        assert channel_feature.registry.get("whatsapp") is None
        assert feature._traffic_gate.sealed is True
        assert feature._stopping is True
    finally:
        if lock_held:
            feature._reload_lock.release()
        if shutdown_task is not None and not shutdown_task.done():
            shutdown_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await shutdown_task
        await feature.shutdown()


@pytest.mark.asyncio
async def test_quarantine_repeated_cancellation_waits_for_lock_and_retires_child(
    monkeypatch, tmp_path
):
    """Terminal quarantine cannot leave its cleanup task behind the lifecycle lock."""

    class ChannelClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=ChannelClient)
    quarantine_task = None
    lock_held = False
    try:
        await feature.initialize()
        client = feature._client
        assert client is not None
        assert channel_feature.registry.get("whatsapp") is not None

        await feature._reload_lock.acquire()
        lock_held = True
        quarantine_task = asyncio.create_task(feature._quarantine_unreconciled_client())
        for _ in range(100):
            if feature._traffic_gate.sealed:
                break
            await asyncio.sleep(0)
        assert feature._traffic_gate.sealed is True
        assert (await feature.call_isolated_tool("ping", {}))["error"] == (
            "isolated feature traffic is unavailable"
        )

        quarantine_task.cancel()
        quarantine_task.cancel()
        await asyncio.sleep(0)
        assert not quarantine_task.done()
        feature._reload_lock.release()
        lock_held = False

        with pytest.raises(asyncio.CancelledError):
            await quarantine_task
        assert client.stopped is True
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._channel_adapter is None
        assert channel_feature.registry.get("whatsapp") is None
        assert feature._traffic_gate.sealed is True
    finally:
        if lock_held:
            feature._reload_lock.release()
        if quarantine_task is not None and not quarantine_task.done():
            quarantine_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await quarantine_task
        await feature.shutdown()


@pytest.mark.asyncio
async def test_initialize_cancellation_after_gate_reset_unpublishes_and_retires_child(
    monkeypatch, tmp_path
):
    """A cancelled post-connect initialize has no tool, channel, or event path."""

    reset_started = asyncio.Event()
    release_reset = asyncio.Event()
    delivered = []

    class ChannelClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = ChannelClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    original_reset = feature._traffic_gate.reset_and_reopen

    async def blocking_reset():
        reset_started.set()
        await release_reset.wait()
        await original_reset()

    async def record_delivery(event):
        delivered.append(event)

    monkeypatch.setattr(feature._traffic_gate, "reset_and_reopen", blocking_reset)
    feature._handle_event_admitted = record_delivery  # type: ignore[method-assign]
    initialize_task = None
    try:
        initialize_task = asyncio.create_task(feature.initialize())
        await asyncio.wait_for(reset_started.wait(), timeout=1)
        client = clients[0]
        assert feature._client is client
        assert channel_feature.registry.get("whatsapp") is not None

        initialize_task.cancel()
        release_reset.set()
        with pytest.raises(asyncio.CancelledError):
            await initialize_task

        assert client.stopped is True
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._channel_adapter is None
        assert channel_feature.registry.get("whatsapp") is None
        assert feature._traffic_gate.sealed is True
        assert (await feature.call_isolated_tool("ping", {}))["error"] == (
            "isolated feature traffic is unavailable"
        )
        assert client.event_handler is not None
        await client.event_handler({"type": "channel.inbound", "payload": {}})
        assert delivered == []
    finally:
        release_reset.set()
        if initialize_task is not None and not initialize_task.done():
            initialize_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await initialize_task
        await feature.shutdown()


@pytest.mark.asyncio
async def test_shutdown_during_event_registration_keeps_initialize_terminal(
    monkeypatch, tmp_path
):
    """A terminal latch during post-publication registration cannot revive the proxy."""

    registration_started = asyncio.Event()
    release_registration = asyncio.Event()
    clients = []

    class ChannelClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

        async def set_event_handler(self, handler):
            if self is clients[0]:
                registration_started.set()
                await release_registration.wait()
            self.event_handler = handler

    def client_factory(**kwargs):
        client = ChannelClient(**kwargs)
        clients.append(client)
        return client

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    initialize_task = None
    shutdown_task = None
    try:
        initialize_task = asyncio.create_task(feature.initialize())
        await asyncio.wait_for(registration_started.wait(), timeout=1)
        first_child = clients[0]
        assert feature._client is first_child
        assert feature.get_tools()
        assert channel_feature.registry.get("whatsapp") is not None

        shutdown_task = asyncio.create_task(feature.shutdown())
        await asyncio.sleep(0)
        assert feature._terminal_lifecycle_latched is True

        release_registration.set()
        with pytest.raises(RuntimeError, match="terminal lifecycle is latched"):
            await asyncio.wait_for(initialize_task, timeout=1)
        await asyncio.wait_for(shutdown_task, timeout=1)

        assert first_child.stopped is True
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._supervision_task is None
        assert feature._channel_adapter is None
        assert channel_feature.registry.get("whatsapp") is None
        assert feature._traffic_gate.sealed is True

        # Cleanup has completed, so a later explicit initialize owns a new
        # lifecycle and may open admission again.
        await feature.initialize()
        assert len(clients) == 2
        assert feature._client is clients[1]
        assert feature._supervision_task is not None
        assert feature._traffic_gate.sealed is False
    finally:
        release_registration.set()
        for task in (initialize_task, shutdown_task):
            if task is not None and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        await feature.shutdown()


@pytest.mark.asyncio
async def test_registration_failure_after_shutdown_retirement_does_not_stop_twice(
    monkeypatch, tmp_path
):
    """Registration cleanup never reclaims a client terminal cleanup already owns."""

    registration_started = asyncio.Event()
    release_registration = asyncio.Event()
    stop_finished = asyncio.Event()
    clients = []

    class RegistrationFailureClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def set_event_handler(self, _handler):
            registration_started.set()
            await release_registration.wait()
            raise RuntimeError("event registration failed")

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True
            stop_finished.set()

    def client_factory(**kwargs):
        client = RegistrationFailureClient(**kwargs)
        clients.append(client)
        return client

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    initialize_task = shutdown_task = None
    try:
        initialize_task = asyncio.create_task(feature.initialize())
        await asyncio.wait_for(registration_started.wait(), timeout=1)
        client = clients[0]

        # Shutdown wins publication ownership while registration remains
        # blocked.  It has retired the exact client, but must wait for the
        # initializer's reload lock before it can complete its final fence.
        shutdown_task = asyncio.create_task(feature.shutdown())
        await asyncio.wait_for(stop_finished.wait(), timeout=1)
        assert feature._client is None
        assert not shutdown_task.done()

        release_registration.set()
        with pytest.raises(IsolatedRuntimePreparationError) as failure:
            await asyncio.wait_for(initialize_task, timeout=1)
        assert "event registration failed" not in str(failure.value)
        assert isinstance(failure.value.__cause__, RuntimeError)
        await asyncio.wait_for(shutdown_task, timeout=1)

        assert client.stop_calls == 1
        assert feature._client is None
        assert feature._terminal_retirement_clients == []
        assert feature._terminal_cleanup_task is None
        assert feature._traffic_gate.sealed is True
        assert feature._supervision_task is None
    finally:
        release_registration.set()
        for task in (initialize_task, shutdown_task):
            if task is not None and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_initialize_gate_reset_failure_unpublishes_and_retires_child(monkeypatch, tmp_path):
    """A post-connect gate failure follows the same terminal cleanup path."""

    class ChannelClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = ChannelClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)

    async def failing_reset():
        raise RuntimeError("gate reset failed")

    monkeypatch.setattr(feature._traffic_gate, "reset_and_reopen", failing_reset)
    try:
        with pytest.raises(RuntimeError, match="gate reset failed"):
            await feature.initialize()

        assert clients[0].stopped is True
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._channel_adapter is None
        assert channel_feature.registry.get("whatsapp") is None
        assert feature._traffic_gate.sealed is True
        assert (await feature.call_isolated_tool("ping", {}))["error"] == (
            "isolated feature traffic is unavailable"
        )
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_reload_candidate_start_failure_quarantines_stopped_old_child(
    monkeypatch, tmp_path
):
    """Direct reload never reopens old tools when candidate startup fails."""

    class ChannelClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

    class StartFailingClient(ChannelClient):
        async def start(self):
            self.started = True
            raise RuntimeError("candidate start failed")

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = ChannelClient(**kwargs) if not clients else StartFailingClient(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    try:
        await feature.initialize()
        old_client = clients[0]
        assert channel_feature.registry.get("whatsapp") is not None

        with pytest.raises(IsolatedRuntimePreparationError) as failure:
            await feature.reload()
        assert "candidate start failed" not in str(failure.value)
        assert isinstance(failure.value.__cause__, RuntimeError)

        assert old_client.stopped is True
        assert clients[1].stopped is True
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._channel_adapter is None
        assert channel_feature.registry.get("whatsapp") is None
        assert feature._traffic_gate.sealed is True
        assert (await feature.call_isolated_tool("ping", {}))["error"] == (
            "isolated feature traffic is unavailable"
        )
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_reload_cancellation_after_old_stop_quarantines_publication(
    monkeypatch, tmp_path
):
    """Reload cancellation after old retirement cannot reopen stale traffic."""

    candidate_started = asyncio.Event()
    release_candidate = asyncio.Event()

    class ChannelClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "whatsapp",
                    "send_tool": "whatsapp_send",
                }
            }

    class BlockingCandidate(ChannelClient):
        async def start(self):
            self.started = True
            candidate_started.set()
            await release_candidate.wait()

    channel_feature = FakeChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    clients = []

    def client_factory(**kwargs):
        client = ChannelClient(**kwargs) if not clients else BlockingCandidate(**kwargs)
        clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    reload_task = None
    try:
        await feature.initialize()
        old_client = clients[0]
        reload_task = asyncio.create_task(feature.reload())
        await asyncio.wait_for(candidate_started.wait(), timeout=1)
        assert old_client.stopped is True
        assert feature._client is None

        reload_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reload_task

        assert clients[1].stopped is True
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature._channel_adapter is None
        assert channel_feature.registry.get("whatsapp") is None
        assert feature._traffic_gate.sealed is True
        assert (await feature.call_isolated_tool("ping", {}))["error"] == (
            "isolated feature traffic is unavailable"
        )
    finally:
        release_candidate.set()
        if reload_task is not None and not reload_task.done():
            reload_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reload_task
        await feature.shutdown()


@pytest.mark.asyncio
async def test_channel_inbound_acknowledges_exact_source_only_after_host_delivery(
    monkeypatch, tmp_path
):
    """A notification's producer waits for post-handler private acknowledgement."""

    dedupe_key = "telegram:v2:bot:42:update:101"
    delivery_started = asyncio.Event()
    release_delivery = asyncio.Event()
    acknowledgement_started = asyncio.Event()
    release_acknowledgement = asyncio.Event()
    delivered = []

    class ChannelFeature(FakeChannelFeature):
        async def handle_inbound(self, message):
            delivery_started.set()
            await release_delivery.wait()
            delivered.append(message.id)
            return SimpleNamespace(durably_admitted=True)

    class AcknowledgingClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "telegram",
                    "send_tool": "telegram_send",
                }
            }

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )
            self.acknowledgements = []

        async def call_host_ingress(self, name, payload=None):
            self.acknowledgements.append((name, payload))
            acknowledgement_started.set()
            await release_acknowledgement.wait()
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": ChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=AcknowledgingClient)
    event_task = None
    try:
        await feature.initialize()
        client = feature._client
        event_task = asyncio.create_task(
            client.event_handler(
                {
                    "type": "channel.inbound",
                    "payload": {
                        "message": {
                            "channel_type": "telegram",
                            "direction": "inbound",
                            "sender": "555",
                            "recipient": "42",
                            "content": "hello",
                            "id": dedupe_key,
                            "metadata": {"dedupe_key": dedupe_key},
                        },
                        "_host_ingress_ack": {
                            "name": "telegram-polling-ack",
                            "payload": {
                                "dedupe_key": dedupe_key,
                                "attempt_token": _TELEGRAM_ATTEMPT_TOKEN,
                            },
                        },
                        "_host_ingress_retry": {
                            "name": "telegram-polling-nack",
                            "payload": {
                                "dedupe_key": dedupe_key,
                                "attempt_token": _TELEGRAM_ATTEMPT_TOKEN,
                            },
                        },
                    },
                }
            )
        )
        await asyncio.wait_for(delivery_started.wait(), timeout=1)
        assert not acknowledgement_started.is_set()

        release_delivery.set()
        await asyncio.wait_for(event_task, timeout=1)
        await asyncio.wait_for(acknowledgement_started.wait(), timeout=1)
        assert delivered == [dedupe_key]
        assert client.acknowledgements == [
            (
                "telegram-polling-ack",
                {
                    "dedupe_key": dedupe_key,
                    "attempt_token": _TELEGRAM_ATTEMPT_TOKEN,
                },
            )
        ]
        # The acknowledgement remains independently in-flight so an SDK event
        # reader can return to consume its JSON-RPC response instead of waiting
        # on that same response inline.
        assert feature._event_ack_tasks
        release_acknowledgement.set()
        for _ in range(20):
            if not feature._event_ack_tasks:
                break
            await asyncio.sleep(0)
        assert not feature._event_ack_tasks
    finally:
        release_delivery.set()
        release_acknowledgement.set()
        if event_task is not None and not event_task.done():
            await event_task
        await feature.shutdown()


@pytest.mark.asyncio
async def test_current_replacement_acknowledged_event_waits_for_reopened_gate(
    monkeypatch, tmp_path
):
    """A replacement poller cannot lose its first update to Core's closed gate."""

    dedupe_key = "telegram:v2:bot:42:update:102"
    delivered = []
    acknowledged = asyncio.Event()

    class ChannelFeature(FakeChannelFeature):
        async def handle_inbound(self, message):
            delivered.append(message.id)
            return SimpleNamespace(durably_admitted=True)

    class AckClient(TelegramChannelClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )

        async def call_host_ingress(self, name, payload=None):
            assert (name, payload) == (
                "telegram-polling-ack",
                {
                    "dedupe_key": dedupe_key,
                    "attempt_token": _TELEGRAM_ATTEMPT_TOKEN,
                },
            )
            acknowledged.set()
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": ChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=AckClient)
    try:
        await feature.initialize()
        await feature._close_traffic_gate()
        await feature._client.event_handler(
            {
                "type": "channel.inbound",
                "payload": {
                    "message": {
                        "channel_type": "telegram",
                        "direction": "inbound",
                        "sender": "555",
                        "recipient": "42",
                        "content": "first after replacement",
                        "id": dedupe_key,
                        "metadata": {"dedupe_key": dedupe_key},
                    },
                    "_host_ingress_ack": {
                        "name": "telegram-polling-ack",
                        "payload": {
                            "dedupe_key": dedupe_key,
                            "attempt_token": _TELEGRAM_ATTEMPT_TOKEN,
                        },
                    },
                    "_host_ingress_retry": {
                        "name": "telegram-polling-nack",
                        "payload": {
                            "dedupe_key": dedupe_key,
                            "attempt_token": _TELEGRAM_ATTEMPT_TOKEN,
                        },
                    },
                },
            }
        )
        assert delivered == []
        assert not acknowledged.is_set()

        await feature._reopen_traffic_gate()
        await asyncio.wait_for(acknowledged.wait(), timeout=1)
        assert delivered == [dedupe_key]
    finally:
        await feature.shutdown()


def _acknowledged_telegram_event(dedupe_key: str) -> dict:
    """Build the exact private ingress envelope shared by polling and hosted dedupe."""

    return {
        "type": "channel.inbound",
        "payload": {
            "message": {
                "channel_type": "telegram",
                "direction": "inbound",
                "sender": "555",
                "recipient": "42",
                "content": "hello",
                "id": dedupe_key,
                "metadata": {"dedupe_key": dedupe_key},
            },
            "_host_ingress_ack": {
                "name": "telegram-polling-ack",
                "payload": {
                    "dedupe_key": dedupe_key,
                    "attempt_token": _TELEGRAM_ATTEMPT_TOKEN,
                },
            },
            "_host_ingress_retry": {
                "name": "telegram-polling-nack",
                "payload": {
                    "dedupe_key": dedupe_key,
                    "attempt_token": _TELEGRAM_ATTEMPT_TOKEN,
                },
            },
        },
    }


def _retryable_telegram_event(dedupe_key: str) -> dict:
    """Build a polling envelope whose provider callback can be NACKed."""

    return _acknowledged_telegram_event(dedupe_key)


def test_telegram_polling_completion_descriptors_require_an_authoritative_pair():
    """Telegram completion semantics come from the registered proxy identity."""

    feature = object.__new__(ProxyFeature)
    feature._authoritative_inbound_channel_type = lambda: "telegram"
    valid = _retryable_telegram_event("telegram:v2:bot:42:update:attempt-fenced")
    assert feature._inbound_event_retry_completion(valid["payload"]) is not None

    tokenless = _retryable_telegram_event("telegram:v2:bot:42:update:tokenless")
    del tokenless["payload"]["_host_ingress_ack"]["payload"]["attempt_token"]
    del tokenless["payload"]["_host_ingress_retry"]["payload"]["attempt_token"]
    _, acknowledgement = feature._split_inbound_event_acknowledgement(
        tokenless["payload"]
    )
    assert acknowledgement is None
    assert feature._inbound_event_retry_completion(tokenless["payload"]) is None

    mismatched = _retryable_telegram_event("telegram:v2:bot:42:update:mismatched")
    mismatched["payload"]["_host_ingress_retry"]["payload"]["attempt_token"] = (
        "n" * 43
    )
    assert feature._inbound_event_retry_completion(mismatched["payload"]) is None

    missing_pair = _acknowledged_telegram_event("telegram:v2:bot:42:update:missing")
    del missing_pair["payload"]["_host_ingress_retry"]
    _, acknowledgement = feature._split_inbound_event_acknowledgement(
        missing_pair["payload"]
    )
    assert acknowledgement is None
    assert feature._inbound_event_retry_completion(missing_pair["payload"]) is None

    renamed = _retryable_telegram_event("telegram:v2:bot:42:update:renamed")
    renamed["payload"]["_host_ingress_ack"]["name"] = "other-provider-ack"
    renamed["payload"]["_host_ingress_retry"]["name"] = "other-provider-nack"
    del renamed["payload"]["_host_ingress_ack"]["payload"]["attempt_token"]
    del renamed["payload"]["_host_ingress_retry"]["payload"]["attempt_token"]
    _, acknowledgement = feature._split_inbound_event_acknowledgement(
        renamed["payload"]
    )
    assert acknowledgement is None
    assert feature._inbound_event_retry_completion(renamed["payload"]) is None

    # A different host-negotiated channel retains its established generic
    # completion contract; the child message field does not choose it.
    feature._authoritative_inbound_channel_type = lambda: "whatsapp"
    non_telegram = _acknowledged_telegram_event("whatsapp:v1:message:1")
    non_telegram["payload"]["message"]["channel_type"] = "whatsapp"
    non_telegram["payload"]["_host_ingress_ack"] = {
        "name": "whatsapp-webhook-ack",
        "payload": {"dedupe_key": "whatsapp:v1:message:1"},
    }
    del non_telegram["payload"]["_host_ingress_retry"]
    _, acknowledgement = feature._split_inbound_event_acknowledgement(
        non_telegram["payload"]
    )
    assert acknowledgement is not None
    assert acknowledgement.name == "whatsapp-webhook-ack"


def test_telegram_terminal_envelope_preserves_the_attempt_fenced_ack_nack_pair():
    """Terminal dispositions are bounded metadata, not a reason to drop ACK/NACK."""

    feature = object.__new__(ProxyFeature)
    feature._authoritative_inbound_channel_type = lambda: "telegram"
    event = _acknowledged_telegram_event("telegram:v2:bot:42:update:terminal")
    event["payload"]["_telegram_terminal_disposition"] = {
        "kind": "unauthorized_sender"
    }
    message, acknowledgement = feature._split_inbound_event_acknowledgement(
        event["payload"]
    )
    assert message["id"] == "telegram:v2:bot:42:update:terminal"
    assert acknowledgement is not None
    assert feature._inbound_event_retry_completion(event["payload"]) is not None
    assert feature._telegram_terminal_disposition(
        event["payload"], cursor_owned_protocol=True
    ) == "unauthorized_sender"


@pytest.mark.asyncio
async def test_hosted_telegram_ingress_is_admitted_only_through_the_durable_route():
    """A host cannot treat the child's normalized event as an HTTP-only success."""

    feature = object.__new__(ProxyFeature)
    feature._is_telegram_runtime = lambda: True
    feature._hosted_telegram_startup_attested = True
    feature._route_validated_inbound = AsyncMock(return_value={"durably_admitted": True})

    payload = {"id": "telegram:v2:bot:42:update:12", "content": "hello"}
    assert await feature.admit_hosted_telegram_ingress(
        payload, terminal_disposition="unsupported_update"
    ) == {"durably_admitted": True}
    feature._route_validated_inbound.assert_awaited_once_with(
        payload,
        cursor_owned_protocol=True,
        telegram_terminal_disposition="unsupported_update",
    )


def test_explicit_isolated_feature_data_directory_wins_over_storage_path(tmp_path):
    """Postgres-backed hosts can isolate venvs without a SQLite storage file."""

    agent = SimpleNamespace(
        storage_path=str(tmp_path / "legacy" / "agent.db"),
        isolated_feature_data_dir=tmp_path / "host-owned" / "agent-a",
    )

    assert resolve_agent_runtime_dir(agent) == (
        tmp_path / "host-owned" / "agent-a"
    ).resolve()

@pytest.mark.parametrize(
    ("allowed_senders", "expected"),
    (
        (["@jason"], []),
        (["@jason", "555", "00555", "0"], ["555"]),
    ),
    ids=("legacy-only", "mixed-migration-list"),
)
def test_telegram_proxy_authorization_keeps_only_canonical_numeric_ids(
    allowed_senders, expected
):
    """Migration-only usernames never reach the host authorization adapter."""

    feature = object.__new__(ProxyFeature)
    feature._host_config = {"enabled": True, "allowed_senders": allowed_senders}

    config = feature._channel_config("telegram")

    assert config.allowed_senders == expected


@pytest.mark.asyncio
async def test_spoofed_child_channel_type_cannot_bypass_proxy_allowlist_or_telegram_pair(
    monkeypatch, tmp_path
):
    """A Telegram child cannot select another adapter/filter with wire data."""

    routed = []

    from kestrel_sovereign.features.channels.feature import ChannelFeature

    async def route_inbound(message):
        routed.append((message.channel_type, message.sender))

    class TelegramClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "telegram",
                    "send_tool": "telegram_send",
                }
            }

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack", "telegram-polling-nack")
            )
            self.completions = []

        async def call_host_ingress(self, name, payload=None):
            self.completions.append((name, payload))
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    channel_agent = SimpleNamespace(
        did=_TEST_AGENT_DID,
        storage=SimpleNamespace(agent_id=_TEST_AGENT_DID),
        dispatcher=None,
        signal_registry=SourceRegistry(),
        features={},
    )
    channel_feature = ChannelFeature(channel_agent)
    await channel_feature.initialize()
    channel_feature.registry.set_inbound_router(route_inbound)
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=TelegramClient)
    host_config = {
        "agent_id": _TEST_AGENT_DID,
        "enabled": True,
        "allowed_senders": ["555"],
    }

    async def load_host_config():
        return host_config

    feature._load_host_config = load_host_config  # type: ignore[method-assign]
    try:
        await feature.initialize()
        event = _acknowledged_telegram_event("telegram:v2:bot:42:update:spoofed")
        event["payload"]["message"]["channel_type"] = "whatsapp"
        event["payload"]["_host_ingress_ack"]["name"] = "renamed-ack"
        event["payload"]["_host_ingress_retry"]["name"] = "renamed-nack"

        await feature._client.event_handler(event)
        await asyncio.gather(
            *tuple(feature._event_ingress_tasks), return_exceptions=True
        )

        # The authoritative Telegram identity rejects the renamed pair; it
        # cannot fall through to WhatsApp's absent adapter/filter *or* the
        # generic legacy router. Telegram requires its paired durable path.
        assert routed == []
        assert feature._client.completions == []

        denied = _acknowledged_telegram_event("telegram:v2:bot:42:update:denied")
        denied["payload"]["message"]["channel_type"] = "whatsapp"
        denied["payload"]["message"]["sender"] = "not-allowed"
        await feature._client.event_handler(denied)
        await asyncio.gather(
            *tuple(feature._event_ingress_tasks), return_exceptions=True
        )
        assert routed == []

        channel_feature.registry.unregister("telegram")
        await feature._client.event_handler(_acknowledged_telegram_event(
            "telegram:v2:bot:42:update:missing-adapter"
        ))
        await asyncio.gather(
            *tuple(feature._event_ingress_tasks), return_exceptions=True
        )
        assert routed == []
    finally:
        await feature.shutdown()
        await channel_feature.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host_config",
    (
        {"enabled": False, "allowed_senders": ["555"]},
        {"enabled": True, "allowed_senders": []},
    ),
    ids=("disabled-adapter", "telegram-default-deny"),
)
async def test_host_telegram_authorization_nacks_faulty_child_notifications(
    monkeypatch, tmp_path, host_config
):
    """The host adapter policy controls ACKs even if an isolated child is faulty."""

    from kestrel_sovereign.features.channels.feature import ChannelFeature

    routed = []
    nacked = asyncio.Event()

    async def route_inbound(message):
        routed.append(message)

    class CompletionClient(TelegramChannelClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack", "telegram-polling-nack")
            )
            self.completions = []

        async def call_host_ingress(self, name, payload=None):
            self.completions.append((name, payload))
            if name == "telegram-polling-nack":
                nacked.set()
                return {"status": "ok", "http_status": 200, "state": "retrying"}
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    channel_agent = SimpleNamespace(
        did=_TEST_AGENT_DID,
        storage=SimpleNamespace(agent_id=_TEST_AGENT_DID),
        dispatcher=None,
        signal_registry=SourceRegistry(),
        features={},
    )
    channel_feature = ChannelFeature(channel_agent)
    await channel_feature.initialize()
    channel_feature.registry.set_inbound_router(route_inbound)
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=CompletionClient)
    configured_host = {"agent_id": _TEST_AGENT_DID, **host_config}

    async def load_host_config():
        return configured_host

    feature._load_host_config = load_host_config  # type: ignore[method-assign]
    dedupe_key = "telegram:v2:bot:42:update:host-policy"
    try:
        await feature.initialize()
        await feature._client.event_handler(_retryable_telegram_event(dedupe_key))
        await asyncio.wait_for(nacked.wait(), timeout=1)

        assert routed == []
        assert feature._client.completions == [
            (
                "telegram-polling-nack",
                {"dedupe_key": dedupe_key, "attempt_token": _TELEGRAM_ATTEMPT_TOKEN},
            )
        ]
    finally:
        await feature.shutdown()
        await channel_feature.shutdown()


@pytest.mark.asyncio
async def test_inbound_reader_returns_before_cognition_and_nacks_retryable_result(
    monkeypatch, tmp_path
):
    """Cognition can make an outbound channel call without pinning the SDK reader."""

    dedupe_key = "telegram:v2:bot:42:update:reader-free"
    cognition_started = asyncio.Event()
    release_cognition = asyncio.Event()
    retry_completed = asyncio.Event()

    class ChannelFeature(FakeChannelFeature):
        async def handle_inbound(self, _message):
            cognition_started.set()
            # This is the same data-plane route cognition uses for
            # ``channels_send``. It must be able to await the child RPC after
            # the notification reader has already returned to its stream.
            await feature._channel_adapter.send_message("555", "reply during cognition")
            await release_cognition.wait()
            return SimpleNamespace(durably_admitted=False)

    class RetryClient(FakeIsolatedClient):
        @property
        def capabilities(self):
            return {
                "channel": {
                    "channel_type": "telegram",
                    "send_tool": "telegram_send",
                }
            }

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack", "telegram-polling-nack")
            )
            self.completions = []

        async def call_tool(self, name, args):
            self.calls.append((name, args))
            return {"status": "ok", "data": {"message_id": "reply-1"}}

        async def call_host_ingress(self, name, payload=None):
            self.completions.append((name, payload))
            retry_completed.set()
            return {"status": "ok", "http_status": 200, "state": "retrying"}

    channel_feature = ChannelFeature()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=RetryClient)
    try:
        await feature.initialize()
        client = feature._client
        # The notification invocation returns before even a deliberately
        # blocked cognition turn, freeing its serial reader for outbound RPC
        # responses and the later provider completion.
        await asyncio.wait_for(
            client.event_handler(_retryable_telegram_event(dedupe_key)), timeout=0.1
        )
        await asyncio.wait_for(cognition_started.wait(), timeout=1)
        assert client.calls == [
            ("telegram_send", {"to": "555", "message": "reply during cognition"})
        ]
        assert client.completions == []

        release_cognition.set()
        await asyncio.wait_for(retry_completed.wait(), timeout=1)
        assert client.completions == [
            (
                "telegram-polling-nack",
                {
                    "dedupe_key": dedupe_key,
                    "attempt_token": _TELEGRAM_ATTEMPT_TOKEN,
                },
            )
        ]
    finally:
        release_cognition.set()
        await feature.shutdown()


@pytest.mark.asyncio
async def test_ack_bearing_ingress_never_acknowledges_legacy_router_fallback(monkeypatch, tmp_path):
    """A legacy/non-durable ChannelFeature result leaves the provider cursor intact."""

    dedupe_key = "telegram:v2:bot:42:update:201"

    class LegacyChannelFeature:
        async def handle_inbound(self, message):
            return SimpleNamespace(durably_admitted=False)

    class AckClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )
            self.acknowledgements = []

        async def call_host_ingress(self, name, payload=None):
            self.acknowledgements.append((name, payload))
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": LegacyChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=AckClient)
    try:
        await feature.initialize()
        client = feature._client
        await client.event_handler(_acknowledged_telegram_event(dedupe_key))
        await asyncio.sleep(0)
        assert client.acknowledgements == []
        assert feature._event_ack_tasks == set()
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("state_name", ("MISMATCH", "INVALID"))
async def test_ack_bearing_ingress_nacks_when_channel_source_contract_is_unusable(
    monkeypatch, tmp_path, state_name
):
    """A rejected channel source contract reaches Telegram as a retry, never an ACK."""

    from kestrel_sovereign.features.channels.feature import ChannelFeature
    from kestrel_sovereign.signals.registry import (
        RegistrationOutcome,
        RegistrationState,
    )

    dedupe_key = f"telegram:v2:bot:42:update:source-{state_name.lower()}"
    nacked = asyncio.Event()
    state = getattr(RegistrationState, state_name)

    class SourceRegistry:
        def register_with_policy(self, _registration, _policy):
            return RegistrationOutcome(
                "channel.message", state, "test source contract failure"
            )

    class RetryClient(TelegramChannelClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack", "telegram-polling-nack")
            )
            self.completions = []

        async def call_host_ingress(self, name, payload=None):
            self.completions.append((name, payload))
            if name == "telegram-polling-nack":
                nacked.set()
                return {"status": "ok", "http_status": 200, "state": "retrying"}
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    channel_agent = SimpleNamespace(
        did=_TEST_AGENT_DID,
        storage=SimpleNamespace(agent_id=_TEST_AGENT_DID),
        dispatcher=None,
        signal_registry=SourceRegistry(),
        features={},
    )
    channel_feature = ChannelFeature(channel_agent)
    await channel_feature.initialize()
    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": channel_feature})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=RetryClient)
    feature._host_config = {
        "agent_id": _TEST_AGENT_DID,
        "enabled": True,
        "allowed_senders": ["555"],
    }
    feature._host_config_loaded = True
    try:
        await feature.initialize()
        assert channel_feature._durable_cognition_registration_failed is True
        await feature._client.event_handler(_retryable_telegram_event(dedupe_key))
        await asyncio.wait_for(nacked.wait(), timeout=1)
        assert feature._client.completions == [
            (
                "telegram-polling-nack",
                {
                    "dedupe_key": dedupe_key,
                    "attempt_token": _TELEGRAM_ATTEMPT_TOKEN,
                },
            )
        ]
    finally:
        await feature.shutdown()
        await channel_feature.shutdown()


@pytest.mark.asyncio
async def test_retired_source_event_is_rejected_under_traffic_admission(monkeypatch, tmp_path):
    """A late callback from a retired child cannot route or ACK after replacement."""

    dedupe_key = "telegram:v2:bot:42:update:202"
    delivered = []

    class ChannelFeature(FakeChannelFeature):
        async def handle_inbound(self, message):
            delivered.append(message.id)
            return SimpleNamespace(durably_admitted=True)

    class AckClient(TelegramChannelClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )
            self.acknowledgements = []

        async def call_host_ingress(self, name, payload=None):
            self.acknowledgements.append((name, payload))
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": ChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=AckClient)
    old_client = None
    try:
        await feature.initialize()
        old_client = feature._client
        # The callback closure still carries old_client, while publication has
        # moved to the replacement. Identity is checked *inside* admission.
        replacement = AckClient()
        feature._client = replacement
        await old_client.event_handler(_acknowledged_telegram_event(dedupe_key))
        await asyncio.sleep(0)
        assert delivered == []
        assert old_client.acknowledgements == []
        assert replacement.acknowledgements == []
    finally:
        if old_client is not None:
            await old_client.stop()
        await feature.shutdown()


@pytest.mark.asyncio
async def test_deferred_acknowledged_ingress_uses_detached_k1_snapshot(monkeypatch, tmp_path):
    """Sol regression: mutating a queued k1 envelope to k2 cannot redirect delivery/ACK."""

    k1 = "telegram:v2:bot:42:update:203"
    k2 = "telegram:v2:bot:42:update:204"
    delivered = []
    acknowledged = []

    class ChannelFeature(FakeChannelFeature):
        async def handle_inbound(self, message):
            delivered.append(message.id)
            return SimpleNamespace(durably_admitted=True)

    class AckClient(TelegramChannelClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )

        async def call_host_ingress(self, name, payload=None):
            acknowledged.append((name, payload))
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": ChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=AckClient)
    try:
        await feature.initialize()
        telemetry_active_counts = []
        schedule_telemetry = feature._schedule_runtime_telemetry

        def record_telemetry_admission(**kwargs):
            telemetry_active_counts.append(feature._traffic_gate._active)
            schedule_telemetry(**kwargs)

        monkeypatch.setattr(
            feature,
            "_schedule_runtime_telemetry",
            record_telemetry_admission,
        )
        event = _acknowledged_telegram_event(k1)
        await feature._close_traffic_gate()
        await feature._client.event_handler(event)
        # This is the caller-owned graph Sol mutated after Core deferred it.
        event["payload"]["message"]["id"] = k2
        event["payload"]["message"]["metadata"]["dedupe_key"] = k2
        event["payload"]["_host_ingress_ack"]["payload"]["dedupe_key"] = k2

        await feature._reopen_traffic_gate()
        for _ in range(20):
            if acknowledged:
                break
            await asyncio.sleep(0)
        assert delivered == [k1]
        assert acknowledged == [
            (
                "telegram-polling-ack",
                {"dedupe_key": k1, "attempt_token": _TELEGRAM_ATTEMPT_TOKEN},
            )
        ]
        for _ in range(20):
            if telemetry_active_counts:
                break
            await asyncio.sleep(0)
        assert telemetry_active_counts
        assert set(telemetry_active_counts) == {0}
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_deferred_acknowledged_ingress_exception_nacks_exact_live_callback(
    monkeypatch, tmp_path
):
    """A reopened deferred route failure releases Telegram for redelivery."""

    dedupe_key = "telegram:v2:bot:42:update:deferred-route-failure"
    nacked = asyncio.Event()

    class FailingChannelFeature(FakeChannelFeature):
        async def handle_inbound(self, _message):
            raise RuntimeError("cognition route failed")

    class CompletionClient(TelegramChannelClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack", "telegram-polling-nack")
            )
            self.completions = []

        async def call_host_ingress(self, name, payload=None):
            self.completions.append((name, payload))
            if name == "telegram-polling-nack":
                nacked.set()
                return {"status": "ok", "http_status": 200, "state": "retrying"}
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    agent = Mock(
        did=_TEST_AGENT_DID,
        features={"ChannelFeature": FailingChannelFeature()},
    )
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=CompletionClient)
    try:
        await feature.initialize()
        client = feature._client
        await feature._close_traffic_gate()
        await client.event_handler(_retryable_telegram_event(dedupe_key))
        await feature._reopen_traffic_gate()
        await asyncio.wait_for(nacked.wait(), timeout=1)
        assert client.completions == [
            (
                "telegram-polling-nack",
                {
                    "dedupe_key": dedupe_key,
                    "attempt_token": _TELEGRAM_ATTEMPT_TOKEN,
                },
            )
        ]
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_ingress_workers_are_bounded_to_one_per_source_under_1000_events(monkeypatch, tmp_path):
    """Sol regression: 1,000 callbacks cannot create a host-memory route queue."""

    release_ack = asyncio.Event()
    ack_started = asyncio.Event()
    delivered = []

    class ChannelFeature(FakeChannelFeature):
        async def handle_inbound(self, message):
            delivered.append(message.id)
            return SimpleNamespace(durably_admitted=True)

    class SlowAckClient(TelegramChannelClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )
            self.acknowledgements = []

        async def call_host_ingress(self, name, payload=None):
            self.acknowledgements.append((name, payload))
            ack_started.set()
            await release_ack.wait()
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": ChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=SlowAckClient)
    try:
        await feature.initialize()
        client = feature._client
        for update_id in range(1_000):
            await client.event_handler(
                _acknowledged_telegram_event(f"telegram:v2:bot:42:update:{1_000 + update_id}")
            )
        await asyncio.wait_for(ack_started.wait(), timeout=1)
        # The serial provider must retain every later update while the first
        # detached route/ACK owns the source. Core intentionally does not turn
        # those notifications into 999 unbounded cognition tasks.
        assert len(delivered) == 1
        assert len(feature._event_ingress_tasks) == 0
        assert len(feature._event_ack_tasks) == 1
        assert feature._non_cursor_event_ingress_queues == []
        assert len(client.acknowledgements) == 1
        release_ack.set()
    finally:
        release_ack.set()
        await feature.shutdown()


@pytest.mark.asyncio
async def test_legacy_inbound_events_drop_across_a_closed_live_transition(
    monkeypatch, tmp_path
):
    """Old non-cursor callbacks never replay after a live gate closes."""

    delivered = []
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)

    async def record(message):
        delivered.append(message["id"])
        return SimpleNamespace(durably_admitted=True)

    def legacy_event(message_id):
        return {
            "type": "channel.inbound",
            "payload": {"id": message_id, "content": message_id},
        }

    try:
        await feature.initialize()
        client = feature._client
        feature._route_inbound = record  # type: ignore[method-assign]
        await feature._close_traffic_gate()

        # Both callbacks originate from the old child while a live transition
        # owns the closed gate. They must be retired rather than replayed.
        await client.event_handler(legacy_event("first"))
        await asyncio.sleep(0)
        await client.event_handler(legacy_event("second"))
        assert delivered == []

        await feature._reopen_traffic_gate()
        await asyncio.sleep(0)
        assert delivered == []
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_legacy_inbound_events_are_serial_and_backpressured_while_gate_open(
    monkeypatch, tmp_path
):
    """Open-gate legacy callbacks retain arrival order without task fan-out."""

    first_started = asyncio.Event()
    release_first = asyncio.Event()
    delivered = []
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)

    async def record(message):
        delivered.append(message["id"])
        if message["id"] == "first":
            first_started.set()
            await release_first.wait()
        return SimpleNamespace(durably_admitted=True)

    def legacy_event(message_id):
        return {
            "type": "channel.inbound",
            "payload": {"id": message_id, "content": message_id},
        }

    try:
        await feature.initialize()
        feature._route_inbound = record  # type: ignore[method-assign]
        await feature._client.event_handler(legacy_event("first"))
        await asyncio.wait_for(first_started.wait(), timeout=1)
        await feature._client.event_handler(legacy_event("second"))
        assert delivered == ["first"]
        assert len(feature._event_ingress_tasks) == 1
        release_first.set()
        for _ in range(20):
            if delivered == ["first", "second"]:
                break
            await asyncio.sleep(0)
        assert delivered == ["first", "second"]
    finally:
        release_first.set()
        await feature.shutdown()


@pytest.mark.asyncio
async def test_cross_process_callbacks_chain_attempt_scoped_completions_per_source(
    monkeypatch, tmp_path
):
    """A second child callback waits for, rather than losing to, a delayed ACK RPC."""

    first_ack_started = asyncio.Event()
    release_first_ack = asyncio.Event()
    second_ack_started = asyncio.Event()
    routed = []

    class ChannelFeature(FakeChannelFeature):
        async def handle_inbound(self, message):
            routed.append(message.id)
            return SimpleNamespace(durably_admitted=True)

    class DelayedAckClient(TelegramChannelClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )
            self.acknowledgements = []

        async def call_host_ingress(self, name, payload=None):
            self.acknowledgements.append((name, payload))
            if len(self.acknowledgements) == 1:
                first_ack_started.set()
                await release_first_ack.wait()
            else:
                second_ack_started.set()
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": ChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=DelayedAckClient)
    first_key = "telegram:v2:bot:42:update:completion-one"
    second_key = "telegram:v2:bot:42:update:completion-two"
    try:
        await feature.initialize()
        client = feature._client
        first = _acknowledged_telegram_event(first_key)
        second = _acknowledged_telegram_event(second_key)
        first["payload"]["_host_ingress_ack"]["payload"]["attempt_token"] = "a" * 43
        second["payload"]["_host_ingress_ack"]["payload"]["attempt_token"] = "b" * 43
        # The retry half is still required to authenticate the Telegram pair.
        first["payload"]["_host_ingress_retry"]["payload"]["attempt_token"] = "a" * 43
        second["payload"]["_host_ingress_retry"]["payload"]["attempt_token"] = "b" * 43

        await client.event_handler(first)
        await asyncio.wait_for(first_ack_started.wait(), timeout=1)
        # Callback one has completed host routing, but its source RPC remains
        # in flight. This mirrors the child process receiving the next callback
        # before its first private RPC response is delivered.
        await client.event_handler(second)
        await asyncio.sleep(0)
        assert routed == [first_key]
        assert client.acknowledgements == [
            ("telegram-polling-ack", {"dedupe_key": first_key, "attempt_token": "a" * 43})
        ]

        release_first_ack.set()
        await asyncio.wait_for(second_ack_started.wait(), timeout=1)
        assert routed == [first_key, second_key]
        assert client.acknowledgements == [
            ("telegram-polling-ack", {"dedupe_key": first_key, "attempt_token": "a" * 43}),
            ("telegram-polling-ack", {"dedupe_key": second_key, "attempt_token": "b" * 43}),
        ]
        await asyncio.gather(*tuple(feature._event_ack_tasks))
        assert feature._event_ack_tasks == set()
    finally:
        release_first_ack.set()
        await feature.shutdown()


@pytest.mark.asyncio
async def test_rejected_ack_retries_then_terminally_retires_exact_source(monkeypatch, tmp_path):
    """A rejected polling ACK retries idempotently, then leaves Telegram's cursor for restart."""

    monkeypatch.setattr(isolated_runtime, "_EVENT_INGRESS_ACK_BACKOFF", 0)
    dedupe_key = "telegram:v2:bot:42:update:205"

    class ChannelFeature(FakeChannelFeature):
        async def handle_inbound(self, message):
            return SimpleNamespace(durably_admitted=True)

    class RejectingAckClient(TelegramChannelClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )
            self.acknowledgements = []

        async def call_host_ingress(self, name, payload=None):
            self.acknowledgements.append((name, payload))
            return {"status": "error", "http_status": 409, "error": "still pending"}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": ChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=RejectingAckClient)
    try:
        await feature.initialize()
        client = feature._client
        await client.event_handler(_acknowledged_telegram_event(dedupe_key))
        for _ in range(50):
            if feature._terminal_lifecycle_latched and client.stopped:
                break
            await asyncio.sleep(0)
        assert client.acknowledgements == [
            (
                "telegram-polling-ack",
                {
                    "dedupe_key": dedupe_key,
                    "attempt_token": _TELEGRAM_ATTEMPT_TOKEN,
                },
            )
        ] * 3
        assert feature._terminal_lifecycle_latched is True
        assert client.stopped is True
        assert feature._client is None
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_hung_ack_times_out_with_bounded_retries_then_retires_source(
    monkeypatch, tmp_path
):
    """A never-completing ACK is fenced; it cannot retain an unbounded task swarm."""

    monkeypatch.setattr(isolated_runtime, "_EVENT_INGRESS_ACK_TIMEOUT", 0.01)
    monkeypatch.setattr(isolated_runtime, "_EVENT_INGRESS_ACK_CANCELLATION_GRACE", 0.01)
    monkeypatch.setattr(isolated_runtime, "_EVENT_INGRESS_ACK_BACKOFF", 0)
    dedupe_key = "telegram:v2:bot:42:update:208"

    class ChannelFeature(FakeChannelFeature):
        async def handle_inbound(self, message):
            return SimpleNamespace(durably_admitted=True)

    class HungAckClient(TelegramChannelClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )
            self.acknowledgements = []

        async def call_host_ingress(self, name, payload=None):
            self.acknowledgements.append((name, payload))
            await asyncio.Event().wait()

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": ChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=HungAckClient)
    try:
        await feature.initialize()
        client = feature._client
        await client.event_handler(_acknowledged_telegram_event(dedupe_key))
        for _ in range(100):
            if feature._terminal_lifecycle_latched and client.stopped:
                break
            await asyncio.sleep(0.01)
        assert client.acknowledgements == [
            (
                "telegram-polling-ack",
                {
                    "dedupe_key": dedupe_key,
                    "attempt_token": _TELEGRAM_ATTEMPT_TOKEN,
                },
            )
        ] * 3
        assert feature._terminal_lifecycle_latched is True
        assert client.stopped is True
        assert feature._event_ack_tasks == set()
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_terminal_owner_joins_ack_and_deferred_workers_and_blocks_new_ack(
    monkeypatch, tmp_path
):
    """Terminal success is fenced until both detached ingress worker kinds settle."""

    ack_started = asyncio.Event()
    release_ack = asyncio.Event()

    class ChannelFeature(FakeChannelFeature):
        async def handle_inbound(self, message):
            return SimpleNamespace(durably_admitted=True)

    class SlowAckClient(TelegramChannelClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("telegram-polling-ack",)
            )

        async def call_host_ingress(self, name, payload=None):
            ack_started.set()
            await release_ack.wait()
            return {"status": "ok", "http_status": 200, "state": "acknowledged"}

    agent = Mock(did=_TEST_AGENT_DID, features={"ChannelFeature": ChannelFeature()})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=SlowAckClient)
    try:
        await feature.initialize()
        client = feature._client
        first = _acknowledged_telegram_event("telegram:v2:bot:42:update:206")
        await client.event_handler(first)
        await asyncio.wait_for(ack_started.wait(), timeout=1)
        await feature._close_traffic_gate()
        await client.event_handler(_acknowledged_telegram_event("telegram:v2:bot:42:update:207"))
        for _ in range(20):
            if feature._deferred_acknowledged_event_tasks:
                break
            await asyncio.sleep(0)
        assert feature._event_ack_tasks
        assert feature._deferred_acknowledged_event_tasks

        feature._latch_terminal_lifecycle()
        release_ack.set()
        await feature._complete_terminal_cleanup()
        assert feature._event_ack_tasks == set()
        assert feature._deferred_acknowledged_event_tasks == set()
        request = feature._event_ingress_acknowledgement(first)
        assert request is not None
        feature._schedule_event_ingress_acknowledgement(client, request)
        assert feature._event_ack_tasks == set()
        assert client.stopped is True
    finally:
        release_ack.set()
        await feature.shutdown()


@pytest.mark.asyncio
async def test_exact_lifecycle_rpc_bypasses_closed_data_plane_and_waits_for_drain(
    monkeypatch, tmp_path
):
    """A slow already-admitted send cannot consume Core's quiesce lifecycle budget."""

    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    lifecycle_calls = []

    class SlowToolClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.host_ingress_capabilities = HostIngressCapabilities(
                names=("external-ingress-quiesce", "external-ingress-resume")
            )

        async def call_tool(self, name, args):
            tool_started.set()
            await release_tool.wait()
            return {"echo": args}

        async def call_host_ingress(self, name, payload=None):
            lifecycle_calls.append((name, payload))
            state = "quiesced" if name == "external-ingress-quiesce" else "resumed"
            return {"status": "ok", "http_status": 200, "state": state}

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=SlowToolClient)
    tool_task = None
    try:
        await feature.initialize()
        tool_task = asyncio.create_task(feature.call_isolated_tool("ping", {}))
        await asyncio.wait_for(tool_started.wait(), timeout=1)
        quiesce = feature._new_external_ingress_quiesce()
        assert quiesce is not None
        await feature._close_traffic_gate_admission()
        assert await asyncio.wait_for(feature._quiesce_external_ingress(quiesce), timeout=1) is None
        assert not tool_task.done()
        assert lifecycle_calls == [
            ("external-ingress-quiesce", {"transition_id": quiesce.transition_id})
        ]
        release_tool.set()
        await tool_task
        await feature._drain_traffic_gate()
        await feature._resume_external_ingress(quiesce)
        await feature._reopen_traffic_gate()
    finally:
        release_tool.set()
        if tool_task is not None and not tool_task.done():
            await tool_task
        await feature.shutdown()


def test_inbound_policy_reads_do_not_mint_external_ingress_transition_tokens(
    monkeypatch,
):
    client = FakeIsolatedClient()
    client.host_ingress_capabilities = HostIngressCapabilities(
        names=("external-ingress-quiesce", "external-ingress-resume")
    )
    agent = Mock(did=_TEST_AGENT_DID, features={})
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    mint = Mock(side_effect=AssertionError("policy read minted transition state"))
    monkeypatch.setattr(isolated_runtime.secrets, "token_urlsafe", mint)

    assert feature._owns_inbound_producer() is True
    assert feature._inbound_producer_role_is_ambiguous() is False
    mint.assert_not_called()


class _HostIngressClient(FakeIsolatedClient):
    """SDK-client double that advertises a typed private ingress contract."""

    def __init__(self, *, ingress_capabilities=None, **kwargs):
        super().__init__(**kwargs)
        self.ingress_capabilities = ingress_capabilities
        self.ingress_calls = []

    @property
    def host_ingress_capabilities(self):
        return self.ingress_capabilities

    async def call_host_ingress(self, name, payload=None):
        self.ingress_calls.append((name, payload))
        return {"accepted": True}


async def _initialized_host_ingress_proxy(
    monkeypatch, tmp_path, client_factory, *, agent=None
):
    agent = agent or Mock(did=_TEST_AGENT_DID, features={})
    agent.features = getattr(agent, "features", {})
    storage = _CASStorage()
    storage.agent_id = agent.did
    agent.storage = storage
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    await feature.initialize()
    return feature, agent


def _walk_exception_chain(error):
    """Yield every exception recursively reachable through chaining links."""

    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend((current.__context__, current.__cause__))


def _assert_host_ingress_error_is_detached(error, *, secret, external=None):
    """A redacted public failure must retain no private exception graph."""

    chain = list(_walk_exception_chain(error))
    assert all(secret not in str(item) for item in chain)
    assert all(secret not in repr(item) for item in chain)
    assert all(type(item) is HostIngressError for item in chain)
    if external is not None:
        assert all(item is not external for item in chain)


def _assert_host_ingress_tracebacks_are_secret_free(error, *, secret):
    """Inspect every internal traceback frame on every chained error.

    ``raise ... from None`` only changes formatting; exception context and
    traceback locals remain directly reachable to debuggers/telemetry.  Keep
    this intentionally stronger than a message-redaction assertion.
    """

    forbidden_frames = {"_call_host_ingress_rpc", "_run_host_ingress"}
    for chained in _walk_exception_chain(error):
        traceback = chained.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if frame.f_code.co_filename == isolated_runtime.__file__:
                assert frame.f_code.co_name not in forbidden_frames
                assert secret not in repr(dict(frame.f_locals))
            traceback = traceback.tb_next


def _safe_traceback_references(value):
    """Yield bounded, non-executing references for traceback graph checks.

    Do not use ``repr()``, ordinary ``getattr()``, or application-defined
    properties here: the adversarial values under test can make each of those
    secret-bearing or executable.  Never expand modules, globals, or classes:
    those are ambient process state rather than a public-error retention path.
    Native runtime objects are different: a public traceback can retain an
    operation task, and a done task can retain its secret-bearing result.  Walk
    only their built-in, non-executing reference edges below.
    """

    value_type = type(value)
    if value_type is dict:
        yield from value.keys()
        yield from value.values()
        return
    if value_type in (list, tuple, set, frozenset):
        yield from value
        return

    if value_type in (asyncio.Task, asyncio.Future):
        # Use the base Future descriptors, never a facade/subclass override.
        # A completed operation owns both its result and exception internally;
        # either can otherwise retain private response/client data.
        try:
            done = asyncio.Future.done(value)
            cancelled = asyncio.Future.cancelled(value)
        except (AttributeError, TypeError):
            return
        if done and not cancelled:
            try:
                yield asyncio.Future.result(value)
            except BaseException as error:  # native task failure graph
                yield error
            try:
                error = asyncio.Future.exception(value)
            except BaseException as error:  # native task failure graph
                yield error
            else:
                if error is not None:
                    yield error
        for attribute in ("_fut_waiter", "_callbacks"):
            try:
                reference = object.__getattribute__(value, attribute)
            except (AttributeError, TypeError):
                continue
            if type(reference) in (list, tuple):
                yield from reference
            elif reference is not None:
                yield reference
        return

    if value_type is types.CoroutineType:
        for attribute in ("cr_frame", "cr_await"):
            try:
                reference = object.__getattribute__(value, attribute)
            except (AttributeError, TypeError):
                continue
            if reference is not None:
                yield reference
        return

    if value_type is types.GeneratorType:
        for attribute in ("gi_frame", "gi_yieldfrom"):
            try:
                reference = object.__getattribute__(value, attribute)
            except (AttributeError, TypeError):
                continue
            if reference is not None:
                yield reference
        return

    if value_type is types.AsyncGeneratorType:
        for attribute in ("ag_frame", "ag_await"):
            try:
                reference = object.__getattribute__(value, attribute)
            except (AttributeError, TypeError):
                continue
            if reference is not None:
                yield reference
        return

    if value_type is types.TracebackType:
        yield value.tb_frame
        if value.tb_next is not None:
            yield value.tb_next
        return

    if value_type is types.FrameType:
        # ``f_locals`` is a runtime mapping, not application code.  Do not
        # inspect ``f_globals``: modules/global graphs are intentionally leaves.
        namespace = value.f_locals
        if type(namespace) is dict:
            yield from namespace.values()
        return

    if issubclass(value_type, BaseException):
        for attribute in ("args", "__cause__", "__context__", "__traceback__"):
            try:
                reference = object.__getattribute__(value, attribute)
            except (AttributeError, TypeError):
                continue
            if reference is not None:
                yield reference

    if value_type is types.CellType:
        try:
            yield value.cell_contents
        except ValueError:
            pass
        return

    if value_type is types.FunctionType:
        closure = object.__getattribute__(value, "__closure__")
        if type(closure) is tuple:
            yield from closure
        return

    if value_type in (types.MethodType, types.BuiltinMethodType):
        try:
            yield object.__getattribute__(value, "__self__")
        except (AttributeError, TypeError):
            pass
        if value_type is types.MethodType:
            yield object.__getattribute__(value, "__func__")
        return

    if isinstance(
        value,
        (
            types.ModuleType,
            types.BuiltinFunctionType,
            type,
            types.CodeType,
        ),
    ):
        return

    # Calling object.__getattribute__ bypasses an instance's hostile
    # __getattribute__ implementation. A class that replaces __dict__ with a
    # property is skipped rather than executed.
    class_namespace = type.__getattribute__(value_type, "__dict__")
    if isinstance(class_namespace.get("__dict__"), property):
        return
    try:
        namespace = object.__getattribute__(value, "__dict__")
    except (AttributeError, TypeError):
        return
    if type(namespace) is dict:
        yield from namespace.keys()
        yield from namespace.values()


def _assert_runtime_traceback_locals_are_detached(
    error, *, forbidden_values, operation=None
):
    """Assert public runtime frames retain no task or private input/output.

    This direct check complements the graph traversal: if a future refactor
    forgets to clear the local operation task before raising, fail immediately
    rather than relying on a particular task implementation's result graph.
    """

    forbidden_ids = {id(value) for value in forbidden_values}
    if operation is not None:
        forbidden_ids.add(id(operation))
    for chained in _walk_exception_chain(error):
        traceback = chained.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if frame.f_code.co_filename == isolated_runtime.__file__:
                local_values = tuple(frame.f_locals.values())
                assert not any(
                    type(value) in (asyncio.Task, asyncio.Future)
                    for value in local_values
                )
                assert not any(id(value) in forbidden_ids for value in local_values)
            traceback = traceback.tb_next


def _traceback_locals_reach_any(
    error, forbidden_values, *, runtime_filename=isolated_runtime.__file__
):
    """Whether runtime traceback locals reach an exact forbidden object."""

    forbidden_ids = {id(value) for value in forbidden_values}
    pending = []
    for chained in _walk_exception_chain(error):
        traceback = chained.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if frame.f_code.co_filename == runtime_filename:
                pending.extend(frame.f_locals.values())
            traceback = traceback.tb_next

    seen = set()
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in forbidden_ids:
            return True
        if identity in seen:
            continue
        seen.add(identity)
        # This limit is a circuit breaker, not a lossy global traversal: the
        # leaf policy above excludes the objects that could otherwise explode.
        assert len(seen) < 4096
        pending.extend(_safe_traceback_references(value))
    return False


@pytest.mark.asyncio
async def test_traceback_reachability_follows_done_operation_task_result():
    """The cancellation guard reaches a secret retained only by a task result."""

    response_secret = "TRACEBACK-TASK-RESULT-SECRET-2755"

    async def raise_with_operation_local():
        operation = asyncio.create_task(
            asyncio.sleep(0, result={"response": response_secret})
        )
        await operation
        raise RuntimeError("synthetic public failure")

    with pytest.raises(RuntimeError) as raised:
        await raise_with_operation_local()
    assert _traceback_locals_reach_any(
        raised.value,
        (response_secret,),
        runtime_filename=__file__,
    )


@pytest.mark.asyncio
async def test_host_ingress_is_private_and_uses_the_negotiated_sdk_contract(
    monkeypatch, tmp_path
):
    """Host ingress is callable without becoming an agent-visible tool."""

    client = _HostIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    try:
        result = await feature.call_host_ingress(
            "telegram-webhook", {"update_id": 7}
        )

        assert result == {"accepted": True}
        assert client.ingress_calls == [("telegram-webhook", {"update_id": 7})]
        # The service's ordinary tools remain the only agent/LLM tool surface.
        assert [tool.name for tool in feature.get_tools()] == ["ping"]
        assert all(
            "telegram-webhook" not in tool.schema.description
            for tool in feature.get_tools()
        )
        assert "telegram-webhook" not in feature.tool_description
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_uses_the_published_wrapper_not_its_inner_transport(
    monkeypatch, tmp_path
):
    """A subprocess-style facade owns both capability passthrough and RPC."""

    inner = _HostIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )

    class Wrapper:
        def __init__(self):
            self.client = inner
            self.wrapper_calls = []

        @property
        def host_ingress_capabilities(self):
            return self.client.host_ingress_capabilities

        async def call_host_ingress(self, name, payload=None):
            self.wrapper_calls.append((name, payload))
            return await self.client.call_host_ingress(name, payload)

        def __getattr__(self, name):
            return getattr(self.client, name)

    wrapper = Wrapper()
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: wrapper
    )
    try:
        assert await feature.call_host_ingress("telegram-webhook", {"id": 1}) == {
            "accepted": True
        }
        assert wrapper.wrapper_calls == [("telegram-webhook", {"id": 1})]
        assert inner.ingress_calls == [("telegram-webhook", {"id": 1})]
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capabilities", "error_type"),
    [
        (None, HostIngressUnsupportedError),
        ({"version": 1, "names": ["telegram-webhook"]}, HostIngressUnsupportedError),
        (
            HostIngressCapabilities(names=("other-webhook",)),
            HostIngressUnknownNameError,
        ),
    ],
)
async def test_host_ingress_fails_closed_for_missing_malformed_or_unknown_capabilities(
    monkeypatch, tmp_path, capabilities, error_type
):
    """No ingress RPC is emitted unless typed metadata names it exactly."""

    client = _HostIngressClient(ingress_capabilities=capabilities)
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    try:
        with pytest.raises(error_type) as raised:
            await feature.call_host_ingress("telegram-webhook", {"update_id": 7})

        assert type(raised.value) is error_type
        assert client.ingress_calls == []
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_rejects_str_subclasses_before_capability_comparison(
    monkeypatch, tmp_path
):
    """A hostile ``str`` subclass cannot equality-match an advertised name."""

    class EqualToEverything(str):
        def __eq__(self, other):
            return True

        def __hash__(self):
            return hash(str(self))

    client = _HostIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("other-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    try:
        with pytest.raises(HostIngressError) as raised:
            await feature.call_host_ingress(
                EqualToEverything("not-an-advertised-webhook"),
                {"update_id": 7},
            )

        assert type(raised.value) is HostIngressError
        assert str(raised.value) == "host ingress failed"
        assert client.ingress_calls == []
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_rejects_mutable_and_malformed_capabilities(
    monkeypatch, tmp_path
):
    """Frozen SDK dataclasses with mutable or corrupted fields fail closed."""

    mutable_names = ["telegram-webhook"]
    mutable = HostIngressCapabilities(names=mutable_names)
    mutable_names.append("unexpected-webhook")
    malformed = HostIngressCapabilities(names=("telegram-webhook",))
    object.__setattr__(malformed, "version", 2)

    for capabilities in (mutable, malformed):
        client = _HostIngressClient(ingress_capabilities=capabilities)
        feature, _ = await _initialized_host_ingress_proxy(
            monkeypatch, tmp_path, lambda client=client, **_: client
        )
        try:
            with pytest.raises(HostIngressUnsupportedError) as raised:
                await feature.call_host_ingress("telegram-webhook", {"update_id": 7})

            assert type(raised.value) is HostIngressUnsupportedError
            assert client.ingress_calls == []
        finally:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_rejects_subclass_with_overridden_supports(
    monkeypatch, tmp_path
):
    """An untrusted capability subclass cannot grant an unadvertised name."""

    class PermissiveCapabilities(HostIngressCapabilities):
        def supports(self, name):
            return True

    client = _HostIngressClient(
        ingress_capabilities=PermissiveCapabilities(names=("other-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    try:
        with pytest.raises(HostIngressUnsupportedError) as raised:
            await feature.call_host_ingress("telegram-webhook", {"update_id": 7})

        assert type(raised.value) is HostIngressUnsupportedError
        assert client.ingress_calls == []
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_does_not_resolve_or_fall_back_to_another_agents_proxy(
    monkeypatch, tmp_path
):
    """The caller's already-resolved proxy is the only ingress authority."""

    alice_client = _HostIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    bob_client = _HostIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    alice_agent = Mock(did="did:test:alice", features={})
    bob_agent = Mock(did="did:test:bob", features={})
    alice, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path / "alice", lambda **_: alice_client, agent=alice_agent
    )
    bob, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path / "bob", lambda **_: bob_client, agent=bob_agent
    )
    # An ambient registry-like reference must not influence a proxy call.
    alice_agent.features["CompanionFeature"] = bob
    try:
        assert (
            await alice.call_host_ingress("telegram-webhook", {"agent": "alice"})
            == {"accepted": True}
        )
        assert alice_client.ingress_calls == [
            ("telegram-webhook", {"agent": "alice"})
        ]
        assert bob_client.ingress_calls == []
    finally:
        await alice.shutdown()
        await bob.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_redacts_invalid_payload_and_transport_failure(
    monkeypatch, tmp_path, caplog
):
    """Host-only request/config/transport details never cross the proxy API."""

    secret = "super-secret-webhook-token"
    transport_external = RuntimeError(
        f"stdio://private/{secret} config={{'token': '{secret}'}}"
    )

    class LeakyClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            raise transport_external

    client = LeakyClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    try:
        with (
            caplog.at_level("WARNING", logger=isolated_runtime.__name__),
            pytest.raises(HostIngressError) as transport_error,
        ):
            await feature.call_host_ingress("telegram-webhook", {"token": secret})
        assert str(transport_error.value) == "host ingress failed"
        _assert_host_ingress_error_is_detached(
            transport_error.value,
            secret=secret,
            external=transport_external,
        )
        assert secret not in str(transport_error.value)
        assert "stdio://" not in str(transport_error.value)
        assert secret not in caplog.text

        with pytest.raises(HostIngressError) as invalid_payload_error:
            await feature.call_host_ingress("telegram-webhook", {"token": object()})
        assert str(invalid_payload_error.value) == "host ingress failed"
        _assert_host_ingress_error_is_detached(
            invalid_payload_error.value,
            secret=secret,
        )
        assert client.ingress_calls == [("telegram-webhook", {"token": secret})]
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_redacts_capability_getter_and_response_errors(
    monkeypatch, tmp_path, caplog
):
    """HostIngressError from an external facade never bypasses redaction."""

    secret = "super-secret-webhook-token"
    getter_external = RuntimeError(f"private capability token={secret}")

    class GetterErrorClient(_HostIngressClient):
        @property
        def host_ingress_capabilities(self):
            raise getter_external

    getter_client = GetterErrorClient()
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: getter_client
    )
    try:
        caplog.clear()
        with (
            caplog.at_level("WARNING", logger=isolated_runtime.__name__),
            pytest.raises(HostIngressError) as getter_error,
        ):
            await feature.call_host_ingress("telegram-webhook", {"update_id": 7})

        assert type(getter_error.value) is HostIngressError
        assert str(getter_error.value) == "host ingress failed"
        _assert_host_ingress_error_is_detached(
            getter_error.value,
            secret=secret,
            external=getter_external,
        )
        assert secret not in caplog.text
        assert getter_client.ingress_calls == []
    finally:
        await feature.shutdown()

    class SecretResponse:
        def __repr__(self):
            return secret

    class InvalidResponseClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            return {"token": SecretResponse()}

    response_client = InvalidResponseClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: response_client
    )
    try:
        caplog.clear()
        with (
            caplog.at_level("WARNING", logger=isolated_runtime.__name__),
            pytest.raises(HostIngressError) as response_error,
        ):
            await feature.call_host_ingress("telegram-webhook", {"update_id": 7})

        assert type(response_error.value) is HostIngressError
        assert str(response_error.value) == "host ingress failed"
        _assert_host_ingress_error_is_detached(
            response_error.value,
            secret=secret,
        )
        assert secret not in caplog.text
        assert response_client.ingress_calls == [
            ("telegram-webhook", {"update_id": 7})
        ]
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_uses_exact_detached_json_request_and_response_snapshots(
    monkeypatch, tmp_path
):
    """Ingress cannot dispatch subclasses or observe post-validation mutation."""

    entered = asyncio.Event()
    release = asyncio.Event()
    original_request = {"outer": {"state": "before"}, "items": [1, 2]}
    original_response = {"outer": {"accepted": True}, "items": [3, 4]}

    class SnapshotClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            entered.set()
            await release.wait()
            self.ingress_calls.append((name, payload))
            return original_response

    client = SnapshotClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    call = None
    try:
        call = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", original_request)
        )
        await asyncio.wait_for(entered.wait(), timeout=1)

        # The client has not consumed its argument yet. Mutating the original
        # after admission must not rewrite the already-snapshotted RPC payload.
        original_request["outer"]["state"] = "after"
        original_request["items"].append(99)
        release.set()
        result = await call

        assert client.ingress_calls == [
            ("telegram-webhook", {"outer": {"state": "before"}, "items": [1, 2]})
        ]
        assert result == original_response
        original_response["outer"]["accepted"] = False
        original_response["items"].append(99)
        assert result == {"outer": {"accepted": True}, "items": [3, 4]}

        class DictSubclass(dict):
            pass

        class ListSubclass(list):
            pass

        class StringSubclass(str):
            pass

        invalid_payloads = (
            DictSubclass({"token": "subclass"}),
            {"items": ListSubclass([1])},
            {"token": StringSubclass("subclass")},
            {StringSubclass("token"): "subclass"},
        )
        for invalid_payload in invalid_payloads:
            with pytest.raises(HostIngressError, match="host ingress failed"):
                await feature.call_host_ingress("telegram-webhook", invalid_payload)
        assert len(client.ingress_calls) == 1

        class ResponseSubclassClient(_HostIngressClient):
            async def call_host_ingress(self, name, payload=None):
                self.ingress_calls.append((name, payload))
                return {"token": StringSubclass("response-subclass")}

        response_client = ResponseSubclassClient(
            ingress_capabilities=HostIngressCapabilities(
                names=("telegram-webhook",)
            )
        )
        response_feature, _ = await _initialized_host_ingress_proxy(
            monkeypatch, tmp_path / "response", lambda **_: response_client
        )
        try:
            with pytest.raises(HostIngressError, match="host ingress failed"):
                await response_feature.call_host_ingress(
                    "telegram-webhook", {"update_id": 1}
                )
            assert response_client.ingress_calls == [
                ("telegram-webhook", {"update_id": 1})
            ]
        finally:
            await response_feature.shutdown()
    finally:
        release.set()
        if call is not None and not call.done():
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_preflights_exact_json_size_before_sdk_serialization(
    monkeypatch, tmp_path
):
    """Dense valid JSON reaches the SDK; oversized escaped values never do."""

    # ``[0,0,...]`` is the densest non-empty JSON tree: 32,767 scalar leaves
    # plus brackets/separators encode to 65,535 bytes, one below the SDK cap.
    dense_payload = [0] * ((MAX_HOST_INGRESS_PAYLOAD_BYTES - 1) // 2)
    dense_response = list(dense_payload)
    oversized_astral = "\U0001f642" * (MAX_HOST_INGRESS_PAYLOAD_BYTES // 12 + 1)
    validator_payloads = []
    sdk_validator = isolated_runtime.validate_host_ingress_payload

    def recording_validator(value):
        validator_payloads.append(value)
        return sdk_validator(value)

    monkeypatch.setattr(
        isolated_runtime,
        "validate_host_ingress_payload",
        recording_validator,
    )

    class DenseClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            return dense_response

    client = DenseClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    try:
        assert await feature.call_host_ingress("telegram-webhook", dense_payload) == dense_response
        assert client.ingress_calls == [("telegram-webhook", dense_payload)]

        # Both a scalar and a key exceed 64 KiB only after the SDK's required
        # ASCII escaping.  They must fail before the SDK validator or facade
        # can materialize/receive them.
        validator_count = len(validator_payloads)
        for oversized_request in (
            {"payload": oversized_astral},
            {oversized_astral: "payload"},
        ):
            with pytest.raises(HostIngressError, match="host ingress failed") as raised:
                await feature.call_host_ingress("telegram-webhook", oversized_request)
            _assert_host_ingress_error_is_detached(
                raised.value,
                secret=oversized_astral,
            )
            _assert_host_ingress_tracebacks_are_secret_free(
                raised.value,
                secret=oversized_astral,
            )
        assert len(validator_payloads) == validator_count
        assert client.ingress_calls == [("telegram-webhook", dense_payload)]

        class OversizedResponseClient(_HostIngressClient):
            async def call_host_ingress(self, name, payload=None):
                self.ingress_calls.append((name, payload))
                return {"payload": oversized_astral}

        response_client = OversizedResponseClient(
            ingress_capabilities=HostIngressCapabilities(
                names=("telegram-webhook",)
            )
        )
        response_feature, _ = await _initialized_host_ingress_proxy(
            monkeypatch, tmp_path / "response", lambda **_: response_client
        )
        try:
            with pytest.raises(HostIngressError, match="host ingress failed") as raised:
                await response_feature.call_host_ingress(
                    "telegram-webhook", {"update_id": 1}
                )
            _assert_host_ingress_error_is_detached(
                raised.value,
                secret=oversized_astral,
            )
            _assert_host_ingress_tracebacks_are_secret_free(
                raised.value,
                secret=oversized_astral,
            )
            assert response_client.ingress_calls == [
                ("telegram-webhook", {"update_id": 1})
            ]
            # The response reaches the host worker, but its oversized value
            # never reaches the SDK validator's allocating serializer.
            assert not any(
                value == {"payload": oversized_astral}
                for value in validator_payloads
            )
        finally:
            await response_feature.shutdown()
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_accepts_shared_alias_dags_as_detached_json_trees(
    monkeypatch, tmp_path
):
    """SDK-valid aliases are charged and copied independently per branch."""

    client = _HostIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    shared = {"items": ["first"]}
    payload = {"left": shared, "right": shared}
    validator_payloads = []
    sdk_validator = isolated_runtime.validate_host_ingress_payload

    def recording_validator(value):
        validator_payloads.append(value)
        return sdk_validator(value)

    monkeypatch.setattr(isolated_runtime, "validate_host_ingress_payload", recording_validator)
    try:
        assert await feature.call_host_ingress("telegram-webhook", payload) == {
            "accepted": True
        }
        dispatched = client.ingress_calls[0][1]
        assert dispatched == payload
        assert dispatched is not payload
        assert dispatched["left"] is not dispatched["right"]
        assert dispatched["left"]["items"] is not dispatched["right"]["items"]
        assert json.dumps(dispatched, ensure_ascii=True, separators=(",", ":")) == json.dumps(
            payload, ensure_ascii=True, separators=(",", ":")
        )
        shared["items"].append("source mutation")
        assert dispatched == {
            "left": {"items": ["first"]},
            "right": {"items": ["first"]},
        }
        assert dispatched in validator_payloads
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_rejects_container_cycles_and_excessive_alias_expansion(
    monkeypatch, tmp_path
):
    """Cycles and aliases that exceed the expanded JSON budget fail closed."""

    client = _HostIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    validator_called = False

    def unexpected_validator(_value):
        nonlocal validator_called
        validator_called = True
        raise AssertionError("invalid graph reached SDK validation")

    monkeypatch.setattr(isolated_runtime, "validate_host_ingress_payload", unexpected_validator)
    recursive_list = []
    recursive_list.append(recursive_list)
    recursive_dict = {}
    recursive_dict["self"] = recursive_dict
    expanded: list[object] = []
    for _ in range(16):
        expanded = [expanded, expanded]
    try:
        for invalid_payload in (
            {"tree": recursive_list},
            {"tree": recursive_dict},
            {"tree": expanded},
        ):
            with pytest.raises(HostIngressError, match="host ingress failed"):
                await feature.call_host_ingress("telegram-webhook", invalid_payload)
        assert validator_called is False
        assert client.ingress_calls == []
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_generic_failures_have_no_secret_traceback_locals(
    monkeypatch, tmp_path
):
    """Validation/facade/RPC failures expose neither chains nor frame locals."""

    secret = "TRACEBACK-ONLY-SECRET-2755"

    class SecretValue:
        def __repr__(self):
            return secret

    capability_error = RuntimeError(f"capability={secret}")
    descriptor_error = RuntimeError(f"descriptor={secret}")
    transport_error = RuntimeError(f"transport={secret}")

    class GetterFailureClient(_HostIngressClient):
        @property
        def host_ingress_capabilities(self):
            raise capability_error

    class DescriptorFailureClient(_HostIngressClient):
        @property
        def call_host_ingress(self):
            raise descriptor_error

    class TransportFailureClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            raise transport_error

    class ResponseFailureClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            return {"secret": SecretValue()}

    clients_and_payloads = (
        (
            _HostIngressClient(
                ingress_capabilities=HostIngressCapabilities(
                    names=("telegram-webhook",)
                )
            ),
            {"secret": SecretValue()},
        ),
        (GetterFailureClient(), {"secret": secret}),
        (
            DescriptorFailureClient(
                ingress_capabilities=HostIngressCapabilities(
                    names=("telegram-webhook",)
                )
            ),
            {"secret": secret},
        ),
        (
            TransportFailureClient(
                ingress_capabilities=HostIngressCapabilities(
                    names=("telegram-webhook",)
                )
            ),
            {"secret": secret},
        ),
        (
            ResponseFailureClient(
                ingress_capabilities=HostIngressCapabilities(
                    names=("telegram-webhook",)
                )
            ),
            {"secret": secret},
        ),
    )

    for index, (client, payload) in enumerate(clients_and_payloads):
        feature, _ = await _initialized_host_ingress_proxy(
            monkeypatch,
            tmp_path / str(index),
            lambda client=client, **_: client,
        )
        try:
            with pytest.raises(HostIngressError) as raised:
                await feature.call_host_ingress("telegram-webhook", payload)
            assert type(raised.value) is HostIngressError
            assert str(raised.value) == "host ingress failed"
            _assert_host_ingress_error_is_detached(raised.value, secret=secret)
            _assert_host_ingress_tracebacks_are_secret_free(
                raised.value, secret=secret
            )
            _assert_runtime_traceback_locals_are_detached(
                raised.value,
                forbidden_values=(secret, client),
            )
        finally:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_caller_cancelled_host_ingress_discards_successful_secret_outcome(
    monkeypatch, tmp_path
):
    """A cancellation traceback cannot reach a response that won the RPC race."""

    request_secret = "HOST-INGRESS-REQUEST-SECRET-2755"
    response_secret = "HOST-INGRESS-RESPONSE-SECRET-2755"
    client_secret = "HOST-INGRESS-CLIENT-SECRET-2755"

    class CancelAfterSuccessClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.cancel_target = None
            self.client_secret = client_secret

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            # Deliver cancellation after this task has reached its successful
            # response path but before the public caller can return it.
            self.cancel_target.cancel("first caller cancellation")
            return {"response": response_secret}

    client = CancelAfterSuccessClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    call = None
    try:
        call = asyncio.create_task(
            feature.call_host_ingress(
                "telegram-webhook", {"request": request_secret}
            )
        )
        client.cancel_target = call
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await call

        assert cancelled.value.args == ("first caller cancellation",)
        assert client.ingress_calls == [
            ("telegram-webhook", {"request": request_secret})
        ]
        assert not _traceback_locals_reach_any(
            cancelled.value,
            (request_secret, response_secret, client_secret, client),
        )
        _assert_runtime_traceback_locals_are_detached(
            cancelled.value,
            forbidden_values=(request_secret, response_secret, client_secret, client),
            operation=call,
        )
    finally:
        if call is not None and not call.done():
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_worker_tasks_drop_private_results_on_all_terminal_paths(
    monkeypatch, tmp_path
):
    """Success, failure, and caller cancellation leave no payload in worker tasks."""

    request_secret = "WORKER-REQUEST-SECRET-2755"
    response_secret = "WORKER-RESPONSE-SECRET-2755"
    delayed_started = asyncio.Event()
    release_delayed = asyncio.Event()
    worker_tasks = []

    class TerminalPathClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.mode = "success"
            self.client_secret = "WORKER-CLIENT-SECRET-2755"

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if self.mode == "failure":
                raise RuntimeError(response_secret)
            if self.mode == "delayed":
                delayed_started.set()
                await release_delayed.wait()
            return {"response": response_secret}

    client = TerminalPathClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    real_create_task = asyncio.create_task

    def capture_host_ingress_worker(coro, *, name=None, **kwargs):
        task = real_create_task(coro, name=name, **kwargs)
        if name and name.startswith("isolated-host-ingress:"):
            worker_tasks.append(task)
        return task

    monkeypatch.setattr(
        isolated_runtime.asyncio, "create_task", capture_host_ingress_worker
    )
    delayed_call = None
    try:
        assert await feature.call_host_ingress(
            "telegram-webhook", {"request": request_secret}
        ) == {"response": response_secret}

        client.mode = "failure"
        with pytest.raises(HostIngressError, match="host ingress failed") as failure:
            await feature.call_host_ingress(
                "telegram-webhook", {"request": request_secret}
            )
        _assert_host_ingress_error_is_detached(failure.value, secret=response_secret)

        client.mode = "delayed"
        delayed_call = real_create_task(
            feature.call_host_ingress(
                "telegram-webhook", {"request": request_secret}
            )
        )
        await asyncio.wait_for(delayed_started.wait(), timeout=1)
        delayed_call.cancel("private caller cancellation")
        await asyncio.sleep(0)
        assert delayed_call.done() is False
        release_delayed.set()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await delayed_call
        assert cancelled.value.args == ("private caller cancellation",)

        assert len(worker_tasks) == 3
        for worker in worker_tasks:
            assert worker.done()
            assert worker.cancelled() is False
            # The one-shot slot is consumed before delivery, so the task's
            # native result cannot retain request, response, or client data.
            assert worker.result() is None
            assert worker.get_coro().cr_frame is None
    finally:
        release_delayed.set()
        if delayed_call is not None and not delayed_call.done():
            delayed_call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await delayed_call
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_preserves_sdk_cancellation(monkeypatch, tmp_path):
    """A child-side cancellation is never redacted as a host ingress error."""

    class CancellingClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            raise asyncio.CancelledError()

    client = CancellingClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await feature.call_host_ingress("telegram-webhook", {"update_id": 7})
        assert client.ingress_calls == [("telegram-webhook", {"update_id": 7})]
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_cancellation_drains_remote_call_before_reload(
    monkeypatch, tmp_path
):
    """Caller cancellation cannot release traffic beneath an active RPC."""

    started = asyncio.Event()
    blocker = asyncio.Event()
    clients = []

    class CancellableClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if len(clients) == 1:
                started.set()
                await blocker.wait()
            return {"client": len(clients)}

    def client_factory(**kwargs):
        client = CancellableClient(
            ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",)),
            **kwargs,
        )
        clients.append(client)
        return client

    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, client_factory
    )
    active = reload_task = None
    try:
        active = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 1})
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        active.cancel()
        await asyncio.sleep(0)
        assert not active.done()

        reload_task = asyncio.create_task(feature.reload())
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)
        assert feature._traffic_gate.closed is True
        assert not reload_task.done()
        assert clients[0].stopped is False

        blocker.set()
        with pytest.raises(asyncio.CancelledError):
            await active
        await asyncio.wait_for(reload_task, timeout=1)
        assert clients[0].stopped is True
        assert await feature.call_host_ingress("telegram-webhook", {"sequence": 2}) == {
            "client": 2
        }
    finally:
        blocker.set()
        if reload_task is not None and not reload_task.done():
            reload_task.cancel()
            try:
                await reload_task
            except asyncio.CancelledError:
                pass
        if active is not None and not active.done():
            active.cancel()
            with pytest.raises(asyncio.CancelledError):
                await active
        await feature.shutdown()


@pytest.mark.asyncio
async def test_terminal_stop_failure_retains_exact_client_for_bounded_retry(
    monkeypatch, tmp_path
):
    """A failed terminal stop neither loses its client handle nor drains forever."""

    ingress_started = asyncio.Event()
    terminate_ingress = asyncio.Event()

    class TerminalStopFailure(BaseException):
        pass

    class FirstStopFailsClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            ingress_started.set()
            await terminate_ingress.wait()
            return {"accepted": True}

        async def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise TerminalStopFailure("first terminal stop failed")
            self.stopped = True
            terminate_ingress.set()

    client = FirstStopFailsClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    ingress = None
    supervisor = feature._supervision_task
    try:
        assert supervisor is not None
        ingress = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"update_id": 7})
        )
        await asyncio.wait_for(ingress_started.wait(), timeout=1)

        with pytest.raises(
            RuntimeError, match="isolated feature terminal retirement is incomplete"
        ) as raised:
            await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert type(raised.value) is RuntimeError
        assert "first terminal stop failed" not in str(raised.value)
        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None

        assert feature._client is None
        assert feature._terminal_retirement_clients == [client]
        assert feature._traffic_gate.sealed is True
        assert feature._traffic_gate._active == 1
        assert ingress.done() is False

        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert await ingress == {"accepted": True}
        assert client.stop_calls == 2
        assert client.stopped is True
        assert feature._terminal_retirement_clients == []
        assert feature._traffic_gate._active == 0
        assert feature._client is None
        assert feature._supervision_task is None
        assert feature._terminal_cleanup_task is None
        assert supervisor.done() is True
        assert supervisor.cancelled() is True
    finally:
        terminate_ingress.set()
        if ingress is not None and not ingress.done():
            ingress.cancel()
            with pytest.raises(asyncio.CancelledError):
                await ingress
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_best_effort_terminal_cleanup_cannot_make_concurrent_shutdown_succeed(
    monkeypatch, tmp_path
):
    """A cached best-effort stop never lends its policy to explicit shutdown."""

    ingress_started = asyncio.Event()
    stop_started = asyncio.Event()
    release_first_stop = asyncio.Event()
    terminate_ingress = asyncio.Event()

    class HostileTerminalStop(BaseException):
        pass

    class ConcurrentStopClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.stops_in_flight = 0
            self.maximum_stops_in_flight = 0

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            ingress_started.set()
            await terminate_ingress.wait()
            return {"accepted": True}

        async def stop(self):
            self.stop_calls += 1
            self.stops_in_flight += 1
            self.maximum_stops_in_flight = max(
                self.maximum_stops_in_flight, self.stops_in_flight
            )
            try:
                if self.stop_calls == 1:
                    stop_started.set()
                    await release_first_stop.wait()
                    raise HostileTerminalStop("hostile terminal stop detail")
                self.stopped = True
                terminate_ingress.set()
            finally:
                self.stops_in_flight -= 1

    client = ConcurrentStopClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    ingress = quarantine = shutdown_task = None
    supervisor = feature._supervision_task
    try:
        assert supervisor is not None
        ingress = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"update_id": 7})
        )
        await asyncio.wait_for(ingress_started.wait(), timeout=1)

        # Quarantine owns the cached shared task and is entitled to return to
        # its originating lifecycle failure after its neutral attempt settles.
        quarantine = asyncio.create_task(feature._quarantine_unreconciled_client())
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        assert feature._terminal_cleanup_task is not None
        assert not quarantine.done()

        # An explicit shutdown joins the in-flight neutral work. It must make
        # its own success decision once the hostile stop has settled.
        shutdown_task = asyncio.create_task(feature.shutdown())
        await asyncio.sleep(0)
        assert not shutdown_task.done()
        assert client.maximum_stops_in_flight == 1

        release_first_stop.set()
        await asyncio.wait_for(quarantine, timeout=1)
        with pytest.raises(
            RuntimeError, match="isolated feature terminal retirement is incomplete"
        ) as raised:
            await asyncio.wait_for(shutdown_task, timeout=1)
        assert "hostile terminal stop detail" not in str(raised.value)
        assert raised.value.__cause__ is None

        # The exact facade is still retained and the old ingress is counted,
        # but the sealed proxy makes it unreachable to all new host traffic.
        assert client.stop_calls == 1
        assert client.maximum_stops_in_flight == 1
        assert feature._terminal_retirement_clients == [client]
        assert feature._client is None
        assert feature._traffic_gate.sealed is True
        assert feature._traffic_gate._active == 1
        assert not ingress.done()
        with pytest.raises(HostIngressError, match="host ingress is unavailable"):
            await feature.call_host_ingress("telegram-webhook", {"update_id": 8})

        # A later explicit caller creates a fresh neutral attempt. It retries
        # the exact retained facade, then may truthfully drain and finish.
        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert await ingress == {"accepted": True}
        assert client.stop_calls == 2
        assert client.maximum_stops_in_flight == 1
        assert client.stopped is True
        assert feature._terminal_retirement_clients == []
        assert feature._traffic_gate._active == 0
        assert feature._client is None
        assert feature._supervision_task is None
        assert feature._terminal_cleanup_task is None
        assert supervisor.done() is True
        assert supervisor.cancelled() is True
    finally:
        release_first_stop.set()
        terminate_ingress.set()
        for task in (shutdown_task, quarantine, ingress):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_initialize_reports_generic_failure_for_retained_terminal_client(
    monkeypatch, tmp_path
):
    """Explicit initialize cannot start beside an unsuccessfully stopped child."""

    stop_started = asyncio.Event()
    release_first_stop = asyncio.Event()

    class HostileTerminalStop(BaseException):
        pass

    class NeverStopsClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                stop_started.set()
                await release_first_stop.wait()
            raise HostileTerminalStop("terminal stop must remain private")

    client = NeverStopsClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    quarantine = initialize_task = None
    try:
        quarantine = asyncio.create_task(feature._quarantine_unreconciled_client())
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        initialize_task = asyncio.create_task(feature.initialize())
        await asyncio.sleep(0)
        assert not initialize_task.done()

        release_first_stop.set()
        await asyncio.wait_for(quarantine, timeout=1)
        assert client.stop_calls == 1
        assert feature._terminal_retirement_clients == [client]

        with pytest.raises(
            RuntimeError, match="isolated feature terminal retirement is incomplete"
        ) as raised:
            await asyncio.wait_for(initialize_task, timeout=1)
        assert type(raised.value) is RuntimeError
        assert "terminal stop must remain private" not in str(raised.value)
        assert raised.value.__cause__ is None
        assert client.stop_calls == 1
        assert feature._terminal_retirement_clients == [client]
        assert feature._client is None
        assert feature._traffic_gate.sealed is True
    finally:
        release_first_stop.set()
        for task in (initialize_task, quarantine):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        with pytest.raises(
            RuntimeError, match="isolated feature terminal retirement is incomplete"
        ):
            await feature.shutdown()


@pytest.mark.asyncio
async def test_shutdown_cancellation_preserves_caller_cancel_after_hostile_stop(
    monkeypatch, tmp_path
):
    """A caller cancellation wins over an incomplete neutral stop attempt."""

    stop_started = asyncio.Event()
    release_first_stop = asyncio.Event()

    class HostileTerminalStop(BaseException):
        pass

    class CancelledStopClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                stop_started.set()
                await release_first_stop.wait()
                raise HostileTerminalStop("do not replace caller cancellation")
            self.stopped = True

    client = CancelledStopClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    shutdown_task = None
    try:
        shutdown_task = asyncio.create_task(feature.shutdown())
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        shutdown_task.cancel("explicit shutdown cancellation")
        await asyncio.sleep(0)
        assert not shutdown_task.done()

        release_first_stop.set()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(shutdown_task, timeout=1)
        assert cancelled.value.args == ("explicit shutdown cancellation",)
        assert client.stop_calls == 1
        assert feature._terminal_retirement_clients == [client]
        assert feature._traffic_gate.sealed is True

        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 2
        assert client.stopped is True
        assert feature._terminal_retirement_clients == []
        assert feature._terminal_cleanup_task is None
    finally:
        release_first_stop.set()
        if shutdown_task is not None and not shutdown_task.done():
            shutdown_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await shutdown_task
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_owned_terminal_cleanup_and_shutdown_serialize_exact_client_stop(
    monkeypatch, tmp_path
):
    """A reload owner and shared shutdown never overlap one facade stop."""

    first_stop_started = asyncio.Event()
    release_first_stop = asyncio.Event()

    class HostileTerminalStop(BaseException):
        pass

    class OwnedRaceClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.stops_in_flight = 0
            self.maximum_stops_in_flight = 0

        async def stop(self):
            self.stop_calls += 1
            self.stops_in_flight += 1
            self.maximum_stops_in_flight = max(
                self.maximum_stops_in_flight, self.stops_in_flight
            )
            try:
                if self.stop_calls == 1:
                    first_stop_started.set()
                    await release_first_stop.wait()
                if self.stop_calls < 3:
                    raise HostileTerminalStop("owned cleanup stop failed")
                self.stopped = True
            finally:
                self.stops_in_flight -= 1

    client = OwnedRaceClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    owned = shutdown_task = None
    lock_held = False
    try:
        await feature._reload_lock.acquire()
        lock_held = True
        owned = asyncio.create_task(
            feature._quarantine_unreconciled_client(lifecycle_lock_held=True)
        )
        await asyncio.wait_for(first_stop_started.wait(), timeout=1)

        # The shared shutdown begins while the owned cleanup still awaits its
        # exact client's first stop. It must wait on retirement ownership,
        # rather than start a second facade stop beside it.
        shutdown_task = asyncio.create_task(feature.shutdown())
        await asyncio.sleep(0)
        assert client.maximum_stops_in_flight == 1
        assert not shutdown_task.done()

        release_first_stop.set()
        await asyncio.wait_for(owned, timeout=1)
        with pytest.raises(
            RuntimeError, match="isolated feature terminal retirement is incomplete"
        ):
            await asyncio.wait_for(shutdown_task, timeout=1)
        assert client.stop_calls == 2
        assert client.maximum_stops_in_flight == 1
        assert feature._terminal_retirement_clients == [client]
        assert feature._traffic_gate.sealed is True

        feature._reload_lock.release()
        lock_held = False
        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 3
        assert client.maximum_stops_in_flight == 1
        assert client.stopped is True
        assert feature._terminal_retirement_clients == []
        assert feature._terminal_cleanup_task is None
    finally:
        release_first_stop.set()
        if lock_held:
            feature._reload_lock.release()
        for task in (shutdown_task, owned):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_waits_for_reload_then_uses_replacement_child(
    monkeypatch, tmp_path
):
    """Reload drains a running ingress call before retiring its subprocess."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()
    clients = []

    class BlockingClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if payload == {"sequence": 1}:
                active_started.set()
                await release_active.wait()
            return {"client": len(clients), "sequence": payload["sequence"]}

    def client_factory(**kwargs):
        client = BlockingClient(
            ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",)),
            **kwargs,
        )
        clients.append(client)
        return client

    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, client_factory
    )
    active = reload_task = queued = None
    try:
        active = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 1})
        )
        await asyncio.wait_for(active_started.wait(), timeout=1)
        reload_task = asyncio.create_task(feature.reload())
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)
        assert feature._traffic_gate.closed is True

        queued = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 2})
        )
        await asyncio.sleep(0)
        assert not queued.done()
        assert clients[0].ingress_calls == [("telegram-webhook", {"sequence": 1})]

        release_active.set()
        assert await active == {"client": 1, "sequence": 1}
        await reload_task
        assert clients[0].stopped is True
        assert await queued == {"client": 2, "sequence": 2}
        assert clients[1].ingress_calls == [("telegram-webhook", {"sequence": 2})]
    finally:
        release_active.set()
        for task in (queued, reload_task, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_waits_for_config_transition_before_delivery(
    monkeypatch, tmp_path
):
    """A config transition drains ingress before its lifecycle hook starts."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class TransitionClient(_HostIngressClient):
        supports_config_transition = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prepared = []

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if payload == {"sequence": 1}:
                active_started.set()
                await release_active.wait()
            return {"sequence": payload["sequence"]}

        async def prepare_config_transition(self, config):
            self.prepared.append(dict(config))
            return ConfigTransitionResult.applied()

    client = TransitionClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    active = update = queued = None
    try:
        active = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 1})
        )
        await asyncio.wait_for(active_started.wait(), timeout=1)
        update = asyncio.create_task(feature.set_config({"enabled": False}))
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)
        assert feature._traffic_gate.closed is True
        assert client.prepared == []

        queued = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 2})
        )
        await asyncio.sleep(0)
        assert not queued.done()

        release_active.set()
        assert await active == {"sequence": 1}
        await update
        assert client.prepared == [{"enabled": False}]
        assert await queued == {"sequence": 2}
        assert client.ingress_calls == [
            ("telegram-webhook", {"sequence": 1}),
            ("telegram-webhook", {"sequence": 2}),
        ]
    finally:
        release_active.set()
        for task in (queued, update, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_idle_wake_terminal_revocation_is_not_retryable(
    monkeypatch, tmp_path
):
    client = _HostIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )

    feature._client = None
    feature._idle_retired = True
    wake_entered = asyncio.Event()
    original_wake = feature._wake_idle_runtime

    async def observe_real_wake():
        wake_entered.set()
        await original_wake()

    monkeypatch.setattr(feature, "_wake_idle_runtime", observe_real_wake)
    ingress = None
    try:
        await feature._reload_lock.acquire()
        ingress = asyncio.create_task(
            feature.call_host_ingress(
                "telegram-webhook",
                {"update_id": 8},
            )
        )
        await asyncio.wait_for(wake_entered.wait(), timeout=1)
        assert ingress.done() is False
        feature._latch_terminal_lifecycle()
        feature._reload_lock.release()
        with pytest.raises(HostIngressError, match="host ingress is unavailable"):
            await ingress
        assert client.ingress_calls == []
    finally:
        if feature._reload_lock.locked():
            feature._reload_lock.release()
        if ingress is not None and not ingress.done():
            ingress.cancel()
            await asyncio.gather(ingress, return_exceptions=True)
        feature._client = client
        feature._idle_retired = False
        await feature.shutdown()


@pytest.mark.asyncio
async def test_host_ingress_shutdown_drains_active_call_and_rejects_new_work(
    monkeypatch, tmp_path
):
    """Shutdown cannot stop a child while it serves an admitted ingress RPC."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class BlockingClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            active_started.set()
            await release_active.wait()
            return {"accepted": True}

    client = BlockingClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    active = shutdown_task = None
    try:
        active = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 1})
        )
        await asyncio.wait_for(active_started.wait(), timeout=1)
        shutdown_task = asyncio.create_task(feature.shutdown())
        for _ in range(100):
            if feature._traffic_gate.sealed:
                break
            await asyncio.sleep(0)
        assert feature._traffic_gate.sealed is True

        with pytest.raises(HostIngressError, match="host ingress is unavailable"):
            await feature.call_host_ingress("telegram-webhook", {"sequence": 2})
        assert client.ingress_calls == [("telegram-webhook", {"sequence": 1})]
        # Terminal cleanup stops the selected child before it drains the
        # admitted ingress. A cooperative double still needs the explicit
        # release below, but a production wrapper uses this stop to terminate
        # a permanently wedged subprocess.
        for _ in range(100):
            if client.stopped:
                break
            await asyncio.sleep(0)
        assert client.stopped is True
        assert not shutdown_task.done()

        release_active.set()
        assert await active == {"accepted": True}
        await shutdown_task
        assert client.stopped is True
    finally:
        release_active.set()
        for task in (shutdown_task, active):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_terminal_drain_timeout_keeps_dishonest_stop_sealed_and_owned(
    monkeypatch, tmp_path
):
    """A stop that lies about a wedged RPC cannot hang or report success."""

    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class DishonestStopClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            active_started.set()
            await release_active.wait()
            return {"accepted": True}

        async def stop(self):
            # Deliberately claim success without settling the admitted RPC.
            self.stop_calls += 1
            self.stopped = True

    client = DishonestStopClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    monkeypatch.setattr(isolated_runtime, "_TERMINAL_TRAFFIC_DRAIN_TIMEOUT", 0.02)
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    ingress = None
    try:
        ingress = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 1})
        )
        await asyncio.wait_for(active_started.wait(), timeout=1)

        with pytest.raises(RuntimeError, match="terminal retirement is incomplete"):
            await asyncio.wait_for(feature.shutdown(), timeout=1)

        assert client.stop_calls == 1
        assert feature._client is None
        assert feature._traffic_gate.sealed is True
        assert feature._traffic_gate._active == 1
        assert feature._terminal_traffic_drain_task is not None
        assert ingress.done() is False

        release_active.set()
        assert await ingress == {"accepted": True}
        for _ in range(100):
            if feature._terminal_traffic_drain_task is None:
                break
            await asyncio.sleep(0)
        assert feature._terminal_traffic_drain_task is None

        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert feature._traffic_gate._active == 0
        assert feature._traffic_gate.sealed is True
    finally:
        release_active.set()
        if ingress is not None and not ingress.done():
            ingress.cancel()
            with pytest.raises(asyncio.CancelledError):
                await ingress
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_supervisor_drain_timeout_seals_before_recovery_finalizer_reopens(
    monkeypatch, tmp_path
):
    """A stopped facade cannot admit a queued ingress during drain-timeout unwind."""

    unhealthy = asyncio.Event()
    active_started = asyncio.Event()
    release_active = asyncio.Event()

    class DishonestStopClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def health(self):
            await unhealthy.wait()
            return False

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            if payload == {"sequence": 1}:
                active_started.set()
                await release_active.wait()
            return {"sequence": payload["sequence"]}

        async def stop(self):
            # Claim success without interrupting the already admitted request.
            self.stop_calls += 1
            self.stopped = True

    client = DishonestStopClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    monkeypatch.setattr(isolated_runtime, "_TERMINAL_TRAFFIC_DRAIN_TIMEOUT", 0.02)
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    active = queued = None
    supervisor = feature._supervision_task
    try:
        assert supervisor is not None
        active = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 1})
        )
        await asyncio.wait_for(active_started.wait(), timeout=1)
        unhealthy.set()
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await real_sleep(0)
        assert feature._traffic_gate.closed is True

        queued = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"sequence": 2})
        )
        with pytest.raises(isolated_runtime._TerminalTrafficDrainTimedOut):
            await asyncio.wait_for(supervisor, timeout=1)
        with pytest.raises(HostIngressError, match="host ingress is unavailable"):
            await asyncio.wait_for(queued, timeout=1)

        assert client.stop_calls == 1
        assert client.ingress_calls == [("telegram-webhook", {"sequence": 1})]
        assert feature._traffic_gate.sealed is True
        assert feature._traffic_gate.closed is True
    finally:
        release_active.set()
        for task in (queued, active):
            if task is not None and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        for _ in range(100):
            if feature._terminal_traffic_drain_task is None:
                break
            await real_sleep(0)
        if feature._terminal_traffic_drain_task is None:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_shutdown_revokes_initializer_waiting_for_reload_lock(monkeypatch, tmp_path):
    """A shutdown queued before inner initialize owns the terminal generation."""

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    started_clients = []

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        started_clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    initialize_task = shutdown_task = None
    reload_lock_held = False
    try:
        await feature._reload_lock.acquire()
        reload_lock_held = True
        initialize_task = asyncio.create_task(feature.initialize())
        await asyncio.sleep(0)

        # The outer initialize has created its inner task, but that task has
        # not yet acquired lifecycle ownership.  Shutdown now creates its
        # cleanup transaction before the lock is released.
        shutdown_task = asyncio.create_task(feature.shutdown())
        for _ in range(100):
            if feature._terminal_lifecycle_latched and feature._terminal_cleanup_task:
                break
            await asyncio.sleep(0)
        assert feature._terminal_lifecycle_latched is True
        assert feature._terminal_cleanup_task is not None

        feature._reload_lock.release()
        reload_lock_held = False
        await asyncio.wait_for(shutdown_task, timeout=1)
        with pytest.raises(
            RuntimeError, match="terminal lifecycle changed during initialize"
        ):
            await asyncio.wait_for(initialize_task, timeout=1)

        assert started_clients == []
        assert feature._client is None
        assert feature._supervision_task is None
        assert feature._traffic_gate.sealed is True
        assert feature._traffic_gate.closed is True
    finally:
        if reload_lock_held:
            feature._reload_lock.release()
        for task in (initialize_task, shutdown_task):
            if task is not None and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_shutdown_during_initialize_precleanup_revokes_enable_permit(
    monkeypatch, tmp_path
):
    """A newer shutdown cannot be absorbed while re-enable retires old state."""

    stop_started = asyncio.Event()
    release_stop = asyncio.Event()
    started_clients = []

    class SlowRetirementClient(FakeIsolatedClient):
        async def stop(self):
            stop_started.set()
            await release_stop.wait()
            self.stopped = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")

    def client_factory(**kwargs):
        client = FakeIsolatedClient(**kwargs)
        started_clients.append(client)
        return client

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=client_factory)
    retired_client = SlowRetirementClient()
    feature._terminal_lifecycle_latched = True
    feature._terminal_retirement_clients = [retired_client]
    initialize_task = shutdown_task = None
    try:
        initialize_task = asyncio.create_task(feature.initialize())
        await asyncio.wait_for(stop_started.wait(), timeout=1)

        # This terminal request arrives after initialize has begun its old
        # cleanup but before it may acquire the fresh-start ownership lock.
        shutdown_task = asyncio.create_task(feature.shutdown())
        await asyncio.sleep(0)
        release_stop.set()

        with pytest.raises(
            RuntimeError, match="terminal lifecycle changed during initialize"
        ):
            await asyncio.wait_for(initialize_task, timeout=1)
        await asyncio.wait_for(shutdown_task, timeout=1)

        assert started_clients == []
        assert feature._client is None
        assert feature._traffic_gate.sealed is True
    finally:
        release_stop.set()
        for task in (initialize_task, shutdown_task):
            if task is not None and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_supervisor_cancellation_seals_waiting_ingress_before_terminal_finally(
    monkeypatch, tmp_path
):
    """A waiter cannot slip through recovery's cancellation-finally boundary."""

    close_admission_entered = asyncio.Event()

    class UnhealthyIngressClient(_HostIngressClient):
        async def health(self):
            return False

    client = UnhealthyIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    original_close_admission = feature._close_traffic_gate_admission

    async def block_after_close_admission():
        await original_close_admission()
        close_admission_entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    monkeypatch.setattr(feature, "_close_traffic_gate_admission", block_after_close_admission)
    supervisor = queued = None
    try:
        supervisor = asyncio.create_task(feature._supervise())
        feature._supervision_task = supervisor
        await asyncio.wait_for(close_admission_entered.wait(), timeout=1)
        assert feature._traffic_gate.closed is True

        queued = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"update_id": 7})
        )
        await asyncio.sleep(0)
        assert queued.done() is False

        supervisor.cancel("close-admission cancellation")
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(supervisor, timeout=1)
        assert cancelled.value.args == ("close-admission cancellation",)
        with pytest.raises(HostIngressError, match="host ingress is unavailable"):
            await asyncio.wait_for(queued, timeout=1)

        assert client.ingress_calls == []
        assert client.stopped is True
        assert feature._client is None
        assert feature._traffic_gate.sealed is True
    finally:
        for task in (queued, supervisor):
            if task is not None and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_fenced_recovery_replays_first_of_repeated_cancellations(
    monkeypatch, tmp_path
):
    """The shared recovery drain retains the caller's first cancellation args."""

    recovery_started = asyncio.Event()
    release_recovery = asyncio.Event()
    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)

    async def controlled_recovery(_transition):
        recovery_started.set()
        await release_recovery.wait()

    monkeypatch.setattr(
        feature,
        "_recover_fenced_transition_uninterrupted",
        controlled_recovery,
    )
    recovery = None
    try:
        recovery = asyncio.create_task(
            feature._recover_fenced_transition(object(), RuntimeError("original"))
        )
        await asyncio.wait_for(recovery_started.wait(), timeout=1)
        recovery.cancel("first recovery cancellation")
        await asyncio.sleep(0)
        recovery.cancel("second recovery cancellation")
        release_recovery.set()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(recovery, timeout=1)
        assert cancelled.value.args == ("first recovery cancellation",)
    finally:
        release_recovery.set()
        if recovery is not None and not recovery.done():
            recovery.cancel()
            with pytest.raises(asyncio.CancelledError):
                await recovery


@pytest.mark.asyncio
async def test_supervisor_does_not_stop_terminal_client_after_retirement_lock_wait(
    monkeypatch, tmp_path
):
    """Shutdown ownership won while a health restart waited for the stop lock."""

    class UnhealthyClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.start_calls = 0

        async def health(self):
            return False

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

        async def start(self):
            self.start_calls += 1
            self.started = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = UnhealthyClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    supervisor = shutdown_task = None
    retirement_lock_held = False
    try:
        await feature._terminal_retirement_lock.acquire()
        retirement_lock_held = True
        supervisor = asyncio.create_task(feature._supervise())
        for _ in range(100):
            if feature._reload_lock.locked() and feature._traffic_gate.closed:
                break
            await real_sleep(0)
        assert feature._reload_lock.locked()
        assert feature._traffic_gate.closed is True

        # Keep this manually-driven supervisor alive long enough to acquire
        # the stop lock after shutdown has latched and unpublished the client.
        feature._supervision_task = None
        shutdown_task = asyncio.create_task(feature.shutdown())
        for _ in range(100):
            if feature._stopping and feature._client is None:
                break
            await real_sleep(0)
        assert feature._stopping is True
        assert feature._terminal_lifecycle_latched is True
        assert feature._client is None

        feature._terminal_retirement_lock.release()
        retirement_lock_held = False
        await asyncio.wait_for(supervisor, timeout=1)
        await asyncio.wait_for(shutdown_task, timeout=1)

        assert client.stop_calls == 1
        assert client.start_calls == 0
        assert feature._terminal_retirement_clients == []
        assert feature._traffic_gate.sealed is True
    finally:
        if retirement_lock_held:
            feature._terminal_retirement_lock.release()
        for task in (supervisor, shutdown_task):
            if task is not None and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_supervisor_does_not_start_client_after_terminal_backoff_race(
    monkeypatch, tmp_path
):
    """A terminal latch during supervisor backoff prevents the stale restart."""

    backoff_started = asyncio.Event()
    release_backoff = asyncio.Event()

    class UnhealthyClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.start_calls = 0

        async def health(self):
            return False

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

        async def start(self):
            self.start_calls += 1
            self.started = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = UnhealthyClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    real_sleep = asyncio.sleep
    sleep_calls = 0

    async def controlled_sleep(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 2:
            backoff_started.set()
            await release_backoff.wait()
            return
        await real_sleep(0)

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", controlled_sleep)
    supervisor = shutdown_task = None
    try:
        supervisor = asyncio.create_task(feature._supervise())
        await asyncio.wait_for(backoff_started.wait(), timeout=1)
        assert client.stop_calls == 1

        # Shutdown can retire the stopped facade while the supervisor still
        # owns the reload lock.  Do not cancel the manually-driven task: it
        # must reach the immediate pre-start ownership revalidation itself.
        feature._supervision_task = None
        shutdown_task = asyncio.create_task(feature.shutdown())
        for _ in range(100):
            if feature._stopping and feature._client is None:
                break
            await real_sleep(0)
        assert feature._stopping is True
        assert feature._client is None

        release_backoff.set()
        await asyncio.wait_for(supervisor, timeout=1)
        await asyncio.wait_for(shutdown_task, timeout=1)

        # The supervisor had already stopped this exact facade before the
        # terminal transaction unpublished it during backoff.
        assert client.stop_calls == 1
        assert client.start_calls == 0
        assert feature._terminal_retirement_clients == []
        assert feature._traffic_gate.sealed is True
    finally:
        release_backoff.set()
        for task in (supervisor, shutdown_task):
            if task is not None and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_supervisor_cancellation_replays_original_args_after_exact_stop(
    monkeypatch, tmp_path
):
    """Cancellation after an owned exact stop is replayed without a second stop."""

    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    class SlowStoppingClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.start_calls = 0

        async def health(self):
            return False

        async def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                stop_started.set()
                await release_stop.wait()
            self.stopped = True

        async def start(self):
            self.start_calls += 1
            self.started = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = SlowStoppingClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    supervisor = None
    try:
        supervisor = asyncio.create_task(feature._supervise())
        feature._supervision_task = supervisor
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        supervisor.cancel("agent shutdown cancellation")
        await asyncio.sleep(0)
        assert supervisor.done() is False
        release_stop.set()

        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(supervisor, timeout=1)
        assert cancelled.value.args == ("agent shutdown cancellation",)
        assert client.stop_calls == 1
        assert client.start_calls == 0
        assert feature._client is None
        assert feature._terminal_retirement_clients == []
        assert feature._terminal_stop_completed_client_markers == []
        assert feature._traffic_gate.sealed is True
    finally:
        release_stop.set()
        if supervisor is not None and not supervisor.done():
            supervisor.cancel()
            with pytest.raises(asyncio.CancelledError):
                await supervisor
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_supervisor_fences_nonreturning_facade_stop_without_terminal_hang(
    monkeypatch, tmp_path
):
    """A timed-out legacy stop is sealed and retained, never reported as success."""

    stop_started = asyncio.Event()
    cancellation_observed = asyncio.Event()
    release_stop = asyncio.Event()

    class TimeoutSuppressingClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.start_calls = 0

        async def health(self):
            return False

        async def stop(self):
            self.stop_calls += 1
            stop_started.set()
            while not release_stop.is_set():
                try:
                    await release_stop.wait()
                except asyncio.CancelledError:
                    # A genuinely hostile facade can consume the timeout
                    # cancellation and keep waiting. The proxy must still
                    # return at its bounded fence with this task explicitly
                    # owned, rather than orphaned in the event loop.
                    cancellation_observed.set()
            self.stopped = True

        async def start(self):
            self.start_calls += 1

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = TimeoutSuppressingClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_OPERATION_TIMEOUT", 0.02)
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_CANCELLATION_GRACE", 0.02)
    supervisor = None
    try:
        supervisor = asyncio.create_task(feature._supervise())
        feature._supervision_task = supervisor
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        await asyncio.wait_for(supervisor, timeout=1)

        assert cancellation_observed.is_set()
        assert client.stop_calls == 1
        assert client.start_calls == 0
        assert feature._client is None
        assert feature._terminal_retirement_clients == [client]
        assert feature._terminal_cleanup_uncertain is True
        assert len(feature._terminal_lifecycle_tasks) == 1
        assert feature._traffic_gate.sealed is True
        assert feature._terminal_lifecycle_tasks[0].task.get_name().startswith(
            "isolated-supervisor-stop:"
        )
        with pytest.raises(HostIngressError, match="host ingress is unavailable"):
            await feature.call_host_ingress("telegram-webhook", {"update_id": 7})

        # The retained task's eventual outcome is consumed and released; it
        # never becomes an orphaned background exception. Its first timeout
        # remains uncertain, so an explicit cleanup still performs the one
        # fresh exact retirement attempt before it can report success.
        release_stop.set()
        for _ in range(100):
            if not feature._terminal_lifecycle_tasks:
                break
            await real_sleep(0)
        assert feature._terminal_lifecycle_tasks == []
        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 2
        assert client.stopped is True
        assert feature._terminal_retirement_clients == []
        assert feature._terminal_lifecycle_tasks == []
        assert feature._traffic_gate.sealed is True
    finally:
        release_stop.set()
        if supervisor is not None and not supervisor.done():
            supervisor.cancel()
            with pytest.raises(asyncio.CancelledError):
                await supervisor


@pytest.mark.asyncio
async def test_timed_out_terminal_stop_stays_owned_and_blocks_exact_retry(
    monkeypatch, tmp_path
):
    """A bounded caller never detaches or races the still-running exact stop."""

    stop_started = asyncio.Event()
    cancellation_observed = asyncio.Event()
    release_stop = asyncio.Event()

    class CancellationHostileClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.secret = "TERMINAL-STOP-SECRET-2755"

        async def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                stop_started.set()
                while not release_stop.is_set():
                    try:
                        await release_stop.wait()
                    except asyncio.CancelledError:
                        cancellation_observed.set()
                return
            self.stopped = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage = _CASStorage()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_OPERATION_TIMEOUT", 0.01)
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_CANCELLATION_GRACE", 0.01)
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    client = CancellationHostileClient()
    feature._client = client
    try:
        with pytest.raises(RuntimeError, match="terminal retirement is incomplete"):
            await asyncio.wait_for(feature.shutdown(), timeout=1)
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        assert cancellation_observed.is_set()
        assert client.stop_calls == 1
        assert feature._terminal_retirement_clients == [client]
        assert len(feature._terminal_lifecycle_tasks) == 1

        # The still-running first stop owns this exact facade. A fresh caller
        # remains fail-closed instead of starting a competing stop coroutine.
        with pytest.raises(RuntimeError, match="terminal retirement is incomplete"):
            await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 1

        release_stop.set()
        for _ in range(100):
            if not feature._terminal_lifecycle_tasks:
                break
            await asyncio.sleep(0)
        assert feature._terminal_lifecycle_tasks == []

        # The original timeout remains uncertain, so the first honest success
        # path makes one fresh, serialized exact-client retirement attempt.
        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 2
        assert client.stopped is True
        assert feature._terminal_retirement_clients == []
    finally:
        release_stop.set()
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_cancellation_resistant_health_probe_is_owned_until_terminal_retry(
    monkeypatch, tmp_path
):
    """Shutdown never succeeds or re-enables beside a stale health coroutine."""

    health_started = asyncio.Event()
    cancellation_observed = asyncio.Event()
    release_health = asyncio.Event()
    started_clients = []

    class CancellationResistantHealthClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.credential = "HEALTH-PROBE-CREDENTIAL-2755"

        async def health(self):
            health_started.set()
            while not release_health.is_set():
                try:
                    await release_health.wait()
                except asyncio.CancelledError:
                    cancellation_observed.set()
            return True

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    monkeypatch.setattr(isolated_runtime, "_HEALTH_PROBE_TIMEOUT", 0.5)
    monkeypatch.setattr(isolated_runtime, "_HEALTH_PROBE_CANCELLATION_GRACE", 0.01)
    client = CancellationResistantHealthClient()
    client_ref = weakref.ref(client)

    def factory(**kwargs):
        replacement = FakeIsolatedClient(**kwargs)
        started_clients.append(replacement)
        return replacement

    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=factory)
    feature._client = client
    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    supervisor = asyncio.create_task(feature._supervise())
    feature._supervision_task = supervisor
    try:
        await asyncio.wait_for(health_started.wait(), timeout=1)

        # Cancelling the supervisor through the direct shutdown path must not
        # turn the still-running probe into an unowned task or false success.
        with pytest.raises(RuntimeError, match="terminal retirement is incomplete"):
            await asyncio.wait_for(feature.shutdown(), timeout=1)
        await asyncio.wait_for(cancellation_observed.wait(), timeout=1)
        probe = feature._terminal_health_probe_task
        assert probe is not None and not probe.done()
        assert probe.get_name().startswith("isolated-health-probe:")
        assert client.stop_calls == 0
        assert feature._client is None
        assert feature._traffic_gate.sealed is True

        # Explicit re-enable must first settle the exact old probe; it cannot
        # construct a new facade alongside credentials retained by that task.
        with pytest.raises(RuntimeError, match="terminal retirement is incomplete"):
            await asyncio.wait_for(feature.initialize(), timeout=1)
        assert started_clients == []

        release_health.set()
        for _ in range(100):
            if feature._terminal_health_probe_task is None:
                break
            await real_sleep(0)
        assert feature._terminal_health_probe_task is None

        # Once the callback consumed the probe outcome, a fresh terminal retry
        # can retire the exact facade and a later explicit initialize is safe.
        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 1
        await asyncio.wait_for(feature.initialize(), timeout=1)
        assert len(started_clients) == 1
        assert feature._client is started_clients[0]

        await feature.shutdown()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(supervisor, timeout=1)
        del probe
        del client
        gc.collect()
        assert client_ref() is None
    finally:
        release_health.set()
        if not supervisor.done():
            supervisor.cancel()
            with pytest.raises(asyncio.CancelledError):
                await supervisor
        if not feature._stopping:
            await feature.shutdown()


def test_terminal_stop_completion_marker_is_weak_and_drops_dead_identity(tmp_path):
    """Stopped facades and their secrets are never retained by a completion mark."""

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    client = FakeIsolatedClient()
    client.secret = "STOPPED-FACADE-SECRET-2755"
    client_ref = weakref.ref(client)

    feature._mark_terminal_stop_completed(client)
    assert len(feature._terminal_stop_completed_client_markers) == 1
    del client
    gc.collect()

    assert client_ref() is None
    # A dead weak marker cannot match a later object, even if Python later
    # reuses the old object's id(). Consuming it also removes the stale entry.
    assert feature._forget_terminal_stop_completion(FakeIsolatedClient()) is False
    assert feature._terminal_stop_completed_client_markers == []


@pytest.mark.asyncio
@pytest.mark.parametrize("nonweakrefable", [False, True])
async def test_terminal_cleanup_claims_inflight_supervisor_stop_completion_once(
    tmp_path, nonweakrefable
):
    """A late supervisor callback cannot recreate a retired-cycle marker."""

    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    if nonweakrefable:

        class Client:
            __slots__ = ("stop_calls", "secret")

            def __init__(self):
                self.stop_calls = 0
                self.secret = "NONWEAK-STOP-SECRET-2755"

            async def stop(self):
                self.stop_calls += 1
                stop_started.set()
                await release_stop.wait()

    else:

        class Client:
            def __init__(self):
                self.stop_calls = 0
                self.secret = "WEAK-STOP-SECRET-2755"

            async def stop(self):
                self.stop_calls += 1
                stop_started.set()
                await release_stop.wait()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    client = Client()
    marker = feature._begin_terminal_stop_completion(client)
    assert marker is not None

    async def supervisor_stop():
        await client.stop()
        feature._mark_terminal_stop_completed(marker)

    owner = asyncio.create_task(supervisor_stop())
    try:
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        # This is the terminal-cleanup ordering: it has consumed no completed
        # marker yet, so it claims the in-flight callback and retains the
        # exact facade until that callback proves the one stop completed.
        assert feature._forget_terminal_stop_completion(
            client, terminal_retirement=True
        ) is False
        feature._retain_terminal_retirement_client(client)

        release_stop.set()
        await asyncio.wait_for(owner, timeout=1)

        assert client.stop_calls == 1
        assert feature._terminal_retirement_clients == []
        assert feature._terminal_stop_completed_client_markers == []
        # A later lifecycle reuse cannot consume the retired completion.
        assert feature._forget_terminal_stop_completion(client) is False
    finally:
        release_stop.set()
        if not owner.done():
            owner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await owner


@pytest.mark.asyncio
async def test_nonweakrefable_stop_marker_avoids_duplicate_stop_and_is_consumed(
    monkeypatch, tmp_path
):
    """A supervisor-proven stop survives the shutdown/backoff race exactly once."""

    class NonWeakrefableClient:
        __slots__ = ("stop_calls", "stopped")

        def __init__(self):
            self.stop_calls = 1  # Supervisor already completed this exact stop.
            self.stopped = True

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    client = NonWeakrefableClient()
    feature._client = client
    feature._mark_terminal_stop_completed(client)
    assert len(feature._terminal_stop_completed_client_markers) == 1

    await feature.shutdown()

    assert client.stop_calls == 1
    assert feature._terminal_stop_completed_client_markers == []
    assert feature._terminal_retirement_clients == []
    assert feature._client is None


@pytest.mark.asyncio
async def test_unhealthy_supervisor_stops_wedged_host_ingress_before_draining(
    monkeypatch, tmp_path
):
    """Health recovery reaches the wrapper stop path before gate drain."""

    permit_unhealthy_probe = asyncio.Event()
    ingress_started = asyncio.Event()
    terminate_ingress = asyncio.Event()
    restarted = asyncio.Event()

    class WedgedIngressClient(_HostIngressClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.start_calls = 0
            self.stop_calls = 0
            self.active_when_stopped = None
            self.active_when_restarted = None

        async def start(self):
            self.start_calls += 1
            if self.start_calls > 1:
                self.active_when_restarted = feature._traffic_gate._active
                restarted.set()

        async def health(self):
            await permit_unhealthy_probe.wait()
            return self.start_calls > 1

        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            ingress_started.set()
            await terminate_ingress.wait()
            return {"accepted": True}

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True
            self.active_when_stopped = feature._traffic_gate._active
            terminate_ingress.set()

    client = WedgedIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    ingress = None
    try:
        ingress = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"update_id": 7})
        )
        await asyncio.wait_for(ingress_started.wait(), timeout=1)
        permit_unhealthy_probe.set()

        await asyncio.wait_for(restarted.wait(), timeout=1)
        assert await ingress == {"accepted": True}
        assert client.stop_calls == 1
        assert client.active_when_stopped == 1
        assert client.active_when_restarted == 0
        # The owned facade operation returns before the supervisor's matching
        # gate-finally reopens admission; wait for that public boundary rather
        # than treating the client-side start callback as a host-ready signal.
        for _ in range(100):
            if feature._traffic_gate.closed is False:
                break
            await real_sleep(0)
        assert feature._traffic_gate.closed is False
        assert feature._traffic_gate.sealed is False
        assert feature._client is client
    finally:
        terminate_ingress.set()
        if ingress is not None and not ingress.done():
            ingress.cancel()
            with pytest.raises(asyncio.CancelledError):
                await ingress
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_cancelled_supervisor_uses_terminal_stop_before_draining_host_ingress(
    monkeypatch, tmp_path
):
    """Agent-side supervisor cancellation composes through terminal cleanup."""

    ingress_started = asyncio.Event()
    terminate_ingress = asyncio.Event()

    class WedgedIngressClient(_HostIngressClient):
        async def call_host_ingress(self, name, payload=None):
            self.ingress_calls.append((name, payload))
            ingress_started.set()
            await terminate_ingress.wait()
            return {"accepted": True}

        async def stop(self):
            self.stopped = True
            terminate_ingress.set()

    client = WedgedIngressClient(
        ingress_capabilities=HostIngressCapabilities(names=("telegram-webhook",))
    )
    feature, _ = await _initialized_host_ingress_proxy(
        monkeypatch, tmp_path, lambda **_: client
    )
    ingress = None
    supervisor = feature._supervision_task
    try:
        assert supervisor is not None
        ingress = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"update_id": 7})
        )
        await asyncio.wait_for(ingress_started.wait(), timeout=1)
        ingress.cancel("caller cancellation")
        await asyncio.sleep(0)
        assert ingress.done() is False
        assert feature._traffic_gate._active == 1

        supervisor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(supervisor, timeout=1)
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(ingress, timeout=1)
        assert cancelled.value.args == ("caller cancellation",)
        assert client.stopped is True
        assert feature._traffic_gate.sealed is True
        assert feature._traffic_gate._active == 0
        assert feature._client is None
        assert feature._supervision_task is None
    finally:
        terminate_ingress.set()
        if ingress is not None and not ingress.done():
            ingress.cancel()
            with pytest.raises(asyncio.CancelledError):
                await ingress
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("awaitable_kind", ("future", "task"))
async def test_owned_facade_lifecycle_accepts_completed_future_and_task(
    awaitable_kind, monkeypatch
):
    """A valid facade awaitable always runs through the named host task."""

    async def completed_task():
        return "stopped"

    loop = asyncio.get_running_loop()
    if awaitable_kind == "future":
        operation = loop.create_future()
        operation.set_result("stopped")
    else:
        operation = asyncio.create_task(completed_task(), name="facade-stop-task")
        await operation

    class FutureReturningFacade:
        def stop(self):
            return operation

    created_tasks = []
    real_create = isolated_runtime._create_host_owned_facade_task

    def capture_task(awaitable, *, name):
        task = real_create(awaitable, name=name)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(isolated_runtime, "_create_host_owned_facade_task", capture_task)
    assert await isolated_runtime._await_owned_facade_lifecycle_operation(
        FutureReturningFacade().stop(),
        name="test-facade-stop-awaitable",
        on_late_task=lambda _task: pytest.fail("completed facade must not detach"),
    ) == "stopped"
    assert [task.get_name() for task in created_tasks] == ["test-facade-stop-awaitable"]


@pytest.mark.asyncio
@pytest.mark.parametrize("awaitable_kind", ("future", "task"))
async def test_owned_facade_lifecycle_cancellation_keeps_pending_future_and_task_owned(
    awaitable_kind, monkeypatch
):
    """Caller cancellation waits for the host owner instead of detaching it."""

    release = asyncio.Event()

    async def pending_task():
        await release.wait()
        return "stopped"

    loop = asyncio.get_running_loop()
    operation = (
        loop.create_future()
        if awaitable_kind == "future"
        else asyncio.create_task(pending_task(), name="facade-pending-stop-task")
    )

    class FutureReturningFacade:
        def stop(self):
            return operation

    created_tasks = []
    real_create = isolated_runtime._create_host_owned_facade_task

    def capture_task(awaitable, *, name):
        task = real_create(awaitable, name=name)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(isolated_runtime, "_create_host_owned_facade_task", capture_task)
    owner = asyncio.create_task(
        isolated_runtime._await_owned_facade_lifecycle_operation(
            FutureReturningFacade().stop(),
            name="test-pending-facade-stop-awaitable",
            on_late_task=lambda _task: pytest.fail("pending facade must settle"),
        )
    )
    await asyncio.sleep(0)
    owner.cancel("caller cancellation")
    await asyncio.sleep(0)
    assert owner.done() is False
    if awaitable_kind == "future":
        operation.set_result("stopped")
    else:
        release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await owner
    assert cancelled.value.args == ("caller cancellation",)
    assert created_tasks[0].get_name() == "test-pending-facade-stop-awaitable"


@pytest.mark.asyncio
async def test_owned_facade_lifecycle_timeout_retains_task_returned_by_facade(monkeypatch):
    """A cancellation-resistant Task return remains attached to its host owner.

    The two short budgets below are patched because this asserts THAT the
    timeout fires. Everything the timeout causes is awaited as an event, not
    assumed to have happened inside those windows (#3077).

    ``started`` is the load-sensitive part: a task cancelled before its first
    step is cancelled outright, its body never runs, and then nothing observes
    the cancellation and nothing is handed over — the runtime is right to
    record no late task for a settled one, and the test was asserting a
    scheduling race rather than the behaviour.
    """

    started = asyncio.Event()
    cancellation_observed = asyncio.Event()
    release = asyncio.Event()

    async def pending_task():
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellation_observed.set()

    operation = asyncio.create_task(pending_task(), name="facade-hostile-stop-task")
    # RUNNING before the settlement budget starts, so the cancellation lands in
    # the body rather than on a task that never began.
    await asyncio.wait_for(started.wait(), timeout=5)

    class TaskReturningFacade:
        def stop(self):
            return operation

    late_tasks = []
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_OPERATION_TIMEOUT", 0.01)
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_CANCELLATION_GRACE", 0.01)
    try:
        with pytest.raises(isolated_runtime._FacadeLifecycleOperationTimedOut):
            await isolated_runtime._await_owned_facade_lifecycle_operation(
                TaskReturningFacade().stop(),
                name="test-hostile-facade-stop-task",
                on_late_task=late_tasks.append,
            )
        assert len(late_tasks) == 1
        assert late_tasks[0].get_name() == "test-hostile-facade-stop-task"
        # The cancellation was DELIVERED before the timeout raised; the task
        # observing it is its own next scheduling, which is not the grace
        # window's to contain. The bound here only fails a genuine hang.
        await asyncio.wait_for(cancellation_observed.wait(), timeout=5)
    finally:
        release.set()
        await asyncio.wait_for(late_tasks[0] if late_tasks else operation, timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("awaitable_kind", ("future", "task"))
async def test_terminal_shutdown_owns_future_and_task_returned_by_facade_once(
    awaitable_kind, monkeypatch, tmp_path
):
    """Terminal retirement never retries a successfully owned facade stop."""

    async def completed_task():
        return None

    loop = asyncio.get_running_loop()
    if awaitable_kind == "future":
        operation = loop.create_future()
        operation.set_result(None)
    else:
        operation = asyncio.create_task(completed_task(), name="terminal-facade-stop-task")
        await operation

    class FutureReturningStopClient(FakeIsolatedClient):
        def __init__(self):
            super().__init__()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            return operation

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = FutureReturningStopClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    await feature.shutdown()
    await feature.shutdown()
    assert client.stop_calls == 1
    assert feature._terminal_retirement_clients == []
    assert feature._terminal_lifecycle_tasks == []


@pytest.mark.asyncio
@pytest.mark.parametrize("awaitable_kind", ("future", "task"))
async def test_owned_health_probe_accepts_completed_future_and_task(awaitable_kind):
    """Completed facade probes are healthy rather than TypeError failures."""

    async def completed_task():
        return {"healthy": True}

    loop = asyncio.get_running_loop()
    if awaitable_kind == "future":
        operation = loop.create_future()
        operation.set_result({"healthy": True})
    else:
        operation = asyncio.create_task(completed_task(), name="facade-health-task")
        await operation

    class FutureReturningFacade:
        def health(self):
            return operation

    started_tasks = []
    assert await isolated_runtime._await_owned_health_probe(
        FutureReturningFacade().health(),
        name="test-facade-health-awaitable",
        on_started=started_tasks.append,
        on_late_task=lambda _task: pytest.fail("completed health must not detach"),
    ) == {"healthy": True}
    assert [task.get_name() for task in started_tasks] == ["test-facade-health-awaitable"]


@pytest.mark.asyncio
@pytest.mark.parametrize("awaitable_kind", ("future", "task"))
async def test_owned_health_probe_accepts_pending_future_and_task(awaitable_kind):
    """Pending valid facade probes settle normally through host ownership."""

    release = asyncio.Event()

    async def pending_task():
        await release.wait()
        return {"healthy": True}

    loop = asyncio.get_running_loop()
    operation = (
        loop.create_future()
        if awaitable_kind == "future"
        else asyncio.create_task(pending_task(), name="facade-pending-health-task")
    )

    class FutureReturningFacade:
        def health(self):
            return operation

    started_tasks = []
    owner = asyncio.create_task(
        isolated_runtime._await_owned_health_probe(
            FutureReturningFacade().health(),
            name="test-pending-facade-health-awaitable",
            on_started=started_tasks.append,
            on_late_task=lambda _task: pytest.fail("pending health must settle"),
        )
    )
    await asyncio.sleep(0)
    if awaitable_kind == "future":
        operation.set_result({"healthy": True})
    else:
        release.set()
    assert await owner == {"healthy": True}
    assert started_tasks[0].get_name() == "test-pending-facade-health-awaitable"


@pytest.mark.asyncio
async def test_host_owned_facade_callbacks_deliver_each_registration_once():
    """Settlement has one callback path across registration timing and reentry."""

    loop = asyncio.get_running_loop()
    source = loop.create_future()
    operation = isolated_runtime._create_host_owned_facade_task(
        source, name="callback-once"
    )
    delivered = []

    def duplicate(completed):
        delivered.append(("duplicate", completed))

    def reentrant(completed):
        delivered.append(("reentrant", completed))
        completed.add_done_callback(lambda settled: delivered.append(("nested", settled)))

    # Multiple registrations of the same callback retain asyncio's one-call
    # per registration contract; none may be redelivered by _notify_settled.
    operation.add_done_callback(duplicate)
    operation.add_done_callback(duplicate)
    operation.add_done_callback(reentrant)
    source.set_result(None)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert [name for name, _ in delivered] == [
        "duplicate",
        "duplicate",
        "reentrant",
        "nested",
    ]
    assert all(completed is operation for _, completed in delivered)
    assert operation._done_callbacks == []

    # Registration after notification is queued once and not retained.
    operation.add_done_callback(lambda completed: delivered.append(("post", completed)))
    await asyncio.sleep(0)
    assert [name for name, _ in delivered].count("post") == 1
    assert operation._done_callbacks == []


@pytest.mark.asyncio
async def test_host_owned_facade_pre_settled_callback_is_not_delivered_twice():
    """A source settled before facade construction still has one callback path."""

    source = asyncio.get_running_loop().create_future()
    source.set_result(None)
    operation = isolated_runtime._create_host_owned_facade_task(
        source, name="pre-settled-callback"
    )
    delivered = []
    operation.add_done_callback(delivered.append)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert delivered == [operation]
    assert operation._done_callbacks == []


@pytest.mark.asyncio
async def test_foreign_facade_stopped_or_closed_loop_remains_durably_fenced(
    monkeypatch,
):
    """No cross-thread observer/cancel is queued onto stopped or closed loops."""

    stopped_loop = asyncio.new_event_loop()
    stopped_source = stopped_loop.create_future()
    stopped_dispatches = []
    monkeypatch.setattr(
        stopped_loop,
        "call_soon_threadsafe",
        lambda *args: stopped_dispatches.append(args),
    )
    stopped = isolated_runtime._create_host_owned_facade_task(
        stopped_source, name="stopped-foreign-facade"
    )
    stopped.cancel()
    assert stopped.done() is False
    assert stopped_dispatches == []

    # Even a terminal Future stays fenced until its owner loop consumes it.
    # Future subclasses may require that ``result()`` runs in that loop.
    completed_source = stopped_loop.create_future()
    completed_source.set_result(None)
    completed = isolated_runtime._create_host_owned_facade_task(
        completed_source, name="completed-stopped-foreign-facade"
    )
    completed.cancel()
    assert completed.done() is False
    assert completed.foreign_settlement_disposition is None
    assert completed._source is completed_source
    assert stopped_dispatches == []

    closed_loop = asyncio.new_event_loop()
    closed_source = closed_loop.create_future()
    closed_loop.close()
    closed = isolated_runtime._create_host_owned_facade_task(
        closed_source, name="closed-foreign-facade"
    )
    closed.cancel()
    assert closed.done() is False
    stopped_loop.close()


@pytest.mark.asyncio
async def test_precompleted_foreign_future_is_consumed_only_by_owner_after_resume():
    """A stopped pre-settled source fences, then settles on its owner thread."""

    foreign_loop = asyncio.new_event_loop()
    state = {"result_threads": []}
    running = threading.Event()

    class OwnerThreadFuture(asyncio.Future):
        def result(self):
            state["result_threads"].append(threading.get_ident())
            assert threading.get_ident() == state["owner_thread"]
            return super().result()

    source = OwnerThreadFuture(loop=foreign_loop)
    source.set_result("stopped")
    operation = isolated_runtime._create_host_owned_facade_task(
        source, name="precompleted-stopped-foreign-owner-thread"
    )
    notified = asyncio.Event()
    operation.add_done_callback(lambda _completed: notified.set())
    assert operation.done() is False
    assert state["result_threads"] == []

    def run_foreign_loop():
        asyncio.set_event_loop(foreign_loop)
        state["owner_thread"] = threading.get_ident()
        running.set()
        foreign_loop.run_forever()
        foreign_loop.close()

    foreign_thread = threading.Thread(target=run_foreign_loop)
    foreign_thread.start()
    assert running.wait(timeout=1)
    try:
        # Retrying the retained operation asks the now-running owner loop to
        # acknowledge the exact source; it does not issue another facade call.
        operation.cancel()
        await asyncio.wait_for(notified.wait(), timeout=1)
        assert operation.done() is True
        assert operation.foreign_settlement_disposition == "succeeded"
        assert state["result_threads"] == [state["owner_thread"]]
    finally:
        if foreign_thread.is_alive():
            foreign_loop.call_soon_threadsafe(foreign_loop.stop)
            await asyncio.to_thread(foreign_thread.join)


@pytest.mark.asyncio
async def test_terminal_retirement_retries_stopped_foreign_operation_after_owner_restarts(
    monkeypatch, tmp_path
):
    """A retained foreign stop is retried on restart without another facade stop."""

    foreign_loop = asyncio.new_event_loop()
    foreign_running = threading.Event()
    state = {"result_threads": []}

    class OwnerThreadFuture(asyncio.Future):
        def result(self):
            state["result_threads"].append(threading.get_ident())
            assert threading.get_ident() == state["owner_thread"]
            return super().result()

    foreign_stop = OwnerThreadFuture(loop=foreign_loop)

    class CrossLoopStopClient(FakeIsolatedClient):
        def __init__(self):
            super().__init__()
            self.stop_calls = 0
            self.stopped = False

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                return foreign_stop
            self.stopped = True
            return None

    def run_foreign_loop():
        asyncio.set_event_loop(foreign_loop)
        state["owner_thread"] = threading.get_ident()
        foreign_running.set()
        foreign_loop.run_forever()
        foreign_loop.close()

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    client = CrossLoopStopClient()
    feature._retain_terminal_retirement_client(client)

    foreign_thread = None
    try:
        # Initial retirement fences the exact source because its owner loop is
        # stopped.  The source's result must not be consumed from this thread.
        assert await feature._retire_terminal_clients() is False
        assert client.stop_calls == 1
        assert len(feature._terminal_lifecycle_tasks) == 1
        operation = feature._terminal_lifecycle_tasks[0].task
        assert operation.done() is False
        assert state["result_threads"] == []

        foreign_thread = threading.Thread(target=run_foreign_loop)
        foreign_thread.start()
        assert foreign_running.wait(timeout=1)

        # The next cleanup retries cancellation plus observation on the same
        # retained operation.  It remains fenced for this pass, never issuing
        # a duplicate facade stop while owner-loop acknowledgement is pending.
        assert await feature._retire_terminal_clients() is False
        assert client.stop_calls == 1

        for _ in range(100):
            if not feature._terminal_lifecycle_tasks:
                break
            await asyncio.sleep(0)
        assert feature._terminal_lifecycle_tasks == []
        assert operation.foreign_settlement_disposition == "cancelled"
        assert state["result_threads"] == [state["owner_thread"]]
        assert feature._terminal_retirement_clients == [client]

        # A cancelled foreign stop leaves the client fail-closed, but after
        # acknowledgement a bounded later retirement may make the one retry.
        assert await feature._retire_terminal_clients() is True
        assert client.stop_calls == 2
        assert client.stopped is True
        assert feature._terminal_retirement_clients == []
    finally:
        if foreign_thread is not None and foreign_thread.is_alive():
            foreign_loop.call_soon_threadsafe(foreign_loop.stop)
            await asyncio.to_thread(foreign_thread.join)
        elif not foreign_loop.is_closed():
            foreign_loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("result", "exception"))
async def test_foreign_terminal_source_is_detached_when_host_delivery_is_stranded(
    monkeypatch, caplog, outcome
):
    """A host callback failure cannot retain a foreign tenant terminal value."""

    foreign_loop = asyncio.new_event_loop()
    secret = "STRANDED-FOREIGN-TERMINAL-SECRET-2755"

    class TenantValue:
        pass

    class TenantError(RuntimeError):
        pass

    source = foreign_loop.create_future()
    value = TenantValue()
    value.secret = secret
    value_ref = weakref.ref(value)
    operation = isolated_runtime._create_host_owned_facade_task(
        source, name=f"stranded-foreign-{outcome}"
    )
    # The foreign owner loop is stopped, so construction cannot queue an
    # observer. Model its eventual owner-loop consumption below while the host
    # is already unable to accept the resulting callback.
    monkeypatch.setattr(
        operation._host_loop,
        "call_soon_threadsafe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("closing")),
    )
    if outcome == "result":
        source.set_result(value)
        error = None
    else:
        error = TenantError(secret)
        error.value = value
        source.set_exception(error)

    # The owner-loop disposition is terminal, but the closing host loop cannot
    # deliver its release callback. Detachment must precede that attempt.
    operation._observe_foreign_source_settlement(source)

    assert operation.done() is True
    assert operation._source is None
    del source, value, error
    gc.collect()
    assert value_ref() is None
    assert secret not in caplog.text
    foreign_loop.close()


@pytest.mark.asyncio
async def test_foreign_facade_dispatch_race_fails_closed_without_escaping(monkeypatch):
    """A loop that closes between inspection and dispatch cannot lose ownership."""

    foreign_loop = asyncio.new_event_loop()
    source = foreign_loop.create_future()
    original_is_running = foreign_loop.is_running
    monkeypatch.setattr(foreign_loop, "is_running", lambda: True)
    monkeypatch.setattr(
        foreign_loop,
        "call_soon_threadsafe",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("loop is closing")),
    )
    operation = isolated_runtime._create_host_owned_facade_task(
        source, name="racing-foreign-facade"
    )

    operation.cancel()
    assert operation.done() is False
    monkeypatch.setattr(foreign_loop, "is_running", original_is_running)
    foreign_loop.close()


@pytest.mark.asyncio
async def test_foreign_facade_owner_loop_consumes_secret_error_before_release(caplog):
    """Foreign failure consumption happens in its owner loop with no warning leak."""

    foreign_loop = asyncio.new_event_loop()
    ready = threading.Event()
    state = {}
    secret = "FOREIGN-FACADE-SECRET-2755"

    class ForeignFacadeError(RuntimeError):
        pass

    def run_foreign_loop():
        asyncio.set_event_loop(foreign_loop)

        def create_source():
            state["source"] = foreign_loop.create_future()
            ready.set()

        foreign_loop.call_soon(create_source)
        foreign_loop.run_forever()
        foreign_loop.close()

    foreign_thread = threading.Thread(target=run_foreign_loop)
    foreign_thread.start()
    assert ready.wait(timeout=1)
    operation = isolated_runtime._create_host_owned_facade_task(
        state["source"], name="foreign-secret-consumption"
    )
    notified = asyncio.Event()
    operation.add_done_callback(lambda _completed: notified.set())

    def fail_source():
        state["source"].set_exception(ForeignFacadeError(secret))
        foreign_loop.call_soon(foreign_loop.stop)

    foreign_loop.call_soon_threadsafe(fail_source)
    try:
        await asyncio.wait_for(notified.wait(), timeout=1)
        await asyncio.wait_for(asyncio.to_thread(foreign_thread.join), timeout=1)
        assert operation.done() is True
    finally:
        if foreign_thread.is_alive():
            foreign_loop.call_soon_threadsafe(foreign_loop.stop)
            await asyncio.to_thread(foreign_thread.join)

    state.clear()
    del operation
    gc.collect()
    assert secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected_disposition"),
    (("success", "succeeded"), ("error", "failed"), ("cancel", "cancelled")),
)
async def test_foreign_pre_settled_source_is_observed_before_owner_loop_stops(
    outcome, expected_disposition, caplog
):
    """A pre-settled foreign source cannot strand its observer behind stop."""

    foreign_loop = asyncio.new_event_loop()
    ready = threading.Event()
    state = {}
    secret = "FOREIGN-PRESETTLED-SECRET-2755"

    class ForeignPresettledError(RuntimeError):
        pass

    def run_foreign_loop():
        asyncio.set_event_loop(foreign_loop)

        def create_source():
            source = foreign_loop.create_future()
            if outcome == "success":
                source.set_result({"secret": secret})
            elif outcome == "error":
                source.set_exception(ForeignPresettledError(secret))
            else:
                source.cancel()
            state["source"] = source
            ready.set()

        foreign_loop.call_soon(create_source)
        foreign_loop.run_forever()
        foreign_loop.close()

    foreign_thread = threading.Thread(target=run_foreign_loop)
    foreign_thread.start()
    assert ready.wait(timeout=1)
    operation = isolated_runtime._create_host_owned_facade_task(
        state["source"], name=f"foreign-pre-settled-{outcome}"
    )
    notified = asyncio.Event()
    operation.add_done_callback(lambda _completed: notified.set())
    # The observer registration is already queued ahead of this stop.  Its
    # owner-loop inline path must consume a completed source in that same turn
    # rather than enqueueing one more callback which this stop would strand.
    foreign_loop.call_soon_threadsafe(foreign_loop.stop)
    try:
        await asyncio.wait_for(notified.wait(), timeout=1)
        await asyncio.wait_for(asyncio.to_thread(foreign_thread.join), timeout=1)
        assert operation.done() is True
        assert operation.foreign_settlement_disposition == expected_disposition
        assert state["source"]._log_traceback is False
    finally:
        if foreign_thread.is_alive():
            foreign_loop.call_soon_threadsafe(foreign_loop.stop)
            await asyncio.to_thread(foreign_thread.join)

    state.clear()
    del operation
    gc.collect()
    assert secret not in caplog.text


@pytest.mark.asyncio
async def test_owned_health_probe_child_cancellation_is_an_ordinary_failure():
    """A self-cancelled facade health task must not cancel its supervisor."""

    async def cancelled_health():
        asyncio.current_task().cancel()
        await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="health probe was cancelled") as failed:
        await isolated_runtime._await_owned_health_probe(
            cancelled_health(),
            name="self-cancelled-health",
            on_started=lambda _task: None,
            on_late_task=lambda _task: pytest.fail("settled child must not detach"),
        )

    assert failed.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("repeat", (False, True))
async def test_lifecycle_argless_first_cancellation_preserves_count_without_redelivery(
    repeat,
):
    """An arg-less first cancel is authoritative and leaves no synthetic wakeup."""

    owner_ref = {}

    async def child():
        owner = owner_ref["owner"]
        owner.cancel()
        if repeat:
            owner.cancel("later cancellation reason")
        asyncio.current_task().cancel("child cancellation")
        await asyncio.sleep(0)

    async def owner():
        task = asyncio.current_task()
        try:
            await isolated_runtime._await_owned_facade_lifecycle_operation(
                child(),
                name="argless-first-lifecycle-cancellation",
                on_late_task=lambda _task: pytest.fail("child must settle promptly"),
            )
        except asyncio.CancelledError as cancelled:
            args = cancelled.args
            count = task.cancelling()
            while task.cancelling():
                task.uncancel()
            await asyncio.sleep(0)
            return args, count

    owner_task = asyncio.create_task(owner())
    owner_ref["owner"] = owner_task
    args, count = await asyncio.wait_for(owner_task, timeout=1)
    assert args == ()
    assert count == (2 if repeat else 1)


@pytest.mark.asyncio
async def test_stale_cancellation_count_does_not_reclassify_self_cancelled_health():
    """A caught historical cancel is not a parent cancel for a later probe."""

    owner = asyncio.current_task()
    owner.cancel("historical cancellation")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.sleep(0)
    assert owner.cancelling() == 1

    async def self_cancelled_health():
        asyncio.current_task().cancel()
        await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="health probe was cancelled"):
        await isolated_runtime._await_owned_health_probe(
            self_cancelled_health(),
            name="stale-cancellation-health",
            on_started=lambda _task: None,
            on_late_task=lambda _task: pytest.fail("settled child must not detach"),
        )

    assert owner.cancelling() == 1
    owner.uncancel()


@pytest.mark.asyncio
async def test_stale_cancellation_count_does_not_reclassify_lifecycle_success():
    """Historical cancellation state cannot turn a settled facade into cancel."""

    owner = asyncio.current_task()
    owner.cancel("historical lifecycle cancellation")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.sleep(0)
    assert owner.cancelling() == 1

    source = asyncio.get_running_loop().create_future()
    source.set_result("stopped")
    assert await isolated_runtime._await_owned_facade_lifecycle_operation(
        source,
        name="stale-cancellation-lifecycle-success",
        on_late_task=lambda _task: pytest.fail("settled facade must not detach"),
    ) == "stopped"
    assert owner.cancelling() == 1
    owner.uncancel()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_stale_cancellation_count_does_not_reclassify_lifecycle_timeout_ack(
    monkeypatch,
):
    """A historical count cannot replace the timeout during child acknowledgement."""

    owner = asyncio.current_task()
    owner.cancel("historical lifecycle cancellation")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.sleep(0)
    assert owner.cancelling() == 1

    async def stop():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return None

    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_OPERATION_TIMEOUT", 0.01)
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_CANCELLATION_GRACE", 0.1)
    with pytest.raises(isolated_runtime._FacadeLifecycleOperationTimedOut):
        await isolated_runtime._await_owned_facade_lifecycle_operation(
            stop(),
            name="stale-cancellation-lifecycle-timeout-ack",
            on_late_task=lambda _task: pytest.fail("stop should settle promptly"),
        )
    assert owner.cancelling() == 1
    owner.uncancel()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_terminal_health_failure_after_cancel_does_not_abort_retirement(
    monkeypatch, tmp_path, caplog
):
    """A terminal probe's ordinary post-cancel error still permits one stop."""

    probe_started = asyncio.Event()
    secret = "TERMINAL-HEALTH-SECRET-2755"

    async def health():
        probe_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise RuntimeError(secret)

    class CountingStopClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = CountingStopClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    probe = isolated_runtime._create_host_owned_facade_task(
        health(), name="terminal-health-post-cancel-error"
    )
    feature._own_health_probe_task(probe)
    await asyncio.wait_for(probe_started.wait(), timeout=1)

    await asyncio.wait_for(feature.shutdown(), timeout=1)

    assert client.stop_calls == 1
    assert feature._terminal_health_probe_task is None
    assert feature._terminal_retirement_clients == []
    assert secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("child_cancels_first", (False, True))
@pytest.mark.parametrize("cancel_count", (1, 2))
async def test_lifecycle_parent_cancellation_payload_wins_same_turn_as_child(
    child_cancels_first, cancel_count
):
    """A child CancelledError never supplies a same-turn parent reason."""

    owner_ref = {}

    async def child():
        if child_cancels_first:
            asyncio.current_task().cancel()
        owner_ref["owner"].cancel("first parent cancellation")
        if cancel_count == 2:
            owner_ref["owner"].cancel("later parent cancellation")
        if not child_cancels_first:
            asyncio.current_task().cancel()
        await asyncio.sleep(0)

    owner = asyncio.create_task(
        isolated_runtime._await_owned_facade_lifecycle_operation(
            child(),
            name="same-turn-parent-lifecycle-cancellation",
            on_late_task=lambda _task: pytest.fail("child must settle promptly"),
        )
    )
    owner_ref["owner"] = owner

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await asyncio.wait_for(owner, timeout=1)

    assert cancelled.value.args == ("first parent cancellation",)
    assert owner.cancelled() is True
    assert owner.cancelling() == cancel_count


@pytest.mark.asyncio
@pytest.mark.parametrize("child_cancels_first", (False, True))
@pytest.mark.parametrize("cancel_count", (1, 2))
async def test_health_parent_cancellation_payload_wins_same_turn_as_child(
    child_cancels_first, cancel_count
):
    """Health retains ordinary child classification absent an accepted parent."""

    owner_ref = {}

    async def child():
        if child_cancels_first:
            asyncio.current_task().cancel()
        owner_ref["owner"].cancel("first health parent cancellation")
        if cancel_count == 2:
            owner_ref["owner"].cancel("later health parent cancellation")
        if not child_cancels_first:
            asyncio.current_task().cancel()
        await asyncio.sleep(0)

    owner = asyncio.create_task(
        isolated_runtime._await_owned_health_probe(
            child(),
            name="same-turn-parent-health-cancellation",
            on_started=lambda _task: None,
            on_late_task=lambda _task: pytest.fail("child must settle promptly"),
        )
    )
    owner_ref["owner"] = owner

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await asyncio.wait_for(owner, timeout=1)

    assert cancelled.value.args == ("first health parent cancellation",)
    assert owner.cancelled() is True
    assert owner.cancelling() == cancel_count


@pytest.mark.asyncio
async def test_supervisor_restarts_after_child_cancelled_health_probe(monkeypatch, tmp_path):
    """A child cancellation follows normal health recovery rather than terminal unwind."""

    first_health = asyncio.Event()
    restarted = asyncio.Event()
    real_sleep = asyncio.sleep

    class ChildCancelledHealthClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.health_calls = 0
            self.stop_calls = 0
            self.start_calls = 0

        async def health(self):
            self.health_calls += 1
            if self.health_calls == 1:
                first_health.set()
                asyncio.current_task().cancel()
                await real_sleep(0)
            return True

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

        async def start(self):
            self.start_calls += 1
            self.started = True
            restarted.set()

    async def immediate_sleep(_delay):
        await real_sleep(0)

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    client = ChildCancelledHealthClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    start_idle_monitor = Mock()
    monkeypatch.setattr(feature, "_start_idle_monitor", start_idle_monitor)
    supervisor = asyncio.create_task(feature._supervise())
    feature._supervision_task = supervisor
    try:
        await asyncio.wait_for(first_health.wait(), timeout=1)
        await asyncio.wait_for(restarted.wait(), timeout=1)
        for _ in range(100):
            if start_idle_monitor.called:
                break
            await real_sleep(0)
        assert client.stop_calls == 1
        assert client.start_calls == 1
        start_idle_monitor.assert_called_once_with()
        assert feature._traffic_gate.sealed is False
    finally:
        feature._stopping = True
        if not supervisor.done():
            supervisor.cancel()
        await asyncio.gather(supervisor, return_exceptions=True)


@pytest.mark.asyncio
async def test_supervisor_logs_neutral_health_and_restart_failures(monkeypatch, tmp_path, caplog):
    """Compatible facade failures never put tenant exception text in logs."""

    health_attempted = asyncio.Event()
    restart_attempted = asyncio.Event()
    secret = "ISOLATED-FACADE-TENANT-SECRET-2755"
    real_sleep = asyncio.sleep

    class SecretFailingClient(FakeIsolatedClient):
        def health(self):
            health_attempted.set()
            raise RuntimeError(secret)

        async def stop(self):
            return None

        def start(self):
            restart_attempted.set()
            raise RuntimeError(secret)

    async def immediate_sleep(_delay):
        await real_sleep(0)

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = SecretFailingClient()
    supervisor = asyncio.create_task(feature._supervise())
    feature._supervision_task = supervisor
    try:
        await asyncio.wait_for(health_attempted.wait(), timeout=1)
        await asyncio.wait_for(restart_attempted.wait(), timeout=1)
        assert "health check failed" in caplog.text
        assert "restart failed" in caplog.text
        assert secret not in caplog.text
    finally:
        feature._stopping = True
        if not supervisor.done():
            supervisor.cancel()
        await asyncio.gather(supervisor, return_exceptions=True)


@pytest.mark.asyncio
async def test_owned_health_probe_timeout_retains_task_returned_by_facade(monkeypatch):
    """A timed-out Task-returning health facade cannot become an unowned probe."""

    cancellation_observed = asyncio.Event()
    release = asyncio.Event()

    async def pending_task():
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellation_observed.set()

    operation = asyncio.create_task(pending_task(), name="facade-hostile-health-task")

    class TaskReturningFacade:
        def health(self):
            return operation

    late_tasks = []
    monkeypatch.setattr(isolated_runtime, "_HEALTH_PROBE_TIMEOUT", 0.01)
    monkeypatch.setattr(isolated_runtime, "_HEALTH_PROBE_CANCELLATION_GRACE", 0.01)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await isolated_runtime._await_owned_health_probe(
                TaskReturningFacade().health(),
                name="test-hostile-facade-health-task",
                on_started=lambda _task: None,
                on_late_task=late_tasks.append,
            )
        assert cancellation_observed.is_set()
        assert len(late_tasks) == 1
        assert late_tasks[0].get_name() == "test-hostile-facade-health-task"
    finally:
        release.set()
        await asyncio.wait_for(late_tasks[0] if late_tasks else operation, timeout=1)


@pytest.mark.asyncio
async def test_same_loop_health_parent_cancellation_claims_original_before_wrapper_runs():
    """Health cancellation reaches the facade Task, never only an observer."""

    started = asyncio.Event()
    release = asyncio.Event()
    cancellation_observed = asyncio.Event()

    async def hostile_health():
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellation_observed.set()

    original = asyncio.create_task(hostile_health(), name="facade-original-health")
    started_tasks = []
    late_tasks = []
    owner = asyncio.create_task(
        isolated_runtime._await_owned_health_probe(
            original,
            name="host-owned-prestart-health",
            on_started=started_tasks.append,
            on_late_task=late_tasks.append,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        owner.cancel("health parent cancellation")
        await asyncio.wait_for(cancellation_observed.wait(), timeout=1)

        assert original.done() is False
        assert len(started_tasks) == 1
        assert late_tasks == started_tasks
        assert late_tasks[0].get_name() == "host-owned-prestart-health"

        release.set()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(owner, timeout=1)
        assert cancelled.value.args == ("health parent cancellation",)
    finally:
        release.set()
        if not original.done():
            original.cancel()
        await asyncio.gather(original, return_exceptions=True)


@pytest.mark.asyncio
async def test_cross_loop_health_task_fails_closed_and_consumes_late_error():
    """A foreign health Task cannot trigger an ordinary retry while it runs."""

    foreign_loop = asyncio.new_event_loop()
    foreign_ready = threading.Event()
    release_foreign = threading.Event()
    cancellation_observed = threading.Event()
    foreign_state = {}

    class ForeignHealthError(RuntimeError):
        pass

    async def foreign_health():
        while not release_foreign.is_set():
            try:
                await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                cancellation_observed.set()
        raise ForeignHealthError("FOREIGN-HEALTH-SECRET")

    def run_foreign_loop():
        asyncio.set_event_loop(foreign_loop)
        task = foreign_loop.create_task(foreign_health(), name="foreign-facade-health")
        foreign_state["task"] = task
        foreign_ready.set()
        try:
            foreign_loop.run_until_complete(task)
        except ForeignHealthError:
            # The host observer has already retrieved this private outcome.
            pass
        finally:
            foreign_loop.close()

    foreign_thread = threading.Thread(target=run_foreign_loop)
    foreign_thread.start()
    assert foreign_ready.wait(timeout=1)

    started_tasks = []
    late_tasks = []
    try:
        with pytest.raises(isolated_runtime._CrossLoopFacadeOperationError):
            await isolated_runtime._await_owned_health_probe(
                foreign_state["task"],
                name="host-owned-foreign-health",
                on_started=started_tasks.append,
                on_late_task=late_tasks.append,
            )
        assert late_tasks == started_tasks
        assert len(late_tasks) == 1
        assert late_tasks[0].foreign_loop is True
        assert foreign_state["task"].done() is False
        assert cancellation_observed.wait(timeout=1)

        release_foreign.set()
        await asyncio.wait_for(asyncio.to_thread(foreign_thread.join), timeout=1)
        for _ in range(100):
            if late_tasks[0].done():
                break
            await asyncio.sleep(0)
        assert late_tasks[0].done() is True
    finally:
        release_foreign.set()
        if foreign_thread.is_alive():
            await asyncio.to_thread(foreign_thread.join)


@pytest.mark.asyncio
async def test_terminal_cleanup_never_stops_beside_unacknowledged_foreign_health(
    monkeypatch, tmp_path
):
    """A cross-loop health fence remains terminally incomplete, never retried."""

    foreign_loop = asyncio.new_event_loop()
    foreign_probe = foreign_loop.create_future()

    class StopCountingClient(FakeIsolatedClient):
        def __init__(self):
            super().__init__()
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    client = StopCountingClient()
    feature._client = client
    probe = isolated_runtime._create_host_owned_facade_task(
        foreign_probe, name="unacknowledged-foreign-terminal-health"
    )
    feature._own_health_probe_task(probe)

    try:
        with pytest.raises(RuntimeError, match="terminal retirement is incomplete"):
            await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 0
        assert feature._terminal_cleanup_uncertain is True
        assert feature._terminal_health_probe_task is probe
        assert probe.done() is False
    finally:
        foreign_loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("nonweak_facade", (False, True))
@pytest.mark.parametrize("foreign_outcome", ("success", "error", "cancel"))
async def test_cross_loop_facade_task_is_retained_fenced_and_released_before_retry(
    nonweak_facade, foreign_outcome, monkeypatch, tmp_path
):
    """Only a foreign success releases its exact retained stop ownership."""

    foreign_loop = asyncio.new_event_loop()
    foreign_ready = threading.Event()
    release_foreign = threading.Event()
    cancellation_observed = threading.Event()
    foreign_state = {}

    class ForeignStopError(RuntimeError):
        pass

    async def foreign_stop():
        while not release_foreign.is_set():
            try:
                await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                cancellation_observed.set()
        if foreign_outcome == "error":
            raise ForeignStopError("FOREIGN-STOP-SECRET-2755")
        if foreign_outcome == "cancel":
            raise asyncio.CancelledError()

    def run_foreign_loop():
        asyncio.set_event_loop(foreign_loop)
        task = foreign_loop.create_task(foreign_stop(), name="foreign-facade-stop")
        foreign_state["task"] = task
        foreign_ready.set()
        try:
            foreign_loop.run_until_complete(task)
        except (ForeignStopError, asyncio.CancelledError):
            # The host's owner-loop observer has already consumed the private
            # disposition; the loop runner only needs to terminate cleanly.
            pass
        finally:
            foreign_loop.close()

    foreign_thread = threading.Thread(target=run_foreign_loop)
    foreign_thread.start()
    assert foreign_ready.wait(timeout=1)

    if nonweak_facade:
        class CrossLoopStopClient:
            __slots__ = ("stop_calls", "stopped")

            def __init__(self):
                self.stop_calls = 0
                self.stopped = False

            def stop(self):
                self.stop_calls += 1
                if self.stop_calls == 1:
                    return foreign_state["task"]
                self.stopped = True
                return None
    else:
        class CrossLoopStopClient(FakeIsolatedClient):
            def __init__(self):
                super().__init__()
                self.stop_calls = 0

            def stop(self):
                self.stop_calls += 1
                if self.stop_calls == 1:
                    return foreign_state["task"]
                self.stopped = True
                return None

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    client = CrossLoopStopClient()
    feature._client = client
    try:
        with pytest.raises(RuntimeError, match="terminal retirement is incomplete"):
            await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 1
        assert len(feature._terminal_lifecycle_tasks) == 1
        assert feature._terminal_lifecycle_tasks[0].task.foreign_loop is True
        assert foreign_state["task"].done() is False
        assert cancellation_observed.wait(timeout=1)

        # A fresh terminal caller must not retry while the foreign operation
        # still owns the facade's lifecycle handle.
        with pytest.raises(RuntimeError, match="terminal retirement is incomplete"):
            await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == 1

        release_foreign.set()
        await asyncio.wait_for(asyncio.to_thread(foreign_thread.join), timeout=1)
        for _ in range(100):
            if not feature._terminal_lifecycle_tasks:
                break
            await asyncio.sleep(0)
        assert feature._terminal_lifecycle_tasks == []

        # A successful foreign source releases its exact retained client. A
        # failed/cancelled source remains fail-closed, but only after its first
        # source has settled may a later shutdown attempt a fresh ``stop()``.
        await asyncio.wait_for(feature.shutdown(), timeout=1)
        assert client.stop_calls == (1 if foreign_outcome == "success" else 2)
        assert client.stopped is (foreign_outcome != "success")
        assert feature._terminal_retirement_clients == []
    finally:
        release_foreign.set()
        if foreign_thread.is_alive():
            await asyncio.to_thread(foreign_thread.join)
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_foreign_success_keeps_terminal_stop_owned_until_host_disposition_runs(
    monkeypatch, tmp_path
):
    """A host cleanup between owner acknowledgement and callback cannot re-stop."""

    foreign_loop = asyncio.new_event_loop()
    foreign_stop = foreign_loop.create_future()

    class DeferredForeignStopClient(FakeIsolatedClient):
        def __init__(self):
            super().__init__()
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                return foreign_stop
            pytest.fail("terminal cleanup issued a duplicate foreign stop")

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    client = DeferredForeignStopClient()
    feature._retain_terminal_retirement_client(client)

    try:
        with pytest.raises(isolated_runtime._CrossLoopFacadeOperationError):
            await isolated_runtime._await_owned_facade_lifecycle_operation(
                client.stop(),
                name="foreign-success-host-disposition-gap",
                on_completed=lambda: feature._release_terminal_retirement_client(client),
                on_timeout=lambda: feature._fence_terminal_retirement_timeout(client),
                on_late_task=lambda task: feature._retain_terminal_lifecycle_task(
                    task, client
                ),
            )
        operation = feature._terminal_lifecycle_tasks[0].task

        # Model the source loop consuming its successful result just before
        # the host handles its queued disposition callback.  No host turn is
        # allowed between these two assertions.
        foreign_stop.set_result(None)
        operation._observe_foreign_source_settlement(foreign_stop)
        assert operation.done() is True
        assert operation.settlement_delivery_complete is False

        assert await feature._retire_terminal_clients() is False
        assert client.stop_calls == 1

        for _ in range(4):
            await asyncio.sleep(0)
        assert feature._terminal_lifecycle_tasks == []
        assert feature._terminal_retirement_clients == []
        assert client.stop_calls == 1
    finally:
        foreign_loop.close()


@pytest.mark.asyncio
async def test_owned_facade_stop_replays_cancellation_before_late_secret_error():
    """A late facade error cannot replace a remembered caller cancellation."""

    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    class SecretStopError(RuntimeError):
        pass

    async def stop():
        stop_started.set()
        await release_stop.wait()
        # Keep the owner in its next shielded await while this facade fails.
        # Without that scheduling edge, the owner can observe task.done() at
        # the top of its loop and miss the path this test protects.
        await asyncio.sleep(0)
        raise SecretStopError("TENANT-STOP-SECRET")

    owner = asyncio.create_task(
        isolated_runtime._await_owned_facade_lifecycle_operation(
            stop(),
            name="test-late-secret-stop",
            on_late_task=lambda _task: pytest.fail("stop should settle promptly"),
        )
    )
    await asyncio.wait_for(stop_started.wait(), timeout=1)
    owner.cancel("reload cancellation")
    await asyncio.sleep(0)
    assert owner.done() is False
    release_stop.set()

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await asyncio.wait_for(owner, timeout=1)

    assert cancelled.value.args == ("reload cancellation",)
    assert "TENANT-STOP-SECRET" not in repr(cancelled.value)
    assert cancelled.value.__context__ is None
    _assert_runtime_traceback_locals_are_detached(
        cancelled.value,
        forbidden_values=(),
    )


@pytest.mark.asyncio
async def test_owned_facade_cancellation_traceback_drops_late_secret_result():
    """Cancellation does not retain a completed facade task's result."""

    stop_started = asyncio.Event()
    release_stop = asyncio.Event()
    secret_result = {"token": "TENANT-STOP-RESULT-SECRET"}

    async def stop():
        stop_started.set()
        await release_stop.wait()
        return secret_result

    owner = asyncio.create_task(
        isolated_runtime._await_owned_facade_lifecycle_operation(
            stop(),
            name="test-late-secret-stop-result",
            on_late_task=lambda _task: pytest.fail("stop should settle promptly"),
        )
    )
    await asyncio.wait_for(stop_started.wait(), timeout=1)
    owner.cancel("reload result cancellation")
    await asyncio.sleep(0)
    release_stop.set()

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await asyncio.wait_for(owner, timeout=1)

    assert cancelled.value.args == ("reload result cancellation",)
    assert not _traceback_locals_reach_any(
        cancelled.value,
        (secret_result,),
    )


@pytest.mark.asyncio
async def test_owned_facade_timeout_acknowledges_child_cancellation_as_timeout(
    monkeypatch,
):
    """The helper's own task.cancel() never becomes a caller cancellation."""

    stop_started = asyncio.Event()
    secret = {"credential": "TIMED-OUT-STOP-SECRET-2755"}

    async def stop():
        stop_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # The acknowledgement receives the child's cancellation first.
            # A late secret failure must stay private behind the timeout.
            await asyncio.sleep(0)
            raise RuntimeError(secret)

    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_OPERATION_TIMEOUT", 0.01)
    monkeypatch.setattr(isolated_runtime, "_FACADE_LIFECYCLE_CANCELLATION_GRACE", 0.1)
    with pytest.raises(isolated_runtime._FacadeLifecycleOperationTimedOut) as timed_out:
        await isolated_runtime._await_owned_facade_lifecycle_operation(
            stop(),
            name="test-child-cancellation-timeout",
            on_late_task=lambda _task: pytest.fail("stop should settle promptly"),
        )

    assert stop_started.is_set()
    assert "TIMED-OUT-STOP-SECRET-2755" not in repr(timed_out.value)
    assert timed_out.value.__context__ is None
    _assert_runtime_traceback_locals_are_detached(
        timed_out.value,
        forbidden_values=(secret,),
    )


def test_facade_lifecycle_timeout_covers_locked_sdk_stop_sequence():
    """The host deadline tracks every sequential SDK 0.36.0 stop observation."""

    assert isolated_runtime._SDK_SUBPROCESS_STOP_BUDGET == 33.0
    assert isolated_runtime._FACADE_LIFECYCLE_OPERATION_TIMEOUT == 36.0


@pytest.mark.asyncio
async def test_health_timeout_acknowledgement_does_not_swallow_supervisor_cancellation(
    monkeypatch, tmp_path
):
    """A cancellation concurrent with probe acknowledgement never restarts."""

    health_started = asyncio.Event()
    supervisor: asyncio.Task[None] | None = None

    class TimeoutAcknowledgingClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0
            self.start_calls = 0

        async def health(self):
            health_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                assert supervisor is not None
                supervisor.cancel("supervisor health cancellation")
                raise

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

        async def start(self):
            self.start_calls += 1
            self.started = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    monkeypatch.setattr(isolated_runtime, "_HEALTH_PROBE_TIMEOUT", 0.01)
    client = TimeoutAcknowledgingClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    try:
        supervisor = asyncio.create_task(feature._supervise())
        feature._supervision_task = supervisor
        await asyncio.wait_for(health_started.wait(), timeout=1)
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(supervisor, timeout=1)

        assert cancelled.value.args == ("supervisor health cancellation",)
        assert client.start_calls == 0
        assert client.stop_calls == 1
        assert feature._client is None
        assert feature._traffic_gate.sealed is True
    finally:
        if supervisor is not None and not supervisor.done():
            supervisor.cancel()
            with pytest.raises(asyncio.CancelledError):
                await supervisor
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_health_timeout_acknowledgement_consumes_late_facade_secret(
    monkeypatch,
):
    """A post-timeout health failure stays private behind TimeoutError."""

    probe_started = asyncio.Event()
    secret = {"credential": "TENANT-HEALTH-SECRET"}

    async def health():
        probe_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # Force the helper to await the task after cancellation before it
            # fails, rather than observing the completed task at loop entry.
            await asyncio.sleep(0)
            raise RuntimeError(secret)

    monkeypatch.setattr(isolated_runtime, "_HEALTH_PROBE_TIMEOUT", 0.01)
    monkeypatch.setattr(isolated_runtime, "_HEALTH_PROBE_CANCELLATION_GRACE", 0.1)
    with pytest.raises(asyncio.TimeoutError) as timed_out:
        await isolated_runtime._await_owned_health_probe(
            health(),
            name="test-late-secret-health",
            on_started=lambda _task: None,
            on_late_task=lambda _task: pytest.fail("health should settle promptly"),
        )

    assert probe_started.is_set()
    assert "TENANT-HEALTH-SECRET" not in repr(timed_out.value)
    _assert_runtime_traceback_locals_are_detached(
        timed_out.value,
        forbidden_values=(secret,),
    )


@pytest.mark.asyncio
async def test_reload_cancellation_after_successful_stop_never_restores_or_retires_twice(
    monkeypatch, tmp_path
):
    """Reload keeps a stop proven during cancellation out of quarantine retry."""

    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    class SlowStopClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1
            stop_started.set()
            await release_stop.wait()
            self.stopped = True

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = SlowStopClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    reload_task = asyncio.create_task(feature.reload())
    try:
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        reload_task.cancel("reload stop cancellation")
        await asyncio.sleep(0)
        assert reload_task.done() is False
        release_stop.set()

        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(reload_task, timeout=1)

        assert cancelled.value.args == ("reload stop cancellation",)
        assert client.stop_calls == 1
        assert feature._client is None
        assert feature._terminal_retirement_clients == []
        assert feature._traffic_gate.sealed is True
    finally:
        release_stop.set()
        if not reload_task.done():
            reload_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reload_task
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_fenced_recovery_later_failure_does_not_retry_successfully_stopped_facade(
    monkeypatch, tmp_path
):
    """Fenced recovery quarantines later failure without a duplicate old stop."""

    class CountingStopClient(FakeIsolatedClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

    async def promotion_failure(_transition):
        raise RuntimeError("later promotion failure")

    async def clear_pending(_transition):
        return None

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = CountingStopClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    monkeypatch.setattr(feature, "_promote_config", promotion_failure)
    monkeypatch.setattr(feature, "_clear_owned_pending_before_quarantine", clear_pending)
    try:
        with pytest.raises(RuntimeError, match="later promotion failure"):
            await feature._recover_fenced_transition_uninterrupted(Mock())

        assert client.stop_calls == 1
        assert feature._client is None
        assert feature._terminal_retirement_clients == []
        assert feature._traffic_gate.sealed is True
    finally:
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
async def test_fenced_recovery_nonweak_facade_does_not_retry_successful_stop(
    monkeypatch, tmp_path
):
    """The proven-stop fence is identity-safe for non-weakrefable facades."""

    class NonWeakCountingStopClient:
        __slots__ = ("stop_calls", "stopped")

        def __init__(self):
            self.stop_calls = 0
            self.stopped = False

        async def stop(self):
            self.stop_calls += 1
            self.stopped = True

    async def promotion_failure(_transition):
        raise RuntimeError("later promotion failure")

    async def clear_pending(_transition):
        return None

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = NonWeakCountingStopClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    monkeypatch.setattr(feature, "_promote_config", promotion_failure)
    monkeypatch.setattr(feature, "_clear_owned_pending_before_quarantine", clear_pending)
    try:
        with pytest.raises(RuntimeError, match="later promotion failure"):
            await feature._recover_fenced_transition_uninterrupted(Mock())

        assert client.stop_calls == 1
        assert feature._client is None
        assert feature._terminal_retirement_clients == []
        assert feature._traffic_gate.sealed is True
    finally:
        if not feature._stopping:
            await feature.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("nonweakrefable", [False, True])
async def test_fenced_recovery_cancellation_does_not_restore_proven_stopped_facade(
    monkeypatch, tmp_path, nonweakrefable
):
    """Cancellation after the exact stop settles cannot schedule a second stop."""

    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    if nonweakrefable:

        class SlowStopClient:
            __slots__ = ("stop_calls", "stopped")

            def __init__(self):
                self.stop_calls = 0
                self.stopped = False

            async def stop(self):
                self.stop_calls += 1
                stop_started.set()
                await release_stop.wait()
                self.stopped = True

    else:

        class SlowStopClient(FakeIsolatedClient):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.stop_calls = 0

            async def stop(self):
                self.stop_calls += 1
                stop_started.set()
                await release_stop.wait()
                self.stopped = True

    async def clear_pending(_transition):
        return None

    agent = Mock(did=_TEST_AGENT_DID, features={})
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    monkeypatch.setenv("KESTREL_FEATURE_TESTFEATURE_BIN", "/bin/test-service")
    client = SlowStopClient()
    feature = ProxyFeature(agent, _cfg_runtime(), client_factory=FakeIsolatedClient)
    feature._client = client
    monkeypatch.setattr(feature, "_clear_owned_pending_before_quarantine", clear_pending)
    recovery = asyncio.create_task(
        feature._recover_fenced_transition_uninterrupted(Mock())
    )
    try:
        await asyncio.wait_for(stop_started.wait(), timeout=1)
        recovery.cancel("fenced recovery cancellation")
        await asyncio.sleep(0)
        assert recovery.done() is False
        release_stop.set()

        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(recovery, timeout=1)

        assert cancelled.value.args == ("fenced recovery cancellation",)
        assert client.stop_calls == 1
        assert feature._client is None
        assert feature._terminal_retirement_clients == []
        assert feature._traffic_gate.sealed is True
    finally:
        release_stop.set()
        if not recovery.done():
            recovery.cancel()
            with pytest.raises(asyncio.CancelledError):
                await recovery
        if not feature._stopping:
            await feature.shutdown()
