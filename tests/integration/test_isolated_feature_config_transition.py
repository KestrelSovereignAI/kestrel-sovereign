"""Real SDK config-transition exchange through the isolated host proxy."""

import asyncio
import json
import signal
import sys
from unittest.mock import Mock

import pytest

from kestrel_sdk.isolated_feature import (
    ConfigTransitionResult,
    HostIngressError,
    IsolatedFeatureClient,
    IsolatedFeatureService,
    SubprocessIsolatedFeatureClient,
)

from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sovereign.features.isolated_runtime import (
    ProxyFeature,
    configure_hosted_isolated_runtime_lifecycle,
)


_TEST_AGENT_DID = "did:test:isolated-integration"


def _config_node_id(feature_name: str) -> str:
    return f"feature_config:v2:{_TEST_AGENT_DID}:{feature_name}"


class _QueueReader:
    def __init__(self):
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def readline(self) -> bytes:
        return await self._queue.get()


class _QueueWriter:
    def __init__(self, target: _QueueReader):
        self._target = target

    def write(self, data: bytes) -> None:
        self._target._queue.put_nowait(data)

    async def drain(self) -> None:
        return None


class _Storage:
    def __init__(self):
        self.nodes = {}
        self.agent_id = _TEST_AGENT_DID

    async def add_node(self, node) -> None:
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


class _SDKWireClient:
    """Proxy-compatible lifecycle adapter over a real SDK JSON-RPC client."""

    def __init__(self, rpc: IsolatedFeatureClient, config: dict[str, object]):
        self._rpc = rpc
        self._config = config
        self.stopped = False

    @property
    def supports_config_transition(self) -> bool:
        return self._rpc.supports_config_transition

    @property
    def replacement_required(self) -> bool:
        return self._rpc.replacement_required

    async def start(self) -> None:
        await self._rpc.initialize(config=self._config)
        await self._rpc.health()

    async def stop(self) -> None:
        self.stopped = True
        await self._rpc.shutdown()
        await self._rpc.close()

    async def health(self):
        return await self._rpc.health()

    async def list_tools(self):
        return await self._rpc.list_tools()

    async def prepare_config_transition(self, next_config):
        return await self._rpc.prepare_config_transition(next_config)

    @property
    def host_ingress_capabilities(self):
        return self._rpc.host_ingress_capabilities

    async def call_host_ingress(self, name, payload=None):
        return await self._rpc.call_host_ingress(name, payload)

    def on_event(self, handler) -> None:
        self._rpc.on_event(handler)


@pytest.mark.asyncio
async def test_proxy_forwards_empty_config_through_real_subprocess_wrapper(
    monkeypatch, tmp_path
):
    """The production wrapper must send ``config: {}`` during initialize.

    ``IsolatedFeatureService`` invokes ``configure`` only when the initialize
    request contains a config object. This drives the actual proxy → subprocess
    wrapper → SDK JSON-RPC path and records that the service observed the
    explicit empty object.
    """

    configured = tmp_path / "configured.json"
    service_script = tmp_path / "empty_config_service.py"
    service_script.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                "import asyncio",
                "import json",
                "import os",
                "from pathlib import Path",
                "from kestrel_sdk.isolated_feature import IsolatedFeatureService",
                "",
                "class EmptyConfigService(IsolatedFeatureService):",
                "    def __init__(self):",
                "        super().__init__(name='empty-config', version='1.0.0')",
                "",
                "    async def configure(self, config):",
                "        Path(os.environ['EMPTY_CONFIG_MARKER']).write_text(json.dumps(config))",
                "",
                "async def main():",
                "    await EmptyConfigService().run_stdio()",
                "",
                "asyncio.run(main())",
                "",
            ]
        )
    )
    service_script.chmod(0o755)

    runtime = InstalledFeatureRuntime(
        class_name="EmptyConfigFeature",
        entry_point="test_pkg.feature:EmptyConfigFeature",
        distribution="test-pkg",
        runtime="isolated-venv",
        service="empty-config-service",
    )
    agent = Mock(did=_TEST_AGENT_DID, storage_path=str(tmp_path / "agent" / "kestrel_prime.db"), features={})
    agent.storage = _Storage()
    monkeypatch.setenv("EMPTY_CONFIG_MARKER", str(configured))
    monkeypatch.setenv("KESTREL_FEATURE_EMPTYCONFIGFEATURE_BIN", str(service_script))
    feature = ProxyFeature(agent, runtime)

    try:
        await feature.initialize()
        assert json.loads(configured.read_text()) == {}
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_hosted_idle_retirement_reaps_and_cold_starts_real_subprocess(
    monkeypatch, tmp_path
):
    """Idle lifecycle telemetry and restart evidence come from a real child."""

    service_script = tmp_path / "idle_lifecycle_service.py"
    service_script.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                "import asyncio",
                "from kestrel_sdk.isolated_feature import IsolatedFeatureService",
                "",
                "class IdleLifecycleService(IsolatedFeatureService):",
                "    def __init__(self):",
                "        super().__init__(name='idle-lifecycle', version='1.0.0')",
                "        self.advertise_inbound_producer(False)",
                "        self.register_host_ingress('poke', self.poke)",
                "",
                "    async def poke(self, payload):",
                "        return {'generation': payload['generation']}",
                "",
                "async def main():",
                "    await IdleLifecycleService().run_stdio()",
                "",
                "asyncio.run(main())",
                "",
            ]
        )
    )
    service_script.chmod(0o755)

    runtime = InstalledFeatureRuntime(
        class_name="IdleLifecycleFeature",
        entry_point="test_pkg.feature:IdleLifecycleFeature",
        distribution="test-pkg",
        runtime="isolated-venv",
        service="idle-lifecycle-service",
    )
    snapshots = []
    agent = Mock(
        did=_TEST_AGENT_DID,
        storage_path=str(tmp_path / "agent" / "kestrel_prime.db"),
        features={},
    )
    agent.storage = _Storage()
    agent.isolated_runtime_root = tmp_path / "runtimes"
    agent.isolated_runtime_namespace = "tenant/agent"
    configure_hosted_isolated_runtime_lifecycle(
        agent,
        idle_timeout_seconds=3600,
        telemetry_observer=snapshots.append,
    )
    monkeypatch.setenv(
        "KESTREL_FEATURE_IDLELIFECYCLEFEATURE_BIN", str(service_script)
    )
    feature = ProxyFeature(agent, runtime)
    first_process = second_process = None

    try:
        await feature.initialize()
        first_process = feature._client.process
        running = await feature.sample_runtime_telemetry()
        assert first_process is not None
        assert running.state == "running"
        assert running.active_processes == 1
        assert running.process_count == 1
        assert running.rss_bytes is not None and running.rss_bytes > 0
        assert running.open_fds is not None and running.open_fds > 0

        assert feature._last_used_monotonic is not None
        feature._last_used_monotonic -= 7200
        retired = await feature._retire_idle_generation(
            expected_activity_generation=feature._activity_generation,
            expected_last_used=feature._last_used_monotonic,
        )

        assert retired is True
        assert first_process.returncode is not None
        idle = feature.runtime_telemetry_snapshot()
        assert idle.state == "idle"
        assert idle.active_processes == 0
        assert idle.cleanup_eligible is True
        assert idle.rss_bytes is None

        work_dir = feature._feature_runtime_dir() / "work"
        work_dir.rmdir()
        assert work_dir.exists() is False

        assert await feature.call_host_ingress("poke", {"generation": 2}) == {
            "generation": 2
        }
        assert work_dir.is_dir()
        second_process = feature._client.process
        restarted = await feature.sample_runtime_telemetry()
        assert second_process is not None
        assert second_process.pid != first_process.pid
        assert restarted.state == "running"
        assert restarted.lifecycle_generation == 2
        assert restarted.restart_count == 0
        assert restarted.idle_wake_count == 1
        assert restarted.process_count == 1
        assert {snapshot.feature for snapshot in snapshots} == {
            "IdleLifecycleFeature"
        }
    finally:
        await feature.shutdown()

    assert second_process is not None
    assert second_process.returncode is not None


@pytest.mark.asyncio
async def test_hosted_idle_monitor_reclaims_and_reprovisions_real_managed_venv(
    tmp_path,
):
    """The real deadline path reaps, reclaims, provisions, and cold-wakes."""

    project = tmp_path / "idle-managed-project"
    package = project / "idle_managed_feature"
    package.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "idle-managed-feature"
version = "1.0.0"
dependencies = ["kestrel-sovereign-sdk>=0.37.0,<0.38"]

[project.scripts]
idle-managed-service = "idle_managed_feature.service:main"
"""
    )
    (package / "__init__.py").write_text("")
    (package / "service.py").write_text(
        """\
import asyncio

from kestrel_sdk.isolated_feature import IsolatedFeatureService


class IdleManagedService(IsolatedFeatureService):
    def __init__(self):
        super().__init__(name="idle-managed", version="1.0.0")
        self.advertise_inbound_producer(False)
        self.register_host_ingress("poke", self.poke)

    async def poke(self, payload):
        return {"generation": payload["generation"]}


def main():
    asyncio.run(IdleManagedService().run_stdio())
"""
    )

    runtime = InstalledFeatureRuntime(
        class_name="IdleManagedFeature",
        entry_point="idle_managed_feature.feature:IdleManagedFeature",
        distribution="idle-managed-feature",
        runtime="isolated-venv",
        service="idle-managed-service",
        project=str(project),
    )
    snapshots = []
    agent = Mock(
        did=_TEST_AGENT_DID,
        storage_path=str(tmp_path / "agent" / "kestrel_prime.db"),
        features={},
    )
    agent.storage = _Storage()
    agent.isolated_runtime_root = tmp_path / "runtimes"
    agent.isolated_runtime_namespace = "tenant/agent"
    configure_hosted_isolated_runtime_lifecycle(
        agent,
        idle_timeout_seconds=0.1,
        telemetry_observer=snapshots.append,
    )
    feature = ProxyFeature(agent, runtime)
    first_process = second_process = None

    try:
        await feature.initialize()
        first_process = feature._client.process
        assert first_process is not None
        assert feature._idle_monitor_task is not None
        assert not feature._idle_monitor_task.done()

        for _ in range(300):
            if any(snapshot.state == "idle" for snapshot in snapshots):
                break
            await asyncio.sleep(0.02)
        assert any(snapshot.state == "idle" for snapshot in snapshots)
        assert first_process.returncode is not None

        managed_venv = feature._feature_runtime_dir() / ".venv"
        assert managed_venv.is_dir()
        outcome = await feature.reclaim_idle_workspace()
        assert outcome.value == "removed"
        assert not managed_venv.exists()

        assert await feature.call_host_ingress("poke", {"generation": 2}) == {
            "generation": 2
        }
        second_process = feature._client.process
        assert second_process is not None
        assert second_process.pid != first_process.pid
        assert managed_venv.is_dir()
        assert feature._last_cache_hit is False
    finally:
        await feature.shutdown()

    assert second_process is not None
    assert second_process.returncode is not None


@pytest.mark.asyncio
async def test_proxy_uses_negotiated_sdk_transition_before_replacement(
    monkeypatch, tmp_path
):
    """The host invokes the public SDK lifecycle over real JSON-RPC, not a tool."""

    old_config = {
        "enabled": True,
        "token": "old-token-not-for-logs",
        "transport": "webhook",
    }
    next_config = {
        "enabled": False,
        "token": "new-token-not-for-logs",
        "transport": "polling",
    }
    observed = []
    service_tasks = []
    clients = []

    class RestartService(IsolatedFeatureService):
        def __init__(self) -> None:
            super().__init__(name="transition-integration", version="1.0.0")
            self.advertise_config_transition()

        async def configure(self, config):
            observed.append(("initialize", dict(config)))

        async def on_config_transition(self, config):
            observed.append(("cleanup", dict(self.host_config), dict(config)))
            return ConfigTransitionResult.restart_required()

        async def on_shutdown(self):
            observed.append(("shutdown", dict(self.host_config)))
            return await super().on_shutdown()

    def client_factory(**kwargs):
        host_reader = _QueueReader()
        service_reader = _QueueReader()
        service = RestartService()
        service_tasks.append(
            asyncio.create_task(service.serve(service_reader, _QueueWriter(host_reader)))
        )
        rpc = IsolatedFeatureClient(host_reader, _QueueWriter(service_reader))
        client = _SDKWireClient(rpc, dict(kwargs.get("config") or {}))
        clients.append(client)
        return client

    runtime = InstalledFeatureRuntime(
        class_name="TransitionFeature",
        entry_point="test_pkg.feature:TransitionFeature",
        distribution="test-pkg",
        runtime="isolated-venv",
        service="test-service",
    )
    agent = Mock(did=_TEST_AGENT_DID, storage_path=str(tmp_path / "agent" / "kestrel_prime.db"), features={})
    agent.storage = _Storage()
    monkeypatch.setenv("KESTREL_FEATURE_TRANSITIONFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, runtime, client_factory=client_factory)

    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        await feature.set_config(next_config)

        # The first service sees old config during ordered cleanup. Only after
        # that lifecycle request resolves does the host issue shutdown and
        # initialize a replacement with the full effective next config.
        assert observed == [
            ("initialize", old_config),
            ("cleanup", old_config, next_config),
            ("shutdown", old_config),
            ("initialize", next_config),
        ]
        assert clients[0].stopped is True
        assert clients[1].stopped is False
    finally:
        await feature.shutdown()
        await asyncio.gather(*service_tasks)


@pytest.mark.asyncio
async def test_cancelled_host_ingress_keeps_real_sdk_rpc_admitted_until_child_settles(
    monkeypatch, tmp_path
):
    """A cancelled host waiter cannot retire a live SDK service callback.

    This crosses the public SDK's real in-memory JSON-RPC client/service wire.
    SDK 0.36.0 deliberately drops a locally cancelled request waiter without
    sending a cancellation message to the service, so the host proxy must keep
    its traffic gate admitted until the child handler's late response settles.
    """

    handler_started = asyncio.Event()
    handler_release = asyncio.Event()
    handler_finished = asyncio.Event()
    handler_cancelled = asyncio.Event()
    handler_active = asyncio.Event()
    service_tasks = []
    clients = []

    class BlockingIngressService(IsolatedFeatureService):
        def __init__(self) -> None:
            super().__init__(name="host-ingress-cancellation", version="1.0.0")
            self.register_host_ingress("telegram-webhook", self._handle_ingress)

        async def _handle_ingress(self, payload):
            handler_active.set()
            handler_started.set()
            try:
                await handler_release.wait()
                return {"accepted": True, "update_id": payload["update_id"]}
            except asyncio.CancelledError:
                handler_cancelled.set()
                raise
            finally:
                handler_active.clear()
                handler_finished.set()

    def client_factory(**kwargs):
        host_reader = _QueueReader()
        service_reader = _QueueReader()
        service = BlockingIngressService()
        service_tasks.append(
            asyncio.create_task(service.serve(service_reader, _QueueWriter(host_reader)))
        )
        rpc = IsolatedFeatureClient(host_reader, _QueueWriter(service_reader))
        client = _SDKWireClient(rpc, dict(kwargs.get("config") or {}))
        clients.append(client)
        return client

    runtime = InstalledFeatureRuntime(
        class_name="HostIngressCancellationFeature",
        entry_point="test_pkg.feature:HostIngressCancellationFeature",
        distribution="test-pkg",
        runtime="isolated-venv",
        service="test-service",
    )
    agent = Mock(
        did=_TEST_AGENT_DID,
        storage_path=str(tmp_path / "agent" / "kestrel_prime.db"),
        features={},
    )
    agent.storage = _Storage()
    monkeypatch.setenv(
        "KESTREL_FEATURE_HOSTINGRESSCANCELLATIONFEATURE_BIN", "/bin/test-service"
    )
    feature = ProxyFeature(agent, runtime, client_factory=client_factory)
    ingress = reload_task = None

    try:
        await feature.initialize()
        ingress = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"update_id": 7})
        )
        await asyncio.wait_for(handler_started.wait(), timeout=1)

        # The public caller receives cancellation only after the proxy drains
        # the still-live wire request. Before that, the SDK service handler is
        # active and no remote cancellation has been delivered.
        ingress.cancel()
        await asyncio.sleep(0)
        assert ingress.done() is False
        assert handler_active.is_set()
        assert handler_cancelled.is_set() is False

        reload_task = asyncio.create_task(feature.reload())
        for _ in range(100):
            if feature._traffic_gate.closed:
                break
            await asyncio.sleep(0)
        assert feature._traffic_gate.closed is True
        assert reload_task.done() is False
        assert clients[0].stopped is False
        assert handler_active.is_set()

        # Let the service send its late JSON-RPC response. The public task then
        # re-delivers its original cancellation, releases admission, and only
        # then permits the lifecycle replacement to stop the first child.
        handler_release.set()
        await asyncio.wait_for(handler_finished.wait(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(ingress, timeout=1)
        await asyncio.wait_for(reload_task, timeout=1)

        assert handler_cancelled.is_set() is False
        assert clients[0].stopped is True
        assert len(clients) == 2
        assert await feature.call_host_ingress(
            "telegram-webhook", {"update_id": 8}
        ) == {"accepted": True, "update_id": 8}
    finally:
        handler_release.set()
        for task in (reload_task, ingress):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await feature.shutdown()
        await asyncio.gather(*service_tasks)


@pytest.mark.asyncio
async def test_cancelled_host_ingress_replays_original_cancellation_after_late_wire_failure(
    monkeypatch, tmp_path
):
    """A late SDK failure cannot replace a caller cancellation or leak a task."""

    handler_started = asyncio.Event()
    handler_release = asyncio.Event()
    handler_finished = asyncio.Event()
    service_tasks = []
    loop_contexts = []
    clients = []

    class LateFailingIngressService(IsolatedFeatureService):
        def __init__(self) -> None:
            super().__init__(name="host-ingress-late-failure", version="1.0.0")
            self.register_host_ingress("telegram-webhook", self._handle_ingress)

        async def _handle_ingress(self, payload):
            handler_started.set()
            try:
                await handler_release.wait()
                raise RuntimeError("late host ingress failure")
            finally:
                handler_finished.set()

    def client_factory(**kwargs):
        host_reader = _QueueReader()
        service_reader = _QueueReader()
        service = LateFailingIngressService()
        service_tasks.append(
            asyncio.create_task(service.serve(service_reader, _QueueWriter(host_reader)))
        )
        rpc = IsolatedFeatureClient(host_reader, _QueueWriter(service_reader))
        client = _SDKWireClient(rpc, dict(kwargs.get("config") or {}))
        clients.append(client)
        return client

    runtime = InstalledFeatureRuntime(
        class_name="HostIngressLateFailureFeature",
        entry_point="test_pkg.feature:HostIngressLateFailureFeature",
        distribution="test-pkg",
        runtime="isolated-venv",
        service="test-service",
    )
    agent = Mock(
        did=_TEST_AGENT_DID,
        storage_path=str(tmp_path / "agent" / "kestrel_prime.db"),
        features={},
    )
    agent.storage = _Storage()
    monkeypatch.setenv(
        "KESTREL_FEATURE_HOSTINGRESSLATEFAILUREFEATURE_BIN", "/bin/test-service"
    )
    feature = ProxyFeature(agent, runtime, client_factory=client_factory)
    ingress = None
    loop = asyncio.get_running_loop()
    previous_exception_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_contexts.append(context))

    try:
        await feature.initialize()
        ingress = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"update_id": 7})
        )
        await asyncio.wait_for(handler_started.wait(), timeout=1)

        ingress.cancel("first caller cancellation")
        await asyncio.sleep(0)
        # A second cancellation must not interrupt the drain or replace the
        # first cancellation's message.
        ingress.cancel("second caller cancellation")
        await asyncio.sleep(0)
        assert ingress.done() is False
        assert feature._traffic_gate._active == 1

        handler_release.set()
        await asyncio.wait_for(handler_finished.wait(), timeout=1)
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(ingress, timeout=1)
        assert cancelled.value.args == ("first caller cancellation",)
        assert cancelled.value.__context__ is None
        assert cancelled.value.__cause__ is None
        assert feature._traffic_gate._active == 0
        assert not [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and task.get_name().startswith("isolated-host-ingress:")
            and not task.done()
        ]
        await asyncio.sleep(0)
        assert not [
            context
            for context in loop_contexts
            if "HostIngressError exception in shielded future"
            in str(context.get("message", ""))
        ]
    finally:
        loop.set_exception_handler(previous_exception_handler)
        handler_release.set()
        if ingress is not None and not ingress.done():
            ingress.cancel()
            try:
                await ingress
            except asyncio.CancelledError:
                pass
        await feature.shutdown()
        await asyncio.gather(*service_tasks)


@pytest.mark.asyncio
async def test_terminal_shutdown_kills_permanently_wedged_subprocess_host_ingress(
    monkeypatch, tmp_path
):
    """Terminal shutdown fences a wedged ingress before waiting for its gate.

    The service blocks its own event loop in the callback and ignores SIGTERM.
    That makes graceful JSON-RPC shutdown and process termination both time
    out, so this exercises the production wrapper's final SIGKILL path rather
    than a cooperative service cancellation.
    """

    started_marker = tmp_path / "wedge-started"
    service_script = tmp_path / "wedged_ingress_service.py"
    service_script.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                "import asyncio",
                "import os",
                "import signal",
                "import time",
                "from pathlib import Path",
                "from kestrel_sdk.isolated_feature import IsolatedFeatureService",
                "",
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                "",
                "class WedgedIngressService(IsolatedFeatureService):",
                "    def __init__(self):",
                "        super().__init__(name='wedged-ingress', version='1.0.0')",
                "        self.register_host_ingress('telegram-webhook', self.ingress)",
                "",
                "    async def ingress(self, payload):",
                "        Path(os.environ['WEDGED_INGRESS_MARKER']).write_text('started')",
                "        time.sleep(60)",
                "        return {'accepted': True}",
                "",
                "async def main():",
                "    await WedgedIngressService().run_stdio()",
                "",
                "asyncio.run(main())",
                "",
            ]
        )
    )
    service_script.chmod(0o755)

    runtime = InstalledFeatureRuntime(
        class_name="WedgedIngressFeature",
        entry_point="test_pkg.feature:WedgedIngressFeature",
        distribution="test-pkg",
        runtime="isolated-venv",
        service="wedged-ingress-service",
    )
    agent = Mock(
        did=_TEST_AGENT_DID,
        storage_path=str(tmp_path / "agent" / "kestrel_prime.db"),
        features={},
    )
    agent.storage = _Storage()
    monkeypatch.setenv(
        "KESTREL_FEATURE_WEDGEDINGRESSFEATURE_BIN", str(service_script)
    )
    monkeypatch.setenv("WEDGED_INGRESS_MARKER", str(started_marker))
    feature = ProxyFeature(agent, runtime)
    ingress = shutdown = None
    wrapper = process = None

    async def wait_for_marker() -> None:
        while not started_marker.exists():
            await asyncio.sleep(0.01)

    try:
        await feature.initialize()
        wrapper = feature._client
        assert type(wrapper) is SubprocessIsolatedFeatureClient
        process = wrapper.process
        assert process is not None

        ingress = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"update_id": 7})
        )
        await asyncio.wait_for(wait_for_marker(), timeout=2)

        ingress.cancel("first caller cancellation")
        await asyncio.sleep(0)
        ingress.cancel("second caller cancellation")
        await asyncio.sleep(0)
        assert ingress.done() is False
        assert feature._traffic_gate._active == 1

        shutdown = asyncio.create_task(feature.shutdown())
        await asyncio.wait_for(shutdown, timeout=15)

        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(ingress, timeout=1)
        assert cancelled.value.args == ("first caller cancellation",)
        assert cancelled.value.__context__ is None
        assert feature._traffic_gate.sealed is True
        assert feature._traffic_gate._active == 0
        assert feature._client is None
        assert wrapper.process is None
        # SIGTERM was ignored by the child, so the SDK's bounded fallback had
        # to issue SIGKILL. This proves terminal cleanup reached child fencing
        # before waiting on the traffic gate.
        if sys.platform == "win32":
            assert process.returncode is not None
        else:
            assert process.returncode == -signal.SIGKILL
        with pytest.raises(Exception, match="host ingress is unavailable"):
            await feature.call_host_ingress("telegram-webhook", {"update_id": 8})
        assert not [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and task.get_name().startswith("isolated-host-ingress:")
            and not task.done()
        ]
        assert feature._supervision_task is None
    finally:
        for task in (shutdown, ingress):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if not feature._stopping:
            await feature.shutdown()
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_shutdown_cannot_orphan_subprocess_during_unhealthy_supervisor_stop(
    monkeypatch, tmp_path
):
    """The supervisor's owned stop reaches SIGKILL despite its cancellation.

    SDK 0.36.0 detaches ``process`` before awaiting graceful shutdown. This
    drives a real unhealthy-supervisor stop, then overlaps normal shutdown
    while that exact stop is in flight.  The service wedges its event loop and
    ignores SIGTERM, so observing the original process's SIGKILL proves this
    is not a second-stop success after the handle was lost.
    """

    started_marker = tmp_path / "supervisor-wedge-started"
    service_script = tmp_path / "supervisor_wedged_ingress_service.py"
    service_script.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                "import asyncio",
                "import os",
                "import signal",
                "import time",
                "from pathlib import Path",
                "from kestrel_sdk.isolated_feature import IsolatedFeatureService",
                "",
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                "",
                "class WedgedIngressService(IsolatedFeatureService):",
                "    def __init__(self):",
                "        super().__init__(name='supervisor-wedged-ingress', version='1.0.0')",
                "        self.register_host_ingress('telegram-webhook', self.ingress)",
                "",
                "    async def ingress(self, payload):",
                "        Path(os.environ['SUPERVISOR_WEDGED_INGRESS_MARKER']).write_text('started')",
                "        time.sleep(60)",
                "        return {'accepted': True}",
                "",
                "async def main():",
                "    await WedgedIngressService().run_stdio()",
                "",
                "asyncio.run(main())",
                "",
            ]
        )
    )
    service_script.chmod(0o755)

    runtime = InstalledFeatureRuntime(
        class_name="SupervisorWedgedIngressFeature",
        entry_point="test_pkg.feature:SupervisorWedgedIngressFeature",
        distribution="test-pkg",
        runtime="isolated-venv",
        service="supervisor-wedged-ingress-service",
    )
    agent = Mock(
        did=_TEST_AGENT_DID,
        storage_path=str(tmp_path / "agent" / "kestrel_prime.db"),
        features={},
    )
    agent.storage = _Storage()
    monkeypatch.setenv(
        "KESTREL_FEATURE_SUPERVISORWEDGEDINGRESSFEATURE_BIN", str(service_script)
    )
    monkeypatch.setenv("SUPERVISOR_WEDGED_INGRESS_MARKER", str(started_marker))
    # The first health probe waits for this bounded duration after its normal
    # one-second supervisor cadence.  Keep the integration regression quick
    # while retaining the real wrapper's graceful/terminate/kill behavior.
    monkeypatch.setattr(
        "kestrel_sovereign.features.isolated_runtime._HEALTH_PROBE_TIMEOUT",
        0.05,
    )

    stop_entered = asyncio.Event()
    original_stop = SubprocessIsolatedFeatureClient.stop

    async def observe_stop(self):
        stop_entered.set()
        await original_stop(self)

    monkeypatch.setattr(SubprocessIsolatedFeatureClient, "stop", observe_stop)
    feature = ProxyFeature(agent, runtime)
    ingress = shutdown = None
    wrapper = process = None

    async def wait_for_marker() -> None:
        while not started_marker.exists():
            await asyncio.sleep(0.01)

    try:
        await feature.initialize()
        wrapper = feature._client
        assert type(wrapper) is SubprocessIsolatedFeatureClient
        process = wrapper.process
        assert process is not None

        ingress = asyncio.create_task(
            feature.call_host_ingress("telegram-webhook", {"update_id": 7})
        )
        await asyncio.wait_for(wait_for_marker(), timeout=2)
        await asyncio.wait_for(stop_entered.wait(), timeout=3)

        # ``shutdown()`` cancels the supervisor while its owned child-stop
        # task is already inside SDK process retirement.  It must drain that
        # original task rather than interrupting it after process detachment.
        shutdown = asyncio.create_task(feature.shutdown())
        await asyncio.wait_for(shutdown, timeout=15)
        with pytest.raises((HostIngressError, asyncio.CancelledError)):
            await asyncio.wait_for(ingress, timeout=1)

        assert feature._client is None
        assert feature._traffic_gate.sealed is True
        assert wrapper.process is None
        if sys.platform == "win32":
            assert process.returncode is not None
        else:
            assert process.returncode == -signal.SIGKILL
        assert feature._supervision_task is None
    finally:
        for task in (shutdown, ingress):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if not feature._stopping:
            await feature.shutdown()
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_real_sdk_fenced_transition_promotion_failure_restores_durable_config(
    monkeypatch, tmp_path
):
    """A real SDK cancellation fence cannot start a pending-config child.

    The transition request is cancelled only after the service received it, so
    ``IsolatedFeatureClient`` takes its documented unknown-outcome path and
    fences the client for replacement.  The storage fault then rejects the
    candidate's *promotion* write, not its initial pending write. Recovery
    must restore the active config without ever initializing that pending
    candidate, then preserve the SDK cancellation outcome.
    """

    old_config = {"enabled": True, "token": "old-token-not-for-logs"}
    next_config = {"enabled": False, "token": "next-token-not-for-logs"}
    transition_started = asyncio.Event()
    release_transition = asyncio.Event()
    observed = []
    service_tasks = []
    clients = []

    class PromotionFailingStorage(_Storage):
        def __init__(self):
            super().__init__()
            self.cas_calls = 0

        async def compare_and_swap_node(self, node_id, expected, new_node):
            self.cas_calls += 1
            # The post-reconciliation lease renewal is a distinct CAS; fault
            # the promotion state itself rather than relying on call order.
            if (
                "pending_config" not in new_node.properties
                and new_node.properties.get("config") == next_config
            ):
                raise OSError("storage offline during promotion")
            return await super().compare_and_swap_node(node_id, expected, new_node)

    class FencedRecoveryService(IsolatedFeatureService):
        def __init__(self) -> None:
            super().__init__(name="fenced-recovery", version="1.0.0")
            self.advertise_config_transition()

        async def configure(self, config):
            observed.append(("initialize", dict(config)))

        async def on_config_transition(self, config):
            observed.append(("transition", dict(self.host_config), dict(config)))
            transition_started.set()
            await release_transition.wait()
            return ConfigTransitionResult.restart_required()

        async def on_shutdown(self):
            observed.append(("shutdown", dict(self.host_config)))
            return await super().on_shutdown()

    def client_factory(**kwargs):
        host_reader = _QueueReader()
        service_reader = _QueueReader()
        service = FencedRecoveryService()
        service_tasks.append(
            asyncio.create_task(service.serve(service_reader, _QueueWriter(host_reader)))
        )
        rpc = IsolatedFeatureClient(host_reader, _QueueWriter(service_reader))
        client = _SDKWireClient(rpc, dict(kwargs.get("config") or {}))
        clients.append(client)
        return client

    runtime = InstalledFeatureRuntime(
        class_name="FencedRecoveryFeature",
        entry_point="test_pkg.feature:FencedRecoveryFeature",
        distribution="test-pkg",
        runtime="isolated-venv",
        service="test-service",
    )
    agent = Mock(did=_TEST_AGENT_DID, storage_path=str(tmp_path / "agent" / "kestrel_prime.db"), features={})
    agent.storage = PromotionFailingStorage()
    monkeypatch.setenv("KESTREL_FEATURE_FENCEDRECOVERYFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(agent, runtime, client_factory=client_factory)
    fresh = None

    try:
        await feature.persist_config(old_config)
        await feature.initialize()
        update = asyncio.create_task(feature.set_config(next_config))
        await asyncio.wait_for(transition_started.wait(), timeout=1)

        # The request has reached the service.  Cancelling the host waiter
        # triggers the SDK's unknown-outcome fence before the service is allowed
        # to return its late response and process the replacement shutdown.
        update.cancel()

        async def _wait_for_sdk_fence() -> None:
            while not clients[0].replacement_required:
                await asyncio.sleep(0)

        await asyncio.wait_for(_wait_for_sdk_fence(), timeout=1)
        assert clients[0].replacement_required is True
        release_transition.set()

        with pytest.raises(asyncio.CancelledError):
            await update

        assert observed == [
            ("initialize", old_config),
            ("transition", old_config, next_config),
            ("shutdown", old_config),
            ("initialize", old_config),
        ]
        assert clients[0].stopped is True
        assert clients[1].stopped is False
        assert feature._host_config == old_config
        assert feature._client is clients[1]
        properties = agent.storage.nodes[_config_node_id("FencedRecoveryFeature")].properties
        assert properties["config"] == old_config
        assert "pending_config" not in properties
        assert "_isolated_pending_generation" not in properties
        assert feature._supervision_task is not None
        assert feature._supervision_task.done() is False

        # A fresh host proxy performs the restart handshake with the durable
        # active config; the rejected candidate was removed during cleanup.
        fresh = ProxyFeature(agent, runtime, client_factory=client_factory)
        await fresh.initialize()
        assert fresh._host_config == old_config
        assert observed[-1] == ("initialize", old_config)
    finally:
        release_transition.set()
        if fresh is not None:
            await fresh.shutdown()
        await feature.shutdown()
        await asyncio.gather(*service_tasks)
