"""Real SDK config-transition exchange through the isolated host proxy."""

import asyncio
import json
import sys
from unittest.mock import Mock

import pytest

from kestrel_sdk.isolated_feature import (
    ConfigTransitionResult,
    IsolatedFeatureClient,
    IsolatedFeatureService,
)

from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sovereign.features.isolated_runtime import ProxyFeature


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
