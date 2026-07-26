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

    async def add_node(self, node) -> None:
        self.nodes[node.node_id] = node

    async def get_node(self, node_id):
        return self.nodes.get(node_id)


class _SDKWireClient:
    """Proxy-compatible lifecycle adapter over a real SDK JSON-RPC client."""

    def __init__(self, rpc: IsolatedFeatureClient, config: dict[str, object]):
        self._rpc = rpc
        self._config = config
        self.stopped = False

    @property
    def supports_config_transition(self) -> bool:
        return self._rpc.supports_config_transition

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
    agent = Mock(storage_path=str(tmp_path / "agent" / "kestrel_prime.db"), features={})
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
    agent = Mock(storage_path=str(tmp_path / "agent" / "kestrel_prime.db"), features={})
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
