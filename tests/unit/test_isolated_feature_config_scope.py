"""DID-scoped durable config and pre-hook lease-fencing regressions."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sovereign.features import isolated_runtime
from kestrel_sovereign.features.isolated_runtime import ProxyFeature
from kestrel_sovereign.storage.async_graph_store import GraphNode


_FEATURE_NAME = "ScopedFeature"
_LEGACY_NODE_ID = f"feature_config:{_FEATURE_NAME}"


def _scoped_node_id(did: str) -> str:
    return f"feature_config:v2:{did}:{_FEATURE_NAME}"


def _runtime() -> InstalledFeatureRuntime:
    return InstalledFeatureRuntime(
        class_name=_FEATURE_NAME,
        entry_point="test_pkg.feature:ScopedFeature",
        distribution="test-pkg",
        runtime="isolated-venv",
        service="test-service",
    )


class _GlobalGraph:
    """Global-ID graph backend with production-shaped owner visibility.

    ``AsyncGraphStore`` has one global ``graph_nodes.node_id`` namespace but
    returns only rows carrying the calling agent's ownership witness.  This
    compact double keeps both properties so proxy tests cannot accidentally
    pass against a per-agent dictionary.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.owners: dict[str, str] = {}
        self._cas_lock = asyncio.Lock()
        self.legacy_reads = 0

    def seed(self, owner: str, node: GraphNode) -> None:
        self.nodes[node.node_id] = node
        self.owners[node.node_id] = owner


class _AgentScopedStorage:
    """One agent-bound view of :class:`_GlobalGraph`."""

    def __init__(self, graph: _GlobalGraph, agent_id: str) -> None:
        self._graph = graph
        self.agent_id = agent_id
        self.cas_node_ids: list[str] = []

    async def add_node(self, node: GraphNode) -> None:
        async with self._graph._cas_lock:
            owner = self._graph.owners.get(node.node_id)
            if owner is not None and owner != self.agent_id:
                raise RuntimeError("foreign graph node is not writable")
            self._graph.nodes[node.node_id] = node
            self._graph.owners[node.node_id] = self.agent_id

    async def get_node(self, node_id: str) -> GraphNode | None:
        if node_id == _LEGACY_NODE_ID:
            self._graph.legacy_reads += 1
        if self._graph.owners.get(node_id) != self.agent_id:
            return None
        return self._graph.nodes.get(node_id)

    async def compare_and_swap_node(
        self,
        node_id: str,
        expected: dict[str, Any] | None,
        new_node: GraphNode,
    ) -> str:
        self.cas_node_ids.append(node_id)

        async with self._graph._cas_lock:
            existing = self._graph.nodes.get(node_id)
            owner = self._graph.owners.get(node_id)
            if expected is None:
                if existing is None:
                    self._graph.nodes[node_id] = new_node
                    self._graph.owners[node_id] = self.agent_id
                    return "swapped"
                return "predicate_failed" if owner == self.agent_id else "not_found"
            if existing is None or owner != self.agent_id:
                return "not_found"
            if existing.properties != expected:
                return "predicate_failed"
            self._graph.nodes[node_id] = new_node
            return "swapped"


def _agent(did: str | None, storage: _AgentScopedStorage, tmp_path) -> SimpleNamespace:
    path_component = did.replace(":", "_") if isinstance(did, str) else "invalid"
    return SimpleNamespace(
        did=did,
        storage=storage,
        storage_path=str(tmp_path / path_component / "agent.db"),
        features={},
    )


class _NoopClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.stopped = False

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        self.stopped = True

    async def health(self) -> bool:
        return True

    async def list_tools(self) -> list[dict[str, Any]]:
        return []

    def on_event(self, _handler: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_did_scoped_configs_initialize_and_configure_independently(
    monkeypatch, tmp_path
):
    """Global graph IDs cannot make two DID-bound proxies share config/secrets."""

    did_a = "did:test:config-agent-a"
    did_b = "did:test:config-agent-b"
    graph = _GlobalGraph()
    feature_a = ProxyFeature(
        _agent(did_a, _AgentScopedStorage(graph, did_a), tmp_path),
        _runtime(),
        client_factory=_NoopClient,
    )
    feature_b = ProxyFeature(
        _agent(did_b, _AgentScopedStorage(graph, did_b), tmp_path),
        _runtime(),
        client_factory=_NoopClient,
    )
    monkeypatch.setenv("KESTREL_FEATURE_SCOPEDFEATURE_BIN", "/bin/test-service")

    try:
        await feature_a.initialize()
        await feature_b.initialize()
        await feature_a.set_config({"token": "agent-a-secret", "enabled": True})
        await feature_b.set_config({"token": "agent-b-secret", "enabled": False})

        assert await feature_a.get_config() == {
            "token": "agent-a-secret",
            "enabled": True,
        }
        assert await feature_b.get_config() == {
            "token": "agent-b-secret",
            "enabled": False,
        }
        assert _scoped_node_id(did_a) in graph.nodes
        assert _scoped_node_id(did_b) in graph.nodes
        assert graph.owners[_scoped_node_id(did_a)] == did_a
        assert graph.owners[_scoped_node_id(did_b)] == did_b
        assert await feature_a.agent.storage.get_node(_scoped_node_id(did_b)) is None
        assert await feature_b.agent.storage.get_node(_scoped_node_id(did_a)) is None
    finally:
        await feature_b.shutdown()
        await feature_a.shutdown()


@pytest.mark.asyncio
async def test_visible_legacy_config_is_adopted_in_place_for_its_owner_only(tmp_path):
    """New replicas keep a visible old-replica authority instead of copying it."""

    owner_did = "did:test:legacy-owner"
    other_did = "did:test:legacy-other"
    legacy_config = {"token": "legacy-owner-secret", "enabled": True}
    graph = _GlobalGraph()
    graph.seed(
        owner_did,
        GraphNode(
            node_id=_LEGACY_NODE_ID,
            node_type="feature_config",
            label=f"{_FEATURE_NAME} config",
            properties={"config": dict(legacy_config)},
        ),
    )
    first = ProxyFeature(
        _agent(owner_did, _AgentScopedStorage(graph, owner_did), tmp_path), _runtime()
    )
    second = ProxyFeature(
        _agent(owner_did, _AgentScopedStorage(graph, owner_did), tmp_path), _runtime()
    )
    foreign = ProxyFeature(
        _agent(other_did, _AgentScopedStorage(graph, other_did), tmp_path), _runtime()
    )
    first_config, second_config = await asyncio.gather(
        first.load_persisted_config(raise_on_error=True),
        second.load_persisted_config(raise_on_error=True),
    )

    assert first_config == legacy_config
    assert second_config == legacy_config
    assert first._resolved_config_node_id == _LEGACY_NODE_ID
    assert second._resolved_config_node_id == _LEGACY_NODE_ID
    assert _scoped_node_id(owner_did) not in graph.nodes

    # The Feature-compatible persistence helper follows the same adopted row;
    # it cannot silently create a second scoped authority beside old replicas.
    await first.persist_config({"token": "base-persist", "enabled": True})
    assert graph.nodes[_LEGACY_NODE_ID].properties == {
        "config": {"token": "base-persist", "enabled": True}
    }
    assert _scoped_node_id(owner_did) not in graph.nodes

    # Cached identity does not mean cached data: an old replica can continue
    # promoting the legacy row, so every new-proxy read remains on that row.
    reads_before_repeat = graph.legacy_reads
    assert await first.load_persisted_config(raise_on_error=True) == {
        "token": "base-persist",
        "enabled": True,
    }
    assert graph.legacy_reads > reads_before_repeat
    graph.nodes[_LEGACY_NODE_ID] = GraphNode(
        node_id=_LEGACY_NODE_ID,
        node_type="feature_config",
        label=f"{_FEATURE_NAME} config",
        properties={"config": {"token": "old-replica-update", "enabled": False}},
    )
    assert await first.load_persisted_config(raise_on_error=True) == {
        "token": "old-replica-update",
        "enabled": False,
    }
    assert _scoped_node_id(owner_did) not in graph.nodes

    # A second bound store cannot see (or migrate) the owner's legacy secret.
    assert await foreign.load_persisted_config(raise_on_error=True) is None
    assert _scoped_node_id(other_did) not in graph.nodes
    await foreign.set_config({"token": "other-secret", "enabled": False})
    assert await foreign.get_config() == {"token": "other-secret", "enabled": False}
    assert graph.nodes[_scoped_node_id(other_did)].properties["config"] == {
        "token": "other-secret",
        "enabled": False,
    }


@pytest.mark.asyncio
async def test_visible_legacy_config_wins_over_scoped_during_rolling_overlap(tmp_path):
    """Mixed-version overlap converges on legacy instead of split authority."""

    did = "did:test:scoped-winner"
    graph = _GlobalGraph()
    graph.seed(
        did,
        GraphNode(
            node_id=_LEGACY_NODE_ID,
            node_type="feature_config",
            label=f"{_FEATURE_NAME} config",
            properties={"config": {"token": "legacy-secret"}},
        ),
    )
    graph.seed(
        did,
        GraphNode(
            node_id=_scoped_node_id(did),
            node_type="feature_config",
            label=f"{_FEATURE_NAME} config",
            properties={"config": {}},
        ),
    )
    feature = ProxyFeature(
        _agent(did, _AgentScopedStorage(graph, did), tmp_path), _runtime()
    )

    assert await feature.load_persisted_config(raise_on_error=True) == {
        "token": "legacy-secret"
    }
    assert feature._resolved_config_node_id == _LEGACY_NODE_ID
    assert graph.legacy_reads > 0


@pytest.mark.asyncio
async def test_scoped_row_appearing_during_resolution_does_not_displace_legacy(tmp_path):
    """A newly visible scoped row cannot displace rolling legacy authority."""

    did = "did:test:concurrent-scope"
    graph = _GlobalGraph()
    graph.seed(
        did,
        GraphNode(
            node_id=_LEGACY_NODE_ID,
            node_type="feature_config",
            label=f"{_FEATURE_NAME} config",
            properties={"config": {"revision": "legacy"}},
        ),
    )

    class _ScopedAppearsOnLegacyRead(_AgentScopedStorage):
        def __init__(self) -> None:
            super().__init__(graph, did)
            self._created_scoped = False

        async def get_node(self, node_id: str) -> GraphNode | None:
            node = await super().get_node(node_id)
            if node_id == _LEGACY_NODE_ID and not self._created_scoped:
                self._created_scoped = True
                graph.seed(
                    did,
                    GraphNode(
                        node_id=_scoped_node_id(did),
                        node_type="feature_config",
                        label=f"{_FEATURE_NAME} config",
                        properties={"config": {"revision": "scoped"}},
                    ),
                )
            return node

    feature = ProxyFeature(_agent(did, _ScopedAppearsOnLegacyRead(), tmp_path), _runtime())

    assert await feature.load_persisted_config(raise_on_error=True) == {
        "revision": "legacy"
    }
    assert feature._resolved_config_node_id == _LEGACY_NODE_ID


@pytest.mark.asyncio
async def test_empty_scoped_config_is_authoritative_without_visible_legacy(tmp_path):
    """A scoped empty config is valid after legacy authority is absent."""

    did = "did:test:empty-scoped-authority"
    graph = _GlobalGraph()
    graph.seed(
        did,
        GraphNode(
            node_id=_scoped_node_id(did),
            node_type="feature_config",
            label=f"{_FEATURE_NAME} config",
            properties={"config": {}},
        ),
    )
    feature = ProxyFeature(
        _agent(did, _AgentScopedStorage(graph, did), tmp_path), _runtime()
    )

    assert await feature.load_persisted_config(raise_on_error=True) == {}
    assert feature._resolved_config_node_id == _scoped_node_id(did)
    assert graph.legacy_reads > 0


@pytest.mark.asyncio
async def test_cached_scoped_authority_fences_legacy_before_new_stage_or_hook(
    monkeypatch, tmp_path
):
    """A late old writer is reconciled before this proxy can stage or hook."""

    did = "did:test:late-legacy-before-stage"
    legacy_config = {"revision": "old-binary-winner"}
    graph = _GlobalGraph()
    storage = _AgentScopedStorage(graph, did)
    clients: list[_NoopClient] = []

    class _HookClient(_NoopClient):
        supports_config_transition = True

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.prepare_calls = 0

        async def prepare_config_transition(self, _config: dict[str, Any]) -> Any:
            self.prepare_calls += 1
            raise AssertionError("legacy authority must fence the hook")

    def client_factory(**kwargs: Any) -> _HookClient:
        client = _HookClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setenv("KESTREL_FEATURE_SCOPEDFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(
        _agent(did, storage, tmp_path), _runtime(), client_factory=client_factory
    )
    try:
        # The first new-proxy read sees neither key and caches the scoped
        # candidate. A still-running old binary writes the pre-scoping key
        # before this proxy starts its requested update.
        assert await feature.load_persisted_config(raise_on_error=True) is None
        assert feature._resolved_config_node_id == _scoped_node_id(did)
        await feature.initialize()
        old_storage = _AgentScopedStorage(graph, did)
        old_active = {"revision": "old-binary-active"}
        await old_storage.add_node(
            GraphNode(
                node_id=_LEGACY_NODE_ID,
                node_type="feature_config",
                label=f"{_FEATURE_NAME} config",
                properties={"config": dict(old_active)},
            )
        )
        old_before = await old_storage.get_node(_LEGACY_NODE_ID)
        assert old_before is not None
        old_staged = dict(old_before.properties)
        old_staged.update(
            {
                "pending_config": dict(legacy_config),
                "_isolated_pending_generation": "old-binary-generation",
                "_isolated_pending_owner": "old-binary",
                "_isolated_pending_lease_expires_at": "2026-07-26T12:01:00+00:00",
            }
        )
        assert (
            await old_storage.compare_and_swap_node(
                _LEGACY_NODE_ID,
                old_before.properties,
                GraphNode(
                    node_id=_LEGACY_NODE_ID,
                    node_type="feature_config",
                    label=f"{_FEATURE_NAME} config",
                    properties=old_staged,
                ),
            )
            == "swapped"
        )
        old_promoted = dict(old_staged)
        old_promoted["config"] = dict(legacy_config)
        old_promoted.pop("pending_config")
        old_promoted.pop("_isolated_pending_generation")
        old_promoted.pop("_isolated_pending_owner")
        old_promoted.pop("_isolated_pending_lease_expires_at")
        assert (
            await old_storage.compare_and_swap_node(
                _LEGACY_NODE_ID,
                old_staged,
                GraphNode(
                    node_id=_LEGACY_NODE_ID,
                    node_type="feature_config",
                    label=f"{_FEATURE_NAME} config",
                    properties=old_promoted,
                ),
            )
            == "swapped"
        )

        with pytest.raises(RuntimeError, match="legacy config authority became visible"):
            await feature.set_config({"revision": "new-candidate"})

        # No scoped CAS or live hook ran. The stale child was replaced from the
        # visible old-binary authority while the finite traffic gate was closed.
        assert storage.cas_node_ids == []
        assert all(getattr(client, "prepare_calls", 0) == 0 for client in clients)
        assert clients[0].stopped is True
        assert clients[1].kwargs["config"] == legacy_config
        assert feature._host_config == legacy_config
        assert await feature.get_config() == legacy_config
        assert feature._traffic_gate.sealed is False
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_clientless_authority_change_caches_legacy_until_explicit_initialize(
    monkeypatch, tmp_path
):
    """A pre-initialize PATCH must not start a child to reconcile old authority."""

    did = "did:test:clientless-authority-change"
    legacy_config = {"revision": "old-binary-winner"}
    graph = _GlobalGraph()
    storage = _AgentScopedStorage(graph, did)
    factory_calls = 0
    started: list[_NoopClient] = []
    hook_calls = 0

    class _CountingClient(_NoopClient):
        supports_config_transition = True

        async def start(self) -> None:
            started.append(self)

        async def prepare_config_transition(self, _config: dict[str, Any]) -> Any:
            nonlocal hook_calls
            hook_calls += 1
            return None

    def client_factory(**kwargs: Any) -> _CountingClient:
        nonlocal factory_calls
        factory_calls += 1
        return _CountingClient(**kwargs)

    monkeypatch.setenv("KESTREL_FEATURE_SCOPEDFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(
        _agent(did, storage, tmp_path), _runtime(), client_factory=client_factory
    )
    try:
        # A normal read before old replicas publish config caches the scoped
        # no-row authority and the empty active config.
        assert await feature.get_config() == {}
        assert feature._resolved_config_node_id == _scoped_node_id(did)

        graph.seed(
            did,
            GraphNode(
                node_id=_LEGACY_NODE_ID,
                node_type="feature_config",
                label=f"{_FEATURE_NAME} config",
                properties={"config": dict(legacy_config)},
            ),
        )

        with pytest.raises(RuntimeError, match="legacy config authority became visible"):
            await feature.set_config({"revision": "new-candidate"})

        # The rolling-authority retry remains visible, but config-only
        # set_config has not created or published an external child.
        assert factory_calls == 0
        assert started == []
        assert hook_calls == 0
        assert storage.cas_node_ids == []
        assert feature._client is None
        assert feature.get_tools() == []
        assert feature.get_router() is None
        assert feature._traffic_gate.closed is False
        assert feature._traffic_gate.sealed is False
        assert feature._host_config == legacy_config
        assert await feature.get_config() == legacy_config

        # Explicit initialization is the first operation allowed to launch a
        # child, and it must receive the adopted legacy config exactly once.
        await feature.initialize()
        assert factory_calls == 1
        assert len(started) == 1
        assert hook_calls == 0
        assert started[0].kwargs["config"] == legacy_config
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_read_only_cached_scoped_authority_adopts_legacy_without_start(
    monkeypatch, tmp_path
):
    """GET and later startup safely adopt legacy before a transition is pinned."""

    did = "did:test:read-only-authority-change"
    legacy_config = {"revision": "old-binary-winner"}
    graph = _GlobalGraph()
    storage = _AgentScopedStorage(graph, did)
    factory_calls = 0
    started: list[_NoopClient] = []

    class _CountingClient(_NoopClient):
        async def start(self) -> None:
            started.append(self)

    def client_factory(**kwargs: Any) -> _CountingClient:
        nonlocal factory_calls
        factory_calls += 1
        return _CountingClient(**kwargs)

    monkeypatch.setenv("KESTREL_FEATURE_SCOPEDFEATURE_BIN", "/bin/test-service")
    feature = ProxyFeature(
        _agent(did, storage, tmp_path), _runtime(), client_factory=client_factory
    )
    try:
        assert await feature.get_config() == {}
        assert feature._resolved_config_node_id == _scoped_node_id(did)

        graph.seed(
            did,
            GraphNode(
                node_id=_LEGACY_NODE_ID,
                node_type="feature_config",
                label=f"{_FEATURE_NAME} config",
                properties={"config": dict(legacy_config)},
            ),
        )

        # A read is not a lifecycle transition: it adopts the visible legacy
        # authority without emitting the one-time rolling-authority error or
        # creating a client.
        assert await feature.get_config() == legacy_config
        assert feature._resolved_config_node_id == _LEGACY_NODE_ID
        assert feature._host_config == legacy_config
        assert factory_calls == 0
        assert started == []
        assert feature._client is None
        assert feature.get_tools() == []

        await feature.initialize()
        assert factory_calls == 1
        assert len(started) == 1
        assert started[0].kwargs["config"] == legacy_config
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_legacy_after_scoped_stage_quarantines_before_hook_and_future_proxy_uses_legacy(
    monkeypatch, tmp_path
):
    """A post-CAS old writer leaves an orphan candidate but cannot reach a hook."""

    did = "did:test:late-legacy-after-stage"
    legacy_config = {"revision": "old-binary-winner"}
    candidate_config = {"revision": "new-candidate"}
    graph = _GlobalGraph()
    clients: list[_NoopClient] = []

    class _LegacyAfterStageStorage(_AgentScopedStorage):
        def __init__(self) -> None:
            super().__init__(graph, did)
            self._legacy_written = False

        async def compare_and_swap_node(self, node_id, expected, new_node):
            result = await super().compare_and_swap_node(node_id, expected, new_node)
            if (
                result == "swapped"
                and not self._legacy_written
                and node_id == _scoped_node_id(did)
                and "pending_config" in new_node.properties
            ):
                self._legacy_written = True
                graph.seed(
                    did,
                    GraphNode(
                        node_id=_LEGACY_NODE_ID,
                        node_type="feature_config",
                        label=f"{_FEATURE_NAME} config",
                        properties={"config": dict(legacy_config)},
                    ),
                )
            return result

    class _HookClient(_NoopClient):
        supports_config_transition = True

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.prepare_calls = 0

        async def prepare_config_transition(self, _config: dict[str, Any]) -> Any:
            self.prepare_calls += 1
            raise AssertionError("post-stage legacy authority must fence the hook")

    def client_factory(**kwargs: Any) -> _HookClient:
        client = _HookClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setenv("KESTREL_FEATURE_SCOPEDFEATURE_BIN", "/bin/test-service")
    storage = _LegacyAfterStageStorage()
    feature = ProxyFeature(
        _agent(did, storage, tmp_path), _runtime(), client_factory=client_factory
    )
    try:
        await feature.initialize()

        with pytest.raises(RuntimeError, match="legacy config authority became visible"):
            await feature.set_config(candidate_config)

        # The scoped compare-and-create may already have committed when the old
        # writer publishes legacy. The proxy does not guess across keys: it
        # quarantines before lease renewal/hook/traffic and leaves cleanup for
        # the operator after old replicas have drained.
        scoped_properties = graph.nodes[_scoped_node_id(did)].properties
        assert scoped_properties["pending_config"] == candidate_config
        assert graph.nodes[_LEGACY_NODE_ID].properties["config"] == legacy_config
        assert storage.cas_node_ids == [_scoped_node_id(did)]
        assert clients[0].prepare_calls == 0
        assert clients[0].stopped is True
        assert feature._client is None
        assert feature._traffic_gate.sealed is True
        assert await feature.get_config() == legacy_config

        future_proxy = ProxyFeature(
            _agent(did, _AgentScopedStorage(graph, did), tmp_path), _runtime()
        )
        assert await future_proxy.load_persisted_config(raise_on_error=True) == legacy_config
        assert future_proxy._resolved_config_node_id == _LEGACY_NODE_ID
    finally:
        await feature.shutdown()


@pytest.mark.asyncio
async def test_persist_config_cas_preserves_metadata_and_never_clobbers_new_stage(
    tmp_path,
):
    """The boot compatibility helper cannot erase another replica's metadata."""

    did = "did:test:compatibility-cas"
    graph = _GlobalGraph()
    stable_config = {"token": "preserved-secret", "enabled": True}
    requested_config = {"token": "replacement-secret", "enabled": False}
    winner_pending = {"token": "winner-secret", "enabled": True}
    initial_properties = {
        "config": {"token": "initial-secret", "enabled": True},
        "_isolated_config_generation": "active-generation",
        "unrelated_metadata": "must-survive",
    }
    graph.seed(
        did,
        GraphNode(
            node_id=_scoped_node_id(did),
            node_type="feature_config",
            label=f"{_FEATURE_NAME} config",
            properties=dict(initial_properties),
        ),
    )
    other_storage = _AgentScopedStorage(graph, did)

    class _InterleavingStorage(_AgentScopedStorage):
        def __init__(self) -> None:
            super().__init__(graph, did)
            self._interleaved = False

        async def compare_and_swap_node(self, node_id, expected, new_node):
            if (
                not self._interleaved
                and node_id == _scoped_node_id(did)
                and new_node.properties.get("config") == requested_config
            ):
                self._interleaved = True
                winner_properties = dict(graph.nodes[node_id].properties)
                winner_properties.update(
                    {
                        "pending_config": dict(winner_pending),
                        "_isolated_pending_generation": "winner-generation",
                        "_isolated_pending_owner": "other-replica",
                        "_isolated_pending_lease_expires_at": (
                            "2026-07-26T12:01:00+00:00"
                        ),
                    }
                )
                assert (
                    await other_storage.compare_and_swap_node(
                        node_id,
                        expected,
                        GraphNode(
                            node_id=node_id,
                            node_type="feature_config",
                            label=f"{_FEATURE_NAME} config",
                            properties=winner_properties,
                        ),
                    )
                    == "swapped"
                )
            return await super().compare_and_swap_node(node_id, expected, new_node)

    storage = _InterleavingStorage()
    feature = ProxyFeature(_agent(did, storage, tmp_path), _runtime())

    # A non-contended compatibility write preserves the active generation,
    # unrelated metadata, and the full secret-bearing effective config.
    await feature.persist_config(stable_config)
    assert graph.nodes[_scoped_node_id(did)].properties == {
        "config": stable_config,
        "_isolated_config_generation": "active-generation",
        "unrelated_metadata": "must-survive",
    }

    # Another hosted replica stages after this helper read but before its CAS.
    # The helper returns best-effort without replacing either active or pending
    # state with its stale requested config.
    await feature.persist_config(requested_config)
    properties = graph.nodes[_scoped_node_id(did)].properties
    assert properties["config"] == stable_config
    assert properties["pending_config"] == winner_pending
    assert properties["_isolated_pending_generation"] == "winner-generation"
    assert properties["_isolated_pending_owner"] == "other-replica"
    assert properties["_isolated_config_generation"] == "active-generation"
    assert properties["unrelated_metadata"] == "must-survive"


@pytest.mark.asyncio
async def test_hidden_foreign_scoped_id_collision_fails_closed(tmp_path):
    """A global-ID collision is neither readable nor writable by this DID."""

    did = "did:test:collision-owner"
    foreign_did = "did:test:collision-foreign"
    graph = _GlobalGraph()
    graph.seed(
        foreign_did,
        GraphNode(
            node_id=_scoped_node_id(did),
            node_type="feature_config",
            label=f"{_FEATURE_NAME} config",
            properties={"config": {"token": "foreign-secret"}},
        ),
    )
    storage = _AgentScopedStorage(graph, did)
    feature = ProxyFeature(_agent(did, storage, tmp_path), _runtime())

    with pytest.raises(RuntimeError, match="conflicts with a newer durable state"):
        await feature.set_config({"token": "local-secret"})

    assert await storage.get_node(_scoped_node_id(did)) is None
    assert graph.nodes[_scoped_node_id(did)].properties["config"] == {
        "token": "foreign-secret"
    }


@pytest.mark.asyncio
async def test_live_old_replica_and_new_proxy_share_adopted_legacy_cas_authority(
    tmp_path,
):
    """A rolling upgrade never forks an old legacy writer into a scoped row."""

    did = "did:test:rolling-owner"
    initial_config = {"revision": "initial"}
    old_replica_config = {"revision": "old-replica-winner"}
    graph = _GlobalGraph()
    graph.seed(
        did,
        GraphNode(
            node_id=_LEGACY_NODE_ID,
            node_type="feature_config",
            label=f"{_FEATURE_NAME} config",
            properties={"config": dict(initial_config)},
        ),
    )
    old_storage = _AgentScopedStorage(graph, did)
    stale_stage_result: str | None = None

    async def old_replica_stage_and_promote() -> None:
        """Model the still-live pre-scoping binary's legacy lease lifecycle."""

        before = await old_storage.get_node(_LEGACY_NODE_ID)
        assert before is not None
        staged = dict(before.properties)
        staged.update(
            {
                "pending_config": dict(old_replica_config),
                "_isolated_pending_generation": "old-generation",
                "_isolated_pending_owner": "old-replica",
                "_isolated_pending_lease_expires_at": "2026-07-26T12:01:00+00:00",
            }
        )
        staged_node = GraphNode(
            node_id=_LEGACY_NODE_ID,
            node_type="feature_config",
            label=f"{_FEATURE_NAME} config",
            properties=staged,
        )
        assert (
            await old_storage.compare_and_swap_node(
                _LEGACY_NODE_ID, before.properties, staged_node
            )
            == "swapped"
        )
        promoted = dict(staged)
        promoted["config"] = dict(old_replica_config)
        promoted["_isolated_config_generation"] = "old-generation"
        promoted.pop("pending_config")
        promoted.pop("_isolated_pending_generation")
        promoted.pop("_isolated_pending_owner")
        promoted.pop("_isolated_pending_lease_expires_at")
        assert (
            await old_storage.compare_and_swap_node(
                _LEGACY_NODE_ID,
                staged,
                GraphNode(
                    node_id=_LEGACY_NODE_ID,
                    node_type="feature_config",
                    label=f"{_FEATURE_NAME} config",
                    properties=promoted,
                ),
            )
            == "swapped"
        )

    class _NewReplicaStorage(_AgentScopedStorage):
        def __init__(self) -> None:
            super().__init__(graph, did)
            self._interleaved = False

        async def compare_and_swap_node(self, node_id, expected, new_node):
            nonlocal stale_stage_result
            if (
                not self._interleaved
                and node_id == _LEGACY_NODE_ID
                and expected == {"config": initial_config}
                and "pending_config" in new_node.properties
            ):
                self._interleaved = True
                await old_replica_stage_and_promote()
                stale_stage_result = await super().compare_and_swap_node(
                    node_id, expected, new_node
                )
                return stale_stage_result
            return await super().compare_and_swap_node(node_id, expected, new_node)

    new_storage = _NewReplicaStorage()
    feature = ProxyFeature(_agent(did, new_storage, tmp_path), _runtime())

    # Resolution happens before the old process advances its legacy state.
    assert await feature.load_persisted_config(raise_on_error=True) == initial_config
    assert feature._resolved_config_node_id == _LEGACY_NODE_ID

    with pytest.raises(RuntimeError, match="conflicts with a newer durable state"):
        await feature.set_config({"revision": "new-stale-loser"})

    assert stale_stage_result == "predicate_failed"
    assert await feature.get_config() == old_replica_config
    assert _scoped_node_id(did) not in graph.nodes

    # A fresh CAS transition by the new binary still targets the adopted old
    # row, then promotes normally. No second durable authority is created.
    await feature.set_config({"revision": "new-proxy-winner"})
    assert await feature.get_config() == {"revision": "new-proxy-winner"}
    assert new_storage.cas_node_ids
    assert set(new_storage.cas_node_ids) == {_LEGACY_NODE_ID}
    assert _scoped_node_id(did) not in graph.nodes


@pytest.mark.asyncio
async def test_durable_config_rejects_missing_or_malformed_agent_did(tmp_path):
    """A process/user-global substitute is never accepted as config identity."""

    did = "did:test:storage-owner"
    storage = _AgentScopedStorage(_GlobalGraph(), did)
    for invalid_did in (None, "", "agent-name", "did:test:contains space"):
        feature = ProxyFeature(
            _agent(invalid_did, storage, tmp_path),  # type: ignore[arg-type]
            _runtime(),
        )
        with pytest.raises(RuntimeError, match="failed to load persisted config"):
            await feature.set_config({"enabled": True})


@pytest.mark.asyncio
async def test_lost_lease_during_reconciliation_never_invokes_transition_hook(
    monkeypatch, tmp_path
):
    """A stale replica is fenced before its candidate can touch a live hook."""

    did = "did:test:lease-owner"
    old_config = {"revision": "old"}
    active_config = {"revision": "active"}
    lost_candidate = {"revision": "lost-candidate"}
    winner_config = {"revision": "winner"}
    graph = _GlobalGraph()
    storage = _AgentScopedStorage(graph, did)
    clock = [datetime(2026, 7, 26, 12, tzinfo=timezone.utc)]
    stale_stop_started = asyncio.Event()
    release_stale_stop = asyncio.Event()
    clients: list[_NoopClient] = []

    class _BlockingStaleClient(_NoopClient):
        supports_config_transition = True

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.prepare_calls = 0

        async def stop(self) -> None:
            stale_stop_started.set()
            await release_stale_stop.wait()
            await super().stop()

        async def prepare_config_transition(self, _config: dict[str, Any]) -> Any:
            self.prepare_calls += 1
            raise AssertionError("lost lease must fence the hook before invocation")

    class _PostReconcileClient(_NoopClient):
        """The replacement exposes the same transition capability as production."""

        supports_config_transition = True

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.prepare_calls = 0

        async def prepare_config_transition(self, _config: dict[str, Any]) -> Any:
            self.prepare_calls += 1
            raise AssertionError("lost lease must fence the replacement hook too")

    def first_factory(**kwargs: Any) -> _NoopClient:
        client: _NoopClient
        if not clients:
            client = _BlockingStaleClient(**kwargs)
        else:
            client = _PostReconcileClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setenv("KESTREL_FEATURE_SCOPEDFEATURE_BIN", "/bin/test-service")
    monkeypatch.setattr(isolated_runtime, "_utc_now", lambda: clock[0])
    monkeypatch.setattr(
        isolated_runtime, "_PENDING_CONFIG_LEASE_TTL", timedelta(milliseconds=50)
    )
    monkeypatch.setattr(
        isolated_runtime, "_PENDING_CONFIG_CLOCK_SKEW", timedelta(0)
    )

    seed = ProxyFeature(_agent(did, storage, tmp_path), _runtime())
    first = ProxyFeature(
        _agent(did, storage, tmp_path), _runtime(), client_factory=first_factory
    )
    competing_replica = ProxyFeature(_agent(did, storage, tmp_path), _runtime())
    winner = ProxyFeature(_agent(did, storage, tmp_path), _runtime())
    first_update: asyncio.Task[None] | None = None
    try:
        await seed.set_config(old_config)
        await first.initialize()
        stale_client = clients[0]

        # Replica one is now running old_config while durable state advances.
        # Its next PATCH must reconcile that stale child before it can hook.
        await competing_replica.set_config(active_config)
        first_update = asyncio.create_task(first.set_config(lost_candidate))
        await asyncio.wait_for(stale_stop_started.wait(), timeout=1)

        # Let the staged lease lapse while reconciliation is blocked; another
        # replica removes that exact abandoned generation and promotes a winner.
        clock[0] += timedelta(milliseconds=51)
        await winner.set_config(winner_config)
        release_stale_stop.set()

        with pytest.raises(RuntimeError, match="lease was lost"):
            await first_update

        assert isinstance(stale_client, _BlockingStaleClient)
        assert stale_client.prepare_calls == 0
        assert all(getattr(client, "prepare_calls", 0) == 0 for client in clients)
        assert await first.get_config() == winner_config
        assert graph.nodes[_scoped_node_id(did)].properties["config"] == winner_config
        assert "pending_config" not in graph.nodes[_scoped_node_id(did)].properties
    finally:
        release_stale_stop.set()
        if first_update is not None and not first_update.done():
            await first_update
        await first.shutdown()
