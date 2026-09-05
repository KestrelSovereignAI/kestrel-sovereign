"""Durable spawned-by authority crosses a real SQLite restart boundary."""

import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_sovereign.inception_service import generate_secp256k1_keypair
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.multi_agent.config import LocalAgentConfig
from kestrel_sovereign.spawn.authority_registry import SpawnAuthorityRegistry
from kestrel_sovereign.spawn.mandate import SpawnMandate, sign_mandate
from kestrel_sovereign.spawn.mandate_reload import read_spawn_mandate
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore, GraphNode


def _agent(agent_id, **state):
    return SimpleNamespace(
        agent_id=agent_id,
        did=agent_id,
        identity=None,
        features={},
        shutdown=AsyncMock(),
        **state,
    )


@pytest.mark.asyncio
async def test_spawned_by_restart_descendant_authority_round_trip_sqlite(
    tmp_path,
):
    """Real spawn persistence rehydrates only the signing parent's control."""

    parent_did = "did:pkh:eip155:1:0xIntegrationRestartParent"
    child_did = "did:pkh:eip155:1:0xIntegrationRestartChild"
    private_key, _ = generate_secp256k1_keypair()
    database = await AsyncDatabase.sqlite(tmp_path / "kestrel_prime.db")
    graph = AsyncGraphStore(database, agent_id=child_did)
    try:
        parent_graph = AsyncGraphStore(database, agent_id=parent_did)
        await parent_graph.add_node(
            GraphNode(parent_did, "agent", "Parent", {})
        )
        await graph.add_node(GraphNode(child_did, "agent", "Child", {}))
        parent = _agent(parent_did, _private_key=private_key)
        child = _agent(
            child_did,
            _raw_storage=SimpleNamespace(graph=graph),
        )
        first = AgentManager(base_data_dir=tmp_path)
        first._register_agent("Parent", parent)

        async def create_and_publish(name, **kwargs):
            admission = first._agent_operations[
                first._canonical_agent_name(name)
            ]
            assert admission.before_publish is not None
            admission.spawn_candidate_config = LocalAgentConfig(
                data_dir=tmp_path,
                port=8801,
            )
            pending = first._spawn_authority_registry.reserve_pending(
                child_name=name,
                parent_did=kwargs["parent_did"],
                mandate=kwargs["mandate"],
                config=admission.spawn_candidate_config,
            )
            admission.spawn_authority_pending_id = pending.reservation_id
            await admission.before_publish(child)
            first._register_agent(name, child)
            return child

        first.create_agent = create_and_publish
        proposal = SpawnMandate(
            parent_did=parent_did,
            purpose="restart integration",
            ttl_seconds=0,
            max_child_depth=1,
            additional_constraints={"restricted_tools": ["shell"]},
        )
        proposal_created_at = proposal.created_at
        await first.spawn_agent(
            "Child",
            parent,
            proposal,
        )

        host_witness = SpawnAuthorityRegistry(tmp_path).get(child_did)
        assert host_witness is not None
        assert host_witness.proposal_created_at == proposal_created_at
        assert host_witness.mandate.created_at != proposal_created_at

        durable_projection = SimpleNamespace(
            get_edges_from=lambda agent_did: graph.get_edges(
                agent_did,
                direction="out",
            )
        )
        restored = await read_spawn_mandate(durable_projection, child_did)
        assert restored is not None
        assert restored.additional_constraints == {
            "restricted_tools": ["shell"]
        }

        restarted_parent = _agent(parent_did, _private_key=private_key)
        restarted_child = _agent(
            child_did,
            _persisted_spawn_mandate=restored,
        )
        restarted = AgentManager(base_data_dir=tmp_path)
        restarted._register_agent("Parent", restarted_parent)
        restarted._initialize_agent = AsyncMock(return_value=restarted_child)
        restarted._on_agent_registered = AsyncMock()
        restarted._run_hosted_agent_ready_hooks = AsyncMock()
        loaded = await restarted.load_agent(
            "Child",
            LocalAgentConfig(data_dir=tmp_path, port=8801),
        )

        assert loaded is restarted_child
        assert restarted.get_children(parent_did) == ["Child"]
        assert restarted.get_mandate("Child") is restored
        assert (
            await restarted.terminate_child("did:test:unrelated-peer", "Child")
            is False
        )

        # A second projection is idempotent, and the verified parent retains
        # the real control path after that repeated reload boundary.
        restarted._register_agent("Child", restarted_child)
        assert restarted.get_children(parent_did) == ["Child"]
        assert await restarted.terminate_child(parent_did, "Child") is True
        assert restarted.get_children(parent_did) == []
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_host_witness_repairs_crash_before_signed_child_edge_sqlite(
    tmp_path,
):
    """A real unsigned inception edge is repaired from the signed host rail."""

    parent_did = "did:pkh:eip155:1:0xInterruptedIntegrationParent"
    child_did = "did:pkh:eip155:1:0xInterruptedIntegrationChild"
    child_name = "InterruptedIntegrationChild"
    private_key, _ = generate_secp256k1_keypair()
    mandate = SpawnMandate(
        parent_did=parent_did,
        child_did=child_did,
        purpose="repair interrupted signed receipt",
        ttl_seconds=0,
        created_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
    )
    config = LocalAgentConfig(
        data_dir=tmp_path / "interrupted-child",
        port=8801,
    )
    database = await AsyncDatabase.sqlite(tmp_path / "interrupted.db")
    graph = AsyncGraphStore(database, agent_id=child_did)
    try:
        parent_graph = AsyncGraphStore(database, agent_id=parent_did)
        await parent_graph.add_node(GraphNode(parent_did, "agent", "Parent", {}))
        await graph.add_node(GraphNode(child_did, "agent", "Child", {}))
        unsigned_properties = mandate.to_edge_properties()
        unsigned_properties["parent_signature"] = None
        await graph.add_trusted_cross_agent_edge(
            child_did,
            parent_did,
            "spawned_by",
            properties=unsigned_properties,
        )
        durable_projection = SimpleNamespace(
            get_edges_from=lambda agent_did: graph.get_edges(
                agent_did,
                direction="out",
            )
        )
        unsigned_receipt = await read_spawn_mandate(
            durable_projection,
            child_did,
        )
        assert unsigned_receipt is not None
        assert unsigned_receipt.parent_signature is None

        proposal_created_at = mandate.created_at
        mandate = copy.deepcopy(mandate)
        mandate.created_at = datetime.now(timezone.utc).isoformat()
        sign_mandate(mandate, private_key)
        assert mandate.created_at != proposal_created_at

        SpawnAuthorityRegistry(tmp_path).record_active(
            child_name=child_name,
            child_did=child_did,
            mandate=mandate,
            config=config,
            proposal_created_at=proposal_created_at,
        )
        manager = AgentManager(base_data_dir=tmp_path)
        manager._register_agent(
            "InterruptedIntegrationParent",
            _agent(parent_did, _private_key=private_key),
        )

        class HostedChild:
            def __init__(self, *, did, **_kwargs):
                self.did = did
                self.agent_id = did
                self.identity = None
                self.features = {}
                self._raw_storage = SimpleNamespace(graph=graph)
                self._persisted_spawn_mandate = unsigned_receipt

            async def initialize(self):
                await self._host_authority_preflight(self)

            async def run_agent_ready_hooks(self):
                return None

            async def shutdown(self):
                return None

        with (
            patch.object(LocalAgentConfig, "validate_runtime", return_value=[]),
            patch(
                "kestrel_sovereign.multi_agent.agent_manager.read_anchor_agent_did",
                new=AsyncMock(return_value=child_did),
            ),
            patch(
                "kestrel_sovereign.multi_agent.agent_manager.KestrelAgent",
                HostedChild,
            ),
        ):
            loaded = await manager.load_agent(child_name, config)

        repaired = await read_spawn_mandate(durable_projection, child_did)
        assert repaired is not None
        assert repaired.to_dict() == mandate.to_dict()
        assert loaded._persisted_spawn_mandate.to_dict() == mandate.to_dict()
        assert manager.get_agent(child_name) is loaded
    finally:
        await database.close()
