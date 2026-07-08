"""Enforce a spawned child's features_allowed on every boot path (#2226).

Follow-up to #2137. The mandate's feature allowlist is persisted on the
spawned_by edge but was only enforced via AgentManager's config threading
(#1946). A child booted directly (single-agent server, CLI, restart outside the
original config) instantiates KestrelAgent without that config, so without this
it would load ALL features. KestrelAgent.initialize now reads the persisted
ceiling from the edge and intersects it into the feature-discovery allowlist.
"""

import os

import pytest

from kestrel_sovereign.inception_service import (
    create_kestrel_identity_async,
    generate_secp256k1_keypair,
)
from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES
from kestrel_sovereign.spawn.mandate import SpawnMandate, sign_mandate
from kestrel_sovereign.spawn.mandate_reload import read_spawn_features_allowed
from kestrel_sovereign.storage.async_storage import AsyncStorage

ALLOWED = ["MemoryFeature"]


async def _spawn_child_dir(tmp_path, name, features_allowed):
    parent_private, _ = generate_secp256k1_keypair()
    parent_did = "did:pkh:eip155:1:0xParentFeat"
    mandate = sign_mandate(
        SpawnMandate(
            parent_did=parent_did,
            purpose="scoped worker",
            ttl_seconds=999,
            features_allowed=features_allowed,
        ),
        parent_private,
    )
    return await create_kestrel_identity_async(
        output_dir=str(tmp_path),
        is_test_instance=True,
        agent_name=name,
        parent_did=parent_did,
        spawn_mandate=mandate,
    )


@pytest.mark.asyncio
async def test_read_features_allowed_from_edge(tmp_path):
    creds = await _spawn_child_dir(tmp_path, "FeatChild", ALLOWED)
    storage = AsyncStorage(os.path.join(str(tmp_path), "kestrel_prime.db"))
    await storage.initialize()
    try:
        assert await read_spawn_features_allowed(storage, creds.agent_did) == ALLOWED
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_read_features_allowed_empty_for_root(tmp_path):
    creds = await create_kestrel_identity_async(
        output_dir=str(tmp_path), is_test_instance=True, agent_name="RootFeat"
    )
    storage = AsyncStorage(os.path.join(str(tmp_path), "kestrel_prime.db"))
    await storage.initialize()
    try:
        assert await read_spawn_features_allowed(storage, creds.agent_did) == []
    finally:
        await storage.close()


def _loaded_classes(agent):
    return {type(f).__name__ for f in agent.features.values()}


@pytest.mark.asyncio
async def test_direct_boot_enforces_feature_ceiling(tmp_path):
    """A spawned child booted directly through KestrelAgent.initialize loads only
    its mandate-allowed features plus the always-mandatory ones — not everything."""
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService

    creds = await _spawn_child_dir(tmp_path, "CeilChild", ALLOWED)
    agent = KestrelAgent(
        did=creds.agent_did,
        storage_path=os.path.join(str(tmp_path), "kestrel_prime.db"),
        llm_service=LLMService(),
    )
    await agent.initialize()

    loaded = _loaded_classes(agent)
    allowed_ceiling = set(ALLOWED) | set(MANDATORY_FEATURES)
    # Nothing beyond the mandate ceiling + mandatory features loaded.
    assert loaded <= allowed_ceiling, f"loaded beyond ceiling: {loaded - allowed_ceiling}"
    # The allowed feature and the mandatory ones are present.
    assert "MemoryFeature" in loaded
    assert set(MANDATORY_FEATURES) <= loaded


@pytest.mark.asyncio
async def test_do_spawn_persists_inherited_ceiling():
    """When a mandate omits features_allowed, _do_spawn writes the inherited
    parent ceiling onto the mandate BEFORE signing, so inception persists it on
    the edge and a direct restart still enforces it (not just this process)."""
    from types import SimpleNamespace

    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    mgr = AgentManager()
    parent = SimpleNamespace(
        _private_key=None,  # skip signing
        identity=None,
        agent_id="did:pkh:eip155:1:0xParentInherit",
        features={"MemoryFeature": object(), "ComputeFeature": object()},
    )
    child = SimpleNamespace(agent_id="did:pkh:eip155:1:0xChildInherit")
    captured = {}

    async def fake_create_agent(name, parent_did=None, features=None, mandate=None):
        captured["config_features"] = features
        captured["mandate_features"] = list(mandate.features_allowed)
        return child

    mgr.create_agent = fake_create_agent

    mandate = SpawnMandate(parent_did=parent.agent_id, purpose="inherit test")
    assert mandate.features_allowed == []  # no explicit allowlist
    await mgr._do_spawn("InheritKid", parent, mandate)

    expected = ["ComputeFeature", "MemoryFeature"]  # sorted parent ceiling
    assert mandate.features_allowed == expected
    assert captured["config_features"] == expected
    assert captured["mandate_features"] == expected  # persisted at inception time


@pytest.mark.asyncio
async def test_root_agent_loads_beyond_the_ceiling(tmp_path):
    """Control: a non-spawned agent has no ceiling, so it loads optional features
    the spawned child above was denied — proving the restriction is spawn-scoped."""
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService

    creds = await create_kestrel_identity_async(
        output_dir=str(tmp_path), is_test_instance=True, agent_name="RootFull"
    )
    agent = KestrelAgent(
        did=creds.agent_did,
        storage_path=os.path.join(str(tmp_path), "kestrel_prime.db"),
        llm_service=LLMService(),
    )
    await agent.initialize()

    loaded = _loaded_classes(agent)
    # A root agent loads more than the spawned child's ceiling would permit.
    assert not (loaded <= (set(ALLOWED) | set(MANDATORY_FEATURES)))
