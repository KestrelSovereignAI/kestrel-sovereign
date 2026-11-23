"""A public reader release must leave native first-edition authority alone."""

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from kestrel_sovereign.features.constitution import ConstitutionFeature
from kestrel_sovereign.storage.async_graph_store import GraphNode


@pytest.mark.asyncio
async def test_native_first_edition_full_and_original_preserve_anchor(native_tool_effect_agent):
    agent = await native_tool_effect_agent(did="did:test:first-edition-reader")
    storage = agent._raw_storage
    text = (Path(__file__).resolve().parents[2] / "kestrel_sovereign/data/KESTREL_CONSTITUTION.md").read_bytes()
    digest = await storage.store_file(text, "KESTREL_CONSTITUTION.md")
    assert digest == "6948454b61253548b02b943dff166ba0c2f2eaabe2f542ace37424a8256d6467"
    await storage.add_node(GraphNode(
        node_id=agent.agent_id, node_type="agent", label="First-edition reader control",
        properties={"constitution_hash": digest, "reader_control": "preserve"},
    ))
    await agent._anchor_constitution_governance(digest)
    before = deepcopy((await storage.get_node(agent.agent_id)).properties)
    edges = await storage.get_edges_from(agent.agent_id)
    agent.extension = None
    feature = ConstitutionFeature(agent)
    await feature.initialize()
    for kwargs in ({}, {"article": "original"}):
        result = await feature.get_constitution(**kwargs)
        assert result.status.value == "ok"
        assert result.confirmation.encode() == text
        assert hashlib.sha256(result.confirmation.encode()).hexdigest() == digest
    assert (await storage.get_node(agent.agent_id)).properties == before
    assert await storage.get_edges_from(agent.agent_id) == edges
    assert await storage.retrieve_file(digest) == text
