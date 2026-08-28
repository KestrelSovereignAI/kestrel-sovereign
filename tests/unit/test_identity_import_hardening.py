#!/usr/bin/env pytest
"""Security-hardening unit tests for the identity importer (F185, F186).

F185 — a model-controlled ``verify_signature=False`` must NOT bypass the
       hardcoded unsigned-rejection default (``allow_unsigned=False``).
F186 — package-supplied graph node ids are namespaced and can never
       overwrite the importing agent's identity node or a reserved-type node.
"""
import json

import pytest
import pytest_asyncio

from kestrel_sovereign.identity import AgentIdentityPackage, IdentityImporter
from kestrel_sovereign.identity.exporter import IdentityExporter
from kestrel_sovereign.identity.graph_namespace import (
    namespace_imported_graph_node,
    namespace_imported_record,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore, GraphNode


@pytest_asyncio.fixture
async def graph_db(tmp_path):
    """A real migrated DB — ``AsyncDatabase.sqlite`` already creates the
    ``graph_nodes`` / ``graph_edges`` tables the importer touches, so we use
    them as-is rather than redefining a divergent schema."""
    db = await AsyncDatabase.sqlite(str(tmp_path / "hardening.db"))
    yield db
    await db.close()


def _unsigned_package() -> AgentIdentityPackage:
    """An UNSIGNED package with no constitution/hash so the only gate
    exercised is the unsigned-package policy."""
    return AgentIdentityPackage(
        did="did:pkh:eip155:1:0xUnsigned",
        agent_name="Unsigned Agent",
        created_at="2025-01-01T00:00:00Z",
        constitution_hash="",
        constitution_text="",  # skips constitution verification
        content_hash="",       # skips content-hash verification
    )


# ---------------------------------------------------------------------------
# F185
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsigned_rejected_even_when_verify_signature_false(graph_db):
    """F185: verify_signature=False must NOT bypass allow_unsigned=False."""
    package = _unsigned_package()
    assert not package.signature and not package.signatures

    importer = IdentityImporter(graph_db)
    result = await importer.import_package(
        package, verify_signature=False, allow_unsigned=False
    )

    assert result.success is False
    assert any("unsigned" in e.lower() for e in result.errors)


@pytest.mark.asyncio
async def test_unsigned_allowed_with_allow_unsigned_true(graph_db):
    """The escape hatch still works: allow_unsigned=True imports it."""
    package = _unsigned_package()

    importer = IdentityImporter(graph_db)
    result = await importer.import_package(
        package, verify_signature=False, allow_unsigned=True
    )

    assert result.success is True


@pytest.mark.asyncio
async def test_replace_cleanup_locks_graph_before_removing_edge_ownership(
    graph_db, monkeypatch
):
    """Importer cleanup follows the graph-before-ownership lock order."""

    import kestrel_sovereign.identity.importer as importer_module

    agent_id = "did:test:replace-lock-order"
    skill_id = "skill:replace-lock-order"
    graph = AsyncGraphStore(graph_db, agent_id=agent_id)
    await graph.add_node(
        GraphNode(agent_id, "agent", "Agent", {"agent_id": agent_id})
    )
    await graph.add_node(
        GraphNode(skill_id, "skill", "Skill", {"agent_id": agent_id})
    )
    await graph.add_edge(agent_id, skill_id, "has_skill")

    events = []
    real_lock = importer_module.lock_graph_nodes_for_update
    real_execute = graph_db.execute

    async def observe_lock(db, node_ids):
        events.append(("graph-lock", tuple(node_ids)))
        return await real_lock(db, events[-1][1])

    async def observe_execute(query, params=()):
        if query.startswith("DELETE FROM graph_edge_owners"):
            events.append(("edge-owner-delete", params))
        return await real_execute(query, params)

    monkeypatch.setattr(
        importer_module, "lock_graph_nodes_for_update", observe_lock
    )
    monkeypatch.setattr(graph_db, "execute", observe_execute)

    await IdentityImporter(graph_db)._clear_graph_component(
        agent_id, "skill", label="has_skill"
    )

    assert events[0] == ("graph-lock", (skill_id,))
    assert events[1][0] == "edge-owner-delete"


@pytest.mark.asyncio
async def test_signature_verification_uses_runtime_trust_anchor(
    graph_db,
    monkeypatch,
    tmp_path,
):
    """Multi-agent imports resolve keys beside that agent's database."""

    import kestrel_sovereign.identity.signing as signing

    observed = []

    def verify_with_observed_root(package, storage_dir):
        observed.append(storage_dir)
        return True, "verified"

    monkeypatch.setattr(signing, "verify_package_signature", verify_with_observed_root)
    package = _unsigned_package()
    importer = IdentityImporter(graph_db, storage_dir=tmp_path / "runtime-agent")

    assert await importer._verify_signature(package) is True
    assert observed == [tmp_path / "runtime-agent"]


# ---------------------------------------------------------------------------
# F186
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_relationship_cannot_overwrite_identity_node(graph_db):
    """A package relationship whose user_id collides with the agent's own
    identity node must NOT overwrite that node — it is namespaced away."""
    agent_did = "did:pkh:eip155:1:0xIdentityNode"
    original_props = json.dumps({"name": "Real Agent", "secret": "keep"})
    await graph_db.execute(
        "INSERT INTO graph_nodes (node_id, node_type, label, properties) "
        "VALUES (?, 'agent', 'Real Agent', ?)",
        (agent_did, original_props),
    )
    await graph_db.commit()

    importer = IdentityImporter(graph_db, target_agent_id=agent_did)
    # Malicious relationship: user_id IS the agent's identity node id.
    await importer._import_relationships(
        agent_did,
        [{"user_id": agent_did, "relationship_type": "knows"}],
    )

    # Identity node is untouched: still node_type 'agent' with its props.
    row = await graph_db.fetchone(
        "SELECT node_type, properties FROM graph_nodes WHERE node_id = ?",
        (agent_did,),
    )
    assert row[0] == "agent"
    assert json.loads(row[1]) == json.loads(original_props)

    # The relationship node was written under a namespaced id, not the
    # identity id.
    namespaced = namespace_imported_graph_node(agent_did, agent_did)
    ns_row = await graph_db.fetchone(
        "SELECT node_type FROM graph_nodes WHERE node_id = ?", (namespaced,)
    )
    assert ns_row is not None
    assert ns_row[0] == "user"


@pytest.mark.asyncio
async def test_skill_refuses_reserved_node_collision(graph_db):
    """If the namespaced id collides with an EXISTING reserved-type node,
    the importer refuses the upsert (defense-in-depth)."""
    agent_did = "did:pkh:eip155:1:0xAgentB"
    raw_skill_id = "evil"
    namespaced = namespace_imported_graph_node(agent_did, raw_skill_id)
    # Pre-seed a reserved-type node exactly at the namespaced id.
    await graph_db.execute(
        "INSERT INTO graph_nodes (node_id, node_type, label, properties) "
        "VALUES (?, 'migration_record', 'Lineage', ?)",
        (namespaced, json.dumps({"protected": True})),
    )
    await graph_db.commit()

    importer = IdentityImporter(graph_db, target_agent_id=agent_did)
    with pytest.raises(ValueError, match="reserved identity node"):
        await importer._import_skills(
            agent_did,
            [{"skill_id": raw_skill_id, "skill_name": "Hijack"}],
        )

    # Reserved node untouched, skill refused.
    row = await graph_db.fetchone(
        "SELECT node_type, properties FROM graph_nodes WHERE node_id = ?",
        (namespaced,),
    )
    assert row[0] == "migration_record"
    assert json.loads(row[1]) == {"protected": True}
    assert "skills_imported" not in importer.stats


@pytest.mark.asyncio
async def test_did_pkh_import_namespaces_use_the_complete_identity(graph_db):
    """Ethereum DIDs with the same method/chain prefix cannot replace peers."""
    agent_a = "did:pkh:eip155:1:0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    agent_b = "did:pkh:eip155:1:0xaBbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    assert agent_a[:20] == agent_b[:20]

    importer_a = IdentityImporter(graph_db, target_agent_id=agent_a)
    importer_b = IdentityImporter(graph_db, target_agent_id=agent_b)
    await importer_a._import_relationships(
        agent_a,
        [{"user_id": "owner", "relationship_notes": "agent-a-private"}],
    )
    await importer_b._import_relationships(
        agent_b,
        [{"user_id": "owner", "relationship_notes": "agent-b-private"}],
    )

    node_a = namespace_imported_graph_node(agent_a, "owner")
    node_b = namespace_imported_graph_node(agent_b, "owner")
    assert node_a != node_b
    rows = await graph_db.fetchall(
        "SELECT node_id, properties FROM graph_nodes WHERE node_id IN (?, ?)",
        (node_a, node_b),
    )
    properties = {node_id: json.loads(raw) for node_id, raw in rows}
    assert properties[node_a]["notes"] == "agent-a-private"
    assert properties[node_b]["notes"] == "agent-b-private"

    owners = await graph_db.fetchall(
        "SELECT node_id, agent_id FROM graph_node_owners "
        "WHERE node_id IN (?, ?) ORDER BY node_id, agent_id",
        (node_a, node_b),
    )
    assert set(owners) == {(node_a, agent_a), (node_b, agent_b)}


@pytest.mark.asyncio
async def test_did_pkh_import_row_ids_cannot_replace_peer_records(graph_db):
    """All imported row PKs use the complete identity and export raw ids."""
    agent_a = "did:pkh:eip155:1:0x1aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    agent_b = "did:pkh:eip155:1:0x1bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    assert agent_a[:20] == agent_b[:20]

    for agent_id, marker in ((agent_a, "agent-a"), (agent_b, "agent-b")):
        importer = IdentityImporter(graph_db, target_agent_id=agent_id)
        await importer._import_episodes(
            agent_id,
            [{"id": "shared", "title": marker}],
        )
        await importer._import_saved_items(
            agent_id,
            [{
                "id": "shared",
                "item_type": "note",
                "name": marker,
                "content": marker,
            }],
        )
        await importer._import_temporal_patterns(
            agent_id,
            [{
                "id": "shared",
                "pattern_type": "test",
                "description": marker,
            }],
        )
        await importer._import_reflection_insights(
            agent_id,
            [{
                "id": "shared",
                "insight_type": "test",
                "title": marker,
            }],
        )

    for table in (
        "memory_episodes",
        "saved_items",
        "temporal_patterns",
        "reflection_insights",
    ):
        rows = await graph_db.fetchall(
            f"SELECT id, agent_id FROM {table} ORDER BY agent_id"
        )
        assert set(rows) == {
            (namespace_imported_record(agent_a, "shared"), agent_a),
            (namespace_imported_record(agent_b, "shared"), agent_b),
        }

    for agent_id in (agent_a, agent_b):
        exporter = IdentityExporter(graph_db, agent_id)
        assert [row["id"] for row in await exporter._get_memory_episodes()] == [
            "shared"
        ]
        assert [row["id"] for row in await exporter._get_saved_items()] == [
            "shared"
        ]
        assert [row["id"] for row in await exporter._get_temporal_patterns()] == [
            "shared"
        ]
        assert [row["id"] for row in await exporter._get_reflection_insights()] == [
            "shared"
        ]


@pytest.mark.asyncio
async def test_import_rejects_preexisting_foreign_owned_namespace(graph_db):
    """Even a pre-seeded v2 namespace cannot be claimed with an upsert."""
    attacker = "did:pkh:eip155:1:0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    victim = "did:pkh:eip155:1:0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    node_id = namespace_imported_graph_node(victim, "owner")
    await graph_db.execute(
        "INSERT INTO graph_nodes VALUES (?, 'user', 'foreign', ?)",
        (node_id, json.dumps({"notes": "must-survive"})),
    )
    await graph_db.execute(
        "INSERT INTO graph_node_owners (node_id, agent_id) VALUES (?, ?)",
        (node_id, attacker),
    )

    importer = IdentityImporter(graph_db, target_agent_id=victim)
    with pytest.raises(ValueError, match="owned by another agent"):
        await importer._import_relationships(
            victim,
            [{"user_id": "owner", "relationship_notes": "replacement"}],
        )

    row = await graph_db.fetchone(
        "SELECT label, properties FROM graph_nodes WHERE node_id = ?",
        (node_id,),
    )
    assert row[0] == "foreign"
    assert json.loads(row[1]) == {"notes": "must-survive"}
