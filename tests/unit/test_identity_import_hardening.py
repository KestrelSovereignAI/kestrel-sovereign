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
from kestrel_sovereign.identity.graph_namespace import (
    namespace_imported_graph_node,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase


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
