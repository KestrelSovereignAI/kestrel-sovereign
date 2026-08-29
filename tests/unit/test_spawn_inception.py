"""Tests for inception service integration with SpawnMandate and DID delegation."""

import json
import pytest
import tempfile
import os
from decimal import Decimal
from pathlib import Path

from kestrel_sovereign.inception_service import (
    create_kestrel_identity_async,
    generate_secp256k1_keypair,
)
from kestrel_sovereign.spawn.mandate import SpawnMandate, sign_mandate
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def constitution_path():
    # Anchor from the packaged governing source — the exact bytes the periodic
    # integrity audit recomputes. Inception now refuses a non-authoritative
    # override (e.g. the docs copy) because it would guarantee a next-audit
    # Safe Mode (#2463). These tests exercise parent-DID/controller wiring, not
    # constitution content, so they use the authoritative source.
    return str(
        Path(__file__).resolve().parent.parent.parent
        / "kestrel_sovereign"
        / "data"
        / "KESTREL_CONSTITUTION.md"
    )


@pytest.mark.asyncio
async def test_inception_with_parent_did_adds_controller(tmp_dir, constitution_path):
    """When parent_did is provided, child DID document should have controller field."""
    parent_did = "did:pkh:eip155:1:0xParent123"

    creds = await create_kestrel_identity_async(
        output_dir=tmp_dir,
        constitution_path=constitution_path,
        is_test_instance=True,
        agent_name="ChildAgent",
        parent_did=parent_did,
    )

    # Find the DID document file (born-hybrid default writes <slug>_did.json)
    did_files = list(Path(tmp_dir).glob("*_did.json"))
    assert len(did_files) == 1, f"Expected 1 DID JSON file, found {len(did_files)}"

    with open(did_files[0]) as f:
        did_doc = json.load(f)

    assert did_doc["controller"] == parent_did
    assert did_doc["id"] == creds.agent_did
    assert did_doc["id"] != parent_did


@pytest.mark.asyncio
async def test_inception_with_parent_did_records_spawned_by_edge(tmp_dir, constitution_path):
    """When parent_did is provided, knowledge graph should have spawned_by edge."""
    parent_did = "did:pkh:eip155:1:0xParent456"

    creds = await create_kestrel_identity_async(
        output_dir=tmp_dir,
        constitution_path=constitution_path,
        is_test_instance=True,
        agent_name="ChildAgent2",
        parent_did=parent_did,
    )

    # Re-open the database to check the graph
    db_path = os.path.join(tmp_dir, "kestrel_prime.db")
    db = await AsyncDatabase.sqlite(db_path)
    graph = AsyncGraphStore(db, agent_id=creds.agent_did)

    edges = await graph.get_edges(creds.agent_did, direction="out")
    spawned_edges = [e for e in edges if e.label == "spawned_by"]

    assert len(spawned_edges) == 1
    assert spawned_edges[0].target_id == parent_did
    assert spawned_edges[0].source_id == creds.agent_did
    owner = await db.fetchone(
        "SELECT agent_id FROM graph_edge_owners "
        "WHERE source_id = ? AND target_id = ? AND label = 'spawned_by'",
        (creds.agent_did, parent_did),
    )
    assert owner == (creds.agent_did,)

    await db.close()


@pytest.mark.asyncio
async def test_inception_without_parent_did_has_no_controller(tmp_dir, constitution_path):
    """Without parent_did, DID document should not have controller field."""
    creds = await create_kestrel_identity_async(
        output_dir=tmp_dir,
        constitution_path=constitution_path,
        is_test_instance=True,
        agent_name="StandaloneAgent",
    )

    did_files = list(Path(tmp_dir).glob("*_did.json"))
    assert len(did_files) == 1

    with open(did_files[0]) as f:
        did_doc = json.load(f)

    # controller should not be present at the top level
    # (note: verificationMethod[*].controller is the self-reference, which is normal)
    assert "controller" not in did_doc or did_doc.get("controller") == did_doc["id"]


@pytest.mark.asyncio
async def test_inception_with_spawn_mandate_records_properties(tmp_dir, constitution_path):
    """Spawn mandate properties should be recorded on the spawned_by edge."""
    parent_private, parent_public = generate_secp256k1_keypair()
    parent_did = "did:pkh:eip155:1:0xParent789"

    mandate = SpawnMandate(
        parent_did=parent_did,
        purpose="research helper",
        ttl_seconds=7200,
        max_child_depth=1,
    )
    mandate = sign_mandate(mandate, parent_private)

    creds = await create_kestrel_identity_async(
        output_dir=tmp_dir,
        constitution_path=constitution_path,
        is_test_instance=True,
        agent_name="MandatedChild",
        parent_did=parent_did,
        spawn_mandate=mandate,
    )

    # Check the edge properties
    db_path = os.path.join(tmp_dir, "kestrel_prime.db")
    db = await AsyncDatabase.sqlite(db_path)
    graph = AsyncGraphStore(db, agent_id=creds.agent_did)

    edges = await graph.get_edges(creds.agent_did, direction="out")
    spawned_edges = [e for e in edges if e.label == "spawned_by"]

    assert len(spawned_edges) == 1
    edge = spawned_edges[0]
    assert edge.properties["purpose"] == "research helper"
    assert edge.properties["ttl_seconds"] == 7200
    assert edge.properties["max_child_depth"] == 1
    # This signature was made before inception generated the child's DID and
    # cannot authorize the resulting identity.  The manager replaces it with
    # a final-DID-bound signature on the managed spawn path.
    assert edge.properties["parent_signature"] is None

    await db.close()


@pytest.mark.asyncio
async def test_unrepresentable_mandate_budget_fails_before_identity_creation(
    tmp_path,
    constitution_path,
):
    output = tmp_path / "never-created"
    mandate = SpawnMandate(
        parent_did="did:pkh:eip155:1:0xParentBudget",
        budget_allocation=Decimal("1e-400"),
    )

    with pytest.raises(ValueError, match="JSON numeric range"):
        await create_kestrel_identity_async(
            output_dir=str(output),
            constitution_path=constitution_path,
            identity_method="did:pkh",
            is_test_instance=True,
            parent_did=mandate.parent_did,
            spawn_mandate=mandate,
        )

    assert not output.exists()
