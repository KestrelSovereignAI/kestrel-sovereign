"""Regression tests for ``kestrel_sovereign.graduate_service``.

Context: the script bitrotted silently against ``AsyncGraphStore`` —
``query_nodes`` was removed and replaced by ``get_nodes_by_type`` /
``query_nodes_by_type_and_property``, ``update_node`` was removed
(``add_node`` upserts), and ``get_edges`` / ``add_edge`` keyword
arguments were renamed. The script's own dry-run raised
``AttributeError`` on the first call against the live DB, blocking
graduation of every test agent.

These tests:

1. Pin the contract at every ``AsyncStorage`` / ``AsyncGraphStore``
   method the script calls. If any future refactor renames or removes
   one of these, this test fails before the script does.
2. Exercise the full graduate flow against a real SQLite fixture —
   no mocks. The fixture is a graduate-ready agent: test instance,
   constitution-anchored, has conversation history, DID artifacts on
   disk, sovereignty backup, ≥3 nodes. Validates that all 7 checks
   pass and that the live graduate path writes the expected nodes
   and edges.
"""

import asyncio
import inspect
import tempfile
from pathlib import Path

import pytest

from kestrel_sovereign import graduate_service
from kestrel_sovereign.storage import Storage
from kestrel_sovereign.storage.async_graph_store import GraphNode


# ----------------------------------------------------------------------
# Contract test — every AsyncStorage / AsyncGraphStore method the script
# touches must exist with a signature the script can call.
# ----------------------------------------------------------------------

REQUIRED_GRAPH_METHODS = [
    "get_node",
    "get_nodes_by_type",
    "add_node",
    "add_edge",
    "get_edges",
]

REQUIRED_STORAGE_METHODS = [
    "get_conversation_history",
]


@pytest.mark.asyncio
async def test_graduate_service_required_graph_methods_present():
    """If ``AsyncGraphStore`` loses any of these, ``graduate_service`` breaks."""
    from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
    for name in REQUIRED_GRAPH_METHODS:
        assert hasattr(AsyncGraphStore, name), (
            f"AsyncGraphStore is missing {name!r} which graduate_service.py "
            f"depends on. If this method was renamed, update the script and "
            f"this contract list together."
        )


@pytest.mark.asyncio
async def test_graduate_service_required_storage_methods_present():
    """Same for the AsyncStorage facade methods."""
    from kestrel_sovereign.storage.async_storage import AsyncStorage
    for name in REQUIRED_STORAGE_METHODS:
        assert hasattr(AsyncStorage, name), (
            f"AsyncStorage is missing {name!r} which graduate_service.py "
            f"depends on."
        )


def test_graduate_service_signature_has_no_council_session():
    """Per feedback_graduation_gate_is_codex_review.md, council was removed."""
    sig = inspect.signature(graduate_service.graduate_agent)
    assert "council_session" not in sig.parameters, (
        "graduate_agent should not accept --council-session; the gate is "
        "codex CLI PR review."
    )


def test_resolve_did_prefers_property_then_falls_back_to_node_id():
    """The agent's DID lives on ``node_id`` by convention. ``properties['did']``
    is an optional shadow some agents carry. Resolver must prefer the
    property when present, otherwise fall back to the node_id.

    Regression for Emma's live DB shape (no ``did`` property; DID lives
    only on node_id) — three validator gates failed before this fallback
    landed.
    """
    DID = "did:pkh:eip155:1:0xABC"

    # Case A: only node_id is set
    node_a = GraphNode(node_id=DID, node_type="agent", label="A", properties={})
    assert graduate_service._resolve_did(node_a) == DID

    # Case B: properties has a DID, node_id is something legacy/arbitrary;
    # the property wins so explicit migrations are honored
    node_b = GraphNode(
        node_id="agent:legacy",
        node_type="agent",
        label="B",
        properties={"did": DID},
    )
    assert graduate_service._resolve_did(node_b) == DID

    # Case C: properties has empty-string did — treated as absent, fall back
    node_c = GraphNode(node_id=DID, node_type="agent", label="C", properties={"did": ""})
    assert graduate_service._resolve_did(node_c) == DID


def test_resolve_did_refuses_non_did_node_id_when_property_missing():
    """If neither the property nor the node_id is actually a DID, the
    resolver returns ``""`` instead of laundering an arbitrary id through.

    Without this guard, an agent with ``node_id="agent:test-emma"`` and
    no ``did`` property would have its non-DID id used for on-disk
    file lookups (``kestrel_test-emma.json``) and conversation tenant
    queries — letting any matching files / rows satisfy the gates and
    weakening graduation. Codex caught this on #1325 round 2.
    """
    # node_id is not a DID, no property
    node_legacy = GraphNode(
        node_id="agent:test-emma", node_type="agent", label="L", properties={},
    )
    assert graduate_service._resolve_did(node_legacy) == ""

    # node_id is not a DID, property is also not a DID
    node_double_bad = GraphNode(
        node_id="agent:legacy",
        node_type="agent",
        label="DB",
        properties={"did": "not-a-did-either"},
    )
    assert graduate_service._resolve_did(node_double_bad) == ""


# ----------------------------------------------------------------------
# Functional test — full graduate flow against a real SQLite fixture.
# ----------------------------------------------------------------------

@pytest.fixture
async def graduate_ready_db(tmp_path):
    """Build a graduate-ready agent DB with all 8 validation gates passing.

    Layout mirrors what a live ``KestrelAgent`` produces:

    - The agent node's ``node_id`` *is* the DID. There is no
      ``properties['did']`` field — that mirrors Emma's live DB shape
      where the DID lives only on the node_id. The validator must fall
      back to ``node_id`` when the property is absent.
    - Conversations are written under ``agent_id=did`` so the
      cross-tenant gate exercises the same path the live agent uses.
    """
    db_path = tmp_path / "kestrel_prime.db"
    address = "0xTESTADDRESS"
    did = f"did:pkh:eip155:1:{address}"
    agent_id = did  # node_id IS the DID, per the canonical layout

    # On-disk files the validator looks for
    (tmp_path / f"kestrel_{address}.json").write_text('{"id": "did-doc"}')
    (tmp_path / f"kestrel_{address}.key.enc").write_bytes(b"encrypted")

    from kestrel_sovereign.storage.async_storage import AsyncStorage
    storage = AsyncStorage(db_path=str(db_path), agent_id=did)
    await storage.initialize()
    try:
        # Agent node — note: no ``did`` property. The validator must use
        # ``node_id`` as the DID source.
        await storage.graph.add_node(GraphNode(
            node_id=agent_id,
            node_type="agent",
            label="Test Emma",
            properties={
                "name": "TestEmma",
                "is_test_instance": True,
            },
        ))

        # Constitution node + governed_by edge
        constitution_id = "constitution:test"
        await storage.graph.add_node(GraphNode(
            node_id=constitution_id,
            node_type="constitution",
            label="Constitution",
            properties={"hash": "abc123"},
        ))
        await storage.graph.add_edge(
            source_id=agent_id,
            target_id=constitution_id,
            label="governed_by",
        )

        # Sovereignty backup
        await storage.graph.add_node(GraphNode(
            node_id="backup:1",
            node_type="backup_artifact",
            label="Sovereignty Backup",
            properties={"cid": "Qm..."},
        ))

        # Conversation history — required by gate #3
        await storage.add_conversation(
            role="user", content="hello", session_id="s1"
        )

        yield {
            "db_path": str(db_path),
            "agent_id": agent_id,
            "did": did,
            "data_dir": tmp_path,
        }
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_validate_agent_all_eight_gates_pass(graduate_ready_db):
    """Every validation gate must pass on a properly-prepared DB."""
    db_path = graduate_ready_db["db_path"]
    agent_id = graduate_ready_db["agent_id"]

    async with Storage(db_path=db_path) as storage:
        checklist = await graduate_service.validate_agent(storage, agent_id)

    assert checklist.all_passed, (
        f"Expected all gates to pass on a graduate-ready fixture. "
        f"Failed: {checklist.failed}"
    )
    # 8 gates: agent exists, is_test_instance, constitution anchored,
    # conversation history, DID doc, encrypted key, sovereignty backup,
    # knowledge-graph populated.
    assert len(checklist.passed) == 8, (
        f"Expected exactly 8 passing checks, got {len(checklist.passed)}: "
        f"{checklist.passed}"
    )


@pytest.mark.asyncio
async def test_graduate_agent_dry_run_returns_true_without_mutation(graduate_ready_db):
    db_path = graduate_ready_db["db_path"]
    agent_id = graduate_ready_db["agent_id"]

    ok = await graduate_service.graduate_agent(db_path=db_path, dry_run=True)
    assert ok is True

    # Confirm the flag did NOT flip in dry-run mode
    async with Storage(db_path=db_path) as storage:
        agent = await storage.graph.get_node(agent_id)
    assert agent.properties.get("is_test_instance") is True, (
        "Dry run must not mutate the agent."
    )


@pytest.mark.asyncio
async def test_graduate_agent_live_run_flips_flag_and_writes_event(graduate_ready_db):
    db_path = graduate_ready_db["db_path"]
    agent_id = graduate_ready_db["agent_id"]

    ok = await graduate_service.graduate_agent(db_path=db_path, dry_run=False)
    assert ok is True

    async with Storage(db_path=db_path) as storage:
        # Flag flipped, timestamp stamped
        agent = await storage.graph.get_node(agent_id)
        assert agent.properties.get("is_test_instance") is False
        assert "graduated_at" in agent.properties

        # lifecycle_event node was created
        events = await storage.graph.get_nodes_by_type("lifecycle_event")
        graduations = [
            e for e in events
            if e.properties.get("event_type") == "graduation"
        ]
        assert len(graduations) == 1, (
            f"Expected exactly one graduation event, got {len(graduations)}"
        )
        grad_event = graduations[0]
        assert grad_event.properties["agent_id"] == agent_id
        assert "validation_passed" in grad_event.properties
        assert len(grad_event.properties["validation_passed"]) == 8

        # Edge agent -> graduation event exists
        out_edges = await storage.graph.get_edges(agent_id, direction="out")
        lifecycle_edges = [e for e in out_edges if e.label == "lifecycle_event"]
        assert len(lifecycle_edges) == 1
        assert lifecycle_edges[0].target_id == grad_event.node_id


SYNC_MANIFEST_NAMES = [
    ".gcs_manifest_{did}.json",          # storage/sync/gcs_target.py
    ".lighthouse_manifest_{did}.json",   # storage/sync/lighthouse_target.py
    ".sovereign_ipfs_manifest_{did}.json",  # storage/sync/sovereign_ipfs_target.py
]


@pytest.mark.asyncio
@pytest.mark.parametrize("manifest_template", SYNC_MANIFEST_NAMES)
async def test_sovereignty_gate_accepts_disk_manifest_without_backup_artifact(
    tmp_path, manifest_template
):
    """Gate #6 ('Has sovereignty backup') must accept any of the disk sync
    manifests as proof of sovereignty backup — not require a discrete
    ``backup_artifact`` graph node, and not be limited to one sync target.

    Parametrised across every sync-target manifest filename in the codebase
    so a future sync target that's added without updating this gate fails
    explicitly. (Codex caught the original gap: I only listed two of the
    three manifest filenames in the first draft.)
    """
    db_path = tmp_path / "kestrel_prime.db"
    address = "0xMANIFEST"
    did = f"did:pkh:eip155:1:{address}"
    agent_id = did

    # On-disk DID artifacts
    (tmp_path / f"kestrel_{address}.json").write_text('{"id": "did-doc"}')
    (tmp_path / f"kestrel_{address}.key.enc").write_bytes(b"encrypted")
    # Sync manifest (proof of continuous sovereignty mirroring) — but
    # deliberately NO backup_artifact graph node.
    manifest_name = manifest_template.format(did=did)
    (tmp_path / manifest_name).write_text('{"manifest": "yes"}')

    from kestrel_sovereign.storage.async_storage import AsyncStorage
    storage = AsyncStorage(db_path=str(db_path), agent_id=did)
    await storage.initialize()
    try:
        await storage.graph.add_node(GraphNode(
            node_id=agent_id, node_type="agent", label="Manifest Test",
            properties={"name": "ManifestTest", "is_test_instance": True},
        ))
        constitution_id = "constitution:test"
        await storage.graph.add_node(GraphNode(
            node_id=constitution_id, node_type="constitution",
            label="Constitution", properties={},
        ))
        await storage.graph.add_edge(
            source_id=agent_id, target_id=constitution_id, label="governed_by",
        )
        # Pad node count so gate #8 also passes
        await storage.graph.add_node(GraphNode(
            node_id="pad:1", node_type="concept", label="Pad", properties={},
        ))
        await storage.add_conversation(role="user", content="hi", session_id="s1")

        checklist = await graduate_service.validate_agent(storage, agent_id)

        # All 8 gates pass — including sovereignty gate via disk manifest
        assert checklist.all_passed, (
            f"Gate #6 must accept disk manifest as sovereignty proof. "
            f"Failed: {checklist.failed}"
        )
        backup_check = next(
            c for c in checklist.checks if c["name"] == "Has sovereignty backup"
        )
        assert backup_check["passed"]
        assert manifest_name in backup_check["details"], (
            f"Details line should name the {manifest_name} that satisfied the gate."
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_sovereignty_gate_fails_when_neither_surface_present(tmp_path):
    """Gate #6 must still fail when neither a ``backup_artifact`` node nor a
    sync manifest is present. Broadening shouldn't degenerate the gate.
    """
    db_path = tmp_path / "kestrel_prime.db"
    address = "0xNOBACKUP"
    did = f"did:pkh:eip155:1:{address}"
    agent_id = did

    (tmp_path / f"kestrel_{address}.json").write_text('{}')
    (tmp_path / f"kestrel_{address}.key.enc").write_bytes(b"x")

    from kestrel_sovereign.storage.async_storage import AsyncStorage
    storage = AsyncStorage(db_path=str(db_path), agent_id=did)
    await storage.initialize()
    try:
        await storage.graph.add_node(GraphNode(
            node_id=agent_id, node_type="agent", label="NoBackup",
            properties={"name": "NoBackup", "is_test_instance": True},
        ))
        await storage.graph.add_node(GraphNode(
            node_id="constitution:test", node_type="constitution",
            label="Constitution", properties={},
        ))
        await storage.graph.add_edge(
            source_id=agent_id, target_id="constitution:test", label="governed_by",
        )
        await storage.add_conversation(role="user", content="hi", session_id="s1")

        checklist = await graduate_service.validate_agent(storage, agent_id)
        backup_check = next(
            c for c in checklist.checks if c["name"] == "Has sovereignty backup"
        )
        assert backup_check["passed"] is False
        assert "no backup_artifact nodes" in backup_check["details"]
        assert "no sync manifests on disk" in backup_check["details"]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_graduate_agent_refuses_already_permanent(graduate_ready_db):
    """An agent without is_test_instance=True must not graduate."""
    db_path = graduate_ready_db["db_path"]
    agent_id = graduate_ready_db["agent_id"]

    # Flip the flag manually so the agent is already permanent
    async with Storage(db_path=db_path) as storage:
        agent = await storage.graph.get_node(agent_id)
        await storage.graph.add_node(GraphNode(
            node_id=agent.node_id,
            node_type=agent.node_type,
            label=agent.label,
            properties={**agent.properties, "is_test_instance": False},
        ))

    ok = await graduate_service.graduate_agent(db_path=db_path, dry_run=False)
    assert ok is False, (
        "graduate_agent must return False when the agent is already permanent."
    )
