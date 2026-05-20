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


# ----------------------------------------------------------------------
# Functional test — full graduate flow against a real SQLite fixture.
# ----------------------------------------------------------------------

@pytest.fixture
async def graduate_ready_db(tmp_path):
    """Build a graduate-ready agent DB with all 8 validation gates passing.

    Conversations are written under ``agent_id=did`` so the test exercises
    the same per-tenant path a live ``KestrelAgent`` uses. The validator
    must query under that tenant; passing this fixture means the validator
    is reaching the right tenant, not the empty default. (Codex caught the
    original cross-tenant bug in PR review.)
    """
    db_path = tmp_path / "kestrel_prime.db"
    address = "0xTESTADDRESS"
    did = f"did:pkh:eip155:1:{address}"

    # On-disk files the validator looks for
    (tmp_path / f"kestrel_{address}.json").write_text('{"id": "did-doc"}')
    (tmp_path / f"kestrel_{address}.key.enc").write_bytes(b"encrypted")

    # Open storage scoped to the agent's DID so conversation rows land under
    # the same tenant a live agent would use.
    from kestrel_sovereign.storage.async_storage import AsyncStorage
    storage = AsyncStorage(db_path=str(db_path), agent_id=did)
    await storage.initialize()
    try:
        # Agent node, marked test instance
        agent_id = "agent:test-emma"
        await storage.graph.add_node(GraphNode(
            node_id=agent_id,
            node_type="agent",
            label="Test Emma",
            properties={
                "name": "TestEmma",
                "did": did,
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
