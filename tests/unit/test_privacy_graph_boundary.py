"""Storage-boundary privacy tests for durable graph writes (#2672).

The #1760 gate blocks the post-response/background memory pipeline in volatile
privacy modes, but the *storage* boundary still waved every graph write through
as "structural, not PII" — so a feature calling ``storage.add_node`` (or
reaching the raw store through ``storage.graph`` / ``storage.graph_store``)
could persist user-derived facts, todos, decisions, concepts, and consolidated
episodes directly, outside that gate.

This suite proves the storage boundary now default-denies durable user-derived
and unknown graph writes in EPHEMERAL and ISOLATED — for node writes, edge
writes, and the atomic compare-and-swap — across every surface (the wrapper's
own methods AND the ``.graph`` / ``.graph_store`` proxies).

The admit policy has TWO tiers (review finding P1):

* **Content-free structural types** — ``document`` / ``constitution_amendment_artifact``
  / ``audit_anchor`` (and the ``governed_by`` edge) — are admitted on the
  ordinary path, but every property VALUE is validated content-free-shaped so a
  caller can't smuggle user text into a ``hash`` / ``type`` field.
* **Value-bearing control-plane types** — the ``agent`` identity node (free-text
  ``description``, governance receipts) and ``feature_config`` (arbitrary settings
  dict) — are admitted ONLY through the trusted control-plane path
  (``control_plane=True``), which the identity/governance/bootstrap/feature
  writers pass and no user-facing tool does. An untrusted write to one is
  default-denied, closing the ``config`` / ``description`` smuggling vector.

NORMAL is unchanged.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.async_graph_store import GraphNode, NodeSwapResult
from kestrel_sovereign.storage.privacy_wrapper import (
    CONTROL_PLANE_ONLY_NODE_TYPES,
    PrivacyEnforcingStorage,
    PrivacyViolationError,
    STRUCTURAL_GRAPH_EDGE_LABELS,
    STRUCTURAL_GRAPH_NODE_TYPES,
    _is_content_free_value,
)


AGENT_ID = "did:test:graph-boundary"

# The privacy modes whose contract forbids durable persistence of user data.
VOLATILE_MODES = [PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED]

# The content-free structural types, admitted on the ordinary (untrusted) path.
CONTENT_FREE_STRUCTURAL_TYPES = sorted(
    STRUCTURAL_GRAPH_NODE_TYPES - CONTROL_PLANE_ONLY_NODE_TYPES
)

# A sentinel standing in for user conversation content. It must NEVER reach a
# durable graph row in a volatile mode, and must NEVER appear in any allowed
# structural node.
USER_SECRET = "SECRET-USER-CONTENT-must-never-persist"

# Known content-bearing node types the inventory (#2672) identified as
# conversation-derived. Every one must be default-denied in volatile modes.
CONTENT_NODE_TYPES = [
    "concept",
    "message",
    "action_item",
    "decision",
    "episode",
    "todo_item",
    "learned_fact",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _admit_kwargs(node_type: str) -> dict:
    """The write kwargs needed to ADMIT a structural type in a volatile mode.

    Control-plane types require the trusted capability; content-free types do
    not. Keeps every "allowed" assertion honest about which path admits it.
    """
    if node_type in CONTROL_PLANE_ONLY_NODE_TYPES:
        return {"control_plane": True}
    return {}


def _user_content_node(node_type: str, node_id: str = "user-node-1") -> GraphNode:
    """A node whose payload is user-derived conversation content."""
    return GraphNode(
        node_id=node_id,
        node_type=node_type,
        label="user-derived",
        properties={
            "agent_id": AGENT_ID,
            "text": USER_SECRET,
            "value": USER_SECRET,
            "created_at": _now_iso(),
        },
    )


def _structural_node(node_type: str) -> GraphNode:
    """A minimal, content-free structural node for each allowlisted type.

    Payloads are the agent's own identity/governance state or cryptographic
    hashes — never user conversation text. Used to prove the allowlist admits
    only content-free rows.
    """
    now = _now_iso()
    if node_type == "agent":
        # An agent node's owner is its own node_id; keep it == the bound agent.
        return GraphNode(
            node_id=AGENT_ID,
            node_type="agent",
            label="Kestrel",
            properties={"name": "Kestrel", "constitution_hash": "c0ffee", "created_at": now},
        )
    if node_type == "document":
        return GraphNode(
            node_id="constitution-hash",
            node_type="document",
            label="KESTREL_CONSTITUTION",
            properties={"hash": "constitution-hash", "type": "Constitution", "created_at": now},
        )
    if node_type == "constitution_amendment_artifact":
        return GraphNode(
            node_id="amend-hash",
            node_type="constitution_amendment_artifact",
            label="Signed Constitution Reanchor Artifact",
            properties={"hash": "amend-hash", "type": "SignedConstitutionAmendment", "created_at": now},
        )
    if node_type == "audit_anchor":
        return GraphNode(
            node_id="audit_anchor_1",
            node_type="audit_anchor",
            label="Audit Anchor (3 entries)",
            properties={"anchor_hash": "deadbeef", "entries_count": 3, "created_at": now},
        )
    if node_type == "feature_config":
        return GraphNode(
            node_id="feature_config_todo",
            node_type="feature_config",
            label="todo config",
            properties={"config": {"enabled": True, "default_priority": "medium"}},
        )
    raise AssertionError(f"no structural fixture for {node_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# The governance predicate + allowlist shape
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mode,governed",
    [
        (PrivacyMode.EPHEMERAL, True),
        (PrivacyMode.ISOLATED, True),
        (PrivacyMode.DEIDENTIFIED, True),
        (PrivacyMode.NORMAL, False),
        (PrivacyMode.PUBLIC, False),
        (PrivacyMode.ANONYMOUS, False),
    ],
)
def test_graph_writes_governed_predicate(mode, governed):
    """Durable graph writes are governed in exactly the non-persistent modes."""

    class _Stub:
        agent_id = AGENT_ID

    wrapper = PrivacyEnforcingStorage(_Stub(), mode)
    assert wrapper._graph_writes_governed is governed


def test_allowlist_excludes_every_content_bearing_type():
    """The structural allowlist must never contain a conversation-derived type."""
    for node_type in CONTENT_NODE_TYPES:
        assert node_type not in STRUCTURAL_GRAPH_NODE_TYPES, (
            f"{node_type!r} is user-derived and must not be allowlisted"
        )
    # The governance edge allowlist is likewise minimal — content edges excluded.
    for label in ("knows", "records_action", "records_decision", "co_occurs_with"):
        assert label not in STRUCTURAL_GRAPH_EDGE_LABELS


def test_value_bearing_types_are_control_plane_only():
    """The value-bearing types (free text / arbitrary dict) require the trusted
    path; the hash/timestamp/count types do not (review finding P1)."""
    assert CONTROL_PLANE_ONLY_NODE_TYPES == frozenset({"agent", "feature_config"})
    for node_type in ("document", "constitution_amendment_artifact", "audit_anchor"):
        assert node_type not in CONTROL_PLANE_ONLY_NODE_TYPES


def test_content_free_value_predicate():
    """The value validator admits bounded scalars/containers and rejects blobs."""
    assert _is_content_free_value("deadbeef")
    assert _is_content_free_value(3)
    assert _is_content_free_value(None)
    assert _is_content_free_value(["AGENTS.md", "CLAUDE.md"])
    assert _is_content_free_value({"passed": True, "risk_level": 1})
    # A multi-line paragraph or an oversized string is user content, not a hash.
    assert not _is_content_free_value("line one\nline two")
    assert not _is_content_free_value("x" * 4096)


# ─────────────────────────────────────────────────────────────────────────────
# Node writes: rejected in volatile modes, no durable row (every surface)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("node_type", CONTENT_NODE_TYPES)
async def test_direct_add_node_user_content_rejected(tmp_path, mode, node_type):
    """`storage.add_node(user content)` raises and writes no durable row."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _user_content_node(node_type)

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node)

        # Fail closed: the rejected write created no durable row.
        assert await raw.get_node(node.node_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("surface", ["graph", "graph_store"])
async def test_proxy_add_node_user_content_rejected(tmp_path, mode, surface):
    """`storage.graph.add_node` / `.graph_store.add_node` reject and write nothing.

    This is the bypass the pre-#2672 code left open — the property returned the
    raw store, so feature code could persist user content directly.
    """
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _user_content_node("concept")

        proxy = getattr(wrapper, surface)
        with pytest.raises(PrivacyViolationError):
            await proxy.add_node(node)

        assert await raw.get_node(node.node_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_unknown_node_type_fails_closed(tmp_path, mode):
    """An unknown node_type is default-denied (fail closed), not admitted.

    Even the trusted control-plane capability cannot admit an unknown type — it
    only lifts the gate for the two allowlisted value-bearing types.
    """
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = GraphNode(
            node_id="mystery-1",
            node_type="some_unregistered_future_type",
            label="?",
            properties={"agent_id": AGENT_ID, "text": USER_SECRET},
        )

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node)
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.add_node(node)
        # Not even trusted callers can smuggle an unknown type through.
        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node, control_plane=True)

        assert await raw.get_node("mystery-1") is None


# ─────────────────────────────────────────────────────────────────────────────
# Edge writes: rejected in volatile modes (every surface)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_content_edge_rejected(tmp_path, mode):
    """A content relationship edge is default-denied in volatile modes."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_edge(AGENT_ID, "fact-1", "knows")
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.add_edge(AGENT_ID, "fact-1", "knows")
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph_store.add_edge(AGENT_ID, "fact-1", "records_action")


# ─────────────────────────────────────────────────────────────────────────────
# compare_and_swap: governed without decomposing the atomic primitive
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_cas_user_content_rejected_no_row(tmp_path, mode):
    """CAS of a user-derived node raises and creates no durable row.

    Compare-and-create (`expected is None`) of a user-content node must fail
    closed — the atomic primitive never runs.
    """
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        new_node = _user_content_node("decision", node_id="decision-1")

        with pytest.raises(PrivacyViolationError):
            await wrapper.compare_and_swap_node("decision-1", None, new_node)
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.compare_and_swap_node("decision-1", None, new_node)

        assert await raw.get_node("decision-1") is None


# ─────────────────────────────────────────────────────────────────────────────
# Content-free structural writes: admitted on the ORDINARY path, proven
# content-free (value-validated).
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("node_type", CONTENT_FREE_STRUCTURAL_TYPES)
async def test_content_free_structural_node_allowed(tmp_path, mode, node_type):
    """Each content-free structural type writes through the ORDINARY path AND
    carries no user content.

    Satisfies the acceptance criterion that every explicitly allowed structural
    write is covered by a test proving it contains no user content: the node is
    built from hashes/timestamps only, admitted with no control-plane capability,
    and its persisted payload is asserted free of the user-content sentinel.
    """
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node(node_type)

        # Admitted with NO control_plane capability — content-free by value.
        await wrapper.add_node(node)

        persisted = await raw.get_node(node.node_id)
        assert persisted is not None, f"structural {node_type} node must persist"

        blob = json.dumps(persisted.properties)
        assert USER_SECRET not in blob
        assert "text" not in persisted.properties


# ─────────────────────────────────────────────────────────────────────────────
# Value-bearing control-plane writes: TRUSTED path only.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("node_type", sorted(CONTROL_PLANE_ONLY_NODE_TYPES))
async def test_control_plane_node_requires_trusted_path(tmp_path, mode, node_type):
    """The value-bearing ``agent`` / ``feature_config`` nodes are admitted ONLY
    with the trusted control-plane capability; an untrusted write is denied and
    persists nothing (review finding P1)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node(node_type)

        # Untrusted (ordinary) path: default-denied.
        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node)
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.add_node(node)
        assert await raw.get_node(node.node_id) is None

        # Trusted control-plane path: admitted, content-free.
        await wrapper.add_node(node, control_plane=True)
        persisted = await raw.get_node(node.node_id)
        assert persisted is not None
        assert USER_SECRET not in json.dumps(persisted.properties)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize(
    "smuggled",
    [
        ("feature_config", {"config": {"note": USER_SECRET}}),
        ("agent", {"name": "Kestrel", "description": USER_SECRET}),
    ],
)
async def test_control_plane_type_untrusted_content_smuggling_rejected(
    tmp_path, mode, smuggled
):
    """The exact finding-P1 exploit: a caller stuffs user text into a value-bearing
    field of an allowlisted control-plane type (``feature_config.config`` /
    ``agent.description``) and writes it on the UNTRUSTED path. It must be denied,
    persisting nothing — matching by key does not make the value content-free.
    """
    node_type, props = smuggled
    node_id = AGENT_ID if node_type == "agent" else "fc-smuggle"
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            label="Kestrel" if node_type == "agent" else "todo config",
            properties=dict(props),
        )

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node)
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.add_node(node)

        assert await raw.get_node(node_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_agent_node_full_governance_vocabulary_admitted(tmp_path, mode):
    """The agent identity node carrying its FULL governance vocabulary is admitted
    on the trusted control-plane path in a volatile mode.

    The agent node accretes governance/lifecycle metadata (bootstrap status,
    doctrine + reanchor anchors, genesis audit, graduation) over its life. This
    locks the contract that the canonical-shape check accepts every field the
    production writers set — INCLUDING the doctrine-anchor fields whose omission
    disabled drift detection (review finding P3) — so a born-volatile agent can
    boot and update its governance state.
    """
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = GraphNode(
            node_id=AGENT_ID,
            node_type="agent",
            label="Kestrel",
            properties={
                "agent_id": AGENT_ID,
                "did": AGENT_ID,
                "created_at": _now_iso(),
                "constitution_hash": "c0ffee",
                "constitution_overlay_hash": "0verlay",
                "initialBalance": "1000.0",
                "name": "Kestrel",
                "description": "A sovereign agent.",
                "avatar_hash": "deadbeef",
                "bootstrap_state": "pending",
                "bootstrap_status": "pending",
                "bootstrap_stale_at": _now_iso(),
                "bootstrap_pending_age_seconds": 12,
                "genesis_audit": {"status": "passed", "risk_level": 1},
                "emancipation_contract": {"kind": "none"},
                "constitution_reanchor": {"hash": "abc"},
                "doctrine_bundle_hash": "bundle-hash",
                "doctrine_bundle_files": ["AGENTS.md", "docs/TORTOISE_DOCTRINE.md"],
                "doctrine_bundle_anchored_at": _now_iso(),
                "doctrine_bundle_reanchor": {"hash": "abc"},
                "doctrine_anchored_paths": ["AGENTS.md"],
                "graduated_at": _now_iso(),
                "is_test_instance": True,
                "test_cycle_id": "cycle-1",
                "expected_duration": "unspecified",
                "is_demo": False,
            },
        )
        await wrapper.add_node(node, control_plane=True)
        assert await raw.get_node(AGENT_ID) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_governed_by_edge_allowed_end_to_end(tmp_path, mode):
    """The structural `governed_by` binding still writes in a volatile mode.

    This is what lets the startup constitution audit bind an agent that boots in
    a volatile mode. Exercised end-to-end: the agent endpoint (trusted control-
    plane node) and the constitution anchor (content-free) are written, then the
    governance edge is written through the wrapper and read back.
    """
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)

        await wrapper.add_node(_structural_node("agent"), control_plane=True)
        await wrapper.add_node(_structural_node("document"))

        # No PrivacyViolationError — the governance edge is allowlisted.
        await wrapper.add_edge(AGENT_ID, "constitution-hash", "governed_by")

        edges = await raw.get_edges_from(AGENT_ID)
        assert any(e.label == "governed_by" and e.target_id == "constitution-hash" for e in edges)


# ─────────────────────────────────────────────────────────────────────────────
# NORMAL mode is unchanged
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_normal_mode_user_content_node_persists(tmp_path):
    """NORMAL behavior is unchanged: user-derived graph writes still persist."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        node = _user_content_node("learned_fact", node_id="fact-normal")

        # Both surfaces pass through unchanged in a persistent-write mode.
        await wrapper.add_node(node)
        persisted = await raw.get_node("fact-normal")
        assert persisted is not None
        assert persisted.properties["value"] == USER_SECRET

        # Proxy path is likewise a pass-through in NORMAL.
        node2 = _user_content_node("concept", node_id="concept-normal")
        await wrapper.graph.add_node(node2)
        assert await raw.get_node("concept-normal") is not None


@pytest.mark.asyncio
async def test_normal_mode_control_plane_type_persists_without_capability(tmp_path):
    """NORMAL: a value-bearing control-plane node persists even WITHOUT the
    capability — governance gating is off in persistent modes (no regression)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        node = GraphNode(
            node_id="fc-normal",
            node_type="feature_config",
            label="todo config",
            properties={"config": {"user_note": "anything goes in NORMAL"}},
        )
        await wrapper.add_node(node)  # no control_plane needed
        assert await raw.get_node("fc-normal") is not None


@pytest.mark.asyncio
async def test_normal_mode_content_edge_and_cas_persist(tmp_path):
    """NORMAL: content edges and CAS are unchanged (no governance rejection)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)

        # Compare-and-create a user-content node in NORMAL — should succeed.
        result = await wrapper.compare_and_swap_node(
            "fact-cas", None, _user_content_node("learned_fact", node_id="fact-cas")
        )
        assert result == NodeSwapResult.SWAPPED
        assert await raw.get_node("fact-cas") is not None

        # A content edge between two owned nodes writes in NORMAL.
        await wrapper.add_node(_structural_node("agent"))
        await wrapper.add_edge(AGENT_ID, "fact-cas", "knows")
        edges = await raw.get_edges_from(AGENT_ID)
        assert any(e.label == "knows" for e in edges)


# ─────────────────────────────────────────────────────────────────────────────
# P1: the allowlist enforces content-free SHAPE, not just the type label
#
# Admitting a write on ``node_type`` / edge ``label`` alone let user text ride
# through in an allowlisted type's properties/label (or an edge's payload).
# These prove the policy now rejects user content smuggled into an allowlisted
# node or edge — the case the earlier fixtures did not cover.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("node_type", sorted(STRUCTURAL_GRAPH_NODE_TYPES))
async def test_structural_node_rejects_smuggled_property(tmp_path, mode, node_type):
    """A user-content sentinel under a non-canonical property key is rejected.

    This is a realistic leak vector: a feature stuffs conversation text into a
    ``text`` / ``summary`` / ``note`` property on an otherwise-structural node.
    The per-type key allowlist fails closed on any key outside the canonical set,
    on both the ordinary and (for control-plane types) the trusted path.
    """
    admit = _admit_kwargs(node_type)
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node(node_type)
        node.properties["smuggled_note"] = USER_SECRET  # non-canonical key

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node, **admit)
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.add_node(node, **admit)

        # Fail closed: nothing persisted.
        assert await raw.get_node(node.node_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("node_type", CONTENT_FREE_STRUCTURAL_TYPES)
async def test_content_free_node_rejects_smuggled_value(tmp_path, mode, node_type):
    """User text smuggled into a CANONICAL key's value (not a new key) is rejected.

    A caller writes a content-free type but stuffs a multi-line paragraph into a
    ``hash`` / ``type`` / ``anchor_hash`` field. Value validation fails closed so
    matching a known key is not enough to make the value content-free (P1)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node(node_type)
        # Overwrite a canonical key with a value that is unmistakably user content.
        victim_key = next(iter(node.properties))
        node.properties[victim_key] = f"stolen conversation:\n{USER_SECRET}"

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node)
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.add_node(node)

        assert await raw.get_node(node.node_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize(
    "node_type",
    ["document", "constitution_amendment_artifact", "audit_anchor"],
)
async def test_structural_node_rejects_smuggled_label(tmp_path, mode, node_type):
    """User text in a fixed-label structural node's label is rejected.

    ``document`` / ``constitution_amendment_artifact`` have exact literal labels
    and ``audit_anchor`` a fixed ``Audit Anchor (N entries)`` shape, so a label
    carrying conversation text cannot pass.
    """
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node(node_type)
        node.label = USER_SECRET  # non-canonical label

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node)
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.add_node(node)

        assert await raw.get_node(node.node_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_governed_by_edge_rejects_user_content_properties(tmp_path, mode):
    """A structural governance edge carrying a user-content payload is rejected.

    ``governed_by`` is admitted only as a pure binding: any properties could
    carry user text, so a payload-bearing structural edge fails closed while the
    property-free binding still writes.
    """
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        await wrapper.add_node(_structural_node("agent"), control_plane=True)
        await wrapper.add_node(_structural_node("document"))

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_edge(
                AGENT_ID, "constitution-hash", "governed_by", {"note": USER_SECRET}
            )
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.add_edge(
                AGENT_ID, "constitution-hash", "governed_by", {"note": USER_SECRET}
            )

        # The clean, property-free binding is still admitted.
        await wrapper.add_edge(AGENT_ID, "constitution-hash", "governed_by")
        edges = await raw.get_edges_from(AGENT_ID)
        assert any(e.label == "governed_by" for e in edges)


# ─────────────────────────────────────────────────────────────────────────────
# P2: CAS cannot rewrite an EXISTING user-derived node via a spoofed type
#
# The primitive is properties-only and ignores ``new_node.node_type`` on the
# swap path, so authorizing on that type would let user content ride onto a
# stored ``concept`` / ``episode`` row. The stored-type pin blocks it atomically.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("surface", ["facade", "graph"])
async def test_cas_cannot_overwrite_user_node_via_spoofed_type(
    tmp_path, mode, surface
):
    """CAS-swapping an existing user-derived node under a spoofed structural type
    is rejected, and the stored row is left byte-for-byte intact.

    Even WITH the trusted control-plane capability, the atomic stored-type pin
    refuses because the STORED row is a ``concept`` — the capability admits the
    write intent but the primitive still won't relabel a user node."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        # Seed a durable user-derived concept node, as NORMAL mode would.
        original = {"agent_id": AGENT_ID, "value": "original-concept-text"}
        await raw.add_node(
            GraphNode(
                node_id="concept-x",
                node_type="concept",
                label="c",
                properties=dict(original),
            )
        )

        wrapper = PrivacyEnforcingStorage(raw, mode)
        # Spoof the type to a structural one AND shape the payload like a valid
        # feature_config ({config: ...}) so the shape check passes — the atomic
        # stored-type pin must still refuse because the STORED row is a concept.
        spoof = GraphNode(
            node_id="concept-x",
            node_type="feature_config",
            label="todo config",
            properties={"config": {"leaked": USER_SECRET}},
        )
        target = wrapper if surface == "facade" else wrapper.graph
        with pytest.raises(PrivacyViolationError):
            await target.compare_and_swap_node(
                "concept-x", original, spoof, control_plane=True
            )

        # The stored concept is untouched: same type, same properties, no leak.
        stored = await raw.get_node("concept-x")
        assert stored.node_type == "concept"
        assert stored.properties == original
        assert USER_SECRET not in json.dumps(stored.properties)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_cas_swap_of_existing_structural_node_allowed(tmp_path, mode):
    """A properties swap on an EXISTING content-free structural node still lands
    in a volatile mode — the stored-type pin permits the row's own type.

    ``feature_config`` is a control-plane type, so the trusted capability is
    required; the stored-type pin then admits the swap onto its own type."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        seed_props = {"config": {"enabled": True}}
        await raw.add_node(
            GraphNode(
                node_id="fc-1",
                node_type="feature_config",
                label="todo config",
                properties=dict(seed_props),
            )
        )

        wrapper = PrivacyEnforcingStorage(raw, mode)
        new = GraphNode(
            node_id="fc-1",
            node_type="feature_config",
            label="todo config",
            properties={"config": {"enabled": False}},
        )
        result = await wrapper.compare_and_swap_node(
            "fc-1", seed_props, new, control_plane=True
        )
        assert result == NodeSwapResult.SWAPPED

        stored = await raw.get_node("fc-1")
        assert stored.properties == {"config": {"enabled": False}}


# ─────────────────────────────────────────────────────────────────────────────
# P3: manual/scheduled consolidation cannot persist a durable episode while
# volatile — neither the memory_episodes row nor the KG episode node — AND the
# whole durable pipeline (temporal patterns via the raw DB) is gated.
# ─────────────────────────────────────────────────────────────────────────────


def _episode() -> "object":
    from kestrel_sovereign.storage.memory_models import MemoryEpisode

    now = datetime.now(timezone.utc)
    return MemoryEpisode(
        id="episode:test:secret",
        agent_id=AGENT_ID,
        title="A user secret",
        summary=USER_SECRET,
        timespan_start=now,
        timespan_end=now,
        key_message_ids=[1, 2, 3],
        emotional_arc="neutral",
        importance=0.5,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_volatile_consolidation_persists_no_episode(tmp_path, mode):
    """The consolidator wired exactly as MemorySystem wires it (governed graph +
    persist gate) writes NO durable episode in a volatile mode — closing the
    manual/scheduled ``memory_consolidate`` bypass of the graph proxy (#2672)."""
    from kestrel_sovereign.storage.memory_system import MemorySystem

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        ms = MemorySystem(storage=raw, agent_id=AGENT_ID, privacy_storage=wrapper)
        await ms.initialize()

        # The single durable-write chokepoint both nightly and session/manual
        # consolidation funnel through.
        await ms.consolidator._save_episode(_episode())

        rows = await raw.db.fetchall(
            "SELECT id FROM memory_episodes WHERE agent_id = ?", (AGENT_ID,)
        )
        assert rows == [], "no user-derived episode row may persist while volatile"
        assert await raw.get_node("episode:test:secret") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_volatile_run_consolidation_skips_durable_path(tmp_path, mode):
    """run_consolidation fails closed BEFORE the durable pipeline in a volatile
    mode: no temporal patterns (the raw-DB leak the graph proxy can't see), no
    episodes, and an explicit privacy-blocked report so the manual
    ``memory_consolidate`` tool cannot report success (#2672 finding P2)."""
    from kestrel_sovereign.storage.memory_system import MemorySystem

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        # Seed recent user rows (as a prior NORMAL stint would) so the durable
        # pipeline WOULD detect a temporal pattern / build an episode if it ran.
        now = datetime.now(timezone.utc)
        for i in range(30):
            ts = (now - timedelta(days=1, minutes=i)).isoformat()
            await raw.db.execute(
                "INSERT INTO conversation_history "
                "(agent_id, role, content, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (AGENT_ID, "user", f"late night message {i}", "{}", ts),
            )

        wrapper = PrivacyEnforcingStorage(raw, mode)
        ms = MemorySystem(storage=raw, agent_id=AGENT_ID, privacy_storage=wrapper)
        await ms.initialize()

        report = await ms.consolidate()

        assert report.get("skipped") is True
        assert report.get("privacy_blocked") is True
        assert report.get("episodes_deleted", 0) == 0  # forgetting tier also gated

        patterns = await raw.db.fetchall(
            "SELECT id FROM temporal_patterns WHERE agent_id = ?", (AGENT_ID,)
        )
        assert patterns == [], "no durable temporal pattern may persist while volatile"
        episodes = await raw.db.fetchall(
            "SELECT id FROM memory_episodes WHERE agent_id = ?", (AGENT_ID,)
        )
        assert episodes == []


@pytest.mark.asyncio
async def test_normal_consolidation_persists_episode(tmp_path):
    """NORMAL is unchanged: consolidation still persists the episode row + KG node."""
    from kestrel_sovereign.storage.memory_system import MemorySystem

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        ms = MemorySystem(storage=raw, agent_id=AGENT_ID, privacy_storage=wrapper)
        await ms.initialize()

        await ms.consolidator._save_episode(_episode())

        rows = await raw.db.fetchall(
            "SELECT id FROM memory_episodes WHERE agent_id = ?", (AGENT_ID,)
        )
        assert len(rows) == 1
        assert await raw.get_node("episode:test:secret") is not None
