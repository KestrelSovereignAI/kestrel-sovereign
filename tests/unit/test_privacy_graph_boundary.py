"""Storage-boundary privacy tests for durable graph writes (#2672).

The #1760 gate blocks the post-response/background memory pipeline in volatile
privacy modes, but the *storage* boundary still waved every graph write through
as "structural, not PII" — so a feature calling ``storage.add_node`` (or
reaching the raw store through ``storage.graph`` / ``storage.graph_store``)
could persist user-derived facts, todos, decisions, concepts, and consolidated
episodes directly, outside that gate.

This suite proves the storage boundary now default-denies durable user-derived
and unknown graph writes in EPHEMERAL / ISOLATED / DEIDENTIFIED — for node
writes, edge writes, and the atomic compare-and-swap — across every surface (the
wrapper's own methods AND the ``.graph`` / ``.graph_store`` proxies).

The admit policy is CONTENT-based, not caller-identity-based:

* **Content-free structural types** — ``document`` / ``audit_anchor`` (and the
  ``governed_by`` edge) — are admitted on the ordinary (untrusted) path, but
  every field is validated by a PER-FIELD SEMANTIC validator (finding P1): a
  ``hash`` must be hex of a digest's length, a ``*_at`` must parse as a
  timestamp, an ``entries_count`` must be a non-negative int, ``document.type``
  must be the literal ``"Constitution"``, ``document.hash`` must equal the
  node's own content-hash id. A short secret is none of those.
* **Control-plane types** — ``agent`` / ``constitution_amendment_artifact`` —
  carry a control-plane CAPABILITY marker, but the marker is same-process
  defense-in-depth, NOT an authorization boundary: the governance writers are
  mixin methods on the agent and feature code holds the agent, so any in-process
  caller can obtain the marker (e.g. by ``exec``-ing into a trusted module's
  ``__dict__``). The LOAD-BEARING privacy gate for the ``agent`` node is the
  CARRIED-ALONG identity boundary — user-facing free-text (name / description /
  expected_duration) is admitted only when unchanged from the stored node, so a
  fresh/changed value is refused even to a caller holding a genuine (forged)
  marker (this is what closes the reproduced ``agent.description`` leak). The
  governance-receipt free-text fields are a documented process-isolation residual.

The two user-derived surfaces the review told us not to blanket-trust —
``agent.description`` and ``feature_config.config`` — are gated at their single
source of truth (``persist_agent_description`` / ``Feature.persist_config`` skip
their durable writes while volatile), so ``feature_config`` is not allowlisted at
all. NORMAL / PUBLIC / ANONYMOUS are unchanged.
"""
from __future__ import annotations

import json
import types
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
    _TRUSTED_CONTROL_PLANE_MODULES,
    _ControlPlaneCapability,
    _has_control_plane_capability,
    _is_count,
    _is_enum_token,
    _is_hex_hash,
    _is_id_token,
    _is_iso_timestamp,
    acquire_control_plane_capability,
)


AGENT_ID = "did:test:graph-boundary"


def _capability_from_trusted_module(
    module_name: str = "kestrel_sovereign.bootstrap.service",
):
    """Obtain the real control-plane marker the way a genuine writer does.

    NOTE (honesty): this same technique — binding a probe into a real trusted
    module's ``__dict__`` so its ``f_globals`` matches the issuer's check — is
    ALSO exactly what an in-process feature can do; any code can
    ``import kestrel_sovereign.agent.constitution`` and ``exec`` into its dict.
    That is precisely why the marker is only defense-in-depth and the real privacy
    gate is content-based (see ``test_module_dict_injection_cannot_persist_*``).
    """
    import importlib

    module = importlib.import_module(module_name)
    namespace = module.__dict__
    # Some trusted modules import the acquire fn only locally (inside a method),
    # so temporarily bind it as a module global for the probe's global lookup,
    # then restore. The provenance check is on the frame's globals-dict IDENTITY,
    # not on this name being present, so injecting it changes nothing about trust.
    had_name = "acquire_control_plane_capability" in namespace
    saved = namespace.get("acquire_control_plane_capability")
    namespace["acquire_control_plane_capability"] = acquire_control_plane_capability
    exec("def __cp_probe():\n    return acquire_control_plane_capability()", namespace)
    try:
        return namespace["__cp_probe"]()
    finally:
        namespace.pop("__cp_probe", None)
        if had_name:
            namespace["acquire_control_plane_capability"] = saved
        else:
            namespace.pop("acquire_control_plane_capability", None)


# The one true capability, obtained via a genuine trusted-module acquisition —
# standing in for what a first-party writer presents at its write site.
CAP = _capability_from_trusted_module()

# The privacy modes whose contract forbids durable persistence of user data.
VOLATILE_MODES = [
    PrivacyMode.EPHEMERAL,
    PrivacyMode.ISOLATED,
    PrivacyMode.DEIDENTIFIED,
]

# A realistic 64-hex SHA-256 digest (what ``store_file`` / the identity writers
# produce) and a second distinct one.
VALID_HASH = "0123456789abcdef" * 4
VALID_HASH2 = "fedcba9876543210" * 4

# The content-free structural types, admitted on the ordinary (untrusted) path.
CONTENT_FREE_STRUCTURAL_TYPES = sorted(
    STRUCTURAL_GRAPH_NODE_TYPES - CONTROL_PLANE_ONLY_NODE_TYPES
)

# A sentinel standing in for user conversation content. It must NEVER reach a
# durable graph row in a volatile mode.
USER_SECRET = "SECRET-USER-CONTENT-must-never-persist"

# A SHORT, single-line, alphanumeric secret — the exact case the review flagged:
# the old "any string <= 512 single-line chars is content-free" check admitted
# it into a hash / type / timestamp field. The per-field validators must reject
# it in EVERY content-free field surface.
SHORT_SECRET = "hunter2"

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

    Control-plane types require the unforgeable capability; content-free types do
    not. Keeps every "allowed" assertion honest about which path admits it.
    """
    if node_type in CONTROL_PLANE_ONLY_NODE_TYPES:
        return {"capability": CAP}
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
    """A minimal, VALID content-free structural node for each allowlisted type.

    Payloads are the agent's own identity/governance state or cryptographic
    hashes with realistic per-field shapes — never user conversation text.
    """
    now = _now_iso()
    if node_type == "agent":
        # Governance-state agent node: content-free identity/governance fields
        # ONLY. User-facing identity free-text (name/description/expected_duration)
        # is deliberately absent — in a volatile mode those are admitted only
        # carried along from the stored node (see the carried-along tests), so a
        # fixture that introduces them fresh would be (correctly) refused (#2672).
        return GraphNode(
            node_id=AGENT_ID,
            node_type="agent",
            label="Kestrel",
            properties={
                "constitution_hash": VALID_HASH,
                "created_at": now,
            },
        )
    if node_type == "document":
        return GraphNode(
            node_id=VALID_HASH,
            node_type="document",
            label="KESTREL_CONSTITUTION",
            properties={"hash": VALID_HASH, "type": "Constitution", "created_at": now},
        )
    if node_type == "constitution_amendment_artifact":
        return GraphNode(
            node_id=VALID_HASH2,
            node_type="constitution_amendment_artifact",
            label="Signed Constitution Reanchor Artifact",
            properties={
                "hash": VALID_HASH2,
                "type": "SignedConstitutionAmendment",
                "artifact_type": "reanchor",
                "constitution_hash": VALID_HASH,
                "signer": "did:key:z6MkExampleSigner",
                "source_path": "/data/agent/KESTREL_CONSTITUTION.reanchor.signed.json",
                "created_at": now,
                "anchored_at": now,
                "verification": {"reason": "signature valid"},
            },
        )
    if node_type == "audit_anchor":
        return GraphNode(
            node_id="audit_anchor_1",
            node_type="audit_anchor",
            label="Audit Anchor (3 entries)",
            properties={
                "anchor_hash": VALID_HASH,
                "storage_ref": VALID_HASH2,
                "entries_count": 3,
                "first_entry_at": now,
                "last_entry_at": now,
                "created_at": now,
            },
        )
    raise AssertionError(f"no structural fixture for {node_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# The control-plane capability marker — defense-in-depth only (#2672 P1/P2)
#
# These tests document the marker's behaviour (it stops accidental/trivially-
# forged passes) WITHOUT claiming it is an authorization boundary. The
# ``test_module_dict_injection_cannot_persist_*`` tests below prove the marker IS
# obtainable by in-process forgery, and that the CONTENT boundary refuses the leak
# anyway — that is the load-bearing guarantee.
# ─────────────────────────────────────────────────────────────────────────────


def _forged_name_acquire(module_name: str):
    """A weaker spoof: ``exec`` an acquire call in a namespace whose ``__name__``
    is a trusted module's name but whose dict is a FRESH object, not that module's
    real ``__dict__``. The module-dict-identity check rejects this — but note the
    STRONGER forge in ``test_module_dict_injection_*`` (exec into the REAL dict)
    succeeds, which is why the marker is not relied on."""
    ns = {
        "acquire_control_plane_capability": acquire_control_plane_capability,
        "__name__": module_name,
    }
    exec("def _call():\n    return acquire_control_plane_capability()", ns)
    return ns["_call"]()


@pytest.mark.parametrize("trusted", sorted(_TRUSTED_CONTROL_PLANE_MODULES))
def test_forged_module_name_cannot_obtain_capability(trusted):
    """A forged ``__name__`` on a FRESH dict does NOT yield the marker — the check
    is module-DICT identity, not the ``__name__`` string. (A stronger forge that
    exec's into the REAL module dict DOES succeed; see the injection tests — the
    marker is defense-in-depth, not a boundary.)"""
    with pytest.raises(PrivacyViolationError):
        _forged_name_acquire(trusted)


def test_acquire_from_this_test_module_is_refused():
    """A direct call from this (untrusted) test module is refused — the check is
    on the real caller frame's namespace identity, which this module isn't. This
    only keeps casual imports out; it is not a security boundary."""
    with pytest.raises(PrivacyViolationError):
        acquire_control_plane_capability()


def test_capability_issued_only_from_genuine_trusted_namespace():
    """A call from a trusted module's ACTUAL namespace yields the one marker, and
    every trusted module returns the SAME singleton."""
    for trusted in sorted(_TRUSTED_CONTROL_PLANE_MODULES):
        cap = _capability_from_trusted_module(trusted)
        assert _has_control_plane_capability(cap)
        assert cap is CAP


def test_capability_singleton_is_not_importable():
    """There is NO importable ``_CONTROL_PLANE_CAPABILITY`` module attribute — the
    marker lives only in the closure, so it can't be casually imported (though it
    remains reachable via in-process introspection; see the injection tests)."""
    import kestrel_sovereign.storage.privacy_wrapper as pw

    assert not hasattr(pw, "_CONTROL_PLANE_CAPABILITY")


def test_capability_recognized_by_identity_only():
    """The wrapper accepts ONLY the closure-private singleton, by identity — a
    boolean, an int, an arbitrary object, a freshly-constructed capability, or
    ``None`` are all rejected (stops trivially-forged markers)."""
    assert _has_control_plane_capability(CAP)
    assert not _has_control_plane_capability(True)
    assert not _has_control_plane_capability(1)
    assert not _has_control_plane_capability("control_plane")
    assert not _has_control_plane_capability(object())
    assert not _has_control_plane_capability(_ControlPlaneCapability())  # fresh
    assert not _has_control_plane_capability(None)


# ─────────────────────────────────────────────────────────────────────────────
# The load-bearing boundary: module-dict injection is refused by CONTENT, not
# by caller identity (#2672 review P1 — the regression the review demanded)
# ─────────────────────────────────────────────────────────────────────────────


def _capability_via_module_dict_injection(module_name: str):
    """The flagged forge (#2672 review P1): a feature ``import``s a REAL trusted
    module and ``exec``s a helper into its genuine ``__dict__``, so the helper's
    ``f_globals`` IS that module's dict — the exact identity the issuer checks.
    This SUCCEEDS: any in-process code can do it, which is why caller identity
    cannot be the boundary."""
    import importlib

    module = importlib.import_module(module_name)
    ns = module.__dict__
    had = "acquire_control_plane_capability" in ns
    saved = ns.get("acquire_control_plane_capability")
    ns["acquire_control_plane_capability"] = acquire_control_plane_capability
    exec("def __inj():\n    return acquire_control_plane_capability()", ns)
    try:
        return ns["__inj"]()
    finally:
        ns.pop("__inj", None)
        if had:
            ns["acquire_control_plane_capability"] = saved
        else:
            ns.pop("acquire_control_plane_capability", None)


def test_module_dict_injection_does_obtain_the_marker():
    """Documents the hole honestly: injecting into a real trusted module dict
    DOES yield the genuine marker. The privacy guarantee must therefore not rest
    on it — the next test proves the write is refused anyway."""
    forged = _capability_via_module_dict_injection(
        "kestrel_sovereign.agent.constitution"
    )
    assert _has_control_plane_capability(forged)  # the forge works...


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("surface", ["facade", "graph"])
async def test_module_dict_injection_cannot_persist_fresh_description(
    tmp_path, mode, surface
):
    """The regression the review demanded: code injected into a REAL trusted
    module's ``__dict__`` obtains the marker, yet CANNOT persist a fresh
    ``agent.description`` (or ``name``) in a volatile mode — the carried-along
    CONTENT boundary refuses the changed value regardless of the (genuine, forged)
    marker, and the stored node is left intact (#2672 review P1)."""
    forged = _capability_via_module_dict_injection(
        "kestrel_sovereign.agent.constitution"
    )
    assert _has_control_plane_capability(forged)  # injection yields the real marker

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        # A pre-existing agent node with a stored (inception-written) identity.
        await raw.add_node(GraphNode(
            node_id=AGENT_ID,
            node_type="agent",
            label="Kestrel",
            properties={
                "constitution_hash": VALID_HASH,
                "created_at": _now_iso(),
                "name": "Kestrel",
                "description": "the original, operator-authored bio",
            },
        ))
        wrapper = PrivacyEnforcingStorage(raw, mode)
        evil = GraphNode(
            node_id=AGENT_ID,
            node_type="agent",
            label="Kestrel",
            properties={
                "constitution_hash": VALID_HASH,
                "created_at": _now_iso(),
                "name": "Kestrel",
                "description": f"exfiltrated: {USER_SECRET}",  # fresh user content
            },
        )
        target = wrapper if surface == "facade" else wrapper.graph
        with pytest.raises(PrivacyViolationError):
            await target.add_node(evil, capability=forged)

        stored = await raw.get_node(AGENT_ID)
        assert stored.properties["description"] == "the original, operator-authored bio"
        assert USER_SECRET not in json.dumps(stored.properties)


# ─────────────────────────────────────────────────────────────────────────────
# Carried-along identity boundary on the agent node (#2672 review P1)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("field", ["name", "description", "expected_duration"])
async def test_fresh_identity_field_refused_even_with_capability(
    tmp_path, mode, field
):
    """Introducing a user-facing identity field that is absent from the stored node
    is refused in a volatile mode — WITH the genuine marker — because it is fresh
    user content (there is no stored value to carry)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        await raw.add_node(_structural_node("agent"))  # no user-facing identity
        wrapper = PrivacyEnforcingStorage(raw, mode)

        node = _structural_node("agent")
        node.properties[field] = USER_SECRET
        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node, capability=CAP)

        stored = await raw.get_node(AGENT_ID)
        assert field not in (stored.properties or {})
        assert USER_SECRET not in json.dumps(stored.properties)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_changed_identity_field_refused_even_with_capability(tmp_path, mode):
    """Changing an existing user-facing identity value is refused — only the
    UNCHANGED (carried-along) value is admitted."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        await raw.add_node(GraphNode(
            node_id=AGENT_ID, node_type="agent", label="Kestrel",
            properties={"constitution_hash": VALID_HASH, "created_at": _now_iso(),
                        "name": "Kestrel", "description": "stored bio"},
        ))
        wrapper = PrivacyEnforcingStorage(raw, mode)

        changed = GraphNode(
            node_id=AGENT_ID, node_type="agent", label="Kestrel",
            properties={"constitution_hash": VALID_HASH, "created_at": _now_iso(),
                        "name": "Kestrel", "description": USER_SECRET},
        )
        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(changed, capability=CAP)
        stored = await raw.get_node(AGENT_ID)
        assert stored.properties["description"] == "stored bio"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_carried_along_identity_admitted_with_governance_mutation(
    tmp_path, mode
):
    """A governance write that COPIES the stored node (carrying name/description
    unchanged) and mutates only governance state is admitted — this is the real
    ``mark_stale_bootstrap`` / reanchor shape, so a born-volatile agent stays
    governable."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        await raw.add_node(GraphNode(
            node_id=AGENT_ID, node_type="agent", label="Kestrel",
            properties={"constitution_hash": VALID_HASH, "created_at": _now_iso(),
                        "name": "Kestrel", "description": "stored bio"},
        ))
        wrapper = PrivacyEnforcingStorage(raw, mode)

        stored = await raw.get_node(AGENT_ID)
        updated = GraphNode(
            node_id=AGENT_ID, node_type="agent", label="Kestrel",
            properties={**dict(stored.properties), "bootstrap_state": "complete"},
        )
        await wrapper.add_node(updated, capability=CAP)  # carries name/description
        after = await raw.get_node(AGENT_ID)
        assert after.properties["bootstrap_state"] == "complete"
        assert after.properties["description"] == "stored bio"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_cas_create_with_fresh_identity_refused(tmp_path, mode):
    """A compare-and-CREATE (``expected is None``) of an agent node carrying a
    fresh identity field is refused — there is nothing to carry along, so it is
    fresh user content (atomic, no pre-read)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node("agent")
        node.properties["description"] = USER_SECRET
        with pytest.raises(PrivacyViolationError):
            await wrapper.compare_and_swap_node(AGENT_ID, None, node, capability=CAP)
        assert await raw.get_node(AGENT_ID) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_cas_swap_changing_identity_refused(tmp_path, mode):
    """A CAS swap whose ``new_node`` changes a user-facing identity field vs. the
    ``expected`` snapshot is refused, atomically, without touching the stored row."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        seed = GraphNode(
            node_id=AGENT_ID, node_type="agent", label="Kestrel",
            properties={"constitution_hash": VALID_HASH, "created_at": _now_iso(),
                        "name": "Kestrel", "description": "stored bio"},
        )
        await raw.add_node(seed)
        wrapper = PrivacyEnforcingStorage(raw, mode)
        new = GraphNode(
            node_id=AGENT_ID, node_type="agent", label="Kestrel",
            properties={**dict(seed.properties), "description": USER_SECRET},
        )
        with pytest.raises(PrivacyViolationError):
            await wrapper.compare_and_swap_node(
                AGENT_ID, seed.properties, new, capability=CAP
            )
        stored = await raw.get_node(AGENT_ID)
        assert stored.properties["description"] == "stored bio"


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
    for label in ("knows", "records_action", "records_decision", "co_occurs_with"):
        assert label not in STRUCTURAL_GRAPH_EDGE_LABELS


def test_control_plane_types_are_capability_gated():
    """The single capability-gated type is the ``agent`` identity node; the
    hash/timestamp types stay on the ordinary path, and neither ``feature_config``
    nor ``constitution_amendment_artifact`` is allowlisted at all (the former is
    source-gated, the latter is always-fresh free-text, review P1/P3)."""
    assert CONTROL_PLANE_ONLY_NODE_TYPES == frozenset({"agent"})
    for node_type in ("document", "audit_anchor"):
        assert node_type not in CONTROL_PLANE_ONLY_NODE_TYPES
    assert "feature_config" not in STRUCTURAL_GRAPH_NODE_TYPES
    # The signed reanchor artifact is no longer a wrapper-admissible type: it is
    # always a fresh node whose free-text ``source_path`` could never be carried
    # along, so it is default-denied in volatile modes (the reproduced review
    # exploit surface, #2672 review P1).
    assert "constitution_amendment_artifact" not in STRUCTURAL_GRAPH_NODE_TYPES


def test_per_field_validators_reject_short_secrets():
    """The per-field semantic validators (the P1 fix) reject the short secret the
    old ``<=512 single-line`` check admitted, and accept realistic values."""
    # Hashes: hex of a digest's length only.
    assert _is_hex_hash(VALID_HASH)
    assert not _is_hex_hash(SHORT_SECRET)
    assert not _is_hex_hash("deadbeef")  # valid hex but too short to be a digest
    # Timestamps: must actually parse.
    assert _is_iso_timestamp(_now_iso())
    assert _is_iso_timestamp("2026-07-21 12:34:56")
    assert not _is_iso_timestamp(SHORT_SECRET)
    # Counts: non-negative int, not a numeric string.
    assert _is_count(3)
    assert not _is_count("3")
    assert not _is_count(SHORT_SECRET)
    assert not _is_count(True)
    # Enum tokens: bounded, no spaces — a paragraph is rejected.
    assert _is_enum_token("Constitution")
    assert not _is_enum_token("a paragraph of user text")
    # Id tokens: DID-shaped.
    assert _is_id_token(AGENT_ID)
    assert not _is_id_token("has spaces in it")


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
        # Even presenting the real capability cannot admit a user-derived type.
        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node, capability=CAP)

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
    """An unknown node_type is default-denied — even WITH the capability."""
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
        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node, capability=CAP)

        assert await raw.get_node("mystery-1") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_feature_config_default_denied(tmp_path, mode):
    """``feature_config`` was removed from the allowlist (its ``config`` is
    source-gated at ``Feature.persist_config``), so it is default-denied in
    volatile modes — WITH or WITHOUT the capability (P3)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = GraphNode(
            node_id="feature_config_evil",
            node_type="feature_config",
            label="evil config",
            properties={"config": {"api_key": USER_SECRET}},
        )
        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node)
        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node, capability=CAP)
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.add_node(node)

        assert await raw.get_node("feature_config_evil") is None


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
    """CAS of a user-derived node raises and creates no durable row."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        new_node = _user_content_node("decision", node_id="decision-1")

        with pytest.raises(PrivacyViolationError):
            await wrapper.compare_and_swap_node("decision-1", None, new_node)
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.compare_and_swap_node("decision-1", None, new_node)

        assert await raw.get_node("decision-1") is None


# ─────────────────────────────────────────────────────────────────────────────
# Content-free structural writes: admitted on the ORDINARY path (P1)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("node_type", CONTENT_FREE_STRUCTURAL_TYPES)
async def test_content_free_structural_node_allowed(tmp_path, mode, node_type):
    """Each content-free structural type writes through the ORDINARY path (no
    capability) AND its persisted payload carries no user content."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node(node_type)

        await wrapper.add_node(node)  # no capability — content-free by shape

        persisted = await raw.get_node(node.node_id)
        assert persisted is not None, f"structural {node_type} node must persist"
        blob = json.dumps(persisted.properties)
        assert USER_SECRET not in blob
        assert "text" not in persisted.properties


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_document_hash_must_equal_node_id(tmp_path, mode):
    """A ``document`` whose ``hash`` is a valid 64-hex string but is NOT the
    node's own id is rejected — closing the "arbitrary 32 bytes in a hash field"
    channel (P1)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = GraphNode(
            node_id=VALID_HASH,
            node_type="document",
            label="KESTREL_CONSTITUTION",
            properties={"hash": VALID_HASH2, "type": "Constitution", "created_at": _now_iso()},
        )
        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node)
        assert await raw.get_node(VALID_HASH) is None


# The exact (type, field) surfaces a short secret could be smuggled into on the
# ordinary path. Each must be rejected — the regression the review demanded.
_CONTENT_FREE_FIELD_SURFACES = [
    ("document", "hash"),
    ("document", "type"),
    ("document", "created_at"),
    ("audit_anchor", "anchor_hash"),
    ("audit_anchor", "storage_ref"),
    ("audit_anchor", "entries_count"),
    ("audit_anchor", "first_entry_at"),
    ("audit_anchor", "last_entry_at"),
    ("audit_anchor", "created_at"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("node_type,field", _CONTENT_FREE_FIELD_SURFACES)
async def test_short_secret_rejected_in_every_content_free_field(
    tmp_path, mode, node_type, field
):
    """A SHORT single-line secret placed in ANY content-free field is rejected.

    This is the P1 regression: the old ``<=512 single-line chars`` check admitted
    ``hunter2`` into a hash / type / timestamp / count field; the per-field
    semantic validators reject it everywhere.
    """
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node(node_type)
        node.properties[field] = SHORT_SECRET

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node)
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.add_node(node)

        assert await raw.get_node(node.node_id) is None


# ─────────────────────────────────────────────────────────────────────────────
# Control-plane writes: CAPABILITY required (review finding P2)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("node_type", sorted(CONTROL_PLANE_ONLY_NODE_TYPES))
async def test_control_plane_node_requires_capability(tmp_path, mode, node_type):
    """A control-plane node is admitted ONLY with the unforgeable capability;
    without it (or on the proxy without it) the write is denied and persists
    nothing."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node(node_type)

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node)
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.add_node(node)
        assert await raw.get_node(node.node_id) is None

        # With the capability: admitted, content-free.
        await wrapper.add_node(node, capability=CAP)
        persisted = await raw.get_node(node.node_id)
        assert persisted is not None
        assert USER_SECRET not in json.dumps(persisted.properties)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("forged", [True, 1, "control_plane", None])
async def test_control_plane_node_rejects_forged_capability(tmp_path, mode, forged):
    """A boolean / int / string / ``None`` in the ``capability`` slot cannot
    stand in for the token — the write is denied and nothing persists (P2)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node("agent")

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node, capability=forged)
        assert await raw.get_node(AGENT_ID) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_control_plane_node_rejects_fresh_capability_instance(tmp_path, mode):
    """A freshly-constructed ``_ControlPlaneCapability()`` is NOT the singleton and
    is rejected — the check is object identity, not type (P2)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node("agent")

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node, capability=_ControlPlaneCapability())
        assert await raw.get_node(AGENT_ID) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize(
    "node_type,bad_field",
    [
        ("agent", "constitution_hash"),
        ("agent", "created_at"),
    ],
)
async def test_control_plane_field_validation_is_defense_in_depth(
    tmp_path, mode, node_type, bad_field
):
    """Even WITH the capability, a control-plane node whose content-free field
    holds a short secret is rejected — per-field validation runs behind the
    capability as defense-in-depth."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node(node_type)
        node.properties[bad_field] = SHORT_SECRET

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node, capability=CAP)
        assert await raw.get_node(node.node_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_agent_node_full_governance_vocabulary_admitted(tmp_path, mode):
    """The agent identity node carrying its FULL governance vocabulary — including
    the free-text governance receipts — is admitted in a volatile mode WHEN THE
    FREE-TEXT IS CARRIED ALONG UNCHANGED from the stored node. This is the real
    ``mark_stale_bootstrap`` / doctrine-boot-anchor shape (copy the node, mutate
    only content-free lifecycle state), so a born-volatile agent stays governable
    without ever admitting a fresh free-text write through the wrapper (#2672
    review P1)."""
    stored_receipts = {
        # User-facing identity free-text (inception writes these to the RAW store):
        "name": "Kestrel",
        "description": "A sovereign agent.\nMulti-line bios are fine here.",
        "expected_duration": "1 hour",
        # Governance-receipt free-text (fresh receipts are written to the RAW store
        # by the governance ceremonies; at runtime they are already present here):
        "genesis_audit": {"status": "passed", "risk_level": 1},
        "genesis_audit_history": [
            {"receipt": {"status": "passed"}, "superseded_at": _now_iso()}
        ],
        "emancipation_contract": {"kind": "none"},
        "constitution_reanchor": {"hash": VALID_HASH, "authorization": "sovereign"},
        "doctrine_bundle_reanchor": {"hash": VALID_HASH},
    }
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        await raw.add_node(GraphNode(
            node_id=AGENT_ID,
            node_type="agent",
            label="Kestrel",
            properties={"constitution_hash": VALID_HASH, "created_at": _now_iso(),
                        **stored_receipts},
        ))
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = GraphNode(
            node_id=AGENT_ID,
            node_type="agent",
            label="Kestrel",
            properties={
                "agent_id": AGENT_ID,
                "did": AGENT_ID,
                "created_at": _now_iso(),
                "constitution_hash": VALID_HASH,
                "constitution_overlay_hash": VALID_HASH2,
                "initialBalance": "1000.0",
                # Content-free governance/lifecycle fields freshly written/updated:
                "avatar_hash": VALID_HASH,
                "bootstrap_state": "pending",
                "bootstrap_status": "stale_bootstrap",
                "bootstrap_stale_at": _now_iso(),
                "bootstrap_pending_age_seconds": 12,
                "doctrine_bundle_hash": VALID_HASH,
                "doctrine_bundle_files": ["AGENTS.md", "docs/TORTOISE_DOCTRINE.md"],
                "doctrine_bundle_anchored_at": _now_iso(),
                "doctrine_anchored_paths": ["AGENTS.md"],
                "graduated_at": _now_iso(),
                "is_test_instance": True,
                "test_cycle_id": "cycle-1",
                "is_demo": False,
                # All free-text CARRIED ALONG UNCHANGED from the stored node:
                **stored_receipts,
            },
        )
        await wrapper.add_node(node, capability=CAP)
        assert await raw.get_node(AGENT_ID) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_volatile_reanchor_superseded_receipt_refused_by_wrapper(tmp_path, mode):
    """A signed reanchor CHANGES ``genesis_audit`` and ADDS ``genesis_audit_history``
    (via ``supersede_genesis_audit``). Those are fresh/changed free-text, so the
    wrapper refuses the write in a volatile mode EVEN WITH the genuine capability —
    the carried-along boundary admits governance receipts only unchanged. A real
    reanchor's fresh receipt therefore never rides the feature-facing wrapper; it
    goes to the RAW store (and the runtime reanchor is blocked earlier by the
    volatile ``store_file`` gate anyway) (#2672 review P1). The stored node is left
    intact."""
    from kestrel_sovereign.constitution.genesis_audit import supersede_genesis_audit

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        # Prior stint: an agent node with a completed genesis audit + identity.
        await raw.add_node(GraphNode(
            node_id=AGENT_ID, node_type="agent", label="Kestrel",
            properties={
                "constitution_hash": VALID_HASH, "created_at": _now_iso(),
                "name": "Kestrel",
                "genesis_audit": {"status": "passed", "risk_level": 1},
            },
        ))
        wrapper = PrivacyEnforcingStorage(raw, mode)

        stored = await raw.get_node(AGENT_ID)
        props = dict(stored.properties)
        # Exactly what reanchor does: supersede the receipt (adds history) and
        # point the constitution hash at the new bytes.
        supersede_genesis_audit(
            props, constitution_hash=VALID_HASH2,
            provenance="runtime:constitution_reanchor",
        )
        props["constitution_hash"] = VALID_HASH2
        assert "genesis_audit_history" in props  # fresh free-text field
        updated = GraphNode(
            node_id=AGENT_ID, node_type="agent", label="Kestrel", properties=props,
        )

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(updated, capability=CAP)
        # Stored node unchanged: the changed receipt never landed via the wrapper.
        after = await raw.get_node(AGENT_ID)
        assert after.properties["constitution_hash"] == VALID_HASH
        assert after.properties["genesis_audit_history"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_governed_by_edge_allowed_end_to_end(tmp_path, mode):
    """The structural `governed_by` binding still writes in a volatile mode — what
    lets the startup constitution audit bind a born-volatile agent."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)

        await wrapper.add_node(_structural_node("agent"), capability=CAP)
        await wrapper.add_node(_structural_node("document"))

        await wrapper.add_edge(AGENT_ID, VALID_HASH, "governed_by")

        edges = await raw.get_edges_from(AGENT_ID)
        assert any(e.label == "governed_by" and e.target_id == VALID_HASH for e in edges)


# ─────────────────────────────────────────────────────────────────────────────
# Smuggling: allowlisted type/label is not enough — shape + provenance decide
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("node_type", sorted(STRUCTURAL_GRAPH_NODE_TYPES))
async def test_structural_node_rejects_smuggled_property(tmp_path, mode, node_type):
    """A user-content sentinel under a non-canonical property key is rejected on
    every path (ordinary and, for control-plane types, capability)."""
    admit = _admit_kwargs(node_type)
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node(node_type)
        node.properties["smuggled_note"] = USER_SECRET  # non-canonical key

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node, **admit)
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.add_node(node, **admit)

        assert await raw.get_node(node.node_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("node_type", CONTENT_FREE_STRUCTURAL_TYPES)
async def test_content_free_node_rejects_multiline_smuggled_value(
    tmp_path, mode, node_type
):
    """A multi-line conversation paragraph stuffed into a canonical key's value is
    rejected — matching a known key is not enough (P1)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node(node_type)
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
    "node_type", ["document", "constitution_amendment_artifact", "audit_anchor"]
)
async def test_structural_node_rejects_smuggled_label(tmp_path, mode, node_type):
    """User text in a fixed-label structural node's label is rejected."""
    admit = _admit_kwargs(node_type)
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node(node_type)
        node.label = USER_SECRET  # non-canonical label

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node, **admit)
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.add_node(node, **admit)

        assert await raw.get_node(node.node_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_governed_by_edge_rejects_user_content_properties(tmp_path, mode):
    """A structural governance edge carrying a user-content payload is rejected;
    the clean, property-free binding still writes."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        await wrapper.add_node(_structural_node("agent"), capability=CAP)
        await wrapper.add_node(_structural_node("document"))

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_edge(
                AGENT_ID, VALID_HASH, "governed_by", {"note": USER_SECRET}
            )
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.add_edge(
                AGENT_ID, VALID_HASH, "governed_by", {"note": USER_SECRET}
            )

        await wrapper.add_edge(AGENT_ID, VALID_HASH, "governed_by")
        edges = await raw.get_edges_from(AGENT_ID)
        assert any(e.label == "governed_by" for e in edges)


# ─────────────────────────────────────────────────────────────────────────────
# CAS cannot rewrite an EXISTING user-derived node via a spoofed structural type
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("surface", ["facade", "graph"])
async def test_cas_cannot_overwrite_user_node_via_spoofed_type(
    tmp_path, mode, surface
):
    """CAS-swapping an existing user-derived node under a spoofed structural type
    is rejected, and the stored row is left byte-for-byte intact — EVEN with the
    capability, because the atomic stored-type pin refuses (the stored row is a
    ``concept``)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
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
        # Spoof to a valid, capability-admitted structural type; the stored-type
        # pin must still refuse because the STORED row is a concept.
        spoof = GraphNode(
            node_id="concept-x",
            node_type="constitution_amendment_artifact",
            label="Signed Constitution Reanchor Artifact",
            properties={
                "hash": VALID_HASH,
                "type": "SignedConstitutionAmendment",
                "constitution_hash": VALID_HASH2,
                "created_at": _now_iso(),
                "anchored_at": _now_iso(),
            },
        )
        target = wrapper if surface == "facade" else wrapper.graph
        with pytest.raises(PrivacyViolationError):
            await target.compare_and_swap_node(
                "concept-x", original, spoof, capability=CAP
            )

        stored = await raw.get_node("concept-x")
        assert stored.node_type == "concept"
        assert stored.properties == original
        assert USER_SECRET not in json.dumps(stored.properties)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_cas_swap_of_existing_structural_node_allowed(tmp_path, mode):
    """A properties swap on an EXISTING control-plane structural node still lands in
    a volatile mode with the capability — the stored-type pin admits its own type."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        seed = _structural_node("agent")
        await raw.add_node(seed)

        wrapper = PrivacyEnforcingStorage(raw, mode)
        swapped_props = dict(seed.properties)
        swapped_props["bootstrap_state"] = "complete"
        new = GraphNode(
            node_id=AGENT_ID,
            node_type="agent",
            label="Kestrel",
            properties=swapped_props,
        )
        result = await wrapper.compare_and_swap_node(
            AGENT_ID, seed.properties, new, capability=CAP
        )
        assert result == NodeSwapResult.SWAPPED

        stored = await raw.get_node(AGENT_ID)
        assert stored.properties["bootstrap_state"] == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_cas_control_plane_swap_requires_capability(tmp_path, mode):
    """A compare-and-create of a control-plane node via CAS needs the capability;
    without it the atomic primitive never runs and nothing persists (P2)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node("agent")

        with pytest.raises(PrivacyViolationError):
            await wrapper.compare_and_swap_node(AGENT_ID, None, node)
        assert await raw.get_node(AGENT_ID) is None

        result = await wrapper.compare_and_swap_node(
            AGENT_ID, None, node, capability=CAP
        )
        assert result == NodeSwapResult.SWAPPED
        assert await raw.get_node(AGENT_ID) is not None


# ─────────────────────────────────────────────────────────────────────────────
# NORMAL / PUBLIC / ANONYMOUS are unchanged (no regression)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.NORMAL, PrivacyMode.PUBLIC, PrivacyMode.ANONYMOUS])
async def test_persistent_mode_user_content_node_persists(tmp_path, mode):
    """Persistent modes are unchanged: user-derived graph writes still persist,
    on both the wrapper and the proxy surfaces."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _user_content_node("learned_fact", node_id="fact-persist")

        await wrapper.add_node(node)
        persisted = await raw.get_node("fact-persist")
        assert persisted is not None
        assert persisted.properties["value"] == USER_SECRET

        node2 = _user_content_node("concept", node_id="concept-persist")
        await wrapper.graph.add_node(node2)
        assert await raw.get_node("concept-persist") is not None


@pytest.mark.asyncio
async def test_normal_mode_control_plane_type_persists_without_capability(tmp_path):
    """NORMAL: a control-plane node persists even WITHOUT the capability —
    governance gating is off in persistent modes (no regression)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        await wrapper.add_node(_structural_node("agent"))  # no capability needed
        assert await raw.get_node(AGENT_ID) is not None


@pytest.mark.asyncio
async def test_normal_mode_content_edge_and_cas_persist(tmp_path):
    """NORMAL: content edges and CAS are unchanged (no governance rejection)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)

        result = await wrapper.compare_and_swap_node(
            "fact-cas", None, _user_content_node("learned_fact", node_id="fact-cas")
        )
        assert result == NodeSwapResult.SWAPPED
        assert await raw.get_node("fact-cas") is not None

        await wrapper.add_node(_structural_node("agent"))
        await wrapper.add_edge(AGENT_ID, "fact-cas", "knows")
        edges = await raw.get_edges_from(AGENT_ID)
        assert any(e.label == "knows" for e in edges)


# ─────────────────────────────────────────────────────────────────────────────
# Live-path bypasses (review finding P4): agent_metadata + saved_items
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_persist_agent_description_skips_durable_writes_when_volatile(
    tmp_path, mode
):
    """``persist_agent_description`` writes NOTHING durable in a volatile mode —
    neither the raw ``agent_metadata`` row (the direct write that bypassed the
    graph boundary) nor the identity graph node (#2672 live-path bypass)."""
    from kestrel_sovereign.bootstrap.service import (
        PersistOutcome,
        persist_agent_description,
    )

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        await wrapper.add_node(_structural_node("agent"), capability=CAP)

        wrote = await persist_agent_description(
            raw.db, wrapper, AGENT_ID, f"secret bio {USER_SECRET}"
        )
        assert wrote is PersistOutcome.SKIPPED_PRIVACY

        if await raw.db.table_exists("agent_metadata"):
            rows = await raw.db.fetchall(
                "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = 'description'",
                (AGENT_ID,),
            )
            assert rows == []

        node = await raw.get_node(AGENT_ID)
        assert "description" not in (node.properties or {})


@pytest.mark.asyncio
async def test_persist_agent_description_writes_in_normal(tmp_path):
    """NORMAL: the description persists to both the graph node and
    ``agent_metadata`` (no regression on the persistent path)."""
    from kestrel_sovereign.bootstrap.service import (
        PersistOutcome,
        persist_agent_description,
    )

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        await wrapper.add_node(_structural_node("agent"))

        wrote = await persist_agent_description(
            raw.db, wrapper, AGENT_ID, "A self-authored tagline."
        )
        assert wrote is PersistOutcome.PERSISTED

        node = await raw.get_node(AGENT_ID)
        assert node.properties.get("description") == "A self-authored tagline."
        rows = await raw.db.fetchall(
            "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = 'description'",
            (AGENT_ID,),
        )
        assert rows and rows[0][0] == "A self-authored tagline."


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_feature_persist_config_skips_when_volatile(tmp_path, mode):
    """``Feature.persist_config`` skips the durable ``feature_config`` write in a
    volatile mode (its ``config`` is an arbitrary settings dict, source-gated at
    its single source of truth — #2672 P3), and writes it in NORMAL."""
    from kestrel_sovereign.features.base import Feature

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        volatile_wrapper = PrivacyEnforcingStorage(raw, mode)
        fake = types.SimpleNamespace(
            agent=types.SimpleNamespace(storage=volatile_wrapper),
            name="todo",
            _CONFIG_NODE_TYPE="feature_config",
            _config_node_id=lambda: "feature_config_todo",
        )
        await Feature.persist_config(fake, {"api_key": USER_SECRET})
        assert await raw.get_node("feature_config_todo") is None

        # NORMAL: the same call persists the config node (no regression).
        fake.agent.storage = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        await Feature.persist_config(fake, {"enabled": True})
        assert await raw.get_node("feature_config_todo") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_save_soul_md_writes_nothing_durable_when_volatile(tmp_path, mode):
    """``save_soul_md`` in a volatile mode creates NO durable trace of the
    user-derived SOUL body: no SOUL.md file, no encrypted
    ``agent_identity_resources`` row, and no ``#soul`` graph reference (#2672
    review P1 — the second live-path bypass)."""
    from kestrel_sovereign.bootstrap.service import BootstrapService

    agent_dir = tmp_path / "agentdata"
    agent_dir.mkdir()
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        service = BootstrapService(
            db=raw.db,
            agent_id=AGENT_ID,
            agent_name="Kestrel",
            llm_service=None,
            agent_data_path=str(agent_dir),
            storage=wrapper,
        )

        saved = await service.save_soul_md(f"# SOUL.md\n\n{USER_SECRET}")
        assert saved is False  # nothing durably saved while volatile

        # No SOUL file on disk.
        assert not (agent_dir / "SOUL.md").exists()
        # No encrypted identity-resource row.
        if await raw.db.table_exists("agent_identity_resources"):
            rows = await raw.db.fetchall(
                "SELECT id FROM agent_identity_resources WHERE agent_id = ?",
                (AGENT_ID,),
            )
            assert rows == []
        # No durable graph reference to the private SOUL resource.
        assert await raw.get_node(f"{AGENT_ID}#soul") is None
        # And the SOUL body never reached any graph row.
        assert USER_SECRET not in json.dumps(
            [n.properties for n in await raw.get_nodes_by_type("agent_identity_resource")]
        )


@pytest.mark.asyncio
async def test_save_soul_md_persists_in_normal(tmp_path, monkeypatch):
    """NORMAL: ``save_soul_md`` still writes the SOUL file, promotes the encrypted
    resource, and records the ``#soul`` graph reference (no regression)."""
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-soul-normal-key")
    from kestrel_sovereign.bootstrap.service import BootstrapService

    soul_agent = "did:test:soul-normal"
    agent_dir = tmp_path / "agentdata"
    agent_dir.mkdir()
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=soul_agent) as raw:
        wrapper = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        service = BootstrapService(
            db=raw.db,
            agent_id=soul_agent,
            agent_name="Kestrel",
            llm_service=None,
            agent_data_path=str(agent_dir),
            storage=wrapper,
        )

        saved = await service.save_soul_md("# SOUL.md\n\n## Tagline\nA sovereign agent.\n")
        assert saved is True
        assert (agent_dir / "SOUL.md").exists()
        assert await raw.get_node(f"{soul_agent}#soul") is not None


@pytest.mark.parametrize(
    "mode,hidden",
    [
        (PrivacyMode.EPHEMERAL, True),
        (PrivacyMode.ISOLATED, True),
        (PrivacyMode.DEIDENTIFIED, True),
        (PrivacyMode.NORMAL, False),
        (PrivacyMode.PUBLIC, False),
        (PrivacyMode.ANONYMOUS, False),
    ],
)
def test_hides_persisted_user_content_covers_deidentified(mode, hidden):
    """``hides_persisted_user_content`` must block reads/writes of persisted user
    content in EVERY volatile mode — including DEIDENTIFIED, whose omission let the
    Save feature write user content to ``saved_items`` (#2672 live-path bypass)."""
    from kestrel_sovereign.features.storage_access import hides_persisted_user_content
    from kestrel_sovereign.privacy import privacy_mode_to_config

    agent = types.SimpleNamespace(privacy_config=privacy_mode_to_config(mode))
    assert hides_persisted_user_content(agent) is hidden


# ─────────────────────────────────────────────────────────────────────────────
# Consolidation: the whole durable pipeline is blocked while volatile (P5)
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
    ``memory_consolidate`` tool cannot report success (#2672 P5)."""
    from kestrel_sovereign.storage.memory_system import MemorySystem

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
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


# ─────────────────────────────────────────────────────────────────────────────
# Graph proxy is fail-closed: it does NOT forward the raw ``db`` handle or any
# un-allowlisted attribute (#2672 review P1). The pre-fix ``__getattr__``
# forwarded EVERYTHING, so ``storage.graph.db.execute(...)`` was an ungoverned
# SQL write straight past the boundary. These are mode-independent: the proxy
# refuses raw handles in every mode, because that channel is never the sanctioned
# path (raw access goes through ``_raw_storage`` deliberately).
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED, PrivacyMode.NORMAL, PrivacyMode.PUBLIC],
)
@pytest.mark.parametrize("surface", ["graph", "graph_store"])
async def test_graph_proxy_refuses_raw_db_handle(tmp_path, mode, surface):
    """``storage.graph.db`` / ``.graph_store.db`` fail closed — the raw database
    handle is never forwarded, so a caller cannot run ``.graph.db.execute(...)``
    to bypass the volatile-mode graph-write policy (#2672 review P1)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        proxy = getattr(wrapper, surface)

        with pytest.raises(PrivacyViolationError):
            _ = proxy.db

        # The raw store, reached deliberately, still has its db — the refusal is a
        # property of the GOVERNED proxy surface, not of the store.
        assert raw.graph.db is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize(
    "attr",
    ["db", "_backend", "connection", "execute", "some_future_write_method"],
)
async def test_graph_proxy_refuses_un_allowlisted_attribute(tmp_path, mode, attr):
    """Any attribute NOT on the fixed allowlist — raw handles AND a hypothetical
    future write method — fails closed, so the proxy can never silently forward an
    ungoverned write path (#2672 review P1)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        with pytest.raises(PrivacyViolationError):
            getattr(wrapper.graph, attr)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_graph_proxy_forwards_allowlisted_reads_and_deletes(tmp_path, mode):
    """The allowlisted non-write surface (reads, deletes, ``bind_agent``) still
    forwards through the proxy — closing the ``db`` leak must not break the
    legitimate read/delete/bind uses production code makes of ``.graph`` (#2672)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        # Seed a content-free structural node directly on the raw store.
        await raw.add_node(_structural_node("document"))
        wrapper = PrivacyEnforcingStorage(raw, mode)

        # Reads forward.
        got = await wrapper.graph.get_node(VALID_HASH)
        assert got is not None and got.node_id == VALID_HASH
        by_type = await wrapper.graph.get_nodes_by_type("document")
        assert any(n.node_id == VALID_HASH for n in by_type)
        assert await wrapper.graph.get_edges(VALID_HASH) == []

        # bind_agent forwards (scope binding is not a user-content write).
        wrapper.graph.bind_agent(AGENT_ID)

        # Deletes forward (removal is not a durable user-content WRITE).
        await wrapper.graph.delete_node(VALID_HASH)
        assert await raw.get_node(VALID_HASH) is None


# ─────────────────────────────────────────────────────────────────────────────
# ``openrouter_key_hash`` is a canonical, validated ``agent`` field (#2672 P2).
# The host-owned ``payer_resolver`` persists a delegated-OpenRouter child-key
# hash onto the agent node; without it in the canonical vocabulary, a full
# agent-node governance upsert fails closed once the agent is volatile, breaking
# doctrine/bootstrap/audit persistence for delegated-OpenRouter agents.
# ─────────────────────────────────────────────────────────────────────────────

# A realistic OpenRouter child-key hash (a 64-hex sha256 digest of the key).
VALID_OPENROUTER_KEY_HASH = "abcdef0123456789" * 4


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_agent_node_with_openrouter_key_hash_admitted(tmp_path, mode):
    """An agent node carrying ``openrouter_key_hash`` survives a volatile-mode
    governance upsert — the field is canonical and validated, so it no longer
    trips the non-canonical-key refusal (#2672 review P2)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node("agent")
        node.properties["openrouter_key_hash"] = VALID_OPENROUTER_KEY_HASH

        await wrapper.add_node(node, capability=CAP)  # must NOT raise
        persisted = await raw.get_node(AGENT_ID)
        assert persisted is not None
        assert persisted.properties["openrouter_key_hash"] == VALID_OPENROUTER_KEY_HASH


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_agent_node_openrouter_key_hash_cas_admitted(tmp_path, mode):
    """The same field survives an atomic CAS agent-node update — the path a real
    governance write takes (#2672 review P2)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        stored = _structural_node("agent")
        await raw.add_node(stored)
        wrapper = PrivacyEnforcingStorage(raw, mode)

        # ``expected`` is the PROPERTIES snapshot the caller last read (what a real
        # governance CAS passes), not the GraphNode.
        expected_props = dict((await raw.get_node(AGENT_ID)).properties)
        new_props = dict(expected_props)
        new_props["openrouter_key_hash"] = VALID_OPENROUTER_KEY_HASH
        updated = GraphNode(
            node_id=AGENT_ID, node_type="agent", label="Kestrel", properties=new_props
        )

        result = await wrapper.compare_and_swap_node(
            AGENT_ID, expected_props, updated, capability=CAP
        )
        assert result == NodeSwapResult.SWAPPED
        after = await raw.get_node(AGENT_ID)
        assert after.properties["openrouter_key_hash"] == VALID_OPENROUTER_KEY_HASH


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize(
    "bad_value",
    [
        "not a hash with spaces",       # whitespace → free text
        "line1\nline2",                 # newline → multi-line smuggling
        USER_SECRET + " " + USER_SECRET,  # free-form user content
        "a",                            # too short to be a credential hash
    ],
)
async def test_agent_node_openrouter_key_hash_free_text_rejected(
    tmp_path, mode, bad_value
):
    """A non-credential-shaped ``openrouter_key_hash`` (whitespace, newline, free
    text) is rejected even WITH the capability — per-field validation is
    defense-in-depth so the canonical field can't become a smuggling channel
    (#2672 review P2)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node("agent")
        node.properties["openrouter_key_hash"] = bad_value

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node, capability=CAP)
        assert await raw.get_node(AGENT_ID) is None
