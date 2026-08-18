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
* **Control-plane type** — the ``agent`` identity node (the only capability-gated
  type; ``constitution_amendment_artifact`` was dropped from the allowlist
  entirely, see the note by its former shape entry) — carries a control-plane
  CAPABILITY marker, but the marker is same-process defense-in-depth, NOT an
  authorization boundary: the governance writers are mixin methods on the agent
  and feature code holds the agent, so any in-process caller can obtain the marker
  (e.g. by ``exec``-ing into a trusted module's ``__dict__``). The LOAD-BEARING
  privacy gate for the ``agent`` node is the CARRIED-ALONG identity boundary —
  user-facing free-text (name / description / expected_duration) AND the top-level
  identity ``label`` are admitted only when unchanged from the stored node, so a
  fresh/changed value is refused even to a caller holding a genuine (forged) marker
  (this is what closes the reproduced ``agent.description`` and ``agent.label``
  leaks). The governance-receipt free-text fields are a documented
  process-isolation residual.

The two user-derived surfaces the review told us not to blanket-trust —
``agent.description`` and ``feature_config.config`` — are gated at their single
source of truth (``persist_agent_description`` / ``Feature.persist_config`` skip
their durable writes while volatile), so ``feature_config`` is not allowlisted at
all. NORMAL / PUBLIC / ANONYMOUS are unchanged.
"""
from __future__ import annotations

import asyncio
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
    ReentrantTransitionLock,
    bind_transition_lock_reentry,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("surface", ["facade", "graph"])
async def test_module_dict_injection_cannot_persist_fresh_label(
    tmp_path, mode, surface
):
    """The reproduced P1 label leak: code injected into a REAL trusted module's
    ``__dict__`` obtains the GENUINE marker, yet CANNOT persist arbitrary durable
    user text through the top-level ``GraphNode.label`` in a volatile mode — the
    carried-along boundary now covers the label as well as the properties, so a
    fresh/changed identity label is refused on BOTH the facade and the graph proxy,
    and the stored label is left byte-for-byte intact (#2672 review P1).

    All content-free PROPERTIES are carried along unchanged here, so the ONLY thing
    the write tries to change is the label — isolating the label as the smuggling
    channel the fix closes."""
    forged = _capability_via_module_dict_injection(
        "kestrel_sovereign.agent.constitution"
    )
    assert _has_control_plane_capability(forged)  # injection yields the real marker

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        await raw.add_node(_structural_node("agent"))  # stored label = "Kestrel"
        wrapper = PrivacyEnforcingStorage(raw, mode)

        evil = _structural_node("agent")
        evil.label = f"exfiltrated: {USER_SECRET}"  # arbitrary user text in the label
        target = wrapper if surface == "facade" else wrapper.graph
        with pytest.raises(PrivacyViolationError):
            await target.add_node(evil, capability=forged)

        stored = await raw.get_node(AGENT_ID)
        assert stored.label == "Kestrel"  # stored row unchanged
        assert USER_SECRET not in (stored.label or "")
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
@pytest.mark.parametrize("surface", ["facade", "graph"])
async def test_changed_label_refused_but_unchanged_label_admitted(
    tmp_path, mode, surface
):
    """The top-level identity LABEL follows the same carried-along rule as the
    identity properties: a CHANGED label is refused even with the genuine marker
    (stored label intact), while the SAME label riding a content-free governance
    mutation is admitted. Proven on both the facade and the graph proxy (#2672 P1).
    """
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        await raw.add_node(GraphNode(
            node_id=AGENT_ID, node_type="agent", label="Kestrel",
            properties={"constitution_hash": VALID_HASH, "created_at": _now_iso(),
                        "name": "Kestrel"},
        ))
        wrapper = PrivacyEnforcingStorage(raw, mode)
        target = wrapper if surface == "facade" else wrapper.graph

        stored = await raw.get_node(AGENT_ID)
        changed_label = GraphNode(
            node_id=AGENT_ID, node_type="agent", label="Renamed Live",
            properties={**dict(stored.properties), "bootstrap_state": "complete"},
        )
        with pytest.raises(PrivacyViolationError):
            await target.add_node(changed_label, capability=CAP)
        after = await raw.get_node(AGENT_ID)
        assert after.label == "Kestrel"  # stored label unchanged
        assert "bootstrap_state" not in (after.properties or {})

        # Same label + a content-free mutation → admitted.
        same_label = GraphNode(
            node_id=AGENT_ID, node_type="agent", label="Kestrel",
            properties={**dict(stored.properties), "bootstrap_state": "complete"},
        )
        await target.add_node(same_label, capability=CAP)
        after = await raw.get_node(AGENT_ID)
        assert after.label == "Kestrel"
        assert after.properties["bootstrap_state"] == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("surface", ["facade", "graph"])
async def test_none_label_cannot_clear_stored_label(tmp_path, mode, surface):
    """``add_node`` is a whole-row upsert, so a write carrying ``label=None`` would
    CLEAR the stored user name. On an EXISTING node that is a durable label
    mutation, not carry-along, so it is refused even with the genuine marker and
    the stored label survives byte-for-byte. Proven on both surfaces (#2672 P2).

    Every content-free property is carried along and a governance field is added,
    isolating the ``None`` label as the only mutation the write attempts."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        await raw.add_node(GraphNode(
            node_id=AGENT_ID, node_type="agent", label="Kestrel",
            properties={"constitution_hash": VALID_HASH, "created_at": _now_iso(),
                        "name": "Kestrel"},
        ))
        wrapper = PrivacyEnforcingStorage(raw, mode)
        target = wrapper if surface == "facade" else wrapper.graph

        stored = await raw.get_node(AGENT_ID)
        cleared = GraphNode(
            node_id=AGENT_ID, node_type="agent", label=None,
            properties={**dict(stored.properties), "bootstrap_state": "complete"},
        )
        with pytest.raises(PrivacyViolationError):
            await target.add_node(cleared, capability=CAP)
        after = await raw.get_node(AGENT_ID)
        assert after.label == "Kestrel"  # stored label NOT cleared to None
        assert "bootstrap_state" not in (after.properties or {})


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("surface", ["facade", "graph"])
async def test_did_form_label_cannot_overwrite_stored_user_name(
    tmp_path, mode, surface
):
    """The content-free DID-derived ``Agent {node_id}`` label is admitted ONLY on a
    fresh create (nothing to carry). On an EXISTING node whose stored label is a
    real user name, replacing it with the DID form still DESTROYS the stored name —
    a durable label mutation — so it is refused and the stored user label survives.
    Proven on both surfaces (#2672 P2)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        await raw.add_node(GraphNode(
            node_id=AGENT_ID, node_type="agent", label="Kestrel",
            properties={"constitution_hash": VALID_HASH, "created_at": _now_iso(),
                        "name": "Kestrel"},
        ))
        wrapper = PrivacyEnforcingStorage(raw, mode)
        target = wrapper if surface == "facade" else wrapper.graph

        stored = await raw.get_node(AGENT_ID)
        did_form = GraphNode(
            node_id=AGENT_ID, node_type="agent", label=f"Agent {AGENT_ID}",
            properties={**dict(stored.properties), "bootstrap_state": "complete"},
        )
        with pytest.raises(PrivacyViolationError):
            await target.add_node(did_form, capability=CAP)
        after = await raw.get_node(AGENT_ID)
        assert after.label == "Kestrel"  # stored user name NOT overwritten
        assert "bootstrap_state" not in (after.properties or {})


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
@pytest.mark.parametrize("surface", ["facade", "graph"])
async def test_cas_compare_and_create_agent_label_refused(tmp_path, mode, surface):
    """A compare-and-CREATE (``expected is None``) of an agent node — whose only
    user-derived content is a fresh free-text LABEL (every property is content-free)
    — is refused even with the genuine marker: a create has no stored trusted label
    to carry from, so the label is fresh content and nothing persists. Closes the
    label smuggling channel on the CAS create path, both surfaces (#2672 P1)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        target = wrapper if surface == "facade" else wrapper.graph
        node = _structural_node("agent")  # content-free props, label "Kestrel"
        node.label = f"exfiltrated: {USER_SECRET}"
        with pytest.raises(PrivacyViolationError):
            await target.compare_and_swap_node(AGENT_ID, None, node, capability=CAP)
        assert await raw.get_node(AGENT_ID) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
@pytest.mark.parametrize("surface", ["facade", "graph"])
async def test_born_agent_did_derived_label_create_admitted(tmp_path, mode, surface):
    """The born-agent boot path: first materialising the identity node in a
    volatile mode with the DID-derived ``Agent {node_id}`` label IS admitted even
    on a fresh create — that label is fully determined by the node's own id, so it
    carries no user text. This is the ``KestrelAgent.initialize`` fresh-node write;
    it must not be caught by the label carry-along that refuses a user-authored
    name (#2672 review P1)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, mode)
        target = wrapper if surface == "facade" else wrapper.graph
        node = GraphNode(
            node_id=AGENT_ID,
            node_type="agent",
            label=f"Agent {AGENT_ID}",       # DID-derived, content-free
            properties={"initialBalance": "100.0"},
        )
        await target.add_node(node, capability=CAP)  # must NOT raise
        stored = await raw.get_node(AGENT_ID)
        assert stored is not None
        assert stored.label == f"Agent {AGENT_ID}"


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
    """A control-plane node update is admitted ONLY with the unforgeable
    capability; without it (or on the proxy without it) the write is denied and the
    stored node is left unchanged.

    The agent node exists from inception (written to the RAW store); a volatile
    wrapper only ever CARRIES it along, so the capability gate is exercised here on
    that realistic carried-along-update path. A fresh create through the wrapper is
    independently refused by the label carry-along boundary (see the dedicated
    label tests) — inception never routes through the wrapper."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        await raw.add_node(_structural_node(node_type))
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node(node_type)

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node)
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.add_node(node)
        stored = await raw.get_node(node.node_id)
        assert stored is not None and stored.label == node.label

        # With the capability: admitted (carried-along update), content-free.
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
        # The fresh ``genesis_audit_history`` was only ever added to the rejected
        # copy — it must be absent from the untouched stored node.
        assert "genesis_audit_history" not in after.properties
        assert after.properties["genesis_audit"] == {"status": "passed", "risk_level": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_volatile_agent_write_carries_reanchor_history_along(tmp_path, mode):
    """``constitution_reanchor_history`` is carried along, never freshly written.

    #2963 archives each superseded reanchor receipt here. The key has to be a
    canonical governance field for one reason that is easy to miss: the allowed
    key set IS the validator map's keys, so an agent that accumulated history in
    a persistent stint would fail closed on its NEXT ordinary volatile
    agent-node write — one that changes nothing about the history at all.

    The other half is the boundary itself: appending to that history is a fresh
    free-text change and stays refused, exactly as the superseded genesis
    receipt above. Admitting the key must not admit new content through it.
    """
    prior_receipt = {
        "timestamp": "2026-04-05T00:00:00Z",
        "old_hash": VALID_HASH,
        "new_hash": VALID_HASH2,
        "path": "/prior/KESTREL_CONSTITUTION.md",
        "authorization": "prior_admin",
    }
    history = [{
        "receipt": prior_receipt,
        "superseded_at": "2026-04-06T00:00:00Z",
        "superseded_by_constitution_hash": VALID_HASH2,
        "provenance": "runtime:constitution_reanchor",
    }]

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        await raw.add_node(GraphNode(
            node_id=AGENT_ID, node_type="agent", label="Kestrel",
            properties={
                "constitution_hash": VALID_HASH, "created_at": _now_iso(),
                "name": "Kestrel",
                "constitution_reanchor_history": history,
            },
        ))
        wrapper = PrivacyEnforcingStorage(raw, mode)

        # Unchanged carry-along: admitted, so a prior stint's history does not
        # brick ordinary volatile writes.
        stored = await raw.get_node(AGENT_ID)
        carried = dict(stored.properties)
        carried["bootstrap_state"] = "complete"
        await wrapper.add_node(
            GraphNode(node_id=AGENT_ID, node_type="agent", label="Kestrel",
                      properties=carried),
            capability=CAP,
        )
        after = await raw.get_node(AGENT_ID)
        assert after.properties["constitution_reanchor_history"] == history

        # Appending a new entry is fresh free-text: refused, and nothing lands.
        appended = dict(after.properties)
        appended["constitution_reanchor_history"] = history + [{
            "receipt": {"new_hash": VALID_HASH, "path": "/next.md"},
            "superseded_at": "2026-04-07T00:00:00Z",
            "superseded_by_constitution_hash": VALID_HASH,
            "provenance": "runtime:constitution_reanchor",
        }]
        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(
                GraphNode(node_id=AGENT_ID, node_type="agent", label="Kestrel",
                          properties=appended),
                capability=CAP,
            )
        unchanged = await raw.get_node(AGENT_ID)
        assert unchanged.properties["constitution_reanchor_history"] == history


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_governed_by_edge_allowed_end_to_end(tmp_path, mode):
    """The structural `governed_by` binding still writes in a volatile mode — what
    lets the startup constitution audit bind a born-volatile agent."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        # The agent node exists from inception (raw store); the wrapper carries it
        # along, never freshly creates it in a volatile mode (#2672 P1 label gate).
        await raw.add_node(_structural_node("agent"))
        wrapper = PrivacyEnforcingStorage(raw, mode)

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
@pytest.mark.parametrize("node_type", sorted(STRUCTURAL_GRAPH_NODE_TYPES))
async def test_smuggled_label_refused_on_every_allowlisted_type(
    tmp_path, mode, node_type
):
    """Per-type label audit: NO allowlisted structural type admits arbitrary user
    text in its top-level ``label``. A content-free-labeled type (document /
    audit_anchor) refuses a non-canonical label by shape; the free-text-labeled
    agent node refuses a fresh/changed label by carry-along. Proven on the facade
    AND the graph proxy so no shape can smuggle durable user text through
    ``GraphNode.label`` (#2672 review P1)."""
    admit = _admit_kwargs(node_type)
    control_plane = node_type in CONTROL_PLANE_ONLY_NODE_TYPES
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        # The free-text-labeled agent node exists from inception (raw store), so the
        # ONLY thing the write changes is the label. The content-free-labeled types
        # are fresh creates whose canonical label is checked by shape.
        if control_plane:
            await raw.add_node(_structural_node(node_type))
        wrapper = PrivacyEnforcingStorage(raw, mode)
        node = _structural_node(node_type)
        node.label = f"smuggled {USER_SECRET}"

        with pytest.raises(PrivacyViolationError):
            await wrapper.add_node(node, **admit)
        with pytest.raises(PrivacyViolationError):
            await wrapper.graph.add_node(node, **admit)

        stored = await raw.get_node(node.node_id)
        if control_plane:
            assert stored is not None and USER_SECRET not in (stored.label or "")
        else:
            assert stored is None


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
        # Inception-written agent node (raw store); the wrapper only carries it
        # along in a volatile mode (#2672 P1 label gate).
        await raw.add_node(_structural_node("agent"))
        wrapper = PrivacyEnforcingStorage(raw, mode)
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
    """A CAS swap of a control-plane node needs the capability; without it the
    atomic primitive never runs and the stored node is unchanged (P2).

    The swap is the realistic control-plane CAS: the agent node exists from
    inception (raw store) and a governance write updates its content-free state
    while carrying the identity label along. A compare-and-CREATE with a fresh
    free-text label is refused by the label carry-along boundary — see
    ``test_cas_compare_and_create_agent_label_refused`` (#2672 P1)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        seed = _structural_node("agent")
        await raw.add_node(seed)
        wrapper = PrivacyEnforcingStorage(raw, mode)
        swapped_props = dict(seed.properties)
        swapped_props["bootstrap_state"] = "complete"
        new = GraphNode(
            node_id=AGENT_ID, node_type="agent", label="Kestrel",
            properties=swapped_props,
        )

        with pytest.raises(PrivacyViolationError):
            await wrapper.compare_and_swap_node(AGENT_ID, seed.properties, new)
        assert (await raw.get_node(AGENT_ID)).properties.get("bootstrap_state") != "complete"

        result = await wrapper.compare_and_swap_node(
            AGENT_ID, seed.properties, new, capability=CAP
        )
        assert result == NodeSwapResult.SWAPPED
        assert (await raw.get_node(AGENT_ID)).properties["bootstrap_state"] == "complete"


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
        # Inception-written agent node (raw store); the wrapper carries it along.
        await raw.add_node(_structural_node("agent"))
        wrapper = PrivacyEnforcingStorage(raw, mode)

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


# ─────────────────────────────────────────────────────────────────────────────
# ReentrantTransitionLock — task-reentrant so a durable-write tool nested inside a
# streamed turn (which already holds the lock) does not self-deadlock (#2672 P1).
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reentrant_transition_lock_same_task_reentry():
    """Same-task nested acquire returns immediately (no deadlock) and only the
    OUTERMOST exit releases — the property that fixes the streamed-rename hang."""
    lock = ReentrantTransitionLock()
    async with lock:
        assert lock.locked()
        async with lock:  # would hang forever on a plain asyncio.Lock
            assert lock.locked()
        assert lock.locked()  # inner exit must NOT release while outer holds
    assert not lock.locked()  # outer exit releases


@pytest.mark.asyncio
async def test_reentrant_transition_lock_cross_task_exclusion():
    """Across DIFFERENT tasks the lock is still mutually exclusive — a second task
    waits until the first fully releases the outermost acquire, preserving the
    writer-vs-transition serialization the #2672 race needs."""
    lock = ReentrantTransitionLock()
    order: list[str] = []

    async def holder():
        async with lock:
            order.append("A-acquire")
            for _ in range(6):  # give B a chance to (fail to) acquire
                await asyncio.sleep(0)
            order.append("A-release")

    async def waiter():
        while "A-acquire" not in order:  # ensure A grabs it first
            await asyncio.sleep(0)
        async with lock:
            order.append("B-acquire")

    await asyncio.gather(holder(), waiter())
    assert order == ["A-acquire", "A-release", "B-acquire"]


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic privacy-transition interleaving: a NORMAL→volatile flip that
# lands in the CHECK→PERSIST gap is respected because every direct user-content
# writer holds the agent's privacy-transition lock across both steps (#2672 P1
# race). Covers rename, description, discovery history, discovered user name, and
# SOUL — check-through-persist.
# ─────────────────────────────────────────────────────────────────────────────


async def _flip_lands_in_check_persist_gap(lock, coro, flip):
    """Deterministically land ``flip`` inside a writer's check→persist gap.

    Pre-hold ``lock`` (a privacy transition is "in flight"), start ``coro`` (it
    blocks entering its OWN ``async with lock``), assert it is BLOCKED, apply the
    volatile ``flip``, assert it is STILL blocked (the entire transition happened
    inside the gap), then release. The writer then runs its whole
    check-through-persist under the lock AFTER the flip, so it must observe the
    volatile mode and skip — proving the check and the persist are one serialized
    unit and a transition can never land between them (#2672 review P1 race)."""
    await lock.acquire()
    task = asyncio.create_task(coro)
    for _ in range(6):
        await asyncio.sleep(0)
    assert not task.done(), "writer must block on the held privacy-transition lock"
    flip()
    for _ in range(4):
        await asyncio.sleep(0)
    assert not task.done(), "writer must stay serialized behind the transition"
    lock.release()
    return await task


class _RenameAgent:
    """Minimal agent shape for ``rename_agent_core``: live name, bootstrap-service
    name mirror, the wrapper as ``storage`` (so ``hides_persisted_user_content``
    reads the CURRENT mode from it), the raw db for the metadata write, and the
    shared privacy-transition lock."""

    def __init__(self, wrapper, raw, lock):
        self.agent_id = AGENT_ID
        self._agent_name = "Kestrel"
        self.storage = wrapper
        self._raw_storage = types.SimpleNamespace(db=raw.db)
        self.bootstrap_service = types.SimpleNamespace(
            agent_name="Kestrel", agent_data_path=None
        )
        self._lock = lock

    def _get_privacy_transition_lock(self):
        return self._lock


async def _metadata_value(raw, key):
    if not await raw.db.table_exists("agent_metadata"):
        return None
    rows = await raw.db.fetchall(
        "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
        (AGENT_ID, key),
    )
    return rows[0][0] if rows else None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_transition_in_gap_rename_skips_all_durable_writes(tmp_path, mode):
    """RENAME: a NORMAL→volatile flip landing in the gap makes the rename skip
    every durable store (metadata row, graph node/label, SOUL) and update only the
    live session name."""
    from kestrel_sovereign.features.bootstrap.feature import rename_agent_core

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        await raw.add_node(_structural_node("agent"))  # stored label "Kestrel"
        wrapper = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        lock = asyncio.Lock()
        agent = _RenameAgent(wrapper, raw, lock)

        outcome = await _flip_lands_in_check_persist_gap(
            lock,
            rename_agent_core(agent, "RenamedLive"),
            lambda: wrapper.set_privacy_mode(mode),
        )

        assert outcome.skipped_privacy is True
        assert agent._agent_name == "RenamedLive"          # live/session name only
        assert await _metadata_value(raw, "name") is None  # nothing durable
        node = await raw.get_node(AGENT_ID)
        assert node.label == "Kestrel"                     # stored node untouched
        assert node.properties.get("name") != "RenamedLive"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.NORMAL, PrivacyMode.EPHEMERAL])
async def test_rename_under_held_turn_lock_does_not_deadlock(tmp_path, mode):
    """P1 regression: a streamed turn holds the privacy-transition lock across the
    WHOLE turn (``agent/streaming.py`` acquires it before dispatching tools), and
    ``rename_agent`` runs as a tool WITHIN that turn. With a non-reentrant
    ``asyncio.Lock`` the tool would wait on a lock its OWN task already holds and
    hang the stream forever. The real agent lock (``ReentrantTransitionLock``) is
    task-reentrant, so the nested rename completes and still persists (NORMAL) or
    skips (volatile) correctly. Uses the REAL lock type and drives the write under
    an ``asyncio.timeout`` deadline so the rename stays on the lock-owning task
    while a regression still fails loudly instead of hanging the whole suite
    (#2672 review P1)."""
    from kestrel_sovereign.features.bootstrap.feature import rename_agent_core

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        await raw.add_node(_structural_node("agent"))  # stored label "Kestrel"
        wrapper = PrivacyEnforcingStorage(raw, mode)
        lock = ReentrantTransitionLock()
        agent = _RenameAgent(wrapper, raw, lock)

        # Streaming path: hold the transition lock across the whole "turn", then
        # dispatch the rename tool inside it — exactly the same-task re-acquire
        # that deadlocked with a plain asyncio.Lock.
        async with lock:
            assert lock.locked()
            async with asyncio.timeout(5.0):
                outcome = await rename_agent_core(agent, "RenamedInTurn")

        assert agent._agent_name == "RenamedInTurn"        # live name always updates
        if mode is PrivacyMode.NORMAL:
            assert outcome.skipped_privacy is False
            assert outcome.db_row_written is True
            assert await _metadata_value(raw, "name") == "RenamedInTurn"
            node = await raw.get_node(AGENT_ID)
            assert node.label == "RenamedInTurn"           # durable node renamed
        else:
            assert outcome.skipped_privacy is True
            assert await _metadata_value(raw, "name") is None  # nothing durable
            node = await raw.get_node(AGENT_ID)
            assert node.label == "Kestrel"                 # stored node untouched


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-TASK reentry (#2672 review P1 follow-up). The same-task test above covers
# the ANTHROPIC path, where the inline rename runs on the turn task that holds the
# lock. It does NOT cover the CODEX app-server path: the long-lived app-server
# dispatches each ``item/tool/call`` handler on its OWN reader-spawned task, so the
# nested rename runs on a DIFFERENT task than the lock owner. Same-task reentry
# cannot help it — without the captured per-turn reentry token the turn (blocked
# awaiting the app-server's tool result) and the tool (blocked acquiring the lock
# the turn holds) DEADLOCK. This drives the REAL inline executor
# (``_make_inline_tool_executor``) across a faithful reader-task boundary.
# ─────────────────────────────────────────────────────────────────────────────


from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin  # noqa: E402
from kestrel_sovereign.features.base import Feature  # noqa: E402


class _InlineOrchestratorHost(OrchestratorEngineMixin):
    """Real orchestrator host that builds the PRODUCTION inline executor.

    Combines the orchestrator mixin (so ``_make_inline_tool_executor`` is the real
    code under test — token capture and re-presentation included) with the
    ``_RenameAgent`` identity shape ``rename_agent_core`` needs and the shared
    ``ReentrantTransitionLock``. ``execute_named_tool`` routes to
    ``rename_agent_core`` exactly like the live app-server path
    (server→client ``item/tool/call`` → ``execute_named_tool`` → the named tool).
    """

    def __init__(self, wrapper, raw, lock):
        self.agent_id = AGENT_ID
        self._agent_name = "Kestrel"
        self.storage = wrapper
        self._raw_storage = types.SimpleNamespace(db=raw.db)
        self.bootstrap_service = types.SimpleNamespace(
            agent_name="Kestrel", agent_data_path=None
        )
        self._privacy_transition_lock = lock

    def _get_privacy_transition_lock(self):
        return self._privacy_transition_lock

    async def execute_named_tool(self, name, args, *, session_id, source, _capture):
        assert name == "rename_agent"
        from kestrel_sovereign.features.bootstrap.feature import rename_agent_core

        _capture["effective_args"] = args
        return await rename_agent_core(self, args["new_name"])


class _AppServerReaderHarness:
    """Reproduces the codex app-server reader-task topology WITHOUT a live server.

    The reader loop is spawned once; each dispatched tool call runs in its OWN task
    spawned FROM the reader (mirrors ``codex_app_server`` running each
    ``item/tool/call`` on a per-call task), so the tool never runs on the turn task
    — exactly the topology that deadlocks without the captured reentry token.
    """

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._reader = None

    async def ensure_started(self):
        if self._reader is None:
            self._reader = asyncio.create_task(self._read_loop())

    async def _read_loop(self):
        while True:
            executor, name, args, done = await self._queue.get()
            if executor is None:
                return
            asyncio.create_task(self._handle(executor, name, args, done))

    async def _handle(self, executor, name, args, done):
        try:
            done.set_result(await executor(name, args))
        except Exception as exc:  # pragma: no cover - defensive
            done.set_exception(exc)

    async def dispatch(self, executor, name, args):
        done = asyncio.get_event_loop().create_future()
        await self._queue.put((executor, name, args, done))
        return await done

    async def stop(self):
        if self._reader is not None:
            await self._queue.put((None, None, None, None))
            await self._reader


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.NORMAL, PrivacyMode.EPHEMERAL])
async def test_inline_rename_from_app_server_task_does_not_deadlock(tmp_path, mode):
    """P1 (the CROSS-TASK case the same-task test cannot reach): the turn holds the
    transition lock and dispatches the rename tool through the REAL inline executor
    onto the codex app-server's reader task. That task is NOT the lock owner, so the
    nested rename can only proceed via the per-turn reentry token the executor
    captured on the turn task; without it the turn and tool deadlock. A ``wait_for``
    timeout makes a regression fail loudly instead of hanging the whole suite
    (#2672 review P1)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        await raw.add_node(_structural_node("agent"))  # stored label "Kestrel"
        wrapper = PrivacyEnforcingStorage(raw, mode)
        lock = ReentrantTransitionLock()
        host = _InlineOrchestratorHost(wrapper, raw, lock)
        harness = _AppServerReaderHarness()

        # Streaming path: hold the transition lock across the whole "turn", build
        # the inline executor INSIDE the held lock (as the orchestrator does, so it
        # captures this turn's token), then dispatch the rename onto a reader-spawned
        # task and AWAIT it — the turn is now blocked on the app-server's result
        # exactly as in production, while the tool runs on a foreign task.
        async with lock:
            assert lock.locked()
            executor = host._make_inline_tool_executor("sess-inline")
            await harness.ensure_started()
            _eff, outcome = await asyncio.wait_for(
                harness.dispatch(
                    executor, "rename_agent", {"new_name": "RenamedInline"}
                ),
                timeout=5.0,
            )
        await harness.stop()

        assert host._agent_name == "RenamedInline"  # live name always updates
        if mode is PrivacyMode.NORMAL:
            assert outcome.skipped_privacy is False
            assert outcome.db_row_written is True
            assert await _metadata_value(raw, "name") == "RenamedInline"
            node = await raw.get_node(AGENT_ID)
            assert node.label == "RenamedInline"           # durable node renamed
        else:
            assert outcome.skipped_privacy is True
            assert await _metadata_value(raw, "name") is None  # nothing durable
            node = await raw.get_node(AGENT_ID)
            assert node.label == "Kestrel"                 # stored node untouched


@pytest.mark.asyncio
async def test_concurrent_transition_from_unrelated_task_still_serializes(tmp_path):
    """The token admits ONLY the owning turn's nested write cross-task; a genuinely
    concurrent ``set_privacy_mode`` from an UNRELATED task (which captured no token)
    must still block behind the held lock. Proves the fix preserves the cross-turn
    exclusion the #2672 check-then-write race depends on — it does not turn the lock
    into a free-for-all (#2672 review P1)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        lock = ReentrantTransitionLock()
        order: list[str] = []

        async def turn_holds_lock():
            async with lock:
                # A bound token exists, but it is NEVER handed to the intruder.
                assert lock.current_reentry_token() is not None
                order.append("turn-acquire")
                for _ in range(6):
                    await asyncio.sleep(0)
                order.append("turn-release")

        async def unrelated_transition():
            while "turn-acquire" not in order:
                await asyncio.sleep(0)
            async with lock:  # no captured token → must wait for the outer release
                order.append("transition-acquire")

        await asyncio.wait_for(
            asyncio.gather(turn_holds_lock(), unrelated_transition()), timeout=5.0
        )
        assert order == ["turn-acquire", "turn-release", "transition-acquire"]


@pytest.mark.asyncio
async def test_concurrent_reader_tasks_with_shared_token_serialize():
    """P1 core (the review's requested regression): the codex app-server dispatches
    each inline tool RPC of ONE turn on its OWN reader task, and EVERY such callback
    carries the SAME captured span token. Admitting them purely on token match let
    two durable identity writes run concurrently and interleave their
    metadata/graph/memory/SOUL writes. Two concurrent reader tasks, both bearing the
    turn's token, must therefore SERIALIZE through the per-span reentry mutex: the
    second cannot enter its critical section until the first completes (#2672 review
    P1). Each reader yields control repeatedly INSIDE the lock, so a regression that
    let them overlap would be caught by ``max_concurrent`` > 1."""
    lock = ReentrantTransitionLock()
    events: list[str] = []
    inside = 0
    max_concurrent = 0

    async def reader(token, label):
        nonlocal inside, max_concurrent
        # Faithful to the reader task: bind the captured token, then re-acquire the
        # lock the turn owner holds (as rename_agent_core does inside a volatile
        # transition guard). The token authorizes re-entry; the mutex serializes it.
        with bind_transition_lock_reentry(token):
            async with lock:
                inside += 1
                max_concurrent = max(max_concurrent, inside)
                events.append(f"{label}-enter")
                for _ in range(5):  # a durable write awaits several times
                    await asyncio.sleep(0)
                events.append(f"{label}-exit")
                inside -= 1

    async with lock:  # the streamed turn holds the transition lock across the turn
        assert lock.locked()
        token = lock.current_reentry_token()
        assert token is not None
        r1 = asyncio.create_task(reader(token, "A"))
        # Start B only once A is already INSIDE its critical section, so a broken
        # lock would let B overlap A right here.
        while "A-enter" not in events:
            await asyncio.sleep(0)
        r2 = asyncio.create_task(reader(token, "B"))
        # The turn task now blocks awaiting the "tool results", exactly as the turn
        # blocks on the app-server response in production while tools run cross-task.
        await asyncio.wait_for(asyncio.gather(r1, r2), timeout=5.0)

    assert max_concurrent == 1, "two token-bearing reentrant writes overlapped"
    # A's critical section is fully bracketed before B's — strict serialization.
    assert events == ["A-enter", "A-exit", "B-enter", "B-exit"]


@pytest.mark.asyncio
async def test_owner_cancel_drains_active_cross_task_writer_before_release():
    """P1 cancellation (the review's requested regression): the codex app-server
    dispatches an inline durable-identity write on a DETACHED reader task, admitted
    cross-task under the owning turn's span token. If the streamed turn is cancelled
    while that reader is PAUSED between its privacy check and its persistence, the
    owner's ``__aexit__`` must NOT release the base lock — a concurrent privacy
    transition must stay blocked — until the paused writer finishes. Otherwise the
    transition could acquire the lock and flip to a volatile mode while the
    already-admitted NORMAL-mode write is still persisting (#2672 review P1
    cancellation). Drives the REAL ``ReentrantTransitionLock`` across a faithful
    reader-task boundary and asserts the transition cannot proceed until the writer
    is done."""
    lock = ReentrantTransitionLock()
    writer_inside = asyncio.Event()   # reader has entered its critical section
    release_writer = asyncio.Event()  # let the reader finish its "persist"
    owner_ready = asyncio.Event()     # owner holds the lock and spawned the reader
    transition_acquired = False

    async def reader(token):
        # Faithful reader task: bind the captured token and re-enter the lock the
        # turn owner holds, then PAUSE mid-write — exactly the check→persist gap.
        with bind_transition_lock_reentry(token):
            async with lock:
                writer_inside.set()
                await release_writer.wait()

    async def owner_turn():
        async with lock:
            token = lock.current_reentry_token()
            assert token is not None
            asyncio.create_task(reader(token))
            await writer_inside.wait()  # the reader is now mid-write
            owner_ready.set()
            # Block as the streamed turn blocks awaiting the app-server tool
            # result; the test cancels this task while we are parked here.
            await asyncio.Event().wait()

    async def transition():
        nonlocal transition_acquired
        async with lock:  # an UNRELATED privacy transition (captured no token)
            transition_acquired = True

    owner_task = asyncio.create_task(owner_turn())
    await asyncio.wait_for(owner_ready.wait(), timeout=5.0)

    # Cancel the owning turn while the reader is paused mid-write.
    owner_task.cancel()
    for _ in range(8):  # let the cancellation reach __aexit__ and enter the drain
        await asyncio.sleep(0)
    assert not owner_task.done(), "owner released the lock without draining the writer"
    assert lock.locked(), "base lock must stay held while a cross-task write is live"

    # A concurrent privacy transition must NOT acquire the lock while the
    # already-admitted writer is still paused mid-write.
    transition_task = asyncio.create_task(transition())
    for _ in range(8):
        await asyncio.sleep(0)
    assert not transition_task.done(), "transition acquired the lock during a live write"
    assert transition_acquired is False

    # Let the paused writer finish; the owner's drain now completes, it releases the
    # base lock, its own cancellation propagates, and ONLY THEN may the transition run.
    release_writer.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(owner_task, timeout=5.0)
    await asyncio.wait_for(transition_task, timeout=5.0)
    assert transition_acquired is True


@pytest.mark.asyncio
async def test_owner_normal_exit_does_not_wait_when_no_reentrant_writer():
    """The cancellation drain is a NO-OP on the normal path: with no cross-task
    reentrant writer ever admitted, the outermost owner exit releases the base lock
    immediately (the idle event starts set), so ordinary turns pay zero added
    latency for the #2672 cancellation guarantee."""
    lock = ReentrantTransitionLock()
    async with lock:
        assert lock.locked()
    assert not lock.locked()  # released synchronously on exit, no drain wait


@pytest.mark.asyncio
async def test_reader_cancel_also_unblocks_owner_drain():
    """The drain terminates even if the paused reader is itself CANCELLED rather than
    completing: the reader's ``__aexit__`` sets the idle signal on the cancellation
    path too, so the owner's drain never hangs holding the base lock (#2672 review
    P1 cancellation, cancellation-safe cleanup on both sides)."""
    lock = ReentrantTransitionLock()
    writer_inside = asyncio.Event()
    owner_ready = asyncio.Event()
    reader_holder: dict = {}

    async def reader(token):
        with bind_transition_lock_reentry(token):
            async with lock:
                writer_inside.set()
                await asyncio.Event().wait()  # park until cancelled

    async def owner_turn():
        async with lock:
            token = lock.current_reentry_token()
            reader_holder["task"] = asyncio.create_task(reader(token))
            await writer_inside.wait()
            owner_ready.set()
            await asyncio.Event().wait()

    owner_task = asyncio.create_task(owner_turn())
    await asyncio.wait_for(owner_ready.wait(), timeout=5.0)

    owner_task.cancel()
    for _ in range(8):
        await asyncio.sleep(0)
    assert not owner_task.done(), "owner must block in drain while the reader is live"
    assert lock.locked()

    # Cancelling the reader (not letting it complete) must still release the drain.
    reader_holder["task"].cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(owner_task, timeout=5.0)
    assert not lock.locked(), "owner must release the base lock once the reader is gone"


@pytest.mark.asyncio
async def test_concurrent_inline_renames_serialize_durable_writes(tmp_path):
    """End-to-end P1: two ``rename_agent`` tool calls of one turn are dispatched
    CONCURRENTLY through the REAL inline executor onto the codex app-server's reader
    tasks. The per-span reentry mutex serializes them, so their durable
    metadata/graph writes never interleave and the stores converge on ONE final
    name (matching the live name) rather than a torn mix (#2672 review P1)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        await raw.add_node(_structural_node("agent"))  # stored label "Kestrel"
        wrapper = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        lock = ReentrantTransitionLock()
        host = _InlineOrchestratorHost(wrapper, raw, lock)
        harness = _AppServerReaderHarness()

        async with lock:
            executor = host._make_inline_tool_executor("sess-inline")
            await harness.ensure_started()
            results = await asyncio.wait_for(
                asyncio.gather(
                    harness.dispatch(executor, "rename_agent", {"new_name": "AlphaName"}),
                    harness.dispatch(executor, "rename_agent", {"new_name": "BetaName"}),
                ),
                timeout=5.0,
            )
        await harness.stop()

        # Both renames succeeded (neither deadlocked nor errored).
        assert all(outcome.success for _eff, outcome in results)
        # Durable state is CONSISTENT: metadata, node label, and live name all agree
        # on the same winner — no interleaving left the sources torn apart.
        final_live = host._agent_name
        assert final_live in {"AlphaName", "BetaName"}
        assert await _metadata_value(raw, "name") == final_live
        node = await raw.get_node(AGENT_ID)
        assert node.label == final_live


# ─────────────────────────────────────────────────────────────────────────────
# NESTED cross-task reentry (#2672 review P1 follow-up). The prior cross-task test
# covers ONE reader boundary: the turn dispatches a durable-identity tool directly
# onto the codex app-server's reader task, and the PARENT inline executor
# (``_make_inline_tool_executor``) captures + re-presents the reentry token. It
# does NOT cover a durable-identity write invoked by a FEATURE SUBAGENT: the
# subagent runs its OWN inline tool loop through
# ``Feature._make_feature_inline_tool_executor``, and the app-server dispatches the
# subagent's tools on a SECOND, freshly-spawned reader task that does not inherit
# the parent reader task's token binding. Without the subagent executor ALSO
# capturing the bound token and re-presenting it, that nested write re-acquires the
# transition lock token-less from a foreign task and DEADLOCKS against the turn
# that holds it. This drives the REAL parent AND subagent executors across a
# faithful two-level reader-task boundary.
# ─────────────────────────────────────────────────────────────────────────────


class _NestedRenameTool:
    """Minimal ``AgentTool``-shaped tool whose ``execute`` runs the REAL
    ``rename_agent_core`` — a durable-identity write that re-acquires the agent's
    privacy-transition lock (the deadlock surface under test)."""

    name = "rename_agent"

    def __init__(self, rename_agent):
        self._rename_agent = rename_agent

    async def execute(self, **kwargs):
        from kestrel_sovereign.features.bootstrap.feature import rename_agent_core

        outcome = await rename_agent_core(self._rename_agent, kwargs["new_name"])
        return {
            "success": outcome.success,
            "skipped_privacy": outcome.skipped_privacy,
            "db_row_written": outcome.db_row_written,
        }


class _RenameSubagentFeature(Feature):
    """Real ``Feature`` exercising the PRODUCTION
    ``_make_feature_inline_tool_executor`` / ``_execute_subagent_tool`` across a
    nested reader-task boundary (#2672 review P1 follow-up). Only the subagent
    inline-executor machinery is needed, so ``Feature.__init__`` is deliberately
    bypassed; ``self.agent`` is the ``_RenameAgent`` shape (no ``hooks_manager`` →
    the PRE_TOOL_USE gate is skipped, orthogonal to the deadlock seam)."""

    def __init__(self, rename_agent):
        self.agent = rename_agent
        self.name = "rename_feature"
        self._tools = [_NestedRenameTool(rename_agent)]

    async def initialize(self):  # abstract
        pass

    @property
    def tool_description(self) -> str:  # abstract
        return "Renames the agent."

    def get_tools(self):
        return self._tools


class _NestedSubagentOrchestratorHost(OrchestratorEngineMixin):
    """Parent turn host that builds the REAL parent inline executor, then dispatches
    a feature SUBAGENT whose own inline tool call performs the durable rename on a
    SECOND reader task. ``execute_named_tool`` runs on the parent reader task INSIDE
    the parent executor's bound-token scope — exactly where the subagent executor
    must capture the token for it to survive onto the nested reader task."""

    def __init__(self, lock, feature, harness):
        self._privacy_transition_lock = lock
        self._feature = feature
        self._harness = harness

    def _get_privacy_transition_lock(self):
        return self._privacy_transition_lock

    async def execute_named_tool(self, name, args, *, session_id, source, _capture):
        assert name == "dispatch_subagent"
        _capture["effective_args"] = args
        # Built here (on the parent reader task, inside the parent's bound-token
        # scope) so the REAL subagent executor captures the owning turn's token,
        # then the durable rename is dispatched onto a NESTED reader task.
        subagent_executor = self._feature._make_feature_inline_tool_executor(
            parts_sink=[]
        )
        _eff, result = await self._harness.dispatch(
            subagent_executor, "rename_agent", {"new_name": args["new_name"]}
        )
        return result


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.NORMAL, PrivacyMode.EPHEMERAL])
async def test_nested_subagent_rename_from_app_server_task_does_not_deadlock(
    tmp_path, mode
):
    """P1 follow-up (the NESTED subagent case neither prior cross-task test reaches):
    the streamed turn holds the transition lock and dispatches a feature subagent
    through the REAL parent inline executor onto the codex app-server's reader task;
    that subagent's OWN inline tool loop then dispatches a durable ``rename_agent``
    onto a SECOND reader task. Both reader tasks are foreign to the lock owner, so
    the nested rename can only proceed if the subagent executor
    (``_make_feature_inline_tool_executor``) captured the owning turn's reentry token
    and re-presented it — WITHOUT it the turn (blocked awaiting the app-server) and
    the tool (blocked acquiring the lock the turn holds) DEADLOCK. A ``wait_for``
    timeout makes a regression fail loudly instead of hanging the suite (#2672 review
    P1 follow-up)."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        await raw.add_node(_structural_node("agent"))  # stored label "Kestrel"
        wrapper = PrivacyEnforcingStorage(raw, mode)
        lock = ReentrantTransitionLock()
        rename_agent = _RenameAgent(wrapper, raw, lock)  # _get_privacy_transition_lock → lock
        feature = _RenameSubagentFeature(rename_agent)
        harness = _AppServerReaderHarness()
        host = _NestedSubagentOrchestratorHost(lock, feature, harness)

        # Streaming path: hold the transition lock across the whole "turn", build the
        # parent inline executor INSIDE the held lock (so it captures this turn's
        # token), then dispatch the subagent onto a reader-spawned task and AWAIT it —
        # the turn is now blocked on the app-server result exactly as in production
        # while BOTH the subagent dispatch and its nested rename run on foreign tasks.
        async with lock:
            assert lock.locked()
            parent_executor = host._make_inline_tool_executor("sess-nested")
            await harness.ensure_started()
            _eff, result = await asyncio.wait_for(
                harness.dispatch(
                    parent_executor,
                    "dispatch_subagent",
                    {"new_name": "NestedRenamed"},
                ),
                timeout=5.0,
            )
        await harness.stop()

        assert rename_agent._agent_name == "NestedRenamed"  # live name always updates
        if mode is PrivacyMode.NORMAL:
            assert result["success"] is True
            assert result["skipped_privacy"] is False
            assert result["db_row_written"] is True
            assert await _metadata_value(raw, "name") == "NestedRenamed"
            node = await raw.get_node(AGENT_ID)
            assert node.label == "NestedRenamed"           # durable node renamed
        else:
            assert result["skipped_privacy"] is True
            assert await _metadata_value(raw, "name") is None  # nothing durable
            node = await raw.get_node(AGENT_ID)
            assert node.label == "Kestrel"                 # stored node untouched


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_transition_in_gap_description_skips_durable_write(tmp_path, mode):
    """DESCRIPTION: a flip in the gap makes ``persist_agent_description`` report
    SKIPPED_PRIVACY and leave the durable stores untouched."""
    from kestrel_sovereign.bootstrap.service import (
        PersistOutcome,
        persist_agent_description,
    )

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        await raw.add_node(_structural_node("agent"))
        wrapper = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        lock = asyncio.Lock()

        outcome = await _flip_lands_in_check_persist_gap(
            lock,
            persist_agent_description(
                raw.db, wrapper, AGENT_ID, f"bio {USER_SECRET}", transition_lock=lock
            ),
            lambda: wrapper.set_privacy_mode(mode),
        )

        assert outcome is PersistOutcome.SKIPPED_PRIVACY
        assert await _metadata_value(raw, "description") is None
        node = await raw.get_node(AGENT_ID)
        assert "description" not in (node.properties or {})


def _bootstrap_service(raw, wrapper, lock, agent_data_path=None):
    from kestrel_sovereign.bootstrap.service import BootstrapService

    return BootstrapService(
        db=raw.db,
        agent_id=AGENT_ID,
        agent_name="Kestrel",
        llm_service=None,
        agent_data_path=agent_data_path,
        storage=wrapper,
        privacy_transition_lock=lock,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_transition_in_gap_discovery_history_skips_durable_write(tmp_path, mode):
    """DISCOVERY HISTORY: a flip in the gap makes ``_save_discovery_history`` skip
    the durable ``agent_metadata`` row and hold the raw conversation in the
    session-only store instead."""
    from kestrel_sovereign.bootstrap.service import PersistOutcome

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        lock = asyncio.Lock()
        service = _bootstrap_service(raw, wrapper, lock)
        history = [{"role": "user", "content": f"my secret is {USER_SECRET}"}]

        outcome = await _flip_lands_in_check_persist_gap(
            lock,
            service._save_discovery_history(history),
            lambda: wrapper.set_privacy_mode(mode),
        )

        assert outcome is PersistOutcome.SKIPPED_PRIVACY
        assert await _metadata_value(raw, service.DISCOVERY_HISTORY_KEY) is None
        assert service._session_discovery_history == history  # session-only


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_transition_in_gap_user_name_skips_durable_write(tmp_path, mode):
    """DISCOVERED USER NAME: a flip in the gap makes ``_save_user_name`` skip the
    durable ``agent_metadata`` row entirely."""
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        lock = asyncio.Lock()
        service = _bootstrap_service(raw, wrapper, lock)

        await _flip_lands_in_check_persist_gap(
            lock,
            service._save_user_name(f"User {USER_SECRET}"),
            lambda: wrapper.set_privacy_mode(mode),
        )

        assert await _metadata_value(raw, service.USER_NAME_KEY) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_transition_in_gap_soul_skips_all_durable_writes(tmp_path, mode):
    """SOUL: a flip in the gap makes ``save_soul_md`` skip the disk file, the
    encrypted resource, the ``#soul`` graph reference, and the SOUL-derived
    description."""
    agent_dir = tmp_path / "agentdata"
    agent_dir.mkdir()
    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        wrapper = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        lock = asyncio.Lock()
        service = _bootstrap_service(raw, wrapper, lock, agent_data_path=str(agent_dir))

        saved = await _flip_lands_in_check_persist_gap(
            lock,
            service.save_soul_md(f"# SOUL.md\n\n{USER_SECRET}"),
            lambda: wrapper.set_privacy_mode(mode),
        )

        assert saved is False
        assert not (agent_dir / "SOUL.md").exists()
        assert await raw.get_node(f"{AGENT_ID}#soul") is None
        assert await _metadata_value(raw, "description") is None


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
        # Inception-written agent node (raw store); the volatile-mode governance
        # upsert carries the identity label along and adds a content-free field.
        await raw.add_node(_structural_node("agent"))
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
