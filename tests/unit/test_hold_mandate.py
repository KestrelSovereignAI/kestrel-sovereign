"""Mandate-scoped Hold: an ancestor latches its signed descendants (#3168).

What these pin, each with the mutation it catches:

* authority is the verified signed lineage and nothing else — a peer, a
  child holding its parent, an expired, tampered, wrong-parent, ambiguous, or
  cyclic lineage is refused (mutation: skip the descendant check);
* the door reads the one descendant query Stop's cascade reads (mutation:
  authorize from ``_parent_children`` / ``get_authoritative_children``);
* a mandate latch is its own latch, keyed ``(target, holder)``: releasing it
  never releases the sovereign's agent latch or another ancestor's latch, and
  a holder can never release another holder's latch (mutation: drop the
  store's holder check);
* enforcement refuses a turn on a mandate latch alone (mutation: drop the
  mandate source from ``EffectiveHoldState``);
* authorize-and-write runs under the topology execution lease, so a topology
  writer can neither interleave nor be overtaken (mutation: write outside it);
* the latch is durable, read by target DID whatever is loaded, and a v2
  anchored host upgrades to the v3 schema and boots.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.hold import (
    HoldAuthority,
    HoldAuthorityMismatch,
    HoldCorruptStateError,
    HoldDisposition,
    HoldScope,
    HoldStore,
    HoldTurnRefusal,
    MandateHoldRefusal,
    hold_descendant,
    hold_latch_payload,
    mandate_latch_key,
    release_descendant_hold,
    require_turn_start_allowed,
)
from kestrel_sovereign.hold.state import (
    _HISTORY_ANCHOR_V3_MIGRATION,
    _HistoryAnchorFormat,
    validate_sqlite_hold_readiness,
)
from kestrel_sovereign.inception_service import generate_secp256k1_keypair
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.multi_agent.config import LocalAgentConfig
from kestrel_sovereign.spawn.authority_registry import SpawnAuthorityRegistry
from kestrel_sovereign.spawn.mandate import SpawnMandate, sign_mandate
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.unit.test_agent_manager import _make_mock_agent
from tests.unit.test_hold_anchor_v2 import anchor_backend  # noqa: F401 - fixture
from tests.unit.test_hold_endpoints import _app, _authenticated, _sovereign
from tests.utils.hold_history_v1 import rewind_to_v2_history_anchor

SOVEREIGN = "did:sovereign:operator"
ROOT = "did:pkh:eip155:1:0xMandateRoot"
CHILD = "did:pkh:eip155:1:0xMandateChild"
GRANDCHILD = "did:pkh:eip155:1:0xMandateGrandchild"
PEER = "did:pkh:eip155:1:0xMandatePeer"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def hold_store(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "host-features.db"))
    store = HoldStore(db)
    await store.ensure_schema()
    try:
        yield store
    finally:
        await db.close()


def _signer(did: str):
    private_key, _ = generate_secp256k1_keypair()
    agent = _make_mock_agent(did)
    agent._private_key = private_key
    agent.identity = None
    agent._persisted_spawn_mandate = None
    return agent, private_key


def _lineage(tmp_path, **child_mandate_overrides):
    """Root -> Child -> Grandchild, plus an unrelated Peer root."""

    root, root_key = _signer(ROOT)
    child, child_key = _signer(CHILD)
    child._persisted_spawn_mandate = sign_mandate(
        SpawnMandate(
            parent_did=ROOT,
            child_did=CHILD,
            ttl_seconds=0,
            max_child_depth=1,
            **child_mandate_overrides,
        ),
        root_key,
    )
    grandchild = _make_mock_agent(GRANDCHILD)
    grandchild._persisted_spawn_mandate = sign_mandate(
        SpawnMandate(parent_did=CHILD, child_did=GRANDCHILD, ttl_seconds=0),
        child_key,
    )
    peer, _peer_key = _signer(PEER)
    manager = AgentManager(base_data_dir=tmp_path)
    for name, agent in (
        ("Root", root),
        ("Child", child),
        ("Grandchild", grandchild),
        ("Peer", peer),
    ):
        manager._register_agent(name, agent)
    return SimpleNamespace(
        manager=manager,
        root=root,
        root_key=root_key,
        child=child,
        child_key=child_key,
        grandchild=grandchild,
        peer=peer,
    )


async def _hold(manager, store, holder, target, *, op, reason="runaway"):
    return await hold_descendant(
        manager=manager,
        store=store,
        holder_did=holder,
        target_did=target,
        reason=reason,
        operation_id=op,
    )


async def _release(manager, store, holder, target, *, op, reason="resume"):
    return await release_descendant_hold(
        manager=manager,
        store=store,
        holder_did=holder,
        target_did=target,
        reason=reason,
        operation_id=op,
    )


# ---------------------------------------------------------------------------
# Authority: the signed lineage, downward only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parent_holds_direct_and_cascading_descendants(tmp_path, hold_store):
    tree = _lineage(tmp_path)
    # The unsigned projection is not consulted: clearing it changes nothing.
    tree.manager._parent_children.clear()

    direct = await _hold(tree.manager, hold_store, ROOT, CHILD, op="direct")
    deep = await _hold(tree.manager, hold_store, ROOT, GRANDCHILD, op="deep")

    for mutation, target in ((direct, CHILD), (deep, GRANDCHILD)):
        assert mutation.receipt.disposition is HoldDisposition.APPLIED
        assert mutation.receipt.authority is HoldAuthority.MANDATE
        assert mutation.receipt.actor_id == ROOT
        assert mutation.current.subject_id == target
        assert mutation.current.holder_id == ROOT
    effective = await hold_store.get_effective(GRANDCHILD)
    assert effective.held is True
    assert effective.sources == (HoldScope.MANDATE,)
    assert [latch.holder_id for latch in effective.mandates] == [ROOT]
    assert (await hold_store.get_effective(ROOT)).held is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("holder", "target"),
    [
        (CHILD, ROOT),  # a child never holds its parent
        (GRANDCHILD, ROOT),  # nor any ancestor
        (PEER, CHILD),  # a peer has no mandate
        (CHILD, PEER),
    ],
    ids=["child-parent", "grandchild-root", "peer-child", "child-peer"],
)
async def test_only_a_signed_ancestor_may_hold(tmp_path, hold_store, holder, target):
    tree = _lineage(tmp_path)

    with pytest.raises(MandateHoldRefusal) as refused:
        await _hold(tree.manager, hold_store, holder, target, op="refused")

    assert refused.value.code == MandateHoldRefusal.NOT_A_DESCENDANT
    assert (await hold_store.get_effective(target)).held is False
    assert (await hold_store.list_receipts(limit=10)).entries == ()


@pytest.mark.asyncio
async def test_an_expired_mandate_confers_no_hold(tmp_path, hold_store, monkeypatch):
    root, root_key = _signer(ROOT)
    child = _make_mock_agent(CHILD)
    child._persisted_spawn_mandate = sign_mandate(
        SpawnMandate(parent_did=ROOT, child_did=CHILD, ttl_seconds=111),
        root_key,
    )
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("Root", root)
    manager._register_agent("Child", child)
    # Proven live first, then expired: the query omits the edge.
    await _hold(manager, hold_store, ROOT, CHILD, op="while-live")
    monkeypatch.setattr(
        "kestrel_sovereign.multi_agent.agent_manager.remaining_spawn_ttl_seconds",
        lambda _created_at, ttl_seconds, **_kwargs: 0 if ttl_seconds == 111 else 999,
    )

    with pytest.raises(MandateHoldRefusal) as refused:
        await _hold(manager, hold_store, ROOT, CHILD, op="expired")
    assert refused.value.code == MandateHoldRefusal.NOT_A_DESCENDANT
    # A lapsed holder cannot release either; its latch stays, visible, until
    # the sovereign releases it (holds never expire).
    with pytest.raises(MandateHoldRefusal):
        await _release(manager, hold_store, ROOT, CHILD, op="expired-release")
    [latch] = (await hold_store.get_effective(CHILD)).mandates
    assert latch.holder_id == ROOT


@pytest.mark.asyncio
async def test_a_tampered_mandate_fails_closed(tmp_path, hold_store):
    tree = _lineage(tmp_path)
    tree.child._persisted_spawn_mandate.purpose = "tampered after signing"

    with pytest.raises(MandateHoldRefusal) as refused:
        await _hold(tree.manager, hold_store, ROOT, GRANDCHILD, op="tampered")

    assert refused.value.code == MandateHoldRefusal.LINEAGE_UNVERIFIABLE
    assert (await hold_store.get_effective(GRANDCHILD)).held is False


@pytest.mark.asyncio
async def test_a_mandate_signed_by_the_wrong_parent_fails_closed(tmp_path, hold_store):
    tree = _lineage(tmp_path)
    # Names ROOT as parent but is signed with another key.
    impostor_key, _ = generate_secp256k1_keypair()
    tree.child._persisted_spawn_mandate = sign_mandate(
        SpawnMandate(parent_did=ROOT, child_did=CHILD, ttl_seconds=0),
        impostor_key,
    )

    with pytest.raises(MandateHoldRefusal) as refused:
        await _hold(tree.manager, hold_store, ROOT, CHILD, op="wrong-parent")

    assert refused.value.code == MandateHoldRefusal.LINEAGE_UNVERIFIABLE
    assert (await hold_store.get_effective(CHILD)).held is False


@pytest.mark.asyncio
async def test_a_signed_cycle_fails_closed(tmp_path, hold_store):
    first, first_key = _signer(CHILD)
    second, second_key = _signer(GRANDCHILD)
    first._persisted_spawn_mandate = sign_mandate(
        SpawnMandate(parent_did=GRANDCHILD, child_did=CHILD), second_key
    )
    second._persisted_spawn_mandate = sign_mandate(
        SpawnMandate(parent_did=CHILD, child_did=GRANDCHILD), first_key
    )
    manager = AgentManager(base_data_dir=tmp_path)
    manager._agents.update({"First": first, "Second": second})
    manager._agent_names.update({CHILD: "First", GRANDCHILD: "Second"})
    manager._child_mandates.update(
        {
            "First": first._persisted_spawn_mandate,
            "Second": second._persisted_spawn_mandate,
        }
    )

    with pytest.raises(MandateHoldRefusal) as refused:
        await _hold(manager, hold_store, CHILD, GRANDCHILD, op="cycle")

    assert refused.value.code == MandateHoldRefusal.LINEAGE_UNVERIFIABLE
    assert (await hold_store.get_effective(GRANDCHILD)).held is False


@pytest.mark.asyncio
async def test_an_ambiguous_descendant_graph_fails_closed(tmp_path, hold_store):
    root, root_key = _signer(ROOT)
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("Root", root)
    mandate = sign_mandate(
        SpawnMandate(parent_did=ROOT, child_did=CHILD, ttl_seconds=0), root_key
    )
    witness = SimpleNamespace(
        child_did=CHILD,
        parent_did=ROOT,
        child_name="ColdChild",
        mandate=mandate,
        active=True,
    )
    # One child DID recorded twice with different routing: ambiguous.
    manager._spawn_authority_registry = SimpleNamespace(
        records=lambda: (witness, SimpleNamespace(**{**vars(witness), "child_name": "Other"}))
    )

    with pytest.raises(MandateHoldRefusal) as refused:
        await _hold(manager, hold_store, ROOT, CHILD, op="ambiguous")

    assert refused.value.code == MandateHoldRefusal.LINEAGE_UNVERIFIABLE


@pytest.mark.asyncio
async def test_authority_comes_only_from_the_stop_descendant_query(
    tmp_path, hold_store, monkeypatch
):
    """The door reads Stop's signed query, never the unsigned projections."""

    tree = _lineage(tmp_path)
    tree.manager._parent_children[ROOT] = ["Child", "Grandchild", "Peer"]

    async def no_descendants(_parent_did, _live):
        return []

    monkeypatch.setattr(
        tree.manager, "_authoritative_stop_descendants_under_lease", no_descendants
    )
    assert await tree.manager.get_authoritative_stop_descendants(ROOT) == []

    with pytest.raises(MandateHoldRefusal):
        await _hold(tree.manager, hold_store, ROOT, CHILD, op="projection")
    # A projection or the direct-children query naming the target is not
    # authority either.
    assert "Child" in await tree.manager.get_authoritative_children(ROOT)
    with pytest.raises(MandateHoldRefusal):
        await _hold(tree.manager, hold_store, ROOT, PEER, op="projection-peer")


@pytest.mark.asyncio
async def test_a_cold_descendant_can_be_held_and_stays_held_across_load_order(
    tmp_path,
):
    """A registry-only signed child: the parent holds it before it is loaded."""

    root, root_key = _signer(ROOT)
    mandate = sign_mandate(
        SpawnMandate(parent_did=ROOT, child_did=CHILD, ttl_seconds=0), root_key
    )
    SpawnAuthorityRegistry(tmp_path).record_active(
        child_name="ColdChild",
        child_did=CHILD,
        mandate=mandate,
        config=LocalAgentConfig(
            data_dir=Path("agent_data") / "ColdChild", port=8802, autostart=False
        ),
    )
    manager = AgentManager(base_data_dir=tmp_path)
    manager._register_agent("Root", root)
    path = tmp_path / "host-features.db"
    db = await AsyncDatabase.sqlite(str(path))
    try:
        store = HoldStore(db)
        await store.ensure_schema()
        assert manager.get_agent("ColdChild") is None
        await _hold(manager, store, ROOT, CHILD, op="cold")
    finally:
        await db.close()

    # Restart with nothing loaded: the child's latch is keyed by its DID, so
    # it holds whether or not its parent is loaded, cold, or elsewhere.
    reopened = await AsyncDatabase.sqlite(str(path))
    try:
        restarted = HoldStore(reopened)
        await restarted.ensure_schema()
        [boot_latch] = await restarted.read_boot_state()
        assert boot_latch.scope is HoldScope.MANDATE
        assert (boot_latch.subject_id, boot_latch.holder_id) == (CHILD, ROOT)
        effective = await restarted.get_effective(CHILD)
        assert effective.sources == (HoldScope.MANDATE,)
    finally:
        await reopened.close()


# ---------------------------------------------------------------------------
# Independent latches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_parent_release_leaves_the_sovereign_hold_in_place(
    tmp_path, hold_store
):
    tree = _lineage(tmp_path)
    sovereign = await hold_store.set_hold(
        scope=HoldScope.AGENT,
        target_id=CHILD,
        actor_id=SOVEREIGN,
        reason="sovereign investigation",
        operation_id="sovereign-hold",
        authority=HoldAuthority.SOVEREIGN,
    )
    await _hold(tree.manager, hold_store, ROOT, CHILD, op="parent-hold")
    assert (await hold_store.get_effective(CHILD)).sources == (
        HoldScope.AGENT,
        HoldScope.MANDATE,
    )

    released = await _release(tree.manager, hold_store, ROOT, CHILD, op="parent-release")

    assert released.receipt.disposition is HoldDisposition.APPLIED
    effective = await hold_store.get_effective(CHILD)
    assert effective.agent == sovereign.current
    assert effective.mandates == ()
    assert effective.held is True
    # And the parent has nothing further to release.
    assert await _release(tree.manager, hold_store, ROOT, CHILD, op="again") is None


@pytest.mark.asyncio
async def test_two_ancestors_hold_one_grandchild_independently(tmp_path, hold_store):
    tree = _lineage(tmp_path)
    await _hold(tree.manager, hold_store, ROOT, GRANDCHILD, op="root-hold")
    await _hold(tree.manager, hold_store, CHILD, GRANDCHILD, op="child-hold")
    effective = await hold_store.get_effective(GRANDCHILD)
    assert sorted(latch.holder_id for latch in effective.mandates) == [CHILD, ROOT]

    await _release(tree.manager, hold_store, CHILD, GRANDCHILD, op="child-release")

    [remaining] = (await hold_store.get_effective(GRANDCHILD)).mandates
    assert remaining.holder_id == ROOT


@pytest.mark.asyncio
async def test_the_store_refuses_a_holder_releasing_another_holders_latch(
    hold_store,
):
    held = await hold_store.set_hold(
        scope=HoldScope.MANDATE,
        target_id=GRANDCHILD,
        holder_id=ROOT,
        actor_id=ROOT,
        reason="root hold",
        operation_id="root",
        authority=HoldAuthority.MANDATE,
    )

    with pytest.raises(HoldAuthorityMismatch):
        await hold_store.release_hold(
            scope=HoldScope.MANDATE,
            target_id=GRANDCHILD,
            holder_id=ROOT,
            actor_id=CHILD,
            reason="not mine",
            operation_id="forged-release",
            expected_hold_receipt_id=held.receipt.receipt_id,
            authority=HoldAuthority.MANDATE,
        )
    assert (await hold_store.get_effective(GRANDCHILD)).mandates == (held.current,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "target", "holder", "actor", "authority"),
    [
        # The sovereign never sets a mandate latch; it holds with scope agent.
        (HoldScope.MANDATE, CHILD, ROOT, SOVEREIGN, HoldAuthority.SOVEREIGN),
        # A mandate authority never reaches a host or agent latch.
        (HoldScope.AGENT, CHILD, None, ROOT, HoldAuthority.MANDATE),
        (HoldScope.HOST, None, None, ROOT, HoldAuthority.MANDATE),
        # A holder never sets a latch in another holder's name.
        (HoldScope.MANDATE, CHILD, ROOT, PEER, HoldAuthority.MANDATE),
    ],
)
async def test_the_store_refuses_an_authority_that_cannot_set_the_latch(
    hold_store, scope, target, holder, actor, authority
):
    kwargs = {"holder_id": holder} if holder is not None else {}
    with pytest.raises(HoldAuthorityMismatch):
        await hold_store.set_hold(
            scope=scope,
            target_id=target,
            actor_id=actor,
            reason="refused",
            operation_id="refused",
            authority=authority,
            **kwargs,
        )
    assert (await hold_store.list_receipts(limit=10)).entries == ()


@pytest.mark.asyncio
async def test_the_sovereign_releases_a_mandate_latch(hold_store):
    held = await hold_store.set_hold(
        scope=HoldScope.MANDATE,
        target_id=CHILD,
        holder_id=ROOT,
        actor_id=ROOT,
        reason="root hold",
        operation_id="root",
        authority=HoldAuthority.MANDATE,
    )

    released = await hold_store.release_hold(
        scope=HoldScope.MANDATE,
        target_id=CHILD,
        holder_id=ROOT,
        actor_id=SOVEREIGN,
        reason="sovereign resumes",
        operation_id="sovereign-release",
        expected_hold_receipt_id=held.receipt.receipt_id,
        authority=HoldAuthority.SOVEREIGN,
    )

    assert released.receipt.disposition is HoldDisposition.APPLIED
    assert (await hold_store.get_effective(CHILD)).held is False


# ---------------------------------------------------------------------------
# Durable evidence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forged_key",
    [
        "did:agent:unkeyed",  # not a pair at all
        '["did:a","did:a"]',  # a holder holding itself
        '[ "did:a","did:b"]',  # not canonical
    ],
)
async def test_a_noncanonical_mandate_key_is_corruption(hold_store, forged_key):
    await hold_store.set_hold(
        scope=HoldScope.MANDATE,
        target_id=CHILD,
        holder_id=ROOT,
        actor_id=ROOT,
        reason="root hold",
        operation_id="root",
        authority=HoldAuthority.MANDATE,
    )
    key = mandate_latch_key(CHILD, ROOT)
    for table in (
        "hold_latches",
        "hold_receipts",
        "hold_receipt_witnesses",
        "hold_receipt_content_witnesses",
    ):
        await hold_store._db.execute(
            f"UPDATE {table} SET target_id = ? WHERE target_id = ?",
            (forged_key, key),
        )

    with pytest.raises(HoldCorruptStateError):
        await hold_store.read_boot_state()


@pytest.mark.asyncio
async def test_a_rewritten_mandate_actor_is_caught(hold_store):
    await hold_store.set_hold(
        scope=HoldScope.MANDATE,
        target_id=CHILD,
        holder_id=ROOT,
        actor_id=ROOT,
        reason="root hold",
        operation_id="root",
        authority=HoldAuthority.MANDATE,
    )
    await hold_store._db.execute(
        "UPDATE hold_receipts SET actor_id = ? WHERE operation_id = ?",
        (PEER, "root"),
    )

    with pytest.raises(HoldCorruptStateError, match="authority"):
        await hold_store.get_effective(CHILD)


@pytest.mark.asyncio
async def test_mandate_history_is_readable_by_subject(hold_store):
    for holder, op in ((ROOT, "a"), (CHILD, "b")):
        await hold_store.set_hold(
            scope=HoldScope.MANDATE,
            target_id=GRANDCHILD,
            holder_id=holder,
            actor_id=holder,
            reason=f"by {holder}",
            operation_id=op,
            authority=HoldAuthority.MANDATE,
        )
    await hold_store.set_hold(
        scope=HoldScope.MANDATE,
        target_id=CHILD,
        holder_id=ROOT,
        actor_id=ROOT,
        reason="other subject",
        operation_id="c",
        authority=HoldAuthority.MANDATE,
    )

    page = await hold_store.list_receipts(
        scope=HoldScope.MANDATE, subject_id=GRANDCHILD, limit=10
    )

    assert sorted(entry.receipt.holder_id for entry in page.entries) == [CHILD, ROOT]
    assert {entry.receipt.subject_id for entry in page.entries} == {GRANDCHILD}


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_mandate_latch_alone_refuses_the_next_turn(hold_store):
    held = await hold_store.set_hold(
        scope=HoldScope.MANDATE,
        target_id=CHILD,
        holder_id=ROOT,
        actor_id=ROOT,
        reason="parent paused me",
        operation_id="root",
        authority=HoldAuthority.MANDATE,
    )
    agent = KestrelAgent.__new__(KestrelAgent)
    agent.did = CHILD
    agent._hold_store = hold_store

    with pytest.raises(HoldTurnRefusal) as refused:
        await require_turn_start_allowed(agent)

    payload = refused.value.wire_payload()
    assert payload["host_hold"] is None and payload["agent_hold"] is None
    assert payload["mandate_holds"] == [hold_latch_payload(held.current)]
    assert payload["mandate_holds"][0]["target_id"] == CHILD
    assert payload["mandate_holds"][0]["holder_id"] == ROOT
    json.dumps(payload)


# ---------------------------------------------------------------------------
# Atomic authorize-and-mutate under the topology lease
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_topology_writer_waits_for_an_authorized_hold_to_commit(
    tmp_path, hold_store, monkeypatch
):
    tree = _lineage(tmp_path)
    writing = asyncio.Event()
    finish = asyncio.Event()
    original = hold_store.set_hold

    async def paused_set_hold(**kwargs):
        writing.set()
        await finish.wait()
        return await original(**kwargs)

    monkeypatch.setattr(hold_store, "set_hold", paused_set_hold)
    hold = asyncio.create_task(_hold(tree.manager, hold_store, ROOT, CHILD, op="race"))
    await asyncio.wait_for(writing.wait(), 5)

    lease = tree.manager.a2a_lifecycle_lease()
    writer = asyncio.create_task(lease.acquire())
    await asyncio.sleep(0.05)
    assert not writer.done(), "a terminate slipped between authorize and write"

    finish.set()
    await asyncio.wait_for(hold, 5)
    await asyncio.wait_for(writer, 5)
    lease.release()
    assert (await hold_store.get_effective(CHILD)).held is True


@pytest.mark.asyncio
async def test_a_hold_waiting_behind_a_terminate_sees_the_withdrawn_lineage(
    tmp_path, hold_store
):
    tree = _lineage(tmp_path)
    lease = tree.manager.a2a_lifecycle_lease()
    await lease.acquire()
    hold = asyncio.create_task(_hold(tree.manager, hold_store, ROOT, CHILD, op="late"))
    await asyncio.sleep(0.05)
    assert not hold.done()

    # The topology writer withdraws the child's signed receipt, then yields.
    tree.child._persisted_spawn_mandate = None
    lease.release()

    with pytest.raises(MandateHoldRefusal):
        await asyncio.wait_for(hold, 5)
    assert (await hold_store.get_effective(CHILD)).held is False


# ---------------------------------------------------------------------------
# Upgrade: a v2-anchored host boots on the v3 schema
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_v2_anchored_history_boots_after_the_mandate_migration(
    anchor_backend,  # noqa: F811 - the imported dual-backend fixture
):
    db = anchor_backend.db
    store = anchor_backend.open_store()
    await store.ensure_schema()
    agent = await store.set_hold(
        scope=HoldScope.AGENT,
        target_id=CHILD,
        actor_id=SOVEREIGN,
        reason="before upgrade",
        operation_id="v2-agent",
        authority=HoldAuthority.SOVEREIGN,
    )
    host = await store.set_hold(
        scope=HoldScope.HOST,
        actor_id=SOVEREIGN,
        reason="fleet",
        operation_id="v2-host",
        authority=HoldAuthority.SOVEREIGN,
    )
    v2_anchor = await rewind_to_v2_history_anchor(db, store)
    assert v2_anchor.startswith(_HistoryAnchorFormat.V2.header)
    # The v2 schema really does refuse a mandate row.
    with pytest.raises(Exception):
        await db.execute(
            "INSERT INTO hold_latches (scope, target_id) VALUES ('mandate', 'x')"
        )
    if anchor_backend.sqlite_path is not None:
        for member in anchor_backend.sqlite_path.parent.glob(
            f"{anchor_backend.sqlite_path.name}*"
        ):
            member.chmod(0o600)
        assert set(validate_sqlite_hold_readiness(anchor_backend.sqlite_path)) == {
            agent.current,
            host.current,
        }

    upgraded = anchor_backend.open_store()
    await upgraded.ensure_schema()

    stable = await upgraded._read_history_anchor()
    assert stable.startswith(_HistoryAnchorFormat.V3.header)
    assert stable == await upgraded._current_history_anchor_payload()
    assert await upgraded._read_external_history_candidate() is None
    assert await db.fetchall(
        "SELECT name FROM hold_schema_migrations WHERE name = ?",
        (_HISTORY_ANCHOR_V3_MIGRATION,),
    )
    assert set(await upgraded.read_boot_state()) == {agent.current, host.current}
    mandate = await upgraded.set_hold(
        scope=HoldScope.MANDATE,
        target_id=CHILD,
        holder_id=ROOT,
        actor_id=ROOT,
        reason="after upgrade",
        operation_id="v3-mandate",
        authority=HoldAuthority.MANDATE,
    )
    if anchor_backend.sqlite_path is not None:
        assert set(validate_sqlite_hold_readiness(anchor_backend.sqlite_path)) == {
            agent.current,
            host.current,
            mandate.current,
        }

    restarted = anchor_backend.open_store()
    await restarted.ensure_schema()
    effective = await restarted.get_effective(CHILD)
    assert effective.sources == (HoldScope.HOST, HoldScope.AGENT, HoldScope.MANDATE)
    assert effective.mandates == (mandate.current,)
    # The rebuilt receipt table still numbers every receipt in commit order:
    # a SQLite rebuild drops the feed trigger with the old table.
    page = await restarted.list_receipts(limit=10)
    sequences = [entry.feed_seq for entry in page.entries]
    assert len(sequences) == 3 and sequences == sorted(set(sequences))
    assert page.entries[-1].receipt == mandate.receipt


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_mandate_latches_compose_on_each_backend(
    anchor_backend,  # noqa: F811 - the imported dual-backend fixture
):
    store = anchor_backend.open_store()
    await store.ensure_schema()
    for holder, op in ((ROOT, "root"), (CHILD, "child")):
        await store.set_hold(
            scope=HoldScope.MANDATE,
            target_id=GRANDCHILD,
            holder_id=holder,
            actor_id=holder,
            reason=f"by {holder}",
            operation_id=op,
            authority=HoldAuthority.MANDATE,
        )
    [child_latch] = [
        latch
        for latch in (await store.get_effective(GRANDCHILD)).mandates
        if latch.holder_id == CHILD
    ]

    await store.release_hold(
        scope=HoldScope.MANDATE,
        target_id=GRANDCHILD,
        holder_id=CHILD,
        actor_id=CHILD,
        reason="child resumes",
        operation_id="child-release",
        expected_hold_receipt_id=child_latch.hold_receipt_id,
        authority=HoldAuthority.MANDATE,
    )

    restarted = anchor_backend.open_store()
    await restarted.ensure_schema()
    [latch] = await restarted.read_boot_state()
    assert (latch.subject_id, latch.holder_id) == (GRANDCHILD, ROOT)
    page = await restarted.list_receipts(
        scope=HoldScope.MANDATE, subject_id=GRANDCHILD, limit=10
    )
    assert len(page.entries) == 3


# ---------------------------------------------------------------------------
# Doors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_sovereign_holds_without_an_admin_agent_role(hold_store):
    """Host and agent Hold stay on the sovereign door; no agent role exists."""

    class _Agent:
        """Carries a DID and nothing else, so no role can be read from it."""

        def __init__(self, did):
            self.did = did

    app, _ = _app(agents=(), caller=_sovereign(), store=hold_store)
    app.state.agent_manager.list_agents.return_value = {"Child": _Agent(CHILD)}
    client = TestClient(app)

    host = client.post(
        "/api/host/hold",
        json={"scope": "host", "reason": "fleet", "operation_id": "sov-host"},
    )
    agent = client.post(
        "/api/host/hold",
        json={
            "scope": "agent",
            "target_id": CHILD,
            "reason": "one agent",
            "operation_id": "sov-agent",
        },
    )

    assert host.status_code == 200 and agent.status_code == 200
    assert host.json()["receipt"]["authority"] == "sovereign"
    assert agent.json()["receipt"]["authority"] == "sovereign"
    effective = await hold_store.get_effective(CHILD)
    assert effective.sources == (HoldScope.HOST, HoldScope.AGENT)

    # An authenticated agent-or-user caller is refused at the same door.
    app_other, _ = _app(agents=(), caller=_authenticated(), store=hold_store)
    refused = TestClient(app_other).post(
        "/api/host/hold",
        json={"scope": "host", "reason": "x", "operation_id": "not-sovereign"},
    )
    assert refused.status_code == 403


@pytest.mark.asyncio
async def test_the_host_door_shows_and_releases_mandate_latches(hold_store):
    held = await hold_store.set_hold(
        scope=HoldScope.MANDATE,
        target_id=CHILD,
        holder_id=ROOT,
        actor_id=ROOT,
        reason="parent paused",
        operation_id="root",
        authority=HoldAuthority.MANDATE,
    )
    # The target is no longer hosted here: its latch must still be visible.
    app, _ = _app(agents=(("Peer", PEER),), caller=_sovereign(), store=hold_store)
    client = TestClient(app)

    state = client.get("/api/host/hold").json()
    assert state["mandate_holds"] == [hold_latch_payload(held.current)]
    assert state["agents"][0]["mandate_holds"] == []

    refused_set = client.post(
        "/api/host/hold",
        json={
            "scope": "mandate",
            "target_id": CHILD,
            "reason": "sovereign cannot mandate",
            "operation_id": "sov-mandate",
        },
    )
    assert refused_set.status_code == 400
    missing_holder = client.post(
        "/api/host/hold/release",
        json={
            "scope": "mandate",
            "target_id": CHILD,
            "reason": "resume",
            "operation_id": "release-no-holder",
            "expected_hold_receipt_id": held.receipt.receipt_id,
        },
    )
    assert missing_holder.status_code == 400
    unknown = client.post(
        "/api/host/hold/release",
        json={
            "scope": "mandate",
            "target_id": CHILD,
            "holder_id": PEER,
            "reason": "resume",
            "operation_id": "release-wrong-holder",
            "expected_hold_receipt_id": held.receipt.receipt_id,
        },
    )
    assert unknown.status_code == 404

    released = client.post(
        "/api/host/hold/release",
        json={
            "scope": "mandate",
            "target_id": CHILD,
            "holder_id": ROOT,
            "reason": "sovereign resumes",
            "operation_id": "sov-release",
            "expected_hold_receipt_id": held.receipt.receipt_id,
        },
    )
    assert released.status_code == 200
    body = released.json()
    assert body["receipt"]["authority"] == "sovereign"
    assert body["receipt"]["holder_id"] == ROOT
    assert body["current"] is None

    receipts = client.get(
        "/api/host/hold/receipts", params={"scope": "mandate", "agent_id": CHILD}
    ).json()["receipts"]
    assert [receipt["action"] for receipt in receipts] == ["hold", "release"]


@pytest.mark.asyncio
async def test_the_spawn_tools_bind_the_holder_from_the_runtime(tmp_path, hold_store):
    import inspect

    from kestrel_sovereign.features.spawn.feature import SpawnFeature

    for name in ("hold_descendant", "release_descendant_hold"):
        parameters = inspect.signature(getattr(SpawnFeature, name)).parameters
        assert list(parameters) == ["self", "target_did", "reason"]

    tree = _lineage(tmp_path)
    tree.root._hold_store = hold_store
    tree.root.did = ROOT
    feature = SpawnFeature.__new__(SpawnFeature)
    feature.agent = tree.root

    async def ready_manager():
        return tree.manager

    feature._get_ready_agent_manager = ready_manager

    held = await feature.hold_descendant(target_did=GRANDCHILD, reason="pause")
    assert held.status is ToolResultStatus.OK, held
    assert held.data["receipt"]["actor_id"] == ROOT
    assert held.data["current"]["holder_id"] == ROOT

    refused = await feature.hold_descendant(target_did=PEER, reason="not mine")
    assert refused.status is ToolResultStatus.ERROR
    assert refused.data["refusal"] == MandateHoldRefusal.NOT_A_DESCENDANT

    released = await feature.release_descendant_hold(
        target_did=GRANDCHILD, reason="resume"
    )
    assert released.status is ToolResultStatus.OK
    assert released.data["released"] is True
    assert (await hold_store.get_effective(GRANDCHILD)).held is False


@pytest.mark.asyncio
async def test_the_held_descendant_sees_its_mandate_hold_by_role_only(hold_store):
    from kestrel_sovereign.hold import inspect_self_hold

    await hold_store.set_hold(
        scope=HoldScope.MANDATE,
        target_id=CHILD,
        holder_id=ROOT,
        actor_id=ROOT,
        reason="parent paused me",
        operation_id="root",
        authority=HoldAuthority.MANDATE,
    )
    agent = KestrelAgent.__new__(KestrelAgent)
    agent.did = CHILD
    agent._hold_store = hold_store

    result = await inspect_self_hold(agent)

    assert result["state"] == "held"
    assert result["sources"] == ["mandate"]
    [view] = result["latches"]["mandate"]
    assert view["actor_role"] == "mandate"
    assert view["reason"] == "parent paused me"
    [episode] = result["history"]["episodes"]
    assert (episode["scope"], episode["status"], episode["set_by_role"]) == (
        "mandate",
        "active",
        "mandate",
    )
    # The redaction policy withholds actor identity: the holder DID never
    # reaches the held subject.
    assert ROOT not in json.dumps(result)
