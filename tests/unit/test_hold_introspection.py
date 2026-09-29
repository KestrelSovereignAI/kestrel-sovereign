"""Self-scoped Hold introspection and the recorded Hold authority (#3166).

These pin the authority boundary, not only the happy path:

* the subject is the runtime-bound DID, so another agent's history is never
  readable (mutation: drop the ``target_id`` binding in ``_scope_history``);
* a read failure is ``unknown``, never ``not_held``;
* the actor role is the RECORDED authority, never a guess from the actor
  string (mutation: classify by DID prefix);
* the census: every production caller of ``set_hold``/``release_hold`` is the
  sovereign host door or the mandate descendant door, and names its authority.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.features.identity.feature import IdentityFeature
from kestrel_sovereign.hold import (
    SELF_HOLD_HISTORY_LIMIT,
    SELF_HOLD_REDACTION_POLICY,
    HoldAuthority,
    HoldCorruptStateError,
    HoldScope,
    HoldStateError,
    HoldStore,
    hold_receipt_payload,
    inspect_self_hold,
)
from kestrel_sovereign.hold import introspection as introspection_module
from kestrel_sovereign.hold.state import _receipt_content_digest, _receipt_from_row
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.utils.hold_history_v1 import rewind_to_v1_history_anchor

SELF = "did:web:agents.example:kite"
PEER = "did:web:agents.example:peer"
# The real sovereign here is a did:pkh address: nothing about its shape says
# "sovereign". A prefix classifier would misname it.
SOVEREIGN = "did:pkh:eip155:1:0x1111111111111111111111111111111111111111"

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Agent:
    """The runtime's own agent object: its DID and its bound Hold store."""

    def __init__(self, did, store):
        self.did = did
        self._hold_store = store


@pytest.fixture
async def hold_db(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "host.db"))
    store = HoldStore(db)
    await store.ensure_schema()
    try:
        yield db, store
    finally:
        await db.close()


async def _hold(store, *, scope, target_id=None, reason, operation_id, actor=SOVEREIGN):
    return await store.set_hold(
        scope=scope,
        target_id=target_id,
        actor_id=actor,
        reason=reason,
        operation_id=operation_id,
        authority=HoldAuthority.SOVEREIGN,
    )


async def _release(store, mutation, *, reason, operation_id, actor=SOVEREIGN):
    return await store.release_hold(
        scope=mutation.receipt.scope,
        target_id=mutation.receipt.target_id,
        actor_id=actor,
        reason=reason,
        operation_id=operation_id,
        expected_hold_receipt_id=mutation.current.hold_receipt_id,
        authority=HoldAuthority.SOVEREIGN,
    )


# ---------------------------------------------------------------------------
# History: a resumed agent can explain why it was silent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_released_hold_reads_back_as_an_episode_after_resume(hold_db):
    _db, store = hold_db
    held = await _hold(
        store,
        scope=HoldScope.AGENT,
        target_id=SELF,
        reason="runaway heartbeat loop",
        operation_id="op-hold",
    )
    await _release(store, held, reason="loop fixed", operation_id="op-release")

    result = await inspect_self_hold(_Agent(SELF, store))

    assert result["state"] == "not_held"
    assert result["held"] is False
    assert result["history"]["status"] == "read"
    [episode] = result["history"]["episodes"]
    assert episode["scope"] == "agent"
    assert episode["status"] == "released"
    assert episode["reason"] == "runaway heartbeat loop"
    assert episode["set_by_role"] == "sovereign"
    assert episode["ended_by_role"] == "sovereign"
    assert episode["end_reason"] == "loop fixed"
    assert episode["set_at"] == held.receipt.occurred_at
    assert episode["ended_at"] is not None
    actions = [receipt["action"] for receipt in result["history"]["receipts"]]
    assert actions == ["release", "hold"], "history is newest first"


@pytest.mark.asyncio
async def test_tool_reports_the_episode_to_the_resumed_agent(hold_db):
    _db, store = hold_db
    held = await _hold(
        store,
        scope=HoldScope.AGENT,
        target_id=SELF,
        reason="runaway heartbeat loop",
        operation_id="op-hold",
    )
    await _release(store, held, reason="loop fixed", operation_id="op-release")
    feature = IdentityFeature(_Agent(SELF, store))

    result = await feature.inspect_hold_state()

    assert result.status is ToolResultStatus.OK
    assert "No host or agent Hold currently applies" in result.confirmation
    assert "runaway heartbeat loop" in result.confirmation
    assert "sovereign" in result.confirmation
    assert "released" in result.confirmation


@pytest.mark.asyncio
async def test_non_applied_receipts_are_part_of_the_history(hold_db):
    _db, store = hold_db
    held = await _hold(
        store, scope=HoldScope.AGENT, target_id=SELF, reason="r", operation_id="a"
    )
    # Same actor, same reason: already_in_state.
    await _hold(store, scope=HoldScope.AGENT, target_id=SELF, reason="r", operation_id="b")
    # A release naming a receipt that is not the current one: refused_stale.
    await store.release_hold(
        scope=HoldScope.AGENT,
        target_id=SELF,
        actor_id=SOVEREIGN,
        reason="stale",
        operation_id="c",
        expected_hold_receipt_id="not-the-current-receipt",
        authority=HoldAuthority.SOVEREIGN,
    )

    result = await inspect_self_hold(_Agent(SELF, store))

    dispositions = [r["disposition"] for r in result["history"]["receipts"]]
    assert dispositions == ["refused_stale", "already_in_state", "applied"]
    [episode] = result["history"]["episodes"]
    assert episode["status"] == "active"
    assert result["latches"]["agent"]["set_at"] == held.receipt.occurred_at


@pytest.mark.asyncio
async def test_history_is_bounded_newest_first_and_offers_a_cursor(hold_db):
    _db, store = hold_db
    for index in range(4):
        await _hold(
            store,
            scope=HoldScope.AGENT,
            target_id=SELF,
            reason=f"reason-{index}",
            operation_id=f"op-{index}",
        )

    result = await inspect_self_hold(_Agent(SELF, store), history_limit=2)

    reasons = [r["reason"] for r in result["history"]["receipts"]]
    assert reasons == ["reason-3", "reason-2"]
    assert result["history"]["next_cursor"] is not None
    # reason-1 began before the page but ended ON it (replaced by reason-2):
    # its episode is complete here, start included.
    episodes = result["history"]["episodes"]
    assert [(e["reason"], e["status"]) for e in episodes] == [
        ("reason-3", "active"),
        ("reason-2", "replaced"),
        ("reason-1", "replaced"),
    ]

    older = await inspect_self_hold(
        _Agent(SELF, store),
        cursor=result["history"]["next_cursor"],
        history_limit=2,
    )
    assert [r["reason"] for r in older["history"]["receipts"]] == [
        "reason-1",
        "reason-0",
    ]
    assert older["history"]["next_cursor"] is None
    # reason-1's episode was reported with its end on the newer page; only
    # reason-0 (ended by reason-1, on this page) remains.
    assert [(e["reason"], e["status"]) for e in older["history"]["episodes"]] == [
        ("reason-0", "replaced"),
    ]


# ---------------------------------------------------------------------------
# The result channel: history fits the orchestrator cap, and pages cover all
# ---------------------------------------------------------------------------

# The host Hold door's own ceiling on a reason: the largest a real receipt has.
_MAX_REASON = 1024


def _orchestrator_view(envelope):
    """Exactly what the orchestrator measures against its cap."""

    from kestrel_sovereign.features.base import _serialize_tool_result

    return json.dumps(_serialize_tool_result(envelope))


async def _run_tool(agent, **kwargs):
    feature = IdentityFeature(agent)
    [tool] = [t for t in feature.get_tools() if t.name == "inspect_hold_state"]
    return await tool.execute(**kwargs)


def _long_reason(label):
    return (label + ":").ljust(_MAX_REASON, "x")


@pytest.mark.asyncio
async def test_maximum_history_pages_within_the_orchestrator_cap(hold_db):
    from kestrel_sovereign.kestrel_agent import MAX_TOOL_RESULT_CHARS

    _db, store = hold_db
    await _hold(
        store, scope=HoldScope.HOST, reason=_long_reason("host"), operation_id="h"
    )
    expected = [("host", "hold", _long_reason("host"))]
    for index in range(SELF_HOLD_HISTORY_LIMIT):
        held = await _hold(
            store,
            scope=HoldScope.AGENT,
            target_id=SELF,
            reason=_long_reason(f"hold-{index}"),
            operation_id=f"hold-{index}",
        )
        await _release(
            store,
            held,
            reason=_long_reason(f"release-{index}"),
            operation_id=f"release-{index}",
        )
        expected += [
            ("agent", "hold", _long_reason(f"hold-{index}")),
            ("agent", "release", _long_reason(f"release-{index}")),
        ]
    await _hold(
        store,
        scope=HoldScope.AGENT,
        target_id=SELF,
        reason=_long_reason("current"),
        operation_id="current",
    )
    expected.append(("agent", "hold", _long_reason("current")))
    agent = _Agent(SELF, store)

    served, episodes, cursor, pages = [], [], None, 0
    while True:
        envelope = await _run_tool(agent, **({"cursor": cursor} if cursor else {}))
        pages += 1
        assert len(_orchestrator_view(envelope)) <= MAX_TOOL_RESULT_CHARS, (
            f"page {pages} would be replaced by the orchestrator's preview"
        )
        assert envelope["status"] == "ok", envelope
        history = envelope["data"]["history"]
        assert history["status"] == "read"
        served += [
            (r["scope"], r["action"], r["reason"]) for r in history["receipts"]
        ]
        episodes += [(e["reason"], e["status"]) for e in history["episodes"]]
        cursor = history["next_cursor"]
        if cursor is None:
            break
        assert pages < 200, "paging must make forward progress"

    assert pages > 1
    # Every receipt exactly once, newest first by commit order.
    assert served == list(reversed(expected))
    # Every hold exactly once as an episode, whichever page it spanned.
    assert sorted(episodes) == sorted(
        [(_long_reason("host"), "active"), (_long_reason("current"), "active")]
        + [
            (_long_reason(f"hold-{index}"), "released")
            for index in range(SELF_HOLD_HISTORY_LIMIT)
        ]
    )


@pytest.mark.asyncio
async def test_ordinary_25_hold_history_is_served_whole_not_cut(hold_db):
    from kestrel_sovereign.kestrel_agent import MAX_TOOL_RESULT_CHARS

    _db, store = hold_db
    for index in range(SELF_HOLD_HISTORY_LIMIT):
        held = await _hold(
            store,
            scope=HoldScope.AGENT,
            target_id=SELF,
            reason=f"maintenance window {index}",
            operation_id=f"hold-{index}",
        )
        await _release(
            store, held, reason=f"window {index} over", operation_id=f"rel-{index}"
        )

    envelope = await _run_tool(_Agent(SELF, store))

    assert len(_orchestrator_view(envelope)) <= MAX_TOOL_RESULT_CHARS
    history = envelope["data"]["history"]
    assert history["next_cursor"] is not None, "the rest must stay reachable"
    assert history["receipts"], "a page that fits must carry history"


@pytest.mark.asyncio
async def test_an_entry_too_large_for_the_channel_is_withheld_not_cut(
    hold_db, monkeypatch
):
    from kestrel_sovereign import kestrel_agent

    monkeypatch.setattr(kestrel_agent, "MAX_TOOL_RESULT_CHARS", 1000)
    _db, store = hold_db
    held = await _hold(
        store,
        scope=HoldScope.AGENT,
        target_id=SELF,
        reason=_long_reason("oversize"),
        operation_id="big",
    )
    await _release(store, held, reason="done", operation_id="big-release")
    agent = _Agent(SELF, store)

    envelope = await _run_tool(agent)

    assert len(_orchestrator_view(envelope)) <= 1000
    assert envelope["status"] == "partial"
    assert "withheld" in envelope["error"]
    history = envelope["data"]["history"]
    assert history["status"] == "withheld_oversize"
    assert history["withheld_receipts"] == 1
    # Forward progress: the withheld receipt is passed, not re-served forever.
    cursor = history["next_cursor"]
    assert cursor is not None
    older = await _run_tool(agent, cursor=cursor)
    assert older["data"]["history"]["next_cursor"] is None


@pytest.mark.parametrize(
    "cursor",
    [
        "garbage",
        "1",
        "1:2",
        "1:2:3:4",
        "0:new:new",
        "-1:new:new",
        "new:1.5:new",
        "new:new:" + "9" * 20,
    ],
)
@pytest.mark.asyncio
async def test_a_cursor_this_tool_did_not_issue_is_refused(hold_db, cursor):
    _db, store = hold_db

    envelope = await _run_tool(_Agent(SELF, store), cursor=cursor)

    assert envelope["status"] == "error"
    assert "cursor" in envelope["error"]


@pytest.mark.asyncio
async def test_a_forged_cursor_cannot_reach_another_agents_history(hold_db):
    _db, store = hold_db
    await _hold(
        store, scope=HoldScope.AGENT, target_id=SELF, reason="mine", operation_id="s"
    )
    await _hold(
        store,
        scope=HoldScope.AGENT,
        target_id=PEER,
        reason="peer-only secret reason",
        operation_id="p",
    )

    # A cursor positioned above every receipt in the store.
    result = await inspect_self_hold(_Agent(SELF, store), cursor="new:999999:end")

    assert [r["reason"] for r in result["history"]["receipts"]] == ["mine"]
    assert "peer-only secret reason" not in json.dumps(result)


# ---------------------------------------------------------------------------
# Binding: the runtime chooses the subject, never a payload
# ---------------------------------------------------------------------------


def test_tool_takes_no_subject_parameter():
    parameters = inspect.signature(IdentityFeature.inspect_hold_state).parameters
    # A history cursor only; the subject is never a parameter.
    assert list(parameters) == ["self", "cursor"]


@pytest.mark.asyncio
async def test_a_peer_hold_is_never_visible_to_the_subject(hold_db):
    _db, store = hold_db
    await _hold(
        store,
        scope=HoldScope.AGENT,
        target_id=PEER,
        reason="peer-only secret reason",
        operation_id="peer-op",
    )

    result = await inspect_self_hold(_Agent(SELF, store))

    assert result["state"] == "not_held"
    assert result["latches"] == {"host": None, "agent": None, "mandate": []}
    assert result["history"]["receipts"] == []
    assert result["history"]["episodes"] == []
    assert "peer-only secret reason" not in json.dumps(result)


@pytest.mark.asyncio
async def test_the_peer_reads_only_its_own_hold(hold_db):
    _db, store = hold_db
    await _hold(
        store, scope=HoldScope.AGENT, target_id=PEER, reason="peer", operation_id="p"
    )
    await _hold(
        store, scope=HoldScope.AGENT, target_id=SELF, reason="self", operation_id="s"
    )

    peer = await inspect_self_hold(_Agent(PEER, store))

    assert [r["reason"] for r in peer["history"]["receipts"]] == ["peer"]
    assert peer["latches"]["agent"]["reason"] == "peer"


@pytest.mark.asyncio
async def test_a_foreign_receipt_from_the_store_fails_closed(hold_db):
    _db, store = hold_db
    await _hold(
        store, scope=HoldScope.AGENT, target_id=PEER, reason="peer", operation_id="p"
    )

    class _UnfilteredStore:
        """A store that ignores the subject filter it was given."""

        async def get_effective(self, agent_id):
            return await store.get_effective(agent_id)

        async def get_receipt_by_id(self, receipt_id):
            return await store.get_receipt_by_id(receipt_id)

        async def list_receipts(
            self, *, scope, after, limit, newest_first, target_id=None, subject_id=None
        ):
            return await store.list_receipts(
                scope=scope, after=after, limit=limit, newest_first=newest_first
            )

    result = await inspect_self_hold(_Agent(SELF, _UnfilteredStore()))

    assert result["state"] == "not_held"
    assert result["history"]["status"] == "unknown"
    assert result["history"]["cause_type"] == "HoldCorruptStateError"


# ---------------------------------------------------------------------------
# Host and agent latches remain independently visible
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_host_and_agent_latches_are_reported_independently(hold_db):
    _db, store = hold_db
    host = await _hold(store, scope=HoldScope.HOST, reason="fleet pause", operation_id="h")
    await _hold(
        store, scope=HoldScope.AGENT, target_id=SELF, reason="own pause", operation_id="a"
    )

    both = await inspect_self_hold(_Agent(SELF, store))
    assert both["state"] == "held"
    assert both["sources"] == ["host", "agent"]
    assert both["latches"]["host"]["reason"] == "fleet pause"
    assert both["latches"]["agent"]["reason"] == "own pause"

    await _release(store, host, reason="fleet resumed", operation_id="h-release")
    after = await inspect_self_hold(_Agent(SELF, store))

    assert after["state"] == "held", "releasing the host hold is not releasing mine"
    assert after["sources"] == ["agent"]
    assert after["latches"]["host"] is None
    assert after["latches"]["agent"]["reason"] == "own pause"
    by_scope = {episode["scope"]: episode["status"] for episode in after["history"]["episodes"]}
    assert by_scope == {"host": "released", "agent": "active"}


# ---------------------------------------------------------------------------
# Failures are unknown, never not_held
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unbound_store_is_unknown_not_unheld():
    result = await inspect_self_hold(_Agent(SELF, None))

    assert result["state"] == "unknown"
    assert result["held"] is None
    assert result["failure"] == "store_unbound"
    assert result["cause_type"] == "SelfHoldStateUnavailable"


@pytest.mark.asyncio
async def test_effective_read_failure_is_unknown_not_unheld():
    class _BrokenStore:
        async def list_receipts(self, **_kwargs):
            raise HoldStateError("Hold receipt history could not be read")

        async def get_effective(self, agent_id):
            raise RuntimeError("backend down")

    result = await inspect_self_hold(_Agent(SELF, _BrokenStore()))

    assert result["state"] == "unknown"
    assert result["held"] is None
    assert result["failure"] == "read_failed"
    assert result["cause_type"] == "HoldEnforcementUnavailableError"
    assert result["underlying_cause_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_tool_reports_an_unreadable_state_as_a_failure():
    class _CorruptStore:
        async def list_receipts(self, **_kwargs):
            raise HoldCorruptStateError("witness missing")

        async def get_effective(self, agent_id):
            raise HoldCorruptStateError("witness missing")

    result = await IdentityFeature(_Agent(SELF, _CorruptStore())).inspect_hold_state()

    assert result.status is ToolResultStatus.ERROR
    assert "unknown" in result.error
    assert result.data["state"] == "unknown"


@pytest.mark.asyncio
async def test_missing_subject_identity_is_unknown(hold_db):
    _db, store = hold_db

    result = await inspect_self_hold(_Agent("", store))

    assert result["state"] == "unknown"
    assert result["cause_type"] == "HoldEnforcementUnavailableError"


@pytest.mark.asyncio
async def test_history_failure_keeps_current_state_and_marks_history_unknown(hold_db):
    _db, store = hold_db
    await _hold(
        store, scope=HoldScope.AGENT, target_id=SELF, reason="r", operation_id="a"
    )

    class _HistoryDownStore:
        async def get_effective(self, agent_id):
            return await store.get_effective(agent_id)

        async def get_receipt_by_id(self, receipt_id):
            return await store.get_receipt_by_id(receipt_id)

        async def list_receipts(self, **_kwargs):
            raise HoldStateError("Hold receipt history could not be read")

    agent = _Agent(SELF, _HistoryDownStore())
    result = await inspect_self_hold(agent)
    assert result["state"] == "held"
    assert result["history"] == {"status": "unknown", "cause_type": "HoldStateError"}

    tool = await IdentityFeature(agent).inspect_hold_state()
    assert tool.status is ToolResultStatus.PARTIAL


# ---------------------------------------------------------------------------
# Read-only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_introspection_never_mutates(hold_db):
    _db, store = hold_db
    await _hold(
        store, scope=HoldScope.AGENT, target_id=SELF, reason="r", operation_id="a"
    )

    class _ReadOnlyStore:
        def __getattr__(self, name):
            if name in {"set_hold", "release_hold"}:
                raise AssertionError(f"introspection reached {name}")
            return getattr(store, name)

    result = await inspect_self_hold(_Agent(SELF, _ReadOnlyStore()))
    assert result["state"] == "held"
    assert (await store.get_effective(SELF)).agent is not None


# ---------------------------------------------------------------------------
# Role from provenance, and redaction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_role_is_the_recorded_authority_not_the_actor_string(hold_db):
    _db, store = hold_db
    await _hold(
        store, scope=HoldScope.AGENT, target_id=SELF, reason="r", operation_id="a"
    )

    result = await inspect_self_hold(_Agent(SELF, store))

    assert result["latches"]["agent"]["actor_role"] == "sovereign"
    assert result["history"]["receipts"][0]["actor_role"] == "sovereign"


@pytest.mark.asyncio
async def test_actor_equal_to_subject_is_self(hold_db):
    _db, store = hold_db
    await _hold(
        store,
        scope=HoldScope.AGENT,
        target_id=SELF,
        reason="r",
        operation_id="a",
        actor=SELF,
    )

    result = await inspect_self_hold(_Agent(SELF, store))

    assert result["latches"]["agent"]["actor_role"] == "self"
    assert result["history"]["episodes"][0]["set_by_role"] == "self"


def test_actor_role_reads_only_subject_and_authority():
    assert (
        introspection_module.actor_role(
            actor_id="did:sovereign:operator",
            authority=HoldAuthority.SOVEREIGN,
            subject_did=SELF,
        )
        == "sovereign"
    )
    assert (
        introspection_module.actor_role(
            actor_id=SELF, authority=HoldAuthority.SOVEREIGN, subject_did=SELF
        )
        == "self"
    )


@pytest.mark.asyncio
async def test_redaction_omits_every_identity_but_roles(hold_db):
    _db, store = hold_db
    held = await _hold(
        store,
        scope=HoldScope.AGENT,
        target_id=SELF,
        reason="visible reason",
        operation_id="operation-secret",
    )
    await _hold(store, scope=HoldScope.HOST, reason="host reason", operation_id="host-op")

    result = await inspect_self_hold(_Agent(SELF, store))
    rendered = json.dumps(result)

    assert result["redaction_policy"] == SELF_HOLD_REDACTION_POLICY
    assert result["redaction_policy"]["actor_identity"] == "role_only"
    for secret in (SOVEREIGN, SELF, "operation-secret", "host-op", held.receipt.receipt_id):
        assert secret not in rendered
    assert "visible reason" in rendered


# ---------------------------------------------------------------------------
# The recorded authority column
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_receipts_record_their_authority(hold_db):
    db, store = hold_db
    mutation = await _hold(
        store, scope=HoldScope.AGENT, target_id=SELF, reason="r", operation_id="a"
    )

    row = await db.fetchone(
        "SELECT authority FROM hold_receipts WHERE receipt_id = ?",
        (mutation.receipt.receipt_id,),
    )
    assert row[0] == "sovereign"
    assert mutation.receipt.authority is HoldAuthority.SOVEREIGN
    assert hold_receipt_payload(mutation.receipt)["authority"] == "sovereign"


@pytest.mark.asyncio
async def test_a_mutation_without_a_recorded_authority_is_refused(hold_db):
    _db, store = hold_db

    with pytest.raises(TypeError, match="authority"):
        await store.set_hold(
            scope=HoldScope.AGENT,
            target_id=SELF,
            actor_id=SOVEREIGN,
            reason="r",
            operation_id="a",
            authority="sovereign",
        )


@pytest.mark.asyncio
async def test_legacy_receipts_backfill_to_sovereign_with_witnesses_intact(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "host.db"))
    try:
        store = HoldStore(db)
        await store.ensure_schema()
        await _hold(
            store, scope=HoldScope.AGENT, target_id=SELF, reason="r", operation_id="a"
        )
        # Rewind to what an older binary left: no ``authority`` column, no
        # v2 anchor-format marker, and a v1 history anchor and custody
        # marker, with every row and witness preserved exactly.
        await rewind_to_v1_history_anchor(db, store)

        upgraded = HoldStore(db)
        await upgraded.ensure_schema()

        assert await db.column_exists("hold_receipts", "authority")
        result = await inspect_self_hold(_Agent(SELF, upgraded))
        assert result["state"] == "held"
        assert result["latches"]["agent"]["actor_role"] == "sovereign"
        assert result["history"]["receipts"][0]["actor_role"] == "sovereign"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_rewritten_authority_fails_closed(hold_db):
    db, store = hold_db
    await _hold(
        store, scope=HoldScope.AGENT, target_id=SELF, reason="r", operation_id="a"
    )
    await db.execute("UPDATE hold_receipts SET authority = 'admin-agent'")

    result = await inspect_self_hold(_Agent(SELF, store))

    assert result["state"] == "unknown"
    assert result["cause_type"] == "HoldCorruptStateError"


def test_v1_digest_is_unchanged_for_the_sovereign_authority():
    v1 = (
        "receipt-id",
        "operation",
        "hold",
        "applied",
        "agent",
        SELF,
        "reason",
        SOVEREIGN,
        "2026-09-25T00:00:00+00:00",
        "",
        "",
        "receipt-id",
    )
    assert _receipt_content_digest(v1 + ("sovereign",)) == _receipt_content_digest(v1)
    assert _receipt_from_row(v1).authority is HoldAuthority.SOVEREIGN
    with pytest.raises(HoldCorruptStateError):
        _receipt_from_row(v1 + ("mandate",))


# ---------------------------------------------------------------------------
# Census: every door that mutates Hold records its authority
# ---------------------------------------------------------------------------


def _hold_mutation_calls():
    calls = []
    for path in sorted((REPO_ROOT / "kestrel_sovereign").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"set_hold", "release_hold"}
            ):
                calls.append((path.relative_to(REPO_ROOT).as_posix(), node))
    return calls


def test_every_hold_mutation_is_a_known_door_recording_its_authority():
    """Exactly two doors mutate Hold, each recording the authority it acted under.

    The sovereign host door (#3159) and the mandate descendant door (#3168).
    Every existing receipt is backfilled as ``sovereign`` because the host door
    was once the only writer; the mandate door records ``mandate`` explicitly.
    A third door fails here until it is named and records its own authority.
    """

    calls = _hold_mutation_calls()
    by_door: dict[str, list[tuple[str, str]]] = {}
    for path, node in calls:
        [authority] = [kw.value for kw in node.keywords if kw.arg == "authority"]
        by_door.setdefault(path, []).append(
            (node.func.attr, ast.unparse(authority))
        )
    assert {door: sorted(entries) for door, entries in by_door.items()} == {
        "kestrel_sovereign/endpoints/hold.py": [
            ("release_hold", "HoldAuthority.SOVEREIGN"),
            ("set_hold", "HoldAuthority.SOVEREIGN"),
        ],
        "kestrel_sovereign/hold/mandate.py": [
            ("release_hold", "HoldAuthority.MANDATE"),
            ("set_hold", "HoldAuthority.MANDATE"),
        ],
    }
