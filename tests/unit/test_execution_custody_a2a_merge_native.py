"""Native A2A receipts remain terminal across the new #3440 consumers."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.features.peers import feature as peers
from kestrel_sovereign.storage.async_pending_a2a_question_store import (
    PendingA2AQuestionStore,
)
from tests.unit.test_execution_custody_review29 import native_owner, retire
from tests.unit.test_execution_custody_review48_native import terminal
from tests.unit.test_execution_custody_review49_native import KINDS, join


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("route", [
    "claim_check", "renew", "retire", "retire_late", "read_receiver",
    "read_receiver_typed", "settlement_timeout_receipt",
    "expire_claim", "receiver_row", "startup_reclaim", "sweep_reclaim",
    "sweep_list", "sweep_settlement", "recovery_row", "boot_recovery",
])
async def test_actual_a2a_consumer_preserves_original_native_receipt(
    tmp_path, monkeypatch, kind, route,
):
    agent, db = await native_owner(tmp_path)
    store = PendingA2AQuestionStore(db, agent.did)
    agent.pending_a2a_questions = store
    feature = peers.PeersFeature(agent)
    agent.features = {"Peers": feature}
    original = terminal(kind)
    if route in {"read_receiver_typed", "settlement_timeout_receipt"}:
        wrapper = (
            peers.PeerDirectoryError("native receiver failure")
            if route == "read_receiver_typed" else TimeoutError("native receipt")
        )
        wrapper.__cause__ = original
        original = wrapper
    entered = asyncio.Event()
    owner = None

    async def effect(*args, **kwargs):
        await db.execute("INSERT INTO effects VALUES (1)")
        entered.set()
        raise original

    async def empty(*args, **kwargs):
        return []

    async def expired_settlement(*args, **kwargs):
        return peers.ReceiverSettlement(peers.ReceiverSettlementOutcome.UNCONFIRMED)

    row = SimpleNamespace(
        task_id="question", recipient="Peer", recipient_agent_id="did:test:peer",
        receiver_settlement="PENDING",
    )
    claim = peers._QuestionClaim(store, "native-claim")
    try:
        await store.insert(
            task_id="question", recipient="Peer", original_question="question",
            origin_turn_id=None, origin_session_id="session",
            deadline=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        if route == "claim_check":
            monkeypatch.setattr(store, "renew_claim_lease", effect)
            operation = feature._holds_question_claim("question", claim)
        elif route == "renew":
            async def delivered():
                return "delivered"
            monkeypatch.setattr(store, "renew_claim_lease", effect)
            handle = feature._renewing_claim_lease(
                SimpleNamespace(wait=delivered), "question", claim,
            )
            operation = handle.wait()
        elif route in {"retire", "retire_late"}:
            monkeypatch.setattr(store, "confirm_wake_delivered", effect)
            monkeypatch.setattr(store, "confirm_late_answer_delivered", effect)
            operation = feature._retire_delivered_wake(
                "question", claim, late=route == "retire_late",
            )
        elif route.startswith("read_receiver"):
            monkeypatch.setattr(feature, "_question_recipient_agent_id", effect)
            operation = feature._read_settled_peer_task(
                task_id="question", recipient="Peer", recipient_agent_id=None,
            )
        elif route == "settlement_timeout_receipt":
            monkeypatch.setattr(feature, "_settle_question_with_receiver_unbounded", effect)
            operation = feature._settle_question_with_receiver(task_id="question", recipient="Peer")
        elif route == "expire_claim":
            monkeypatch.setattr(feature, "_settle_question_with_receiver", expired_settlement)
            monkeypatch.setattr(store, "mark_expired", effect)
            operation = feature._expire_pending_question(
                store, task_id="question", recipient="Peer",
                original_question="question", sess_id="session", causation_chain=None,
            )
        elif route == "receiver_row":
            async def rows(*args, **kwargs):
                return [row]
            monkeypatch.setattr(store, "list_expired_awaiting_receiver", rows)
            monkeypatch.setattr(feature, "_settle_expired_row_with_receiver", effect)
            operation = feature._settle_expired_questions_with_receivers(store)
        elif route == "startup_reclaim":
            monkeypatch.setattr(store, "reclaim_abandoned_wakes", effect)
            operation = feature._replay_pending_a2a_questions(store)
        elif route.startswith("sweep_"):
            feature.EXPIRY_SWEEP_INTERVAL_SECONDS = 0.001
            monkeypatch.setattr(store, "reclaim_abandoned_wakes", empty)
            monkeypatch.setattr(store, "list_waiting_past_deadline", empty)
            monkeypatch.setattr(store, "list_expired_awaiting_receiver", empty)
            if route == "sweep_reclaim":
                monkeypatch.setattr(store, "reclaim_abandoned_wakes", effect)
            elif route == "sweep_list":
                monkeypatch.setattr(store, "list_waiting_past_deadline", effect)
            else:
                monkeypatch.setattr(store, "list_expired_awaiting_receiver", effect)
            operation = feature._hourly_expiry_sweep_loop(store)
        elif route == "recovery_row":
            async def rows(*args, **kwargs):
                return [row]
            monkeypatch.setattr(store, "reclaim_abandoned_wakes", rows)
            monkeypatch.setattr(feature, "_handle_retry_payload_row", effect)
            operation = feature._redeliver_abandoned_wakes(store)
        else:
            from kestrel_sovereign.storage import async_pending_a2a_question_store
            monkeypatch.setattr(async_pending_a2a_question_store, "QUESTION_CLAIM_LEASE", timedelta(milliseconds=1))
            monkeypatch.setattr(store, "reclaim_abandoned_wakes", effect)
            operation = feature._redeliver_abandoned_wakes_after_boot(store)
        owner = create_owned_task(operation)
        await asyncio.wait_for(entered.wait(), 2)
        done, _ = await asyncio.wait({owner}, timeout=0.1)
        assert done, "native control became a retry/sweep or a false success"
        outcome = await join(owner)
        assert outcome.error is original, "original native receipt was normalized"
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    finally:
        if owner is not None:
            if not owner.done():
                owner.cancel()
            await join(owner)
        await retire(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("route", ["renewing_wait", "released_claim_wait", "settlement_deadline", "delivered_checkpoint"])
async def test_original_a2a_child_physically_joins_after_parent_cancellation(
    tmp_path, monkeypatch, kind, route,
):
    agent, db = await native_owner(tmp_path)
    feature = peers.PeersFeature(agent)
    agent.features = {"Peers": feature}
    original = terminal(kind)
    entered, interrupted, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    owner = None

    async def effect(*args, **kwargs):
        await db.execute("INSERT INTO effects VALUES (1)")
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            interrupted.set()
            await await_owned_task(create_owned_task(release.wait()))
        await db.execute("INSERT INTO effects VALUES (2)")
        raise original

    try:
        if route in {"renewing_wait", "released_claim_wait"}:
            async def renewed():
                return route != "released_claim_wait"
            handle = peers._ClaimLeaseRenewingHandle(
                SimpleNamespace(wait=effect), interval=0.001, renew=renewed,
            )
            operation = handle.wait()
        elif route == "settlement_deadline":
            feature.QUESTION_SETTLEMENT_TIMEOUT_SECONDS = 0.25
            monkeypatch.setattr(feature, "_settle_question_with_receiver_unbounded", effect)
            operation = feature._settle_question_with_receiver(task_id="question", recipient="Peer")
        else:
            # Use the actual public delivery path and its native row retirement,
            # not the private checkpoint helper introduced by this fix.
            from tests.unit.test_send_a2a_question_supervisor import (
                _DeliveredSignalHandle,
            )
            async def queued(*args, **kwargs):
                return _DeliveredSignalHandle()
            store = PendingA2AQuestionStore(db, agent.did)
            monkeypatch.setattr(store, "confirm_wake_delivered", effect)
            agent.dispatcher = SimpleNamespace(enqueue_signal=queued)
            claim = peers._QuestionClaim(store, "original-claim")
            async def holds(*args, **kwargs):
                return True
            monkeypatch.setattr(store, "renew_claim_lease", holds)
            operation = feature._fire_question_answered_signal(
                task_id="question", recipient="Peer", original_question="question",
                sess_id="session", state="completed", reply_text="reply",
                causation_chain=None, claim=claim,
            )
        owner = create_owned_task(operation)
        await asyncio.wait_for(entered.wait(), 2)
        if route == "settlement_deadline":
            await asyncio.wait_for(interrupted.wait(), 2)
        else:
            owner.cancel()
            await asyncio.sleep(0)
            owner.cancel()
        await asyncio.sleep(0.01)
        assert not owner.done(), "original native child escaped its physical owner"
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        release.set()
        outcome = await join(owner)
        assert outcome.error is original, "late native receipt lost to caller cancellation/deadline"
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 2
    finally:
        release.set()
        if owner is not None:
            await join(owner)
        await retire(agent)
