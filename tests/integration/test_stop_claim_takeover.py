"""A Stop claim carries its owner's lease; only a dead owner's is taken (#3356).

These run on both backends. They live in the integration tier because only
that CI job has a PostgreSQL service (#3336): under ``tests/unit`` every
``[postgres]`` leg would skip in CI.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from kestrel_sovereign.stop import (
    CancellationAuthority,
    CooperativeStopTarget,
    StopCleanupRegistry,
    StopDisposition,
    StopDoor,
    StopOutcome,
    StopReceiptConflict,
    StopReceiptStore,
    StopRequest,
    StopScope,
)
from kestrel_sovereign.stop.receipt import StopOperationClaim, opaque_stop_identifier
from kestrel_sovereign.storage.async_database import AsyncDatabase

AGENT_DID = "did:test:claim-agent"
# Older than any lease: an owner whose heartbeat reads this is proven dead.
_DEAD_HEARTBEAT = "2000-01-01T00:00:00.000+00:00"


def _request(correlation_id: str | None = None) -> StopRequest:
    return StopRequest(
        StopScope.AGENT,
        "did:test:operator",
        AGENT_DID,
        reason="runaway loop",
        correlation_id=correlation_id or f"claim-takeover-{uuid4()}",
    )


def _outcomes(request: StopRequest, disposition: StopDisposition):
    return (
        StopOutcome(
            scope=request.scope,
            requested_target=request.target,
            resolved_target="agent-runtime",
            agent_id=AGENT_DID,
            disposition=disposition,
            correlation_id=request.correlation_id,
        ),
    )


def _operation_id(request: StopRequest) -> str:
    return opaque_stop_identifier("operation", request.correlation_id)


async def _stores(db_backend, count: int, **kwargs) -> list[StopReceiptStore]:
    db = AsyncDatabase(db_backend)
    stores = [StopReceiptStore(db, **kwargs) for _ in range(count)]
    await stores[0].ensure_schema()
    return stores


async def _kill_owner(db: AsyncDatabase, request: StopRequest) -> None:
    await db.execute(
        "UPDATE stop_operation_claims SET heartbeat_at = ? WHERE operation_id = ?",
        (_DEAD_HEARTBEAT, _operation_id(request)),
    )


async def _claim_row(db: AsyncDatabase, request: StopRequest):
    return await db.fetchone(
        "SELECT claim_id, owner_id, heartbeat_at FROM stop_operation_claims "
        "WHERE operation_id = ?",
        (_operation_id(request),),
    )


async def _receipt_count(db: AsyncDatabase, request: StopRequest) -> int:
    return int(
        await db.fetchval(
            "SELECT COUNT(*) FROM stop_receipts WHERE operation_id = ?",
            (_operation_id(request),),
        )
    )


class _Work:
    """One agent's live work: cooperative cancel is idempotent."""

    def __init__(self) -> None:
        self.live = True
        self.effects = 0
        self.calls = 0

    async def cancel(self, _request: StopRequest) -> StopDisposition:
        self.calls += 1
        if not self.live:
            return StopDisposition.ALREADY_COMPLETE
        self.live = False
        self.effects += 1
        return StopDisposition.STOPPED


def _authority(store: StopReceiptStore, cancel) -> CancellationAuthority:
    return CancellationAuthority(
        lambda: (CooperativeStopTarget("agent-runtime", AGENT_DID, cancel),),
        cleanup_registry=StopCleanupRegistry(),
        receipt_store=store,
        door=StopDoor.AGENT,
    )


# ---------------------------------------------------------------------------
# Store: liveness, takeover, fencing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_claim_records_its_owner_and_heartbeat(db_backend):
    [owner] = await _stores(db_backend, 1)
    request = _request()

    claim = await owner.claim(request)

    assert isinstance(claim, StopOperationClaim)
    assert claim.taken_over is False
    claim_id, owner_id, heartbeat_at = await _claim_row(owner._db, request)
    assert (claim_id, owner_id) == (claim.claim_id, owner.owner_id)
    assert heartbeat_at


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_live_owners_claim_is_never_taken_over(db_backend):
    """Renewal keeps a claim live well past one lease; retries refuse."""

    owner, retrier = await _stores(db_backend, 2, claim_lease_seconds=0.4)
    request = _request()
    claim = await owner.claim(request)

    assert await retrier.claim(request) is None
    async with owner.hold_claim(claim):
        await asyncio.sleep(1.2)  # three leases, renewed throughout
        assert await retrier.claim(request) is None

    claim_id, owner_id, _ = await _claim_row(owner._db, request)
    assert (claim_id, owner_id) == (claim.claim_id, owner.owner_id)
    receipt = await owner.persist(
        request,
        _outcomes(request, StopDisposition.STOPPED),
        door=StopDoor.AGENT,
        claim_id=claim.claim_id,
    )
    assert receipt.outcomes[0].disposition is StopDisposition.STOPPED


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_an_expired_lease_is_taken_over_by_the_database_clock(db_backend):
    owner, retrier = await _stores(db_backend, 2, claim_lease_seconds=0.3)
    request = _request()
    dead = await owner.claim(request)

    # The owner stops heartbeating; nothing else touches the row.
    await asyncio.sleep(0.8)
    taken = await retrier.claim(request)

    assert isinstance(taken, StopOperationClaim)
    assert taken.taken_over is True
    assert taken.claim_id != dead.claim_id
    claim_id, owner_id, _ = await _claim_row(owner._db, request)
    assert (claim_id, owner_id) == (taken.claim_id, retrier.owner_id)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_renewal_cannot_revive_an_expired_claim(db_backend):
    owner, retrier = await _stores(db_backend, 2)
    request = _request()
    claim = await owner.claim(request)
    assert await owner.renew_claim(claim) is True

    await _kill_owner(owner._db, request)

    assert await owner.renew_claim(claim) is False
    assert (await _claim_row(owner._db, request))[2] == _DEAD_HEARTBEAT
    assert isinstance(await retrier.claim(request), StopOperationClaim)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_resumed_stale_owner_cannot_write_a_receipt(db_backend):
    owner, retrier = await _stores(db_backend, 2)
    request = _request()
    stale = await owner.claim(request)
    await _kill_owner(owner._db, request)
    taken = await retrier.claim(request)
    assert isinstance(taken, StopOperationClaim)

    with pytest.raises(StopReceiptConflict):
        await owner.persist(
            request,
            _outcomes(request, StopDisposition.STOPPED),
            door=StopDoor.AGENT,
            claim_id=stale.claim_id,
        )
    assert await owner.renew_claim(stale) is False
    assert await _receipt_count(owner._db, request) == 0

    receipt = await retrier.persist(
        request,
        _outcomes(request, StopDisposition.ALREADY_COMPLETE),
        door=StopDoor.AGENT,
        claim_id=taken.claim_id,
    )
    # After the new owner's receipt commits, the stale owner only replays it.
    replay = await owner.persist(
        request,
        _outcomes(request, StopDisposition.STOPPED),
        door=StopDoor.AGENT,
        claim_id=stale.claim_id,
    )
    assert replay == receipt
    assert await _receipt_count(owner._db, request) == 1


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_concurrent_retries_take_over_an_expired_claim_exactly_once(
    db_backend,
):
    owner, *retriers = await _stores(db_backend, 5)
    request = _request()
    await owner.claim(request)
    await _kill_owner(owner._db, request)

    results = await asyncio.gather(*(r.claim(request) for r in retriers))

    winners = [r for r in results if isinstance(r, StopOperationClaim)]
    assert len(winners) == 1
    assert results.count(None) == len(retriers) - 1
    claim_id, owner_id, _ = await _claim_row(owner._db, request)
    assert claim_id == winners[0].claim_id
    assert owner_id == retriers[results.index(winners[0])].owner_id


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_pre_liveness_claim_is_live_until_its_grace_ends(db_backend):
    """A pre-#3356 row has no heartbeat to read.

    During a rolling upgrade its writer may still be running its Stop, so a
    recent heartbeat-less claim is refused as in progress; once it is older
    than any Stop that code could still be running, it is retakable.
    """

    [retrier] = await _stores(db_backend, 1)
    request = _request()
    claim = await retrier.claim(request)
    await retrier._db.execute(
        "UPDATE stop_operation_claims SET owner_id = NULL, heartbeat_at = NULL "
        "WHERE operation_id = ?",
        (_operation_id(request),),
    )

    assert await retrier.claim(request) is None

    await retrier._db.execute(
        "UPDATE stop_operation_claims "
        "SET claimed_at = '2000-01-01T00:00:00.000000+00:00' "
        "WHERE operation_id = ?",
        (_operation_id(request),),
    )
    taken = await retrier.claim(request)

    assert isinstance(taken, StopOperationClaim)
    assert taken.taken_over is True
    assert taken.claim_id != claim.claim_id


@pytest.mark.asyncio
async def test_legacy_claim_table_gains_owner_columns_and_its_rows_are_retakable(
    tmp_path,
):
    db = await AsyncDatabase.sqlite(str(tmp_path / "legacy-claims.db"))
    try:
        await db.execute(
            "CREATE TABLE stop_operation_claims ("
            "operation_id TEXT NOT NULL PRIMARY KEY, "
            "request_fingerprint TEXT NOT NULL, "
            "claim_id TEXT NOT NULL UNIQUE, "
            "claimed_at TEXT NOT NULL)"
        )
        writer, retrier = StopReceiptStore(db), StopReceiptStore(db)
        await writer.ensure_schema()
        request = _request()
        # Write a claim, then strip it back to the pre-liveness shape.
        claim = await writer.claim(request)
        await db.execute(
            "UPDATE stop_operation_claims SET owner_id = NULL, heartbeat_at = NULL, "
            "claimed_at = '2000-01-01T00:00:00.000000+00:00'"
        )
        columns = {
            row[1] for row in await db.fetchall(
                "PRAGMA table_info(stop_operation_claims)"
            )
        }
        assert {"owner_id", "heartbeat_at"} <= columns

        taken = await retrier.claim(request)
        assert isinstance(taken, StopOperationClaim)
        assert taken.claim_id != claim.claim_id
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Authority: every door re-executes a dead owner's Stop exactly once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_crash_after_claim_before_cancel_retry_stops_the_work(db_backend):
    dead_owner, retrier = await _stores(db_backend, 2)
    request = _request()
    work = _Work()
    assert isinstance(await dead_owner.claim(request), StopOperationClaim)
    await _kill_owner(dead_owner._db, request)

    outcomes = await _authority(retrier, work.cancel).stop(request)

    assert [o.disposition for o in outcomes] == [StopDisposition.STOPPED]
    assert outcomes[0].receipt_id is not None
    assert work.effects == 1
    assert await _receipt_count(retrier._db, request) == 1
    assert await _claim_row(retrier._db, request) is None
    # Any later retry replays that one receipt.
    assert await _authority(retrier, work.cancel).stop(request) == outcomes
    assert work.calls == 1


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_crash_after_cancel_before_persist_retry_records_already_complete(
    db_backend,
):
    dead_owner, retrier = await _stores(db_backend, 2)
    request = _request()
    work = _Work()
    assert isinstance(await dead_owner.claim(request), StopOperationClaim)
    assert await work.cancel(request) is StopDisposition.STOPPED  # then it dies
    await _kill_owner(dead_owner._db, request)

    outcomes = await _authority(retrier, work.cancel).stop(request)

    assert [o.disposition for o in outcomes] == [
        StopDisposition.ALREADY_COMPLETE
    ]
    assert work.effects == 1
    assert await _receipt_count(retrier._db, request) == 1


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_retry_during_a_live_owners_long_stop_is_refused(db_backend):
    owner, retrier = await _stores(db_backend, 2, claim_lease_seconds=0.4)
    request = _request()
    work = _Work()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_cancel(stop_request):
        entered.set()
        await release.wait()
        return await work.cancel(stop_request)

    authority = CancellationAuthority(
        lambda: (CooperativeStopTarget("agent-runtime", AGENT_DID, slow_cancel),),
        cleanup_registry=StopCleanupRegistry(),
        receipt_store=owner,
        door=StopDoor.AGENT,
        target_timeout_seconds=30.0,
    )
    first = asyncio.create_task(authority.stop(request))
    await asyncio.wait_for(entered.wait(), timeout=5)
    await asyncio.sleep(1.2)  # three leases into a still-running Stop

    retry = await _authority(retrier, work.cancel).stop(request)

    assert [o.disposition for o in retry] == [StopDisposition.REFUSED]
    assert "already in progress" in retry[0].detail
    release.set()
    outcomes = await asyncio.wait_for(first, timeout=10)
    assert [o.disposition for o in outcomes] == [StopDisposition.STOPPED]
    assert work.calls == 1
    assert await _receipt_count(owner._db, request) == 1


def test_a_store_that_claims_must_hold_its_claims_lease():
    class ClaimsWithoutLease:
        async def load(self, _request):
            return None

        async def claim(self, _request):
            return None

        async def persist(self, *_args, **_kwargs):
            raise AssertionError

    with pytest.raises(TypeError, match="hold_claim"):
        _authority(ClaimsWithoutLease(), _Work().cancel)
