"""Tests for the AWAITER_GONE terminal state in the approval queue (#2558).

When the original tool-call awaiter is cancelled (Cloud Run request
timeout, SSE disconnect, client abort), the request is deliberately kept
in ``_pending`` so a late decision still persists the scope rule for
future calls. But the in-flight call is orphaned, so ``submit_decision``
now reports ``awaiter_gone=True`` and pops the request itself.
"""

import asyncio
from datetime import datetime, timedelta, timezone

from kestrel_sovereign.features.security.approval_queue import (
    ApprovalQueue,
    ApprovalRequest,
    ApprovalStatus,
)


async def _start_and_cancel() -> tuple[ApprovalQueue, str]:
    """Start a ``request_approval`` awaiter and cancel it.

    Returns the queue and the request id, with the request left in
    ``_pending`` marked AWAITER_GONE.
    """
    queue = ApprovalQueue()

    task = asyncio.create_task(
        queue.request_approval(
            feature_name="WalletAgent",
            tool_name="send_payment",
            tool_args={"amount": 100, "recipient": "alice"},
        )
    )

    # Let the awaiter reach the resume_event.wait() point.
    while not queue.pending_requests:
        await asyncio.sleep(0)

    request_id = queue.pending_requests[0].id

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    return queue, request_id


async def test_awaiter_cancelled_marks_request_awaiter_gone():
    queue, request_id = await _start_and_cancel()

    request = queue.get_request(request_id)
    assert request is not None
    assert request.status == ApprovalStatus.AWAITER_GONE
    # Kept alive for a late decision.
    assert request_id in queue._pending


async def test_submit_decision_on_awaiter_gone_persists_and_reports():
    queue, request_id = await _start_and_cancel()

    result = await queue.submit_decision(request_id, approved=True, scope="session")

    assert result.awaiter_gone is True
    assert result.in_memory is True
    assert result.persisted is True
    # The awaiter's finally-block is gone, so submit_decision pops it.
    assert request_id not in queue._pending


async def test_submit_decision_on_awaiter_gone_once_scope_still_reports():
    queue, request_id = await _start_and_cancel()

    result = await queue.submit_decision(request_id, approved=True, scope="once")

    assert result.awaiter_gone is True
    assert result.in_memory is True
    # "once" writes no persistence rule, so persistence is a no-op success.
    assert result.persisted is True
    assert request_id not in queue._pending


async def test_normal_approval_still_awaiter_gone_false():
    queue = ApprovalQueue()

    task = asyncio.create_task(
        queue.request_approval(
            feature_name="WalletAgent",
            tool_name="send_payment",
            tool_args={"amount": 100, "recipient": "alice"},
        )
    )

    while not queue.pending_requests:
        await asyncio.sleep(0)

    request_id = queue.pending_requests[0].id

    result = await queue.submit_decision(request_id, approved=True, scope="once")

    assert result.awaiter_gone is False
    assert result.in_memory is True

    approved, scope = await task
    assert approved is True
    assert scope == "once"


async def test_sweep_stale_reaps_awaiter_gone():
    queue = ApprovalQueue()

    request = ApprovalRequest(
        id="stale-awaiter-gone",
        feature_name="WalletAgent",
        tool_name="send_payment",
        tool_args={"amount": 100, "recipient": "alice"},
        created_at=datetime.now(timezone.utc) - timedelta(hours=48),
        status=ApprovalStatus.AWAITER_GONE,
    )
    queue._pending[request.id] = request

    reaped = await queue.sweep_stale(older_than_seconds=3600)

    assert reaped == 1
    assert request.id not in queue._pending


async def test_cancel_request_pops_awaiter_gone():
    # An AWAITER_GONE request has no live waiter — cancel_request must pop it
    # itself, or it lingers forever (sweep_stale skips set events). (#2558)
    queue, request_id = await _start_and_cancel()

    assert queue.cancel_request(request_id) is True
    assert request_id not in queue._pending

    # And a subsequent sweep has nothing left to reap.
    reaped = await queue.sweep_stale(older_than_seconds=0)
    assert reaped == 0


async def test_cancel_all_pops_awaiter_gone():
    queue, request_id = await _start_and_cancel()

    count = queue.cancel_all()

    assert count == 1
    assert request_id not in queue._pending
    assert queue.pending_count == 0


async def test_cancel_request_still_unblocks_live_pending():
    # Regression: a normal PENDING request must still be unblocked (not just
    # popped) so the live awaiter returns denied. (#2558)
    queue = ApprovalQueue()

    task = asyncio.create_task(
        queue.request_approval(
            feature_name="WalletAgent",
            tool_name="send_payment",
            tool_args={"amount": 100, "recipient": "alice"},
        )
    )
    while not queue.pending_requests:
        await asyncio.sleep(0)
    request_id = queue.pending_requests[0].id

    assert queue.cancel_request(request_id) is True

    approved, scope = await task
    assert approved is False
    assert scope == "cancelled"
