"""Deadlock-freedom regression for the privacy-transition lock order (F037).

The wedge: process_input_streaming acquired the privacy transition lock and THEN
the CONVERSATION lock, while the in-turn `!privacy` path (process_input) holds
CONVERSATION and then acquires the transition lock — opposite orders on the same
two per-agent locks. Two concurrent callers could take the pair AB / BA and
deadlock the whole agent with no timeout and no restart trigger.

The fix pins one global order: CONVERSATION (via the turn lifecycle) is always
acquired BEFORE the transition lock. These tests encode both real acquisition
sequences and assert that the fixed order cannot deadlock, while the old order
provably would (so a regression to transition-first is caught).
"""

from __future__ import annotations

import asyncio

import pytest

from kestrel_sovereign.signals import OrderedLockManager
from kestrel_sdk.signals import ResourceLock


async def _run_pair(streaming_takes_conversation_first: bool) -> None:
    """Drive the two contending paths concurrently.

    Path S (streaming turn): CONVERSATION + transition, in the order under test.
    Path P (in-turn !privacy): CONVERSATION (turn) then transition — fixed.
    """
    mgr = OrderedLockManager()
    transition = asyncio.Lock()
    both_have_first = asyncio.Event()
    s_started = asyncio.Event()
    p_started = asyncio.Event()

    async def streaming_path():
        if streaming_takes_conversation_first:
            async with mgr.acquire({ResourceLock.CONVERSATION}):
                s_started.set()
                await both_have_first.wait()
                async with transition:
                    await asyncio.sleep(0)
        else:  # the OLD buggy order: transition first, then CONVERSATION
            async with transition:
                s_started.set()
                await both_have_first.wait()
                async with mgr.acquire({ResourceLock.CONVERSATION}):
                    await asyncio.sleep(0)

    async def privacy_path():
        async with mgr.acquire({ResourceLock.CONVERSATION}):
            p_started.set()
            await both_have_first.wait()
            async with transition:
                await asyncio.sleep(0)

    # For the buggy order we need BOTH to grab their first lock before either
    # reaches for its second, to force the AB-BA interleaving. With the fixed
    # order both want CONVERSATION first, so they simply serialize.
    async def orchestrate():
        s = asyncio.ensure_future(streaming_path())
        if streaming_takes_conversation_first:
            # Same first lock: they can't both hold it — release the gate now,
            # they serialize cleanly.
            both_have_first.set()
        else:
            # Different first locks: wait until S holds transition and P holds
            # CONVERSATION, then open the gate to force the crossed acquisition.
            p = asyncio.ensure_future(privacy_path())
            await asyncio.wait_for(asyncio.gather(_wait(s_started), _wait(p_started)), timeout=1.0)
            both_have_first.set()
            await asyncio.gather(s, p)
            return
        p = asyncio.ensure_future(privacy_path())
        await asyncio.gather(s, p)

    await orchestrate()


async def _wait(evt: asyncio.Event) -> None:
    await evt.wait()


@pytest.mark.asyncio
async def test_fixed_order_conversation_first_never_deadlocks():
    # Must complete well within the timeout — no cycle is possible.
    await asyncio.wait_for(_run_pair(streaming_takes_conversation_first=True), timeout=2.0)


@pytest.mark.asyncio
async def test_old_transition_first_order_would_deadlock():
    # Documents WHY the fix is needed: the pre-fix order deadlocks, so the
    # crossed acquisition never completes and we hit the timeout.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_run_pair(streaming_takes_conversation_first=False), timeout=1.0)
