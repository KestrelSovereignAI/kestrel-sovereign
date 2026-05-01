"""Shared turn lifecycle for non-streaming and streaming entry points.

Per SIGNAL_DISPATCHER.md §Concern 1: `process_input` and
`process_input_streaming` both write conversation history with no
serialization today; two turns can interleave. Heartbeat ([heartbeat.py:247])
fires `process_input` without checking whether a user turn is in flight.

This mixin provides the **single boundary** where the `CONVERSATION` lock
is acquired/released. Both entry points wrap their inner traced bodies in
`async with self._turn_lifecycle():`. The lock manager is the same
instance the SignalDispatcher will use for its registered resource locks
(Phase 1 — already shipped via the SDK), so cross-system invariants hold.

The dispatcher does NOT pre-acquire `CONVERSATION` for COGNITION sources
(Phase 1 §Concern 2). The turn lifecycle here is the sole owner.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

from kestrel_sdk.signals import ResourceLock
from kestrel_sovereign.signals import OrderedLockManager

logger = logging.getLogger(__name__)


class TurnLifecycleMixin:
    """Provides `_turn_lifecycle` and the per-agent state it needs.

    `KestrelAgent.__init__` initializes `self._lock_manager`; the
    `_get_lock_manager` accessor lazy-creates one for tests/callers that
    bypass `__init__` via `KestrelAgent.__new__` (mirrors the existing
    `_get_privacy_transition_lock` pattern in `kestrel_agent.py`).
    """

    _lock_manager: OrderedLockManager

    def _get_lock_manager(self) -> OrderedLockManager:
        """Return the shared OrderedLockManager, lazy-creating one if the
        owning class skipped __init__ (tests using `KestrelAgent.__new__`)."""
        mgr = getattr(self, "_lock_manager", None)
        if mgr is None:
            mgr = OrderedLockManager()
            self._lock_manager = mgr
        return mgr

    @asynccontextmanager
    async def _turn_lifecycle(self) -> AsyncIterator[str]:
        """Enter a turn: acquire CONVERSATION, yield a fresh turn_id,
        release on exit (normal or exception).

        The yielded `turn_id` is opaque to callers today; Phase 5 (#894 —
        A2A causation chain propagation) will plumb it into Signal
        envelopes so dispatcher-driven cognition can mark its CausationFrame
        with the receiving turn.
        """
        turn_id = f"turn_{uuid4().hex[:12]}"
        mgr = self._get_lock_manager()
        async with mgr.acquire({ResourceLock.CONVERSATION}):
            logger.debug("turn_lifecycle: %s begin", turn_id)
            try:
                yield turn_id
            finally:
                logger.debug("turn_lifecycle: %s end", turn_id)
