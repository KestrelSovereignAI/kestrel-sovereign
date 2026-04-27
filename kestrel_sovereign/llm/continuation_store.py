"""Per-conversation continuation cursor store for stateful provider protocols.

Some providers anchor multi-turn behavior on prior-response state. The
ChatGPT-backend Responses API used by ``CodexAdapter`` is the immediate
motivator: it does NOT support ``previous_response_id`` (caught live in #841)
but it DOES accept reasoning items as input items on subsequent turns. Per
the spec, encrypted reasoning items must appear before their corresponding
``function_call`` items in the input list to give the model continuity of
chain-of-thought across tool-result round trips. Without that, the
``include=[reasoning.encrypted_content]`` flag is dead and multi-turn
agent loops on GPT-5 reason from scratch each turn.

This module provides the minimum primitive: a small KV indexed by
``(adapter_name, session_id)`` that stores a cursor with the last response
id, a request signature for drift detection, and a per-turn record of
output items (reasoning + function_call) that the adapter replays as input
on subsequent turns. Anthropic and other adapters that don't need this
ignore the parameter and the cursor stays empty. See #806 / #808 / #842.

Default implementation is process-local and dict-backed. The
``ContinuationStore`` Protocol is the swap point for Redis/SQL backends in
multi-worker deployments — adapters never see the difference.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple


@dataclass(frozen=True)
class ContinuationCursor:
    """Snapshot of conversation state for cross-turn reasoning continuity.

    Attributes:
        last_response_id: The provider's id for the most recently completed
            response on this session. Recorded for diagnostics; not used as
            a continuation token on the ChatGPT backend (rejected — #841).
        last_message_count: ``len(messages)`` after extracting the system
            prompt at the moment the cursor was written. Used in tests to
            reason about turn boundaries; not used to slice input on the
            wire (slicing was removed when continuation was disabled in #841).
        last_request_signature: Stable hash of the prior request's
            ``(instructions, tools)``. If the next call's signature differs,
            the cached reasoning was conditioned on a different prompt and
            must not be replayed — drop and resubmit full context.
        turn_outputs: Tuple of JSON-encoded lists, one per prior turn. Each
            inner list holds the output items emitted by the model on that
            turn (reasoning items + function_call items, in order). The
            adapter splices these back into the input list on subsequent
            turns so encrypted chain-of-thought persists across tool round
            trips. JSON-encoded so the cursor stays a frozen dataclass with
            hashable contents. See #842.
    """

    last_response_id: str
    last_message_count: int
    last_request_signature: Optional[str] = None
    turn_outputs: Tuple[str, ...] = field(default_factory=tuple)


class ContinuationStore(Protocol):
    """KV interface for continuation cursors. Keyed by (adapter_name, session_id)."""

    def get(
        self, adapter_name: str, session_id: str
    ) -> Optional[ContinuationCursor]: ...

    def put(
        self,
        adapter_name: str,
        session_id: str,
        cursor: ContinuationCursor,
    ) -> None: ...

    def clear(self, adapter_name: str, session_id: str) -> None: ...


class InMemoryContinuationStore:
    """Process-local, threadsafe ``ContinuationStore``. Default for single-worker runtimes.

    Multi-worker deployments need a shared backend (Redis, SQL); the Protocol
    above is the seam — swap the implementation, keep the adapter unchanged.
    """

    def __init__(self) -> None:
        self._cursors: Dict[Tuple[str, str], ContinuationCursor] = {}
        self._lock = threading.Lock()

    def get(
        self, adapter_name: str, session_id: str
    ) -> Optional[ContinuationCursor]:
        with self._lock:
            return self._cursors.get((adapter_name, session_id))

    def put(
        self,
        adapter_name: str,
        session_id: str,
        cursor: ContinuationCursor,
    ) -> None:
        with self._lock:
            self._cursors[(adapter_name, session_id)] = cursor

    def clear(self, adapter_name: str, session_id: str) -> None:
        with self._lock:
            self._cursors.pop((adapter_name, session_id), None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cursors)
