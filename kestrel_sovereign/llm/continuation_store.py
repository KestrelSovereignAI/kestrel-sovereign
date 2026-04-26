"""Per-conversation continuation cursor store for stateful provider protocols.

Some providers anchor multi-turn behavior on a server-side response id rather
than re-deriving state from the message history. The OpenAI Responses API
(used by ``CodexAdapter``) is the immediate motivator: each completed response
carries an id, and the next request can pass ``previous_response_id`` plus
only the *new* input items (typically the tool results from the prior turn),
which preserves encrypted reasoning across turns. Without continuation the
``include=[reasoning.encrypted_content]`` flag is dead — the encrypted blob
has nowhere to land on turn N+1.

This module provides the minimum primitive: a small KV indexed by
``(adapter_name, conversation_id)`` that stores a cursor naming the last
response id, the message-list length at that point, and a request signature
used to detect tool/instruction drift mid-conversation. Anthropic and others
do not use it (their continuation is positional in messages); the store is a
no-op for them.

Default implementation is process-local and dict-backed. The ``ContinuationStore``
Protocol can be re-implemented against Redis/SQL when the runtime moves to
multi-worker uvicorn — adapters never see the difference. See epic #806 / #808.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple


@dataclass(frozen=True)
class ContinuationCursor:
    """Snapshot of conversation state needed to send a delta turn.

    Attributes:
        last_response_id: The provider's id for the most recently completed
            response on this conversation. Used as ``previous_response_id`` on
            the next call.
        last_message_count: ``len(messages)`` at the moment the cursor was
            written. The next turn sends only ``messages[last_message_count:]``
            as input.
        last_request_signature: A stable hash of the prior request's
            ``(instructions, tools)``. If the next call's signature differs,
            the adapter must drop continuation and resubmit full context — the
            server's prior reasoning was conditioned on a different prompt.
    """

    last_response_id: str
    last_message_count: int
    last_request_signature: Optional[str] = None


class ContinuationStore(Protocol):
    """KV interface for continuation cursors. Keyed by (adapter_name, conversation_id)."""

    def get(
        self, adapter_name: str, conversation_id: str
    ) -> Optional[ContinuationCursor]: ...

    def put(
        self,
        adapter_name: str,
        conversation_id: str,
        cursor: ContinuationCursor,
    ) -> None: ...

    def clear(self, adapter_name: str, conversation_id: str) -> None: ...


class InMemoryContinuationStore:
    """Process-local, threadsafe ``ContinuationStore``. Default for single-worker runtimes.

    Multi-worker deployments need a shared backend (Redis, SQL); the Protocol
    above is the seam — swap the implementation, keep the adapter unchanged.
    """

    def __init__(self) -> None:
        self._cursors: Dict[Tuple[str, str], ContinuationCursor] = {}
        self._lock = threading.Lock()

    def get(
        self, adapter_name: str, conversation_id: str
    ) -> Optional[ContinuationCursor]:
        with self._lock:
            return self._cursors.get((adapter_name, conversation_id))

    def put(
        self,
        adapter_name: str,
        conversation_id: str,
        cursor: ContinuationCursor,
    ) -> None:
        with self._lock:
            self._cursors[(adapter_name, conversation_id)] = cursor

    def clear(self, adapter_name: str, conversation_id: str) -> None:
        with self._lock:
            self._cursors.pop((adapter_name, conversation_id), None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cursors)
