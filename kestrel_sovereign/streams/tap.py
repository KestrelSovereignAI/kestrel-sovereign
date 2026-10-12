"""
Agent stream tap — shared infrastructure for tapping into active agent streams.

When the agent streams a response via ``/agent/stream``, text chunks are
published to an :class:`asyncio.Queue` keyed by a server-generated
*delivery_id*. A TTS consumer can subscribe to the delivery ID advertised in
the ``X-Stream-Delivery-ID`` response header and receive text as it arrives,
enabling
incremental speech synthesis before the full response is complete.

Usage::

    # Producer side (agent stream endpoint):
    tap = AgentStreamTap.get_instance()
    tap.register(delivery_id)
    for chunk in agent_chunks:
        await tap.publish(delivery_id, chunk)
    await tap.finish(delivery_id)

    # Consumer side (TTS streaming endpoint):
    async for text_chunk in tap.subscribe(delivery_id):
        ...  # synthesize speech
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody,
    bind_execution_custody_snapshot, current_execution_custody,
)

logger = logging.getLogger(__name__)

# Sentinel object that signals the stream is complete.
_STREAM_DONE = object()


@dataclass(frozen=True)
class _TapRequestFence:
    owner: object
    request_id: str
    generation: int
    backend_type = None

    def require_work(self):
        if (self.generation not in self.owner._active_generations_for_request(self.request_id)
            or (self.request_id, self.generation) in self.owner._cancelled_request_generations):
            raise ExecutionAuthorityError("original stream tap request generation retired")

    async def lock_and_validate(self, connection):
        self.require_work()


class AgentStreamTap:
    """Registry of per-delivery async queues for agent text streaming.

    Singleton — use :meth:`get_instance` to obtain the shared instance.
    """

    _instance: Optional["AgentStreamTap"] = None

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}
        self._producers: dict[str, tuple[object, tuple[ExecutionCustody, ...]]] = {}

    @classmethod
    def get_instance(cls) -> "AgentStreamTap":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    # ---- Producer API ----

    def register(self, delivery_id: str, *, owner=None, request_id: str | None = None) -> None:
        """Create a queue for one server-owned stream delivery."""
        if delivery_id not in self._queues:
            self._queues[delivery_id] = asyncio.Queue()
            if owner is not None:
                scopes = current_execution_custody(owner)
                if request_id is not None:
                    generation = owner._request_generation_for_current_task(request_id)
                    if generation is None:
                        self._queues.pop(delivery_id)
                        raise ExecutionAuthorityError("stream tap producer has no original request generation")
                    scopes += (ExecutionCustody(_TapRequestFence(owner, request_id, generation)),)
                self._producers[delivery_id] = (owner, scopes)

    def producer_custody(self, delivery_id: str, owner) -> tuple[ExecutionCustody, ...]:
        """Only the same agent may consume this original producer admission."""
        producer = self._producers.get(delivery_id)
        if producer is None or producer[0] is not owner:
            raise ExecutionAuthorityError("stream tap is not owned by this agent")
        with bind_execution_custody_snapshot(producer[1]):
            pass
        return producer[1]

    async def publish(self, delivery_id: str, text_chunk: str) -> None:
        """Push a text chunk to subscribers of one stream delivery."""
        queue = self._queues.get(delivery_id)
        if queue is not None:
            await queue.put(text_chunk)

    async def finish(self, delivery_id: str) -> None:
        """Signal that one stream delivery is complete."""
        queue = self._queues.get(delivery_id)
        if queue is not None:
            await queue.put(_STREAM_DONE)

    def unregister(self, delivery_id: str) -> None:
        """Remove the queue for one completed stream delivery."""
        self._queues.pop(delivery_id, None)
        self._producers.pop(delivery_id, None)

    def has_stream(self, delivery_id: str) -> bool:
        """Check whether a stream delivery has an active tap."""
        return delivery_id in self._queues

    # ---- Consumer API ----

    async def subscribe(
        self, delivery_id: str, timeout: float = 30.0, *, owner=None,
    ) -> AsyncIterator[str]:
        """Yield text chunks as they arrive for *request_id*.

        Args:
            delivery_id: The server-generated stream delivery to follow.
            timeout: Max seconds to wait for a chunk before giving up.

        Yields:
            Text chunks from the agent stream.
        """
        queue = self._queues.get(delivery_id)
        if queue is None:
            logger.warning("No active stream for delivery_id=%s", delivery_id)
            return
        producer = self.producer_custody(delivery_id, owner) if owner is not None else ()

        try:
            while True:
                with bind_execution_custody_snapshot(producer):
                    pass
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout waiting for stream chunk (delivery_id=%s)", delivery_id
                    )
                    break
                if item is _STREAM_DONE:
                    break
                with bind_execution_custody_snapshot(producer):
                    pass
                yield item
        finally:
            # Consumer done — clean up only if no other subscribers
            self.unregister(delivery_id)
