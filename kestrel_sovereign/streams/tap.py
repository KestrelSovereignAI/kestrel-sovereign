"""
Agent stream tap — shared infrastructure for tapping into active agent streams.

When the agent streams a response via ``/agent/stream``, text chunks are
published to an :class:`asyncio.Queue` keyed by *request_id*.  A TTS consumer
can subscribe to the same request_id and receive text as it arrives, enabling
incremental speech synthesis before the full response is complete.

Usage::

    # Producer side (agent stream endpoint):
    tap = AgentStreamTap.get_instance()
    tap.register(request_id)
    for chunk in agent_chunks:
        await tap.publish(request_id, chunk)
    await tap.finish(request_id)

    # Consumer side (TTS streaming endpoint):
    async for text_chunk in tap.subscribe(request_id):
        ...  # synthesize speech
"""

import asyncio
import logging
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)

# Sentinel object that signals the stream is complete.
_STREAM_DONE = object()


class AgentStreamTap:
    """Registry of per-request async queues for agent text streaming.

    Singleton — use :meth:`get_instance` to obtain the shared instance.
    """

    _instance: Optional["AgentStreamTap"] = None

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}

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

    def register(self, request_id: str) -> None:
        """Create a queue for *request_id*. Idempotent."""
        if request_id not in self._queues:
            self._queues[request_id] = asyncio.Queue()

    async def publish(self, request_id: str, text_chunk: str) -> None:
        """Push a text chunk to all subscribers of *request_id*."""
        queue = self._queues.get(request_id)
        if queue is not None:
            await queue.put(text_chunk)

    async def finish(self, request_id: str) -> None:
        """Signal that the agent stream for *request_id* is complete."""
        queue = self._queues.get(request_id)
        if queue is not None:
            await queue.put(_STREAM_DONE)

    def unregister(self, request_id: str) -> None:
        """Remove the queue for *request_id* (cleanup)."""
        self._queues.pop(request_id, None)

    def has_stream(self, request_id: str) -> bool:
        """Check whether a stream exists for *request_id*."""
        return request_id in self._queues

    # ---- Consumer API ----

    async def subscribe(
        self, request_id: str, timeout: float = 30.0
    ) -> AsyncIterator[str]:
        """Yield text chunks as they arrive for *request_id*.

        Args:
            request_id: The agent request to follow.
            timeout: Max seconds to wait for a chunk before giving up.

        Yields:
            Text chunks from the agent stream.
        """
        queue = self._queues.get(request_id)
        if queue is None:
            logger.warning("No active stream for request_id=%s", request_id)
            return

        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout waiting for stream chunk (request_id=%s)", request_id
                    )
                    break
                if item is _STREAM_DONE:
                    break
                yield item
        finally:
            # Consumer done — clean up only if no other subscribers
            self.unregister(request_id)
