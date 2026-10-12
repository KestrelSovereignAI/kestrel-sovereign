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
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task, raise_owned_outcome
from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody,
    bind_execution_custody_snapshot, current_execution_custody,
    close_execution_stream, execution_terminal_error,
    execution_cleanup_debts, execution_denial_scopes,
)

logger = logging.getLogger(__name__)

# Sentinel object that signals the stream is complete.
_STREAM_DONE = object()


class _TapResponseOwner:
    """Hold original drain authority through the outer HTTP EOF/close."""
    def __init__(self, stream, fence, producer):
        fence.require_work()
        fence.owner.retain_active_request_generation(fence.request_id, fence.generation)
        self.stream, self.fence, self.producer = stream, fence, producer
        self.retired = False
        self._close_task = None
        self._terminal_error = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await anext(self.stream)
        except BaseException as original:
            if not isinstance(original, StopAsyncIteration):
                self._terminal_error = execution_terminal_error(self._terminal_error, original)
            try:
                await self.aclose()
            except BaseException as cleanup:
                raise execution_terminal_error(original, cleanup)
            raise

    async def aclose(self):
        if self._close_task is None:
            async def retire():
                cleanup_failed = False
                try:
                    try:
                        await close_execution_stream(self.fence.owner, self.stream, self.producer)
                    except BaseException as cleanup:
                        cleanup_failed = True
                        self._terminal_error = execution_terminal_error(self._terminal_error, cleanup)
                    if self._terminal_error is not None:
                        raise self._terminal_error
                finally:
                    if not self.retired:
                        self.retired = True
                        # Work refusal remains visible to the original reader.
                        # Successfully closed ordinary scoped Stop is not
                        # physical abandonment; actual close failures, native
                        # debt and irreversible/unscoped receipts still are.
                        ordinary_stop = (execution_denial_scopes(self._terminal_error) is not None
                            and not execution_cleanup_debts(self._terminal_error))
                        disposition = (RequestCompletionDisposition.COMPLETED
                            if not cleanup_failed and (self._terminal_error is None or ordinary_stop)
                            else RequestCompletionDisposition.ABANDONED)
                        self.fence.owner._cleanup_cancelled_request(self.fence.request_id,
                            generation=self.fence.generation, disposition=disposition)
            self._close_task = create_owned_task(retire())
        return raise_owned_outcome(await await_owned_task(self._close_task),
            operation="original tap response retirement")


@dataclass
class _TapRequestFence:
    owner: object
    request_id: str
    generation: int
    drain_retained: bool = False
    privacy_token: object | None = None
    privacy_ready: asyncio.Event = field(default_factory=asyncio.Event)
    @property
    def backend_type(self):
        return self.owner.storage.db.backend_type

    def is_live(self):
        return (type(self.generation) is int
            and self.generation in self.owner._active_generations_for_request(self.request_id)
            and (self.request_id, self.generation) not in self.owner._cancelled_request_generations
            and self.generation not in getattr(self.owner, "_abandoned_request_generations", {}).get(self.request_id, ()))

    def require_work(self):
        if not self.is_live():
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
        self._requests: dict[str, _TapRequestFence] = {}
        self._consumers: set[str] = set()
        self._expiry: dict[str, asyncio.TimerHandle] = {}

    @classmethod
    def get_instance(cls) -> "AgentStreamTap":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        if cls._instance is not None:
            for delivery_id in tuple(cls._instance._queues):
                cls._instance.unregister(delivery_id)
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
                    fence = _TapRequestFence(owner, request_id, generation)
                    self._requests[delivery_id] = fence
                    scopes += (ExecutionCustody(fence),)
                self._producers[delivery_id] = (owner, scopes)

    def producer_custody(self, delivery_id: str, owner) -> tuple[ExecutionCustody, ...]:
        """Only the same agent may consume this original producer admission."""
        producer = self._producers.get(delivery_id)
        if producer is None or producer[0] is not owner:
            raise ExecutionAuthorityError("stream tap is not owned by this agent")
        with bind_execution_custody_snapshot(producer[1]):
            pass
        return producer[1]

    def own_response(self, delivery_id: str, owner, stream):
        """Retain the SAME live generation; never create a replacement turn."""
        producer = self.producer_custody(delivery_id, owner)
        fence = self._requests.get(delivery_id)
        return stream if fence is None else _TapResponseOwner(stream, fence, producer)

    def capture_request_privacy(self, owner, request_id: str, generation: int | None) -> None:
        """The ACTUAL text lock owner transfers only its matching delivery span."""
        from kestrel_sovereign.storage.privacy_wrapper import capture_transition_lock_reentry
        token = capture_transition_lock_reentry(owner)
        if token is None:
            return
        for fence in self._requests.values():
            if fence.owner is owner and fence.request_id == request_id and fence.generation == generation:
                fence.require_work()
                fence.privacy_token = token
                fence.privacy_ready.set()

    async def producer_privacy(self, delivery_id: str, owner):
        self.producer_custody(delivery_id, owner)
        fence = self._requests.get(delivery_id)
        if fence is None:
            return None  # Standalone text taps have no Core turn privacy span.
        await asyncio.wait_for(fence.privacy_ready.wait(), timeout=30.0)
        self.producer_custody(delivery_id, owner)
        fence.require_work()
        return fence.privacy_token

    async def publish(self, delivery_id: str, text_chunk: str) -> None:
        """Push a text chunk to subscribers of one stream delivery."""
        queue = self._queues.get(delivery_id)
        if queue is not None:
            await queue.put(text_chunk)

    async def finish(self, delivery_id: str, *, completed: bool = False) -> None:
        """Queue EOF; only successful completion transfers original drain custody."""
        queue = self._queues.get(delivery_id)
        if queue is not None:
            fence = self._requests.get(delivery_id)
            if completed and fence is not None and not fence.drain_retained and fence.is_live():
                # A Stop arriving after text publication declines the speech
                # handoff; it does not retroactively fail the completed wire
                # response. No retired generation is retained or resurrected.
                fence.owner.retain_active_request_generation(fence.request_id, fence.generation)
                fence.drain_retained = True
            if fence is not None:
                fence.privacy_ready.set()
            if delivery_id not in self._consumers and delivery_id not in self._expiry:
                self._expiry[delivery_id] = asyncio.get_running_loop().call_later(
                    30.0, self.unregister, delivery_id)
            await queue.put(_STREAM_DONE)

    @contextmanager
    def consumer_custody(self, delivery_id: str, owner):
        """One original consumer owns queue drain AND final synthesis flush."""
        producer = self.producer_custody(delivery_id, owner)
        if delivery_id in self._consumers:
            raise ExecutionAuthorityError("stream delivery already has a drain owner")
        self._consumers.add(delivery_id)
        timer = self._expiry.pop(delivery_id, None)
        if timer is not None:
            timer.cancel()
        try:
            yield producer
        finally:
            self._consumers.discard(delivery_id)
            self.unregister(delivery_id)

    def unregister(self, delivery_id: str) -> None:
        """Remove the queue for one completed stream delivery."""
        self._queues.pop(delivery_id, None)
        self._producers.pop(delivery_id, None)
        timer = self._expiry.pop(delivery_id, None)
        if timer is not None:
            timer.cancel()
        fence = self._requests.pop(delivery_id, None)
        if fence is not None:
            fence.privacy_ready.set()
        if fence is not None and fence.drain_retained:
            fence.owner._cleanup_cancelled_request(fence.request_id, generation=fence.generation)

    def has_stream(self, delivery_id: str) -> bool:
        """Check whether a stream delivery has an active tap."""
        return delivery_id in self._queues

    # ---- Consumer API ----

    async def subscribe(
        self, delivery_id: str, timeout: float = 30.0, *, owner=None, retire: bool = True,
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
            if retire:
                self.unregister(delivery_id)
