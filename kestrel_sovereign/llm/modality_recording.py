"""One recording path for the content-free modalities (#3424, #3426).

Chat calls are finalized by ``LLMService._finalize_invocation``. Decisions and
embeddings have no prompt or response worth keeping, so both are described by
a :class:`ModalityCall` and written by
:meth:`ModalityRecordingMixin.record_modality_call` to the same sinks as chat:
``model_usage``, the ``llm_calls`` row, Prometheus and the metering callback,
through the modality-aware ``_log_llm_call``.

Spec: ``docs/architecture/llm/DECISIONS.md`` §8.2–§8.3.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Protocol, Set

from .invocation_context import LLMInvocationContext

logger = logging.getLogger(__name__)

#: How long a caller waits for its call's telemetry record before the write is
#: handed to a background task (§8.2). The caller's outcome is never delayed
#: past this.
USAGE_RECORD_TIMEOUT = 2.0

RecordedModality = Literal["decision", "embedding"]


@dataclass(frozen=True)
class ModalityCall:
    """One dispatched decision or embedding call, without content.

    ``error_class`` is the exception's class name, never its message: a vendor
    error can echo the request. ``metadata`` holds content-free fields only;
    the writer adds ``modality`` and ``usage_available``.
    """

    modality: RecordedModality
    provider: str
    model: str
    duration_ms: int
    success: bool
    context: LLMInvocationContext
    error_class: Optional[str] = None
    input_tokens: Optional[int] = None
    cost: Optional[float] = None
    caller: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ModalityRecorder(Protocol):
    """What a modality service needs from ``LLMService`` to record a call."""

    def snapshot_invocation_context(self) -> LLMInvocationContext: ...

    async def record_modality_call(self, call: ModalityCall) -> None: ...


class ModalityRecordingMixin:
    """Records :class:`ModalityCall` objects for ``LLMService``."""

    def snapshot_invocation_context(self) -> LLMInvocationContext:
        """Freeze the caller's identity before the call's first await (§8.3)."""

        return self._resolve_invocation_context()

    async def record_modality_call(self, call: ModalityCall) -> None:
        """Write ``call`` without letting it alter the caller's outcome.

        The write runs as its own task under ``asyncio.shield``, so cancelling
        the caller does not cancel it. The caller waits at most
        :data:`USAGE_RECORD_TIMEOUT`; a slower write finishes in the
        background. Nothing raised by the write reaches the caller.
        """

        task = asyncio.ensure_future(self._write_modality_record(call))
        self._pending_modality_records().add(task)
        task.add_done_callback(self._modality_record_done)
        try:
            await asyncio.wait_for(asyncio.shield(task), USAGE_RECORD_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(
                "%s record for %s is late; finishing in the background",
                call.modality,
                call.caller or call.model,
            )
        except asyncio.CancelledError:
            # A new cancellation while waiting: the shielded write carries on
            # by itself, and the cancellation propagates to the caller.
            raise
        except Exception:  # noqa: BLE001 - logged by the done callback
            pass

    def _pending_modality_records(self) -> "Set[asyncio.Task[None]]":
        return self.__dict__.setdefault("_modality_record_tasks", set())

    def _modality_record_done(self, task: "asyncio.Task[None]") -> None:
        self._pending_modality_records().discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("LLM telemetry record failed: %s", type(exc).__name__)

    async def _write_modality_record(self, call: ModalityCall) -> None:
        usage_available = call.input_tokens is not None
        if usage_available:
            await self._record_model_usage(
                call.model,
                call.provider,
                tokens=call.input_tokens,
                label=call.modality,
            )
        await self._log_llm_call(
            provider=call.provider,
            model=call.model,
            duration_ms=call.duration_ms,
            success=call.success,
            # §8.3: no content — not the decision state or questions, not the
            # embedded text, not a vector, not a vendor error body.
            system_prompt=None,
            user_prompt=None,
            response=None,
            error_message=call.error_class,
            metadata={
                "modality": call.modality,
                **call.metadata,
                "usage_available": usage_available,
            },
            input_tokens=call.input_tokens,
            output_tokens=None,
            cost=call.cost,
            usage_available=usage_available,
            invocation_context=call.context,
            modality=call.modality,
            caller=call.caller,
        )

    async def drain_modality_records(self) -> None:
        """Wait for late decision/embedding records (called from ``close``)."""

        tasks = set(self._pending_modality_records())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
