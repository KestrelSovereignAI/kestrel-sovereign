"""Operator-signal producer seam for turn-time LLM delivery.

Operator facts are trusted runtime state, not user content. This module
centralizes when those facts become model-visible and whether they travel as
native inline system turns or as an explicit user-visible fallback.

Delivery does not mean the same thing for both transports, so #2530 states one
rule and evaluates it in two places: **a notice is delivered when it is beyond
loss.**

* An inline ``system`` notice is ephemeral by construction — #2009 made it
  deliberately un-persisted so a failed turn cannot leave a durable
  trailing-system poison pill. It is beyond loss only once the provider has
  accepted the request, so that is where it settles: ``generate_with_messages``
  returning, or the first chunk of a stream.
* A fallback ``user`` notice is written to conversation history *before* the
  provider call. It is already durable at that point, so it settles at
  ``add_conversation`` success.

Retry falls out of that boundary instead of being a second, driftable policy:
**requeue exactly what was not durable.** An undelivered inline notice restores
the producer's pending events and dedupe state so the next turn re-emits it. A
persisted fallback notice does not — requeuing it would put a second copy of
the same notice in the user's history.

The lifecycle of each notice (``collected`` → ``injected`` → ``delivered`` /
``failed`` / ``cancelled``) is durably recorded through
:mod:`kestrel_sovereign.storage.operator_notice_store`, which explains why an
operator notice is not a signal and does not belong in ``signal_log``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from kestrel_sovereign.storage.operator_notice_store import (
    OperatorNoticeAuditStore,
    OperatorNoticeState,
    TERMINAL_STATES,
)

logger = logging.getLogger(__name__)

SOURCE_AUTO_MODE = "operator.auto_mode"
SOURCE_TOKEN_BUDGET = "operator.token_budget"
SOURCE_GOVERNANCE_DELTA = "operator.governance_delta"

#: Distinguishes "this session had no dedupe entry yet" from "it had False".
#: Rolling back to a fabricated ``False`` would silently suppress the next
#: genuine budget notice, so the absence has to be restorable too.
_ABSENT = object()


class OperatorSignalProducer:
    """Build trusted operator facts for the next model request.

    The producer is deliberately stateful: collecting a notice drains the
    pending auto-mode queue and advances the per-session budget/governance
    dedupe state, so a fact is emitted at most once — unless the notice is
    explicitly requeued because it was never delivered.

    Persistence is **not** uniform. The original contract here claimed the
    caller always appends the emitted text to conversation history with
    ``metadata.sent_form=True``; that has been false since #2009. Only the
    ``user``-role fallback form is persisted (byte-stable for replay). The
    inline ``system`` form is delivered in-flight and never written to
    history, which is exactly why the two forms settle at different
    boundaries — see the module docstring and
    :class:`OperatorNoticeLifecycle`.
    """

    def __init__(self, agent: Any):
        self._agent = agent
        self._pending_auto_mode: List[str] = []
        self._last_budget_low_by_session: Dict[str, bool] = {}
        self._last_governance_fingerprint_by_session: Dict[str, str] = {}

    def enqueue_auto_mode(self, scope: str) -> None:
        """Queue a global Auto change for the next turn.

        ``scope`` is one of "off", "session", or "always".
        """
        self._pending_auto_mode.append(str(scope))

    async def collect_for_turn(
        self,
        *,
        session_id: Optional[str],
        llm_service: Any,
        model_override: Optional[str],
        force_local_only: bool,
        budget_summary: Optional[Dict[str, Any]],
        state_of_mind: Any,
    ) -> "OperatorSignalBatch":
        use_inline, route_label = supports_inline_system_for_next_route(
            llm_service,
            model_override=model_override,
            force_local_only=force_local_only,
        )
        delivery_role = "system" if use_inline else "user"
        session_key = session_id or "__global__"

        # Snapshot BEFORE consuming. Everything this method is about to drain
        # or advance has to be restorable, or a turn that dies before the
        # delivery boundary loses the notice permanently: the pending events
        # are gone and the dedupe state says "already reported" (#2530).
        rollback = _ProducerStateRollback(
            producer=self,
            session_key=session_key,
            auto_mode=list(self._pending_auto_mode),
            budget_low=self._last_budget_low_by_session.get(session_key, _ABSENT),
            governance=self._last_governance_fingerprint_by_session.get(
                session_key, _ABSENT
            ),
        )

        events: List[OperatorSignalEvent] = []
        for scope in self._pending_auto_mode:
            events.append(
                OperatorSignalEvent(
                    source=SOURCE_AUTO_MODE,
                    content=_auto_mode_notice(scope),
                    payload={"scope": scope, "enabled": scope != "off"},
                )
            )
        self._pending_auto_mode.clear()

        budget_event = _budget_event(
            budget_summary,
            session_key=session_key,
            low_state=self._last_budget_low_by_session,
        )
        if budget_event is not None:
            events.append(budget_event)

        governance_event = _governance_delta_event(
            state_of_mind,
            session_key=session_key,
            last_fingerprints=self._last_governance_fingerprint_by_session,
        )
        if governance_event is not None:
            events.append(governance_event)

        if not events:
            # No notice was collected, so there is nothing to lose and nothing
            # to roll back — the dedupe advance for a no-event turn is correct
            # bookkeeping, not consumed delivery state.
            return OperatorSignalBatch.empty()

        if use_inline:
            content = "\n\n".join(event.content for event in events)
        else:
            content = _fallback_user_notice(
                "\n\n".join(event.content for event in events)
            )
            logger.info(
                "Operator signal fallback: route=%s does not support inline "
                "system messages; delivering %d event(s) as user-visible "
                "operator notice.",
                route_label,
                len(events),
            )

        lifecycle = OperatorNoticeLifecycle(
            notice_id=uuid.uuid4().hex,
            store=self._audit_store(),
            rollback=rollback,
            session_id=session_id,
        )
        batch = OperatorSignalBatch(
            role=delivery_role,
            content=content,
            keep_trailing_system=use_inline,
            events=events,
            route_label=route_label,
            fallback=not use_inline,
            lifecycle=lifecycle,
        )
        await lifecycle.record_collected(batch)
        return batch

    def _audit_store(self) -> Optional[OperatorNoticeAuditStore]:
        """Resolve the operator-notice audit store, or ``None``.

        Degrades to ``None`` so an agent without storage (tests, partial
        boots) still delivers notices. Requeue correctness does **not** depend
        on this: the rollback is held in memory by the lifecycle, so the audit
        being unavailable costs forensics, never truth.
        """
        raw_storage = getattr(self._agent, "_raw_storage", None)
        db = getattr(raw_storage, "db", None)
        if db is None:
            return None
        try:
            return OperatorNoticeAuditStore(
                db, str(getattr(self._agent, "did", "") or "")
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Operator notice audit store unavailable: %s", exc)
            return None


class _ProducerStateRollback:
    """Everything ``collect_for_turn`` consumed, restorable exactly once.

    This is the mechanism behind "requeue exactly what was not durable". It is
    deliberately not a policy of its own — the lifecycle decides *whether* to
    apply it from the delivery boundary, so the two cannot drift apart.

    Pending auto-mode scopes are restored at the FRONT of the queue so an
    older, undelivered change is still reported before anything enqueued while
    the failed turn was in flight.
    """

    def __init__(
        self,
        *,
        producer: OperatorSignalProducer,
        session_key: str,
        auto_mode: List[str],
        budget_low: Any,
        governance: Any,
    ):
        self._producer = producer
        self._session_key = session_key
        self._auto_mode = auto_mode
        self._budget_low = budget_low
        self._governance = governance
        self._applied = False

    @property
    def applied(self) -> bool:
        return self._applied

    def apply(self) -> bool:
        """Restore the pre-collect producer state. Idempotent."""
        if self._applied:
            return False
        self._applied = True
        producer = self._producer
        if self._auto_mode:
            producer._pending_auto_mode[:0] = self._auto_mode
        _restore(
            producer._last_budget_low_by_session,
            self._session_key,
            self._budget_low,
        )
        _restore(
            producer._last_governance_fingerprint_by_session,
            self._session_key,
            self._governance,
        )
        return True


def _restore(state: Dict[str, Any], key: str, value: Any) -> None:
    """Put ``key`` back exactly as it was — including not being there."""
    if value is _ABSENT:
        state.pop(key, None)
    else:
        state[key] = value


class OperatorNoticeLifecycle:
    """Explicit, durably-recorded lifecycle for one collected notice (#2530).

    ``collected`` → ``injected`` → ``delivered`` / ``failed`` / ``cancelled``.

    Two invariants:

    1. **Never claim a state that was not observed.** ``delivered`` is written
       only by a caller that watched the delivery boundary succeed; nothing
       here defaults to it.
    2. **The first terminal settle wins.** A fallback notice settles
       ``delivered`` the moment it is persisted, so the surrounding turn dying
       afterwards must not rewrite that row — the notice is in the user's
       history either way.

    The requeue decision lives here and nowhere else: the producer's consumed
    state is rolled back exactly when a notice reaches a terminal state that
    is not ``delivered``, i.e. when it left nothing durable behind.
    """

    def __init__(
        self,
        *,
        notice_id: str,
        store: Optional[OperatorNoticeAuditStore],
        rollback: _ProducerStateRollback,
        session_id: Optional[str],
    ):
        self.notice_id = notice_id
        self.session_id = session_id
        self.state = OperatorNoticeState.COLLECTED
        self.state_reason = ""
        self.durable_trace = False
        self.requeued = False
        self._store = store
        self._rollback = rollback

    @property
    def settled(self) -> bool:
        return self.state in TERMINAL_STATES

    async def record_collected(self, batch: "OperatorSignalBatch") -> None:
        if self._store is None:
            logger.info(
                "Operator notice %s collected without an audit store "
                "(role=%s, sources=%s)",
                self.notice_id,
                batch.role,
                [event.source for event in batch.events],
            )
            return
        await self._write(
            "collect",
            self._store.record_collected(
                notice_id=self.notice_id,
                session_id=self.session_id,
                delivery_role=batch.role,
                fallback=batch.fallback,
                route=batch.route_label,
                events=[(event.source, event.payload) for event in batch.events],
            ),
        )

    async def mark_injected(self) -> bool:
        """Record that the notice message joined this turn's outbound array."""
        if self.state is not OperatorNoticeState.COLLECTED:
            return False
        self.state = OperatorNoticeState.INJECTED
        if self._store is not None:
            await self._write(
                "inject", self._store.mark_injected(self.notice_id)
            )
        return True

    async def settle_delivered(self, *, durable_trace: bool = False) -> bool:
        """The notice reached its transport's beyond-loss boundary."""
        return await self._settle(
            OperatorNoticeState.DELIVERED,
            reason="",
            durable_trace=durable_trace,
        )

    async def settle_failed(self, reason: str) -> bool:
        return await self._settle(OperatorNoticeState.FAILED, reason=reason)

    async def settle_cancelled(self, reason: str) -> bool:
        return await self._settle(OperatorNoticeState.CANCELLED, reason=reason)

    async def _settle(
        self,
        state: OperatorNoticeState,
        *,
        reason: str,
        durable_trace: bool = False,
    ) -> bool:
        if self.settled:
            return False
        self.state = state
        self.state_reason = reason
        self.durable_trace = durable_trace
        if state is not OperatorNoticeState.DELIVERED:
            # Undelivered means nothing durable was left behind, so the
            # producer state this notice consumed goes back and the next turn
            # re-emits it.
            self.requeued = self._rollback.apply()
        if self._store is not None:
            await self._write(
                "settle",
                self._store.settle(
                    self.notice_id,
                    state=state,
                    reason=reason,
                    durable_trace=durable_trace,
                    requeued=self.requeued,
                ),
            )
        return True

    async def _write(self, phase: str, awaitable: Any) -> None:
        """Await an audit write, never letting it break turn delivery."""
        try:
            await awaitable
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Operator notice audit %s write failed for %s: %s",
                phase,
                self.notice_id,
                exc,
                exc_info=True,
            )


class OperatorSignalEvent:
    def __init__(self, *, source: str, content: str, payload: Dict[str, Any]):
        self.source = source
        self.content = content
        self.payload = payload


class OperatorSignalBatch:
    def __init__(
        self,
        *,
        role: str,
        content: str,
        keep_trailing_system: bool,
        events: List[OperatorSignalEvent],
        route_label: str,
        fallback: bool,
        lifecycle: Optional[OperatorNoticeLifecycle] = None,
    ):
        self.role = role
        self.content = content
        self.keep_trailing_system = keep_trailing_system
        self.events = events
        self.route_label = route_label
        self.fallback = fallback
        # ``None`` for an empty batch and for test doubles that build a batch
        # directly; every settle below then degrades to a no-op.
        self.lifecycle = lifecycle

    @classmethod
    def empty(cls) -> "OperatorSignalBatch":
        return cls(
            role="",
            content="",
            keep_trailing_system=False,
            events=[],
            route_label="unknown",
            fallback=False,
        )

    @property
    def has_events(self) -> bool:
        return bool(self.events)

    @property
    def notice_id(self) -> str:
        return self.lifecycle.notice_id if self.lifecycle is not None else ""

    @property
    def state(self) -> Optional[OperatorNoticeState]:
        return self.lifecycle.state if self.lifecycle is not None else None

    async def mark_injected(self) -> bool:
        if self.lifecycle is None:
            return False
        return await self.lifecycle.mark_injected()

    async def settle_delivered(self, *, durable_trace: bool = False) -> bool:
        if self.lifecycle is None:
            return False
        return await self.lifecycle.settle_delivered(durable_trace=durable_trace)

    async def settle_failed(self, reason: str) -> bool:
        if self.lifecycle is None:
            return False
        return await self.lifecycle.settle_failed(reason)

    async def settle_cancelled(self, reason: str) -> bool:
        if self.lifecycle is None:
            return False
        return await self.lifecycle.settle_cancelled(reason)


class OperatorTurnInjectionResult(NamedTuple):
    """Outcome of adding operator facts to one outbound LLM turn.

    ``injected_message`` is the exact message appended to ``messages``, or
    ``None`` when the producer had nothing to report. Inline system notices
    are deliberately never persisted (#2009).

    The turn owns the delivery boundary, so it also owns settling the notice:
    non-streaming callers wrap the provider call and use
    :meth:`settle_delivered` / :meth:`settle_interrupted`; streaming callers
    iterate through :meth:`watch_stream`, which settles from the stream's own
    lifecycle.
    """

    batch: OperatorSignalBatch
    injected_message: Optional[Dict[str, str]]

    @property
    def keep_trailing_system(self) -> bool:
        """Whether the selected adapter must preserve the inline system turn."""
        return self.batch.keep_trailing_system

    async def settle_delivered(self) -> bool:
        """The provider accepted the request carrying this notice."""
        return await self.batch.settle_delivered()

    async def settle_interrupted(self, exc: BaseException) -> bool:
        """Settle a turn that died before the provider accepted the request.

        Cancellation and failure are distinct terminal states because they
        answer different questions after the fact ("the operator stopped this"
        vs "this broke"). Both requeue an inline notice, because neither one
        put it anywhere it could survive.
        """
        if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
            return await self.batch.settle_cancelled(
                f"turn_cancelled:{type(exc).__name__}"
            )
        return await self.batch.settle_failed(
            f"turn_failed:{type(exc).__name__}"
        )

    async def watch_stream(self, stream: Any) -> Any:
        """Iterate a provider stream, settling the notice from its lifecycle.

        Streaming early-close semantics, stated rather than implied (#2530):

        * **First item yielded → ``delivered``.** The provider accepted the
          request, so the model provably saw any inline notice.
        * **Consumer stops iterating afterwards** (stop button, ``break``, an
          abandoned generator) **→ stays ``delivered``.** A mid-stream cancel
          does not un-send what the model already read, and requeuing there
          would re-deliver a notice the turn genuinely carried.
        * **Raises before the first item → ``failed``**, or ``cancelled`` for
          ``CancelledError`` / ``GeneratorExit``. Nothing reached the model, so
          an inline notice is requeued.
        * **Closes without ever yielding → ``failed``.** An empty stream is not
          an observation of delivery, and this module does not claim states it
          did not observe. Re-emitting an ephemeral notice next turn is the
          safe side of that ambiguity; silently dropping it is not.

        Wrapping the provider stream also makes this generator responsible for
        closing it, so ``stream`` is finalized on the way out instead of being
        left to the async-generator GC hook — see :func:`_aclose_quietly`.
        """
        delivered = False
        try:
            async for item in stream:
                if not delivered:
                    delivered = True
                    await self.batch.settle_delivered()
                yield item
        except (asyncio.CancelledError, GeneratorExit) as exc:
            # Awaiting during GeneratorExit is legal inside an async generator
            # (that is what ``aclose()`` is for); only yielding is not. When the
            # notice already settled ``delivered`` this returns without
            # suspending at all.
            await self.batch.settle_cancelled(
                f"stream_cancelled:{type(exc).__name__}"
            )
            raise
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            await self.batch.settle_failed(
                f"stream_failed:{type(exc).__name__}"
            )
            raise
        finally:
            # Every exit path releases the provider stream, including the
            # ``break``-then-``aclose()`` stop button, which unwinds through
            # the GeneratorExit branch above and would otherwise leave the
            # cancel token and upstream connection to the asyncgen GC hook.
            await _aclose_quietly(stream)
        if not delivered:
            await self.batch.settle_failed("stream_closed_without_output")


async def _aclose_quietly(stream: Any) -> None:
    """Close a provider stream on the way out of :meth:`watch_stream`.

    Before #2530 the streaming path iterated the provider stream directly, so
    a ``break`` (the #1256 stop button) dropped the only reference and CPython
    finalized it promptly. Now it is wrapped, and the wrapper owns that
    release: the inner stream holds the turn's cancel token and upstream
    connection, so waiting for the async-generator GC hook would keep both
    alive past the moment the operator said "stop".

    Anything without ``aclose`` (a plain async iterator, a test double) is
    skipped, and a failure to close is logged rather than raised — the stream
    is already being abandoned, and turning cleanup noise into the turn's
    exception would hide the real reason it ended.
    """
    aclose = getattr(stream, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except (asyncio.CancelledError, GeneratorExit):
        raise
    except Exception as exc:  # noqa: BLE001
        logger.debug("Operator notice stream close failed: %s", exc)


async def inject_operator_turn(
    agent: Any,
    messages: List[Dict[str, Any]],
    context_result: Any,
    session_id: Optional[str],
    model_override: Optional[str],
    force_local_only: bool,
) -> OperatorTurnInjectionResult:
    """Collect and inject operator facts through the canonical turn contract.

    The outbound message is always appended before optional fallback
    persistence. Persistence failure therefore never suppresses in-flight
    delivery. Native inline system notices remain ephemeral so a failed LLM
    turn cannot leave a durable trailing-system poison pill (#2009).

    Injection is where the two delivery boundaries diverge (#2530). The
    fallback ``user`` notice becomes beyond-loss right here, when
    ``add_conversation`` succeeds, so it settles ``delivered`` before the
    provider is ever called. The inline ``system`` notice is still only
    ``injected`` on return — the caller settles it at provider-accept.
    """
    producer = getattr(agent, "operator_signal_producer", None)
    if isinstance(producer, OperatorSignalProducer):
        batch = await producer.collect_for_turn(
            session_id=session_id,
            llm_service=agent.llm_service,
            model_override=model_override,
            force_local_only=force_local_only,
            budget_summary=context_result.budget_summary,
            state_of_mind=getattr(context_result, "state_of_mind", None),
        )
    else:
        batch = OperatorSignalBatch.empty()

    if not batch.has_events:
        return OperatorTurnInjectionResult(
            batch=batch,
            injected_message=None,
        )

    injected_message = {"role": batch.role, "content": batch.content}
    messages.append(injected_message)
    await batch.mark_injected()

    if batch.role != "system":
        try:
            await agent.privacy_agent.add_conversation(
                batch.role,
                batch.content,
                metadata={
                    "sent_form": True,
                    "operator_signal": True,
                    "operator_signal_sources": [
                        event.source for event in batch.events
                    ],
                    "operator_signal_fallback": batch.fallback,
                },
                session_id=session_id,
                rendered_content=batch.content,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to persist operator signal turn; continuing with "
                "in-flight delivery only: %s",
                exc,
                exc_info=True,
            )
            # The fallback form's beyond-loss boundary IS persistence, and it
            # just missed. In-flight delivery still proceeds — the model sees
            # the notice this turn — but the audit must not claim delivery and
            # the producer state must not stay advanced, or the notice is
            # never reported again anywhere the operator can see it.
            await batch.settle_failed(
                f"fallback_persist_failed:{type(exc).__name__}"
            )
        else:
            # In conversation history now: durable, and requeuing it would put
            # a second copy of the same notice in front of the user.
            await batch.settle_delivered(durable_trace=True)

    return OperatorTurnInjectionResult(
        batch=batch,
        injected_message=injected_message,
    )


def supports_inline_system_for_next_route(
    llm_service: Any,
    *,
    model_override: Optional[str],
    force_local_only: bool,
) -> Tuple[bool, str]:
    """Centralized capability decision for operator-signal delivery."""
    provider: Optional[Dict[str, Any]] = None
    target_model: Optional[str] = None
    try:
        if hasattr(llm_service, "resolve_provider_routing"):
            providers, target_model = llm_service.resolve_provider_routing(
                model_override=model_override,
                force_local_only=force_local_only,
            )
        else:
            providers = getattr(llm_service, "_available_providers", lambda: [])()
        provider = providers[0] if providers else None
    except Exception as exc:  # noqa: BLE001
        logger.info("Operator signal route capability probe failed: %s", exc)
        provider = None

    if not provider:
        return False, "unknown"

    model = target_model or provider.get("model") or ""
    route_label = provider.get("name") or provider.get("vendor") or "unknown"
    caps = provider.get("capabilities") or {}
    route_supports = bool(caps.get("supports_inline_system"))
    adapter = provider.get("adapter")
    model_supports = True
    checker = getattr(adapter, "_model_supports_inline_system", None)
    if callable(checker):
        try:
            model_supports = bool(checker(model))
        except Exception:  # noqa: BLE001
            model_supports = False
    return route_supports and model_supports, f"{route_label}/{model or 'auto'}"


def state_of_mind_snapshot(state: Any) -> Dict[str, Any]:
    if state is None:
        return {}
    if isinstance(state, dict):
        return dict(state)
    if is_dataclass(state):
        raw = asdict(state)
    else:
        raw = {
            "provider": getattr(state, "provider", None),
            "model": getattr(state, "model", None),
            "governance_mode": getattr(state, "governance_mode", None),
            "transparency": getattr(state, "transparency", None),
            "delegated_principles": getattr(state, "delegated_principles", None),
            "active_conflicts": getattr(state, "active_conflicts", None),
            "complements": getattr(state, "complements", None),
        }
    raw.pop("prompt_adaptation", None)
    return raw


def _auto_mode_notice(scope: str) -> str:
    if scope == "always":
        return (
            "Operator context: auto-mode is now enabled persistently (the "
            "'always' tier). The operator has given standing consent for "
            "multi-agent workflows and non-denied tool calls to proceed "
            "without additional approval prompts when earlier constitutional, "
            "honesty, and security checks do not flag the action. This consent "
            "survives session resets and server restarts until explicitly "
            "revoked."
        )
    if scope == "session":
        return (
            "Operator context: auto-mode is now enabled for this server "
            "session. The operator has given standing consent for "
            "multi-agent workflows and non-denied tool calls to proceed "
            "without additional approval prompts when earlier constitutional, "
            "honesty, and security checks do not flag the action."
        )
    return (
        "Operator context: auto-mode is now disabled. Standing consent for "
        "unprompted tool use has ended; approval prompts apply according to "
        "the configured permission policy."
    )


def _budget_event(
    budget_summary: Optional[Dict[str, Any]],
    *,
    session_key: str,
    low_state: Dict[str, bool],
) -> Optional[OperatorSignalEvent]:
    if not budget_summary:
        return None
    try:
        total = int(budget_summary.get("total_budget") or 0)
        used = int(budget_summary.get("total_used") or 0)
        external_reserved = max(
            0,
            int(budget_summary.get("external_reserved_tokens") or 0),
        )
    except (TypeError, ValueError):
        return None
    if total <= 0:
        return None
    # ``total_used`` is deliberately the sum of named context-section usage.
    # Tool schemas and other provider payloads consume the same turn ceiling,
    # but ElasticTokenBudget reports them separately so section attribution
    # remains exact.  Count both here: the operator notice promises remaining
    # budget for the whole turn, not merely unspent named sections.
    remaining = max(0, total - used - external_reserved)
    threshold = _budget_threshold(total)
    is_low = remaining <= threshold
    was_low = low_state.get(session_key, False)
    low_state[session_key] = is_low
    if not is_low or was_low:
        return None
    return OperatorSignalEvent(
        source=SOURCE_TOKEN_BUDGET,
        content=(
            "Operator context: remaining token budget for this turn is now "
            f"{remaining} tokens out of {total}. The low-budget threshold is "
            f"{threshold} tokens."
        ),
        payload={
            "remaining_tokens": remaining,
            "total_budget": total,
            "threshold": threshold,
        },
    )


def _budget_threshold(total_budget: int) -> int:
    return max(2048, int(total_budget * 0.10))


def _governance_delta_event(
    state: Any,
    *,
    session_key: str,
    last_fingerprints: Dict[str, str],
) -> Optional[OperatorSignalEvent]:
    snapshot = state_of_mind_snapshot(state)
    if not snapshot:
        return None
    mutable = {
        "provider": snapshot.get("provider"),
        "model": snapshot.get("model"),
        "governance_mode": snapshot.get("governance_mode"),
        "transparency": snapshot.get("transparency"),
        "delegated_principles": snapshot.get("delegated_principles") or [],
        "active_conflicts": snapshot.get("active_conflicts") or [],
    }
    fingerprint = repr(mutable)
    if last_fingerprints.get(session_key) == fingerprint:
        return None
    last_fingerprints[session_key] = fingerprint
    return OperatorSignalEvent(
        source=SOURCE_GOVERNANCE_DELTA,
        content=_format_governance_notice(mutable),
        payload=mutable,
    )


def _format_governance_notice(state: Dict[str, Any]) -> str:
    lines = [
        "Operator context: constitutional governance state has changed.",
        (
            f"Current model route: {state.get('provider') or 'unknown'} / "
            f"{state.get('model') or 'unknown'}."
        ),
        (
            "Governance mode: "
            f"{str(state.get('governance_mode') or 'unknown').upper()}."
        ),
        f"Transparency: {state.get('transparency') or 'unknown'}.",
    ]
    delegated = state.get("delegated_principles") or []
    if delegated:
        lines.append(
            "Delegated principles: "
            + ", ".join(str(p) for p in delegated)
            + "."
        )
    else:
        lines.append("Delegated principles: none.")
    conflicts = state.get("active_conflicts") or []
    if conflicts:
        rendered = []
        for conflict in conflicts:
            if isinstance(conflict, dict):
                principle = conflict.get("principle", "unknown")
                severity = conflict.get("severity", "unknown")
                description = conflict.get("description", "")
                rendered.append(
                    f"{principle} ({severity})"
                    + (f": {description}" if description else "")
                )
            else:
                rendered.append(str(conflict))
        lines.append("Active conflicts: " + "; ".join(rendered) + ".")
    else:
        lines.append("Active conflicts: none.")
    lines.append(f"Observed at: {datetime.now(timezone.utc).isoformat()}.")
    return "\n".join(lines)


def _fallback_user_notice(content: str) -> str:
    return (
        "<operator_notice>\n"
        "The following trusted operator context is being relayed as a "
        "user-visible notice because the selected route/model does not "
        "support inline mid-conversation system messages.\n\n"
        f"{content}\n"
        "</operator_notice>"
    )
