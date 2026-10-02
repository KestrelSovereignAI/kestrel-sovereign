"""``LLMService.decide``: the single front door to decision models (#3424).

Spec: ``docs/architecture/llm/DECISIONS.md``. This mixin owns discovery of
decision models per route, pin canaries, per-call resolution, dispatch,
normalisation and accounting. Pure pieces (config, resolution, fit,
thresholds, normalisation, HTTP) live in :mod:`kestrel_sovereign.llm.decisions`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Set

from kestrel_sdk.llm.decisions import (
    DecisionModelInfo,
    DecisionProtocolError,
    DecisionRequest,
    DecisionResult,
    DecisionTimeout,
    DecisionTransportError,
    NoulQuestion,
    ValidatedDecisionRequest,
    validate_decision_request,
)

from kestrel_sovereign.config import load_section

from .decisions.config import (
    DecisionRouteConfig,
    DecisionSelector,
    parse_decision_selector,
    parse_service_decision_config,
)
from .decisions.fit import RequestFootprint
from .decisions.http import DecisionHTTPError
from .decisions.normalize import normalize_response
from .decisions.resolve import (
    PinStatus,
    RouteDecisionState,
    candidate_routes,
    select_candidate,
)
from .decisions.thresholds import ThresholdBook

logger = logging.getLogger(__name__)

#: How long ``decide`` waits for its telemetry record before handing it to a
#: background task (§8.2). The caller's outcome is never delayed past this.
DECISION_RECORD_TIMEOUT = 2.0

#: Fixed synthetic request for pin canaries (§4.1): no caller content.
CANARY_REQUEST: ValidatedDecisionRequest = validate_decision_request(
    DecisionRequest(
        state={"text": "The sky is blue."},
        questions={"canary": NoulQuestion(instructions="Does `text` mention a colour?")},
    )
)
CANARY_CALLER = "kestrel.canary"


def _route_state(provider: Mapping[str, Any]) -> RouteDecisionState:
    state = provider.get("decision_state")
    if not isinstance(state, RouteDecisionState):
        state = RouteDecisionState(config=DecisionRouteConfig())
        provider["decision_state"] = state  # type: ignore[index]
    return state


def _supports_decisions(provider: Mapping[str, Any]) -> bool:
    return bool((provider.get("capabilities") or {}).get("supports_decisions"))


class DecisionServiceMixin:
    """Decision modality for :class:`~kestrel_sovereign.llm.service.LLMService`."""

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_decisions(self) -> None:
        """Parse ``[llm] decision_*`` and ``[decisions.thresholds]``. Invalid
        configuration raises: an operator typo must not silently disable or
        redirect decisions."""

        llm_cfg = self.config if isinstance(getattr(self, "config", None), dict) else {}
        self._decision_config = parse_service_decision_config(llm_cfg)
        self._decision_thresholds = ThresholdBook.from_config(load_section("decisions") or {})
        self._decision_discovery_locks: Dict[str, asyncio.Lock] = {}

    def _decision_routes(self) -> List[Dict[str, Any]]:
        """Decision-capable routes, skipping routes disabled by auth failure.

        Reads ``providers`` the way embedding reconciliation does, so a
        partially-constructed service (no routes yet) simply has none.
        """
        providers = getattr(self, "providers", None)
        if not isinstance(providers, list):
            return []
        disabled = getattr(self, "_disabled_routes", None) or ()
        return [
            p for p in providers
            if _supports_decisions(p) and p.get("name") not in disabled
        ]

    def _decision_lock(self, name: str) -> asyncio.Lock:
        locks = self.__dict__.setdefault("_decision_discovery_locks", {})
        return locks.setdefault(name, asyncio.Lock())

    # ------------------------------------------------------------------
    # Discovery (§4)
    # ------------------------------------------------------------------

    async def reconcile_decision_capabilities(self, use_cache: bool = True) -> None:
        """Discover decision models on every decision-capable route.

        With ``use_cache`` only routes that have never been discovered (or
        whose pin was never checked) are contacted; otherwise every route is
        refreshed. Routes run concurrently so one slow route cannot stall the
        others past its own deadline.
        """

        routes = self._decision_routes()
        await asyncio.gather(
            *(self._discover_decision_route(p, force=not use_cache) for p in routes)
        )

    async def _discover_decision_route(self, provider: Dict[str, Any], *, force: bool) -> None:
        name = str(provider.get("name"))
        state = _route_state(provider)
        async with self._decision_lock(name):
            if not force and not state.needs_discovery:
                return
            try:
                discovered = await provider["adapter"].list_decision_models(provider.get("client"))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - any discovery failure marks the route stale
                state.record_discovery_failure()
                logger.warning(
                    "Decision discovery failed for %s (%s); keeping the last "
                    "known list", name, type(exc).__name__,
                )
            else:
                state.record_discovery(
                    replace(info, vendor=str(provider.get("vendor")), route=name)
                    for info in discovered
                )
                logger.info("Decision discovery for %s: %d model(s)", name, len(state.models or ()))
            if state.config.pin is not None:
                await self._run_pin_canary(provider, state)

    def _pin_info(self, provider: Mapping[str, Any], state: RouteDecisionState) -> DecisionModelInfo:
        pin = state.config.pin
        assert pin is not None
        for info in state.models or ():
            if info.id == pin:
                if state.config.context_limit is not None:
                    return replace(info, context_limit=state.config.context_limit)
                return info
        return DecisionModelInfo(
            id=pin,
            vendor=str(provider.get("vendor")),
            route=str(provider.get("name")),
            context_limit=state.config.context_limit,
        )

    async def _run_pin_canary(self, provider: Mapping[str, Any], state: RouteDecisionState) -> None:
        """Verify a pinned model with one synthetic decision (§4.1)."""

        pin = state.config.pin
        assert pin is not None
        name = str(provider.get("name"))
        timeout = self._decision_config.canary_timeout_seconds
        started = time.monotonic()
        body: Optional[Mapping[str, Any]] = None
        error: Optional[BaseException] = None
        try:
            async with asyncio.timeout(timeout):
                body = await provider["adapter"].adecide(
                    provider.get("client"), pin, CANARY_REQUEST, timeout=timeout
                )
            normalize_response(CANARY_REQUEST, body)
        except asyncio.CancelledError as exc:
            error = exc
            state.canary_stale_since = state.canary_stale_since or time.time()
            raise
        except DecisionProtocolError as exc:
            error = exc
            self._mark_pin_unverified(state, name, f"canary answer invalid: {exc}")
        except DecisionHTTPError as exc:
            error = exc
            if exc.status_code == 404:
                self._mark_pin_unverified(state, name, "endpoint or model not found (HTTP 404)")
            else:
                state.canary_stale_since = state.canary_stale_since or time.time()
        except (DecisionTransportError, TimeoutError) as exc:
            error = exc
            state.canary_stale_since = state.canary_stale_since or time.time()
            logger.warning("Decision pin canary for %s did not complete (%s)", name, type(exc).__name__)
        else:
            state.pin_status = PinStatus.VERIFIED
            state.pin_reason = None
            state.canary_stale_since = None
            state.pin_info = self._pin_info(provider, state)
        finally:
            await self._finish_decision_record(
                provider=provider,
                model=pin,
                caller=CANARY_CALLER,
                question_count=1,
                duration_ms=int((time.monotonic() - started) * 1000),
                body=body,
                error=error,
                calibrated=None,
                context=self._resolve_invocation_context(),
            )

    @staticmethod
    def _mark_pin_unverified(state: RouteDecisionState, name: str, reason: str) -> None:
        state.pin_status = PinStatus.UNVERIFIED
        state.pin_reason = reason
        state.pin_info = None
        logger.error("Decision pin on %s is unverified: %s", name, reason)

    # ------------------------------------------------------------------
    # decide (§3.1, §5, §6, §8)
    # ------------------------------------------------------------------

    async def decide(
        self,
        request: DecisionRequest,
        *,
        caller: str,
        timeout_seconds: float,
        model_override: Optional[str] = None,
        local_only: bool = False,
        session_id: Optional[str] = None,
    ) -> DecisionResult:
        """Answer ``request`` on the first decision route that can take it.

        Raises a :class:`~kestrel_sdk.llm.decisions.DecisionError` subclass for
        anything other than a complete, normalised result. Never re-sends to
        another route after a dispatch.
        """

        if not isinstance(caller, str) or not caller:
            raise ValueError("decide() requires a non-empty caller id")
        if timeout_seconds <= 0:
            raise ValueError("decide() timeout_seconds must be positive")

        # Frozen before the first await: the snapshot, privacy and identity.
        snapshot = validate_decision_request(request)
        question_ids = tuple(snapshot.questions)
        self._decision_thresholds.check_request(caller, question_ids)
        selector: Optional[DecisionSelector] = (
            parse_decision_selector(model_override) if model_override else None
        )
        effective_local_only = bool(local_only) or self._current_force_local_only()
        context = self._resolve_invocation_context(session_id=session_id)
        footprint = RequestFootprint.of(snapshot)

        routes = candidate_routes(
            self._decision_routes(),
            service_config=self._decision_config,
            selector=selector,
            local_only=effective_local_only,
        )

        dispatched = False
        candidate = None
        started = time.monotonic()
        body: Optional[Mapping[str, Any]] = None
        error: Optional[BaseException] = None
        calibrated: Optional[bool] = None
        try:
            async with asyncio.timeout(timeout_seconds):
                cold = [p for p in routes if _route_state(p).needs_discovery]
                if cold:
                    await asyncio.gather(
                        *(self._discover_decision_route(p, force=False) for p in cold)
                    )
                candidate = select_candidate(
                    routes,
                    footprint=footprint,
                    selector=selector,
                    thresholds=self._decision_thresholds,
                    caller=caller,
                    question_ids=question_ids,
                )
                resolution = self._decision_thresholds.resolve(
                    caller, candidate.model_key, question_ids
                )
                calibrated = resolution.calibrated
                provider = candidate.provider
                dispatched = True
                started = time.monotonic()
                body = await provider["adapter"].adecide(
                    provider.get("client"),
                    candidate.info.id,
                    snapshot,
                    timeout=timeout_seconds,
                )
                normalized = normalize_response(snapshot, body)
        except TimeoutError as exc:
            error = exc
            raise DecisionTimeout(
                f"decision for {caller!r} exceeded {timeout_seconds}s"
            ) from None
        except BaseException as exc:
            error = exc
            raise
        finally:
            if dispatched and candidate is not None:
                await self._finish_decision_record(
                    provider=candidate.provider,
                    model=candidate.info.id,
                    caller=caller,
                    question_count=len(question_ids),
                    duration_ms=int((time.monotonic() - started) * 1000),
                    body=body,
                    error=error,
                    calibrated=calibrated,
                    context=context,
                )

        provider = candidate.provider
        return DecisionResult(
            answers=normalized.answers,
            vendor=str(provider.get("vendor")),
            route=str(provider.get("name")),
            model=candidate.info.id,
            thresholds=resolution.thresholds,
            calibrated=resolution.calibrated,
            input_tokens=normalized.input_tokens,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    # ------------------------------------------------------------------
    # Accounting (§8.2, §8.3)
    # ------------------------------------------------------------------

    async def _finish_decision_record(self, **record: Any) -> None:
        """Write one decision record without letting it alter the outcome.

        The write runs as its own task under ``asyncio.shield`` so cancelling
        the caller does not cancel it. The caller waits at most
        :data:`DECISION_RECORD_TIMEOUT`; a slower write finishes in the
        background. Nothing raised here reaches the caller.
        """

        task = asyncio.ensure_future(self._write_decision_record(**record))
        self._pending_decision_records().add(task)
        task.add_done_callback(self._decision_record_done)
        try:
            await asyncio.wait_for(asyncio.shield(task), DECISION_RECORD_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(
                "Decision record for %s is late; finishing in the background",
                record.get("caller"),
            )
        except asyncio.CancelledError:
            # A new cancellation while waiting: the shielded write carries on
            # by itself, and the cancellation propagates to the caller.
            raise
        except Exception:  # noqa: BLE001 - logged by the done callback
            pass

    def _pending_decision_records(self) -> "Set[asyncio.Task[None]]":
        return self.__dict__.setdefault("_decision_record_tasks", set())

    def _decision_record_done(self, task: "asyncio.Task[None]") -> None:
        self._pending_decision_records().discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("Decision telemetry record failed: %s", type(exc).__name__)

    async def _write_decision_record(
        self,
        *,
        provider: Mapping[str, Any],
        model: str,
        caller: str,
        question_count: int,
        duration_ms: int,
        body: Optional[Mapping[str, Any]],
        error: Optional[BaseException],
        calibrated: Optional[bool],
        context: Any,
    ) -> None:
        usage = body.get("usage") if isinstance(body, Mapping) else None
        input_tokens = None
        cost = None
        if isinstance(usage, Mapping):
            tokens = usage.get("input_tokens")
            if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
                input_tokens = tokens
            raw_cost = usage.get("cost")
            if isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool):
                cost = float(raw_cost)
        success = error is None
        provider_name = str(provider.get("name"))
        if input_tokens is not None:
            await self._track_model_usage(model, provider_name, tokens=input_tokens)
        await self._log_llm_call(
            provider=provider_name,
            model=model,
            duration_ms=duration_ms,
            success=success,
            # §8.3: decision telemetry carries no content — not the state,
            # not the questions, not a vendor error body.
            system_prompt=None,
            user_prompt=None,
            response=None,
            error_message=type(error).__name__ if error is not None else None,
            metadata={
                "modality": "decision",
                "caller": caller,
                "question_count": question_count,
                "calibrated": calibrated,
                "usage_available": input_tokens is not None,
            },
            input_tokens=input_tokens,
            output_tokens=None,
            cost=cost,
            usage_available=input_tokens is not None,
            invocation_context=context,
            modality="decision",
            caller=caller,
        )

    async def drain_decision_records(self) -> None:
        """Wait for late decision telemetry records (called from ``close``)."""

        tasks = set(self._pending_decision_records())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
