"""``LLMService.decide``: the single front door to decision models (#3424).

Spec: ``docs/architecture/llm/DECISIONS.md``. This mixin owns discovery of
decision models per route, pin canaries, per-call resolution, dispatch,
normalisation and accounting. Pure pieces (config, resolution, fit,
thresholds, normalisation, HTTP) live in :mod:`kestrel_sovereign.llm.decisions`.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence

from kestrel_sdk.llm.decisions import (
    DecisionModelInfo,
    DecisionProtocolError,
    DecisionRequest,
    DecisionRequestInvalid,
    DecisionResult,
    DecisionTimeout,
    DecisionTransportError,
    NoulQuestion,
    ValidatedDecisionRequest,
    validate_decision_request,
)

from kestrel_sovereign.config import load_section
from kestrel_sovereign.execution_custody import ExecutionAuthorityError, await_execution_work, bind_execution_cleanup, bind_execution_runtime, execution_work_operation, require_execution_work, is_execution_control_error, execution_commit_outcome, execution_terminal_error
from kestrel_sovereign.execution_custody import await_execution_work_group

from .adapter import ReportedUsage
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
from .modality_recording import ModalityCall

logger = logging.getLogger(__name__)

#: Fixed synthetic request for pin canaries (§4.1): no caller content.
CANARY_REQUEST: ValidatedDecisionRequest = validate_decision_request(
    DecisionRequest(
        state={"text": "The sky is blue."},
        questions={"canary": NoulQuestion(instructions="Does `text` mention a colour?")},
    )
)
CANARY_CALLER = "kestrel.canary"
_THRESHOLD_KEY = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")


def _route_state(provider: Mapping[str, Any]) -> RouteDecisionState:
    state = provider.get("decision_state")
    if not isinstance(state, RouteDecisionState):
        state = RouteDecisionState(config=DecisionRouteConfig())
        provider["decision_state"] = state  # type: ignore[index]
    return state


def _supports_decisions(provider: Mapping[str, Any]) -> bool:
    return bool((provider.get("capabilities") or {}).get("supports_decisions"))


def _threshold_keys(
    question_ids: Sequence[str], mapping: Optional[Mapping[str, str]]
) -> Dict[str, str]:
    """Each question id's calibration key (the id itself unless mapped)."""

    mapping = dict(mapping or {})
    unknown = sorted(set(mapping) - set(question_ids))
    if unknown:
        raise DecisionRequestInvalid(
            "thresholds", f"threshold_keys names unknown question id(s) {unknown!r}"
        )
    for question_id, key in mapping.items():
        if not isinstance(key, str) or not _THRESHOLD_KEY.fullmatch(key):
            raise DecisionRequestInvalid(
                "thresholds", f"threshold key for {question_id!r} is not a valid id: {key!r}"
            )
    return {q: mapping.get(q, q) for q in question_ids}


def _default_thresholds(
    raw: Optional[Mapping[str, float]],
) -> Optional[Mapping[str, float]]:
    if raw is None:
        return None
    values: Dict[str, float] = {}
    for key, value in raw.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise DecisionRequestInvalid(
                "thresholds", f"default threshold for {key!r} must be in [0, 1]"
            )
        values[str(key)] = float(value)
    return MappingProxyType(values)


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

    def describe_decision_routes(self) -> List[Dict[str, Any]]:
        """Operator view of every decision-capable route's state (no I/O).

        One entry per route: name, locality, the pin and its verification
        state, staleness, and the discovered models with their limits. Used by
        ``kestrel decisions models`` and the eval harness.
        """

        out: List[Dict[str, Any]] = []
        for provider in self._decision_routes():
            state = _route_state(provider)
            out.append({
                "route": str(provider.get("name")),
                "vendor": str(provider.get("vendor")),
                "is_local": bool(provider.get("is_local")),
                "pin": state.config.pin,
                "pin_status": state.pin_status.value,
                "pin_reason": state.pin_reason,
                "hints": list(state.config.hints),
                "discovered": state.models is not None,
                "discovery_stale_since": state.discovery_stale_since,
                "canary_stale_since": state.canary_stale_since,
                "models": [
                    {"id": m.id, "context_limit": m.context_limit,
                     "parallel_questions": m.parallel_questions}
                    for m in (state.models or ())
                ],
            })
        return out

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
        await await_execution_work_group(
            self,
            (lambda p=p: self._discover_decision_route(p, force=not use_cache) for p in routes),
        )

    async def _discover_decision_route(self, provider: Dict[str, Any], *, force: bool) -> None:
        name = str(provider.get("name"))
        state = _route_state(provider)
        async with self._decision_lock(name):
            if not force and not state.needs_discovery:
                return
            try:
                discovered = await await_execution_work(self, lambda: provider["adapter"].list_decision_models(provider.get("client")))
            except asyncio.CancelledError:
                raise
            except ExecutionAuthorityError:
                raise
            except Exception as exc:  # noqa: BLE001 - any discovery failure marks the route stale
                if is_execution_control_error(exc):
                    raise
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
                body = await await_execution_work(self, lambda: provider["adapter"].adecide(
                    provider.get("client"), pin, CANARY_REQUEST, timeout=timeout
                ))
            normalize_response(CANARY_REQUEST, body)
        except ExecutionAuthorityError as exc:
            error = exc
            if is_execution_control_error(exc):
                raise
            raise
        except asyncio.CancelledError as exc:
            error = exc
            state.canary_stale_since = state.canary_stale_since or time.time()
            raise
        except DecisionProtocolError as exc:
            error = exc
            if is_execution_control_error(exc):
                raise
            self._mark_pin_unverified(state, name, f"canary answer invalid: {exc}")
        except DecisionHTTPError as exc:
            error = exc
            if is_execution_control_error(exc):
                raise
            if exc.status_code == 404:
                self._mark_pin_unverified(state, name, "endpoint or model not found (HTTP 404)")
            else:
                state.canary_stale_since = state.canary_stale_since or time.time()
        except (DecisionTransportError, TimeoutError) as exc:
            error = exc
            if is_execution_control_error(exc):
                raise
            state.canary_stale_since = state.canary_stale_since or time.time()
            logger.warning("Decision pin canary for %s did not complete (%s)", name, type(exc).__name__)
        except BaseException as exc:
            # Unlisted native/wrapped errors still reach the unconditional
            # finalizer. Never mislabel those as a successful canary call.
            error = exc
            raise
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

    @execution_work_operation
    async def decide(
        self,
        request: DecisionRequest,
        *,
        caller: str,
        timeout_seconds: float,
        model_override: Optional[str] = None,
        local_only: bool = False,
        session_id: Optional[str] = None,
        threshold_keys: Optional[Mapping[str, str]] = None,
        default_thresholds: Optional[Mapping[str, float]] = None,
    ) -> DecisionResult:
        """Answer ``request`` on the first decision route that can take it.

        Raises a :class:`~kestrel_sdk.llm.decisions.DecisionError` subclass for
        anything other than a complete, normalised result. Never re-sends to
        another route after a dispatch.

        ``threshold_keys`` maps question ids to shared calibration keys, so N
        questions of one kind (one per candidate, say) share one threshold;
        unmapped ids are their own key. ``default_thresholds`` (by key) is the
        caller's own pre-calibration threshold; ``[decisions.thresholds]``
        overrides it key by key and calibrated per-model entries replace it.
        ``DecisionResult.thresholds`` is always keyed by question id.
        """

        require_execution_work(self)
        if not isinstance(caller, str) or not caller:
            raise ValueError("decide() requires a non-empty caller id")
        if timeout_seconds <= 0:
            raise ValueError("decide() timeout_seconds must be positive")

        # Frozen before the first await: the snapshot, privacy and identity.
        snapshot = validate_decision_request(request)
        question_ids = tuple(snapshot.questions)
        keys = _threshold_keys(question_ids, threshold_keys)
        defaults = _default_thresholds(default_thresholds)
        self._decision_thresholds.check_request(caller, keys.values(), defaults)
        selector: Optional[DecisionSelector] = (
            parse_decision_selector(model_override) if model_override else None
        )
        effective_local_only = self._effective_force_local_only(local_only)
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
                    await await_execution_work_group(
                        self,
                        (lambda p=p: self._discover_decision_route(p, force=False) for p in cold),
                    )
                candidate = select_candidate(
                    routes,
                    footprint=footprint,
                    selector=selector,
                    thresholds=self._decision_thresholds,
                    caller=caller,
                    question_ids=tuple(keys.values()),
                )
                resolution = self._decision_thresholds.resolve(
                    caller, candidate.model_key, keys.values(), defaults
                )
                calibrated = resolution.calibrated
                provider = candidate.provider
                dispatched = True
                started = time.monotonic()
                body = await await_execution_work(self, lambda: provider["adapter"].adecide(
                    provider.get("client"),
                    candidate.info.id,
                    snapshot,
                    timeout=timeout_seconds,
                ))
                normalized = normalize_response(snapshot, body)
        except TimeoutError as exc:
            error = exc
            if is_execution_control_error(exc):
                raise
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

        require_execution_work(self)
        provider = candidate.provider
        return DecisionResult(
            answers=normalized.answers,
            vendor=str(provider.get("vendor")),
            route=str(provider.get("name")),
            model=candidate.info.id,
            thresholds=MappingProxyType(
                {q: resolution.thresholds[keys[q]] for q in question_ids}
                if resolution.thresholds
                else {}
            ),
            calibrated=resolution.calibrated,
            input_tokens=normalized.input_tokens,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    # ------------------------------------------------------------------
    # Accounting (§8.2, §8.3)
    # ------------------------------------------------------------------

    async def _finish_decision_record(
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
        """Record one dispatched decision through the shared recorder."""

        if error is not None and is_execution_control_error(error):
            # Irreversible control is not a successful/failed provider call.
            # Even absent ambient custody, it must not start ordinary writes.
            return
        usage = ReportedUsage()
        reported = body.get("usage") if isinstance(body, Mapping) else None
        if isinstance(reported, Mapping):
            usage.add(input_tokens=reported.get("input_tokens"), cost=reported.get("cost"))
        with (bind_execution_cleanup(self) if error is not None and (isinstance(error, asyncio.CancelledError) or is_execution_control_error(error)) else bind_execution_runtime(self)):
            try:
                await self.record_modality_call(
                    ModalityCall(
                        modality="decision",
                        provider=str(provider.get("name")),
                        model=model,
                        duration_ms=duration_ms,
                        success=error is None,
                        context=context,
                        error_class=type(error).__name__ if error is not None else None,
                        input_tokens=usage.input_tokens,
                        cost=usage.cost,
                        caller=caller,
                        metadata={
                            "caller": caller,
                            "question_count": question_count,
                            "calibrated": calibrated,
                        },
                    )
                )
            except Exception as accounting_error:
                if (
                    error is None
                    or not is_execution_control_error(accounting_error)
                    or not (isinstance(error, asyncio.CancelledError) or is_execution_control_error(error))
                    or execution_commit_outcome(accounting_error) is not None
                ):
                    raise execution_terminal_error(error, accounting_error)
