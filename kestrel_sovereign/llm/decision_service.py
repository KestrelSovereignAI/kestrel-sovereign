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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from kestrel_sdk.llm.decisions import (
    DecisionModelInfo,
    DecisionProtocolError,
    DecisionRequest,
    DecisionRequestInvalid,
    DecisionResult,
    DecisionTimeout,
    DecisionTransportError,
    DecisionUnavailable,
    NoulQuestion,
    UnavailableReason,
    ValidatedDecisionRequest,
    validate_decision_request,
)

from kestrel_sovereign.config import load_section
from kestrel_sovereign.execution_custody import ExecutionAuthorityError, await_execution_work, bind_execution_cleanup, bind_execution_runtime, execution_work_operation, require_execution_work, is_execution_control_error, execution_terminal_error
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
from .discovery_deadline import DiscoveryStepTimeout
from .modality_recording import ModalityCall
from .provider_policy import (
    OverrideProvenance,
    ProviderPolicyRefusal,
    approved_vendors,
    config_provenance,
    parse_denied_providers,
    provider_is_denied,
    route_priority_entries,
)

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


class DecisionOverrideRefused(DecisionUnavailable):
    """The provider policy refused a decision ``model_override`` (#2652).

    Nothing was sent. A :class:`DecisionUnavailable`, so every caller that
    handles a failed decision handles a refused one; :attr:`refusal` carries
    the structured :class:`~.provider_policy.ProviderPolicyRefusal`.
    """

    def __init__(self, refusal: ProviderPolicyRefusal) -> None:
        super().__init__(UnavailableReason.NO_ROUTE, str(refusal))
        self.refusal = refusal

    def to_dict(self) -> Dict[str, Any]:
        return self.refusal.to_dict()


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
        # Routes ``[llm] denied_providers`` names are never used (#2652).
        denied = parse_denied_providers(getattr(self, "config", None) or {})
        return [
            p for p in providers
            if _supports_decisions(p)
            and p.get("name") not in disabled
            and not provider_is_denied(p, denied)
        ]

    def _admit_decision_override(
        self, model_override: Optional[str], selector: Optional[DecisionSelector]
    ) -> List[Tuple[Dict[str, Optional[str]], OverrideProvenance]]:
        """Admit a decision ``model_override`` as any per-call override (#2652).

        The provider policy's one admission check, under the writer named for
        exactly this override (an operator's configured decision model, say),
        else the agent's. It admits the provider ``selector``, the override as
        :func:`parse_decision_selector` read it, selects: the route ``decide``
        dispatches to is chosen by that parse, so admission reads the same
        one (``" openrouter "`` is OpenRouter to both). A refusal is recorded
        and logged by the check and raised here as
        :class:`DecisionOverrideRefused`.

        Returns the choices ``decide``'s dispatch check honors: the admitted
        selection with the writer it was admitted for, and an explicit
        ``[llm] decision_route``, which is the operator's.
        """
        admitted: List[Tuple[Dict[str, Optional[str]], OverrideProvenance]] = []
        configured = self._decision_config.route
        if configured:
            vendor, _, route = configured.partition(":")
            admitted.append((
                {"vendor": vendor, "route": route or None, "model": None},
                config_provenance(via="decision_route"),
            ))
        if selector is None:
            return admitted
        selection = self._call_override_selection(selector.text)
        try:
            provenance = self._admit_call_selection(
                model_override, selection, via="decide"
            )
        except ProviderPolicyRefusal as refusal:
            raise DecisionOverrideRefused(refusal) from refusal
        if selection is not None and provenance is not None:
            admitted.append((selection, provenance))
        return admitted

    def _chain_decision_routes(
        self, routes: List[Dict[str, Any]], *, local_only: bool
    ) -> List[Dict[str, Any]]:
        """The decision routes automatic routing may use (#2652).

        With ``[llm] route_priority`` declared, only routes whose vendor is in
        that approved chain, plus, under local-only privacy, the local routes it
        forces (which chat routing authorizes too). The chat mandate is a chat
        choice and authorizes nothing here. Without a chain, every decision
        route, as before. An admitted override or an explicit
        ``[llm] decision_route`` is the writer's choice and is not narrowed
        here.
        """
        config = getattr(self, "config", None) or {}
        if not route_priority_entries(config):
            return routes
        approved = approved_vendors(
            config, getattr(self, "providers", None) or [], parse_denied_providers(config)
        )
        kept = [
            p for p in routes
            if p.get("vendor") in approved or (local_only and p.get("is_local"))
        ]
        if routes and not kept:
            raise DecisionUnavailable(
                UnavailableReason.NO_ROUTE,
                "no decision route is in the approved chain ([llm] "
                f"route_priority: {', '.join(sorted(approved)) or 'empty'}); "
                "name one with [llm] decision_route or route_priority",
            )
        return kept

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
        """Discover one route's decision models, then run its pin canary.

        The listing runs under the per-route discovery deadline (#3646): a
        route that never answers is a failed discovery, not a stalled
        reconcile. A route that did not answer this discovery call (this
        listing overran, or was skipped because an earlier step did) gets no
        canary either: it would be one more request to a silent route and one
        more timeout to wait out. Its pin keeps its last verification, marked
        stale. The canary otherwise keeps its own
        ``decision_canary_timeout_seconds``.
        """
        name = str(provider.get("name"))
        state = _route_state(provider)
        async with self._decision_lock(name):
            if not force and not state.needs_discovery:
                return
            silent: Optional[DiscoveryStepTimeout] = None
            try:
                async with self._discovery_deadline(
                    str(provider.get("vendor") or name.split(":", 1)[0]),
                    name,
                    "decision discovery",
                ):
                    discovered = await await_execution_work(
                        self, lambda: provider["adapter"].list_decision_models(provider.get("client"))
                    )
            except asyncio.CancelledError:
                raise
            except ExecutionAuthorityError:
                raise
            except DiscoveryStepTimeout as exc:
                if is_execution_control_error(exc):
                    raise
                # The deadline logged and recorded it; keep the last list.
                silent = exc
                state.record_discovery_failure()
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
            if state.config.pin is None:
                return
            if silent is not None:
                state.canary_stale_since = state.canary_stale_since or time.time()
                logger.debug("Decision pin canary for %s not sent: %s", name, silent)
                return
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
        # Admitted before discovery or dispatch, so a refused override reaches
        # no route (#2652).
        admitted = self._admit_decision_override(model_override, selector)
        effective_local_only = self._effective_force_local_only(local_only)
        context = self._resolve_invocation_context(session_id=session_id)
        footprint = RequestFootprint.of(snapshot)

        decision_routes = self._decision_routes()
        if selector is None and self._decision_config.route is None:
            decision_routes = self._chain_decision_routes(
                decision_routes, local_only=effective_local_only
            )
        routes = candidate_routes(
            decision_routes,
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
                # The provider policy's dispatch check, before the adapter
                # runs (#2652).
                try:
                    self._guard_dispatch(
                        str(provider.get("name")),
                        candidate.info.id,
                        path="decide",
                        force_local_only=effective_local_only,
                        admitted=admitted,
                    )
                except ProviderPolicyRefusal as refusal:
                    raise DecisionOverrideRefused(refusal) from refusal
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
            except BaseException as accounting_error:
                # This finite attempt owns BOTH original body and recording
                # outcomes. No commit marker is needed for native authority
                # or retirement control, nor can it be assumed caller-local.
                raise execution_terminal_error(
                    error, accounting_error, retain_originals=True,
                    prefer_cancellation=True,
                )
