"""Per-call route and model resolution for decisions (§5.2, §5.3, §6).

Two synchronous steps around the service's (async) discovery:

1. :func:`candidate_routes` applies ``decision_route``, the caller's override
   and the privacy filter. Nothing may contact a route this step removed.
2. :func:`select_candidate` walks the surviving routes in order and returns the
   first that passes model pick, calibration and fit, recording a typed
   rejection for every route it skips.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from typing import Any, Collection, List, Mapping, Optional, Sequence

from kestrel_sdk.llm.decisions import (
    DecisionModelInfo,
    DecisionUnavailable,
    RejectionReason,
    RouteRejection,
    UnavailableReason,
)

from .config import DecisionRouteConfig, DecisionSelector, DecisionServiceConfig
from .fit import RequestFootprint, misfit
from .thresholds import ThresholdBook, model_key


class PinStatus(str, enum.Enum):
    UNCHECKED = "unchecked"
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


@dataclass
class RouteDecisionState:
    """Mutable decision state kept on one route dict (``"decision_state"``).

    ``models`` is ``None`` until discovery has succeeded once; a later failed
    discovery keeps the last list and sets ``discovery_stale_since``.
    """

    config: DecisionRouteConfig
    models: Optional[tuple[DecisionModelInfo, ...]] = None
    discovery_stale_since: Optional[float] = None
    pin_status: PinStatus = PinStatus.UNCHECKED
    pin_reason: Optional[str] = None
    canary_stale_since: Optional[float] = None
    pin_info: Optional[DecisionModelInfo] = None

    @property
    def needs_discovery(self) -> bool:
        if self.config.pin is not None:
            return self.pin_status is PinStatus.UNCHECKED
        return self.models is None

    def record_discovery(self, models: Sequence[DecisionModelInfo]) -> None:
        self.models = tuple(models)
        self.discovery_stale_since = None

    def record_discovery_failure(self) -> None:
        if self.discovery_stale_since is None:
            self.discovery_stale_since = time.time()


@dataclass(frozen=True)
class Candidate:
    provider: Mapping[str, Any]
    info: DecisionModelInfo
    model_key: str


def _unavailable(reason: UnavailableReason, message: str, **kw: Any) -> DecisionUnavailable:
    return DecisionUnavailable(reason, message, **kw)


def _match_route_selector(
    providers: Sequence[Mapping[str, Any]], selector: str
) -> List[Mapping[str, Any]]:
    if ":" in selector:
        return [p for p in providers if p.get("name") == selector]
    return [p for p in providers if p.get("vendor") == selector]


def _conflicts(route_selector: str, override: DecisionSelector) -> bool:
    vendor, _, route = route_selector.partition(":")
    if override.vendor != vendor:
        return True
    return bool(route) and override.route is not None and override.route != route


def _supports_decisions(provider: Mapping[str, Any]) -> bool:
    capabilities = provider.get("capabilities") or {}
    return bool(capabilities.get("supports_decisions"))


def candidate_routes(
    providers: Sequence[Mapping[str, Any]],
    *,
    service_config: DecisionServiceConfig,
    selector: Optional[DecisionSelector],
    local_only: bool,
) -> List[Mapping[str, Any]]:
    """Steps 1–2.3: disabled, decision_route, override, privacy. No I/O."""

    if service_config.disabled:
        raise _unavailable(UnavailableReason.DISABLED, 'decisions are disabled ([llm] decision_route = "none")')

    routes = [p for p in providers if _supports_decisions(p)]
    if service_config.route is not None:
        routes = _match_route_selector(routes, service_config.route)
        if not routes:
            raise _unavailable(
                UnavailableReason.NO_ROUTE,
                f"[llm] decision_route {service_config.route!r} matches no "
                "configured route that supports decisions",
            )

    if selector is not None:
        if service_config.route is not None and _conflicts(service_config.route, selector):
            raise _unavailable(
                UnavailableReason.SELECTOR_CONFLICT,
                f"model_override names {selector.vendor}"
                f"{':' + selector.route if selector.route else ''}, but "
                f"[llm] decision_route is {service_config.route!r}",
            )
        routes = [p for p in routes if selector.matches_route(p)]
        if not routes:
            raise _unavailable(
                UnavailableReason.NO_ROUTE,
                f"model_override matches no configured decision route (vendor "
                f"{selector.vendor!r}); bare model ids are not accepted, use "
                "<vendor>/<model>",
            )

    if local_only:
        routes = [p for p in routes if p.get("is_local")]
        if not routes:
            raise _unavailable(
                UnavailableReason.NO_LOCAL_ROUTE,
                "privacy requires a local route and no local decision route is configured",
            )
    return routes


def _pick_model(
    state: RouteDecisionState, selector: Optional[DecisionSelector]
) -> tuple[Optional[DecisionModelInfo], Optional[RejectionReason], Optional[str]]:
    wanted = selector.model if selector is not None else None
    if state.config.pin is not None:
        if state.pin_status is not PinStatus.VERIFIED or state.pin_info is None:
            return None, RejectionReason.UNVERIFIED_PIN, state.pin_reason or "pin not verified"
        if wanted is not None and wanted != state.config.pin:
            return None, RejectionReason.PIN_CONFLICT, f"route pins {state.config.pin!r}"
        return state.pin_info, None, None

    models = state.models or ()
    if not models:
        return None, RejectionReason.NO_MODELS, "discovery found no decision models"
    if wanted is not None:
        for info in models:
            if info.id == wanted:
                return info, None, None
        return None, RejectionReason.NOT_SERVED, f"route does not serve {wanted!r}"
    survivors = [
        info for info in models
        if not state.config.hints or any(h in info.id for h in state.config.hints)
    ]
    if not survivors:
        return None, RejectionReason.NO_MODELS, "no discovered model matches decision_hints"
    if len(survivors) > 1:
        names = ", ".join(sorted(info.id for info in survivors))
        return None, RejectionReason.AMBIGUOUS_MODEL, f"several decision models: {names}"
    return survivors[0], None, None


def select_candidate(
    routes: Sequence[Mapping[str, Any]],
    *,
    footprint: RequestFootprint,
    selector: Optional[DecisionSelector],
    thresholds: ThresholdBook,
    caller: str,
    question_ids: Collection[str],
) -> Candidate:
    """Step 3–4: the first route that passes every check, or NO_CANDIDATE."""

    rejections: List[RouteRejection] = []
    for provider in routes:
        name = str(provider.get("name"))
        state: RouteDecisionState = provider["decision_state"]
        info, reason, detail = _pick_model(state, selector)
        if info is None:
            assert reason is not None
            rejections.append(RouteRejection(route=name, reason=reason, detail=detail))
            continue
        key = model_key(name, info.id)
        if not thresholds.admits(caller, key, question_ids):
            rejections.append(
                RouteRejection(
                    route=name,
                    reason=RejectionReason.NOT_CALIBRATED,
                    model=info.id,
                    detail=f"no complete [decisions.thresholds.{caller}] entry",
                )
            )
            continue
        override = state.config.context_limit if state.config.pin is not None else None
        problem = misfit(info, footprint, context_limit_override=override)
        if problem is not None:
            rejections.append(
                RouteRejection(route=name, reason=RejectionReason.NO_FIT, model=info.id, detail=problem)
            )
            continue
        return Candidate(provider=provider, info=info, model_key=key)

    summary = "; ".join(f"{r.route}: {r.reason.value}" for r in rejections)
    raise DecisionUnavailable(
        UnavailableReason.NO_CANDIDATE,
        f"no decision route can take this request ({summary})",
        rejections=rejections,
    )
