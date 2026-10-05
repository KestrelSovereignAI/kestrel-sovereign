"""Decision configuration: per-route keys, service keys, and the override grammar.

See ``docs/architecture/llm/DECISIONS.md`` §5.1 (config) and §5.3 (grammar).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

DEFAULT_CANARY_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class DecisionRouteConfig:
    """Operator configuration for one route's decision surface.

    ``pin`` is an explicit model id (``decision_model = "<id>"``); ``None``
    means ``"auto"``. ``context_limit`` is only meaningful with a pin whose
    serving limit cannot be discovered.
    """

    pin: Optional[str] = None
    hints: tuple[str, ...] = ()
    context_limit: Optional[int] = None


def parse_route_decision_config(
    vendor: str, route: str, route_cfg: Mapping[str, Any]
) -> DecisionRouteConfig:
    """Validate the ``decision_*`` keys of one ``[llm.vendors.*.routes.*]`` block."""

    where = f"Route {vendor}:{route}"
    raw_model = route_cfg.get("decision_model", "auto")
    if not isinstance(raw_model, str) or not raw_model.strip():
        raise ValueError(f"{where}: decision_model must be \"auto\" or a model id")
    pin = None if raw_model.strip() == "auto" else raw_model.strip()

    raw_hints = route_cfg.get("decision_hints", [])
    if not isinstance(raw_hints, (list, tuple)) or not all(
        isinstance(hint, str) and hint.strip() for hint in raw_hints
    ):
        raise ValueError(f"{where}: decision_hints must be a list of non-empty strings")
    hints = tuple(hint.strip() for hint in raw_hints)

    raw_limit = route_cfg.get("decision_context_limit")
    if raw_limit is not None:
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, int) or raw_limit <= 0:
            raise ValueError(f"{where}: decision_context_limit must be a positive integer")
        if pin is None:
            raise ValueError(
                f"{where}: decision_context_limit applies only to a pinned "
                "decision_model; discovered models report their own limit"
            )
    return DecisionRouteConfig(pin=pin, hints=hints, context_limit=raw_limit)


@dataclass(frozen=True)
class DecisionServiceConfig:
    """``[llm] decision_route`` and ``decision_canary_timeout_seconds``.

    ``route`` is ``None`` for ``"auto"``, ``""`` for ``"none"`` (disabled),
    otherwise a ``<vendor>`` or ``<vendor>:<route>`` selector.
    """

    route: Optional[str] = None
    canary_timeout_seconds: float = DEFAULT_CANARY_TIMEOUT_SECONDS

    @property
    def disabled(self) -> bool:
        return self.route == ""


def parse_service_decision_config(llm_cfg: Mapping[str, Any]) -> DecisionServiceConfig:
    raw_route = llm_cfg.get("decision_route", "auto")
    if not isinstance(raw_route, str) or not raw_route.strip():
        raise ValueError(
            '[llm] decision_route must be "auto", "none", or "<vendor>[:<route>]"'
        )
    raw_route = raw_route.strip()
    if raw_route == "auto":
        route: Optional[str] = None
    elif raw_route == "none":
        route = ""
    else:
        if "/" in raw_route or raw_route.count(":") > 1:
            raise ValueError(
                '[llm] decision_route names a route, not a model: use "<vendor>" '
                'or "<vendor>:<route>"'
            )
        route = raw_route

    raw_timeout = llm_cfg.get(
        "decision_canary_timeout_seconds", DEFAULT_CANARY_TIMEOUT_SECONDS
    )
    if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)) or raw_timeout <= 0:
        raise ValueError("[llm] decision_canary_timeout_seconds must be a positive number")
    return DecisionServiceConfig(route=route, canary_timeout_seconds=float(raw_timeout))


@dataclass(frozen=True)
class DecisionSelector:
    """A parsed ``model_override`` (§5.3). ``route`` and ``model`` may be None."""

    vendor: str
    route: Optional[str] = None
    model: Optional[str] = None

    @property
    def route_name(self) -> Optional[str]:
        return f"{self.vendor}:{self.route}" if self.route else None

    def matches_route(self, provider: Mapping[str, Any]) -> bool:
        if provider.get("vendor") != self.vendor:
            return False
        return self.route is None or provider.get("route") == self.route


def parse_decision_selector(raw: str) -> DecisionSelector:
    """Parse ``<vendor>``, ``<vendor>:<route>``, ``<vendor>/<model>`` or
    ``<vendor>:<route>/<model>``.

    A bare model id is refused: one id can exist under several vendors with
    different calibration. ``cheap`` is refused: it names a chat model.
    """

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("model_override must be a non-empty selector")
    text = raw.strip()
    if text == "cheap":
        raise ValueError("model_override 'cheap' names a chat model, not a decision model")
    left, slash, model = text.partition("/")
    if slash and not model:
        raise ValueError(f"model_override {raw!r} has an empty model after '/'")
    vendor, colon, route = left.partition(":")
    if not vendor or (colon and not route) or ":" in route:
        raise ValueError(
            f"model_override {raw!r} must be <vendor>, <vendor>:<route>, "
            "<vendor>/<model> or <vendor>:<route>/<model>"
        )
    if not slash and not colon:
        # A single token: it is a vendor selector. Bare model ids are written
        # with a vendor (``<vendor>/<model>``); a token that is not a configured
        # vendor will simply match no route.
        return DecisionSelector(vendor=vendor)
    return DecisionSelector(vendor=vendor, route=route or None, model=model or None)
