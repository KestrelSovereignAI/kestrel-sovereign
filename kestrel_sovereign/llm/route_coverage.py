"""Which LLM routes the next turn would try, and what is known of each (#3912).

The ``llm_service`` and ``model_discovery`` health checks judge only the
routes in use, so a provider kept configured and idle on purpose (a standby
route) never degrades the agent, while a route a turn would try still does.

The evidence is passive. A check that sent a completion of its own to find
out whether a route works would bill a paid route on every tick, so every
chat attempt records how it ended (``LLMService._record_route_outcome``) and
the check reads that. Only whether the route answered, and when, is kept:
never an error message, which can repeat provider or request text.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple


@dataclass(frozen=True)
class RouteOutcome:
    """How the most recent real attempt on one route ended."""

    #: Whether the attempt succeeded. A refusal before any adapter ran, a
    #: provider error and a generation the gate rejected are all failures.
    answered: bool
    #: When the attempt ended (UTC).
    at: datetime


@dataclass(frozen=True)
class RouteState:
    """One route, with the passive evidence there is for it."""

    name: str
    vendor: Optional[str]
    #: The most recent real call on the route this session, if any.
    last_call: Optional[RouteOutcome] = None
    #: The startup reachability probe's status for the route
    #: (``reachable``/``unreachable``/``skipped``), if one was recorded.
    #: Only local routes are probed at startup.
    startup_probe: Optional[str] = None
    #: Why a permanent auth failure disabled the route this session.
    disabled_reason: Optional[str] = None
    #: The route refuses every call now, before any adapter runs: a private
    #: inference lease that is draining or whose expiry a call has seen.
    refusing_calls: bool = False


@dataclass(frozen=True)
class RouteCoverage:
    """The routes the next turn would try, and every other configured route."""

    #: The routes a turn with no per-call override would try, in order; the
    #: first is the primary, the rest its fallbacks. ``None`` when routing
    #: refused to select any route (``routing_error`` says why).
    in_use: Optional[Tuple[RouteState, ...]]
    #: Configured routes a permanent auth failure disabled this session.
    #: Never standby: a real call on each failed.
    disabled: Tuple[RouteState, ...] = ()
    #: Configured routes no turn tries until routing changes. Empty when
    #: routing refused, since the routes in use are then unknown.
    standby: Tuple[RouteState, ...] = ()
    routing_error: Optional[str] = None


__all__ = ["RouteCoverage", "RouteOutcome", "RouteState"]
