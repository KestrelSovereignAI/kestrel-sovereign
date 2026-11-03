"""How the most recent real call on each LLM route ended (#3873).

The ``llm_service`` health check reports whether the routes the agent's next
turn would try are working. A check that sent a completion of its own to find
out would bill a paid route on every tick, so it reads what the agent's own
calls already observed instead: every provider attempt records here how it
ended, from the two places an attempt finishes
(``LLMService._run_provider_attempt`` and
``StreamingMixin._stream_adapter_with_usage``).

Only the outcome is kept: whether the route answered, when, and for a failure
the exception's class and HTTP status. Never the exception's message, which
can repeat provider or request text.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RouteCall:
    """How the most recent attempt on one route ended."""

    route: str
    #: Whether the provider answered. An answer the generation gate then
    #: rejected still counts: the route itself worked.
    answered: bool
    at: datetime
    error_type: Optional[str] = None
    status_code: Optional[int] = None

    def to_dict(self, *, now: datetime) -> Dict[str, Any]:
        """The health surface's view of this call, aged against ``now``."""

        return {
            "status": "ok" if self.answered else "failed",
            "evidence": "last_call",
            "at": self.at.isoformat(),
            "age_seconds": round(max(0.0, (now - self.at).total_seconds()), 1),
            "error_type": self.error_type,
            "status_code": self.status_code,
        }


def _status_code(error: BaseException) -> Optional[int]:
    for name in ("status_code", "status"):
        value = getattr(error, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


class RouteCallLedger:
    """The latest :class:`RouteCall` per route name, for one ``LLMService``."""

    def __init__(self) -> None:
        self._latest: Dict[str, RouteCall] = {}

    def record(
        self,
        route: str,
        error: Optional[BaseException] = None,
        *,
        at: Optional[datetime] = None,
    ) -> None:
        """Record that an attempt on ``route`` ended, failing with ``error``."""

        self._latest[route] = RouteCall(
            route=route,
            answered=error is None,
            at=at or datetime.now(timezone.utc),
            error_type=None if error is None else type(error).__name__,
            status_code=None if error is None else _status_code(error),
        )

    def latest(self, route: str) -> Optional[RouteCall]:
        return self._latest.get(route)


def record_route_call(
    host: Any, route: Any, error: Optional[BaseException] = None
) -> None:
    """Record one ended attempt on ``host``'s ledger, when it keeps one.

    A host without a ledger (a streaming-only consumer, a test double built
    without the constructor) records nothing; its routes read as unprobed.
    """

    ledger = getattr(host, "route_calls", None)
    if isinstance(ledger, RouteCallLedger) and isinstance(route, str) and route:
        ledger.record(route, error)


__all__ = ["RouteCall", "RouteCallLedger", "record_route_call"]
