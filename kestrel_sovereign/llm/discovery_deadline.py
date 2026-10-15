"""The deadline each route's model-discovery steps run under (#3646).

Model discovery asks every configured route for its chat catalog, then
reconciles each route's embedding and decision capabilities. A route that
accepts the connection and never answers (an Ollama daemon whose model slot
another process holds, a wedged proxy) used to stall that whole pass, and with
it ``!model-list`` and every turn whose route still resolved ``auto`` through
discovery. Nothing returned until the process was killed.

Each of those steps now runs, per route, under one deadline:
``[llm] discovery_route_deadline_seconds``. A step that overruns it is
cancelled and becomes that route's discovery failure; the rest of discovery
returns without it. Within one discovery call the route's later steps are not
sent, so a silent route costs that call one deadline, not one per step.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

#: The ``[llm]`` key that sets the deadline.
DISCOVERY_ROUTE_DEADLINE_KEY = "discovery_route_deadline_seconds"

#: Seconds one route gets for one discovery step when ``[llm]`` sets none.
#: A healthy route's catalog, embedding or decision listing answers in well
#: under this; a route that has not answered by then is treated as down.
DEFAULT_DISCOVERY_ROUTE_DEADLINE_SECONDS = 5.0


class DiscoveryStepTimeout(TimeoutError):
    """One route's discovery step did not finish within its deadline.

    ``skipped`` marks a step that was never sent, because the same route
    overran an earlier step of the same discovery call.
    """

    def __init__(
        self,
        *,
        vendor: str,
        route: str,
        step: str,
        seconds: float,
        cause: Optional[BaseException] = None,
        skipped: bool = False,
    ) -> None:
        self.vendor = vendor
        self.route = route
        self.step = step
        self.seconds = seconds
        self.skipped = skipped
        if skipped:
            message = (
                f"{step} on {route} skipped: the route did not answer an "
                "earlier discovery step"
            )
        elif cause is None or not str(cause):
            message = f"{step} on {route} did not finish within {seconds:g}s"
        else:
            # The step's own bound fired first (the Ollama adapter bounds its
            # listing calls with the same value), so report what it said.
            message = f"{step} on {route} timed out: {cause}"
        super().__init__(message)


def discovery_route_deadline_seconds(llm_cfg: Any) -> float:
    """The per-route discovery deadline ``llm_cfg`` (the ``[llm]`` section) sets.

    Absent means :data:`DEFAULT_DISCOVERY_ROUTE_DEADLINE_SECONDS`. Anything
    other than a positive, finite number raises: a typo must not leave
    discovery unbounded, nor bound it at zero.
    """
    cfg: Mapping[str, Any] = llm_cfg if isinstance(llm_cfg, Mapping) else {}
    raw = cfg.get(DISCOVERY_ROUTE_DEADLINE_KEY, DEFAULT_DISCOVERY_ROUTE_DEADLINE_SECONDS)
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not math.isfinite(raw)
        or raw <= 0
    ):
        raise ValueError(
            f"[llm] {DISCOVERY_ROUTE_DEADLINE_KEY} must be a positive number of "
            f"seconds, got {raw!r}"
        )
    return float(raw)
