"""Bounded metrics for work dispositioned by durable Hold."""

from __future__ import annotations

import logging

from kestrel_sdk.metrics import PROMETHEUS_AVAILABLE

logger = logging.getLogger(__name__)


if PROMETHEUS_AVAILABLE:
    from kestrel_sdk.metrics import REGISTRY
    from prometheus_client import Counter

    HELD_WORK_DISPOSITIONS_TOTAL = Counter(
        "kestrel_held_work_dispositions_total",
        "Number of work admissions dispositioned by durable Hold",
        ["disposition", "source"],
        registry=REGISTRY,
    )
else:
    HELD_WORK_DISPOSITIONS_TOTAL = None  # type: ignore[assignment]


def record_held_work_disposition(*, disposition: str, source: str) -> None:
    """Record one bounded Hold disposition; metrics remain optional."""

    if HELD_WORK_DISPOSITIONS_TOTAL is None:
        return
    try:
        HELD_WORK_DISPOSITIONS_TOTAL.labels(
            disposition=disposition,
            source=source,
        ).inc()
    except Exception:
        logger.exception(
            "Failed to record held-work disposition: disposition=%s source=%s",
            disposition,
            source,
        )


__all__ = ["record_held_work_disposition"]
