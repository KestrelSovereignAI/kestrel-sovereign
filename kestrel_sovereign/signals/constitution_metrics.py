"""Constitutional-injection Prometheus counters.

Phase 1 ships three counters per CONSTITUTION_INJECTION.md §"Phase 1
Core primitive":

- `kestrel_constitution_echo_verified_total{source}` — fires when a
  COGNITION dispatch with `require_constitution_echo=True` got a
  VERIFIED canary back.
- `kestrel_constitution_echo_missing_total{source}` — same dispatches
  whose canary came back MISSING (the dispatch fails with
  `constitution_not_received`).
- `kestrel_doctrine_bundle_drift_total{source}` — pre-dispatch drift
  detection refused the dispatch with `doctrine_bundle_drift`.

Counters are bucketed by `source` only — keeping label cardinality
bounded (the registry caps the source set; an attacker cannot
explode label dimensionality by emitting arbitrary signals because
unknown sources DROP_VALIDATION before this code path).

The counters use the SDK's shared `REGISTRY` so a single
`/metrics` scrape yields a coherent surface (per the SDK metrics
module's design).

When `prometheus-client` is not installed (no `[metrics]` extra),
all handles are None and `record_*` calls are no-ops — callers
do not need to guard.

kestrel-sovereign#1137 chunk 1H.
"""

from __future__ import annotations

import logging
from typing import Optional

from kestrel_sdk.metrics import PROMETHEUS_AVAILABLE

logger = logging.getLogger(__name__)


if PROMETHEUS_AVAILABLE:
    from prometheus_client import Counter

    from kestrel_sdk.metrics import REGISTRY

    CONSTITUTION_ECHO_VERIFIED_TOTAL = Counter(
        "kestrel_constitution_echo_verified_total",
        "Number of COGNITION dispatches with verified constitutional echo",
        ["source"],
        registry=REGISTRY,
    )
    CONSTITUTION_ECHO_MISSING_TOTAL = Counter(
        "kestrel_constitution_echo_missing_total",
        "Number of COGNITION dispatches that failed echo verification",
        ["source"],
        registry=REGISTRY,
    )
    DOCTRINE_BUNDLE_DRIFT_TOTAL = Counter(
        "kestrel_doctrine_bundle_drift_total",
        "Number of COGNITION dispatches refused for doctrine_bundle_drift",
        ["source"],
        registry=REGISTRY,
    )
else:
    CONSTITUTION_ECHO_VERIFIED_TOTAL = None  # type: ignore[assignment]
    CONSTITUTION_ECHO_MISSING_TOTAL = None  # type: ignore[assignment]
    DOCTRINE_BUNDLE_DRIFT_TOTAL = None  # type: ignore[assignment]


def record_echo_verified(source: str) -> None:
    """Record one VERIFIED echo. No-op when prometheus-client absent."""
    if CONSTITUTION_ECHO_VERIFIED_TOTAL is not None:
        try:
            CONSTITUTION_ECHO_VERIFIED_TOTAL.labels(source=source).inc()
        except Exception:
            logger.exception(
                "Failed to record echo_verified counter for source=%s",
                source,
            )


def record_echo_missing(source: str) -> None:
    """Record one MISSING echo. No-op when prometheus-client absent."""
    if CONSTITUTION_ECHO_MISSING_TOTAL is not None:
        try:
            CONSTITUTION_ECHO_MISSING_TOTAL.labels(source=source).inc()
        except Exception:
            logger.exception(
                "Failed to record echo_missing counter for source=%s",
                source,
            )


def record_doctrine_bundle_drift(source: str) -> None:
    """Record one bundle-drift refusal. No-op when prometheus-client absent."""
    if DOCTRINE_BUNDLE_DRIFT_TOTAL is not None:
        try:
            DOCTRINE_BUNDLE_DRIFT_TOTAL.labels(source=source).inc()
        except Exception:
            logger.exception(
                "Failed to record doctrine_bundle_drift counter for source=%s",
                source,
            )
