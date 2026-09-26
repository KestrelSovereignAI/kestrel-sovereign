"""Durable correlation-key contract adapter.

Durable consumers select on the stored event, and ANONYMOUS storage persists
a PII-anonymized payload. A system-generated identifier that a consumer
correlates on (the wait reconciler's ``payload.ref``) must survive that
projection verbatim, or a handle such as ``ci:owner/repo#12345`` is stored as
``ci:owner/repo#[ZIP_REDACTED]`` and its consumer is never matched (#3295).

Until the SDK grows a first-class constructor field, sovereign producers use
this subclass. It remains a normal ``Signal`` for the dispatcher and
registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kestrel_sdk.signals import Signal


@dataclass
class SignalWithDurableCorrelation(Signal):
    """Signal envelope naming payload keys that are correlation identifiers.

    Each named key must hold a string the producer wrote itself — never user
    or third-party content — because the durable anonymization projection
    persists it unchanged. Payload-eliding privacy modes still elide it.
    """

    durable_correlation_keys: frozenset[str] = frozenset()


def durable_correlation_values(signal: Signal) -> dict[str, str]:
    """The ``signal`` payload values exempt from durable anonymization.

    Only string values under keys the envelope names are returned; anything
    else is anonymized like the rest of the payload.
    """
    keys = getattr(signal, "durable_correlation_keys", frozenset())
    payload: Any = signal.payload
    if not keys or not isinstance(payload, dict):
        return {}
    return {
        key: payload[key]
        for key in keys
        if isinstance(payload.get(key), str)
    }


__all__ = [
    "SignalWithDurableCorrelation",
    "durable_correlation_values",
]
