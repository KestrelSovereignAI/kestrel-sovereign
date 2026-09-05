"""Typed registration for ACTION sources whose durable payload is always elided.

The dispatcher normally holds the agent privacy-transition lock while it
projects and persists an event.  An always-elided ACTION has no mode-dependent
projection: only a fixed marker and the ordinary source-event identity are
durable, while the live handler still receives the validated in-memory
payload.  That lets a control action such as cooperative Stop reach work which
itself holds the privacy lock, without creating a stale-privacy persistence
window. Its rate admission is also durable, and execution requires exactly one
live runtime owner because the handler's active-work inventory is local.
"""

from dataclasses import dataclass

from kestrel_sdk.signals import SourceRegistration


@dataclass
class AlwaysElidedActionSourceRegistration(SourceRegistration):
    """Always-elided, restart-bounded ACTION with one live runtime inventory."""


__all__ = ["AlwaysElidedActionSourceRegistration"]
