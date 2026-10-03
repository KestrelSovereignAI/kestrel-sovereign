"""Typed registration for ACTION sources that act only on in-flight work.

Cooperative Stop is the motivating case (#3169).  Such an action begins no
work of its own, so the dispatcher treats it differently in two ways:

* **Hold.** Hold declines to *begin* work (#3163).  Stop acts on work that is
  already running and needs no new turn, so a held agent must still receive it.
* **Durable projection.** An in-flight control action persists a fixed marker
  (no payload, no caller, no causation chain); the live handler still
  receives the validated in-memory payload.  The marker is the complete
  durable projection; nothing consumes it.  The dispatcher coalesces a
  repeated source event id and never re-executes it; whether the action
  itself completed is the source's own durable evidence to decide (peer Stop
  consults its Stop receipt).

Everything else — validation, causation cycle and depth TTL, per-source rate
limiting, the durable persistence gate, the ``signal_log`` outcome audit —
applies unchanged.  The gate does not queue a Stop behind the turn it is meant
to stop: turns hold the privacy-transition lock, not the gate, and only a
privacy transition holds the gate exclusively (#3316).
"""

from dataclasses import dataclass

from kestrel_sdk.signals import SourceRegistration


@dataclass
class InFlightControlActionRegistration(SourceRegistration):
    """A trusted ACTION on already-running work; see the module docstring."""


__all__ = ["InFlightControlActionRegistration"]
