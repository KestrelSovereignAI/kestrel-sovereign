"""Typed registration for ACTION sources that act only on in-flight work.

Cooperative Stop is the motivating case (#3169).  Such an action begins no
work of its own, so three dispatcher policies that exist to govern *new* work
would defeat it:

* **Privacy transition lock.** Every cognition turn holds the agent's privacy
  transition lock for its whole body, and the dispatcher normally takes the
  same lock while it projects and persists the durable event.  A Stop that
  waited there could only ever run after the turn it was meant to stop.  An
  in-flight control action therefore persists a fixed marker (no payload, no
  caller, no causation chain), which has no privacy-mode-dependent projection
  and so needs no lock.  The live handler still receives the validated
  in-memory payload.
* **Hold.** Hold declines to *begin* work (#3163).  Stop acts on work that is
  already running and needs no new turn, so a held agent must still receive it.
* **Durable replay.** The marker is the complete durable projection; nothing
  consumes it.  The dispatcher coalesces a repeated source event id and never
  re-executes it; whether the action itself completed is the source's own
  durable evidence to decide (peer Stop consults its Stop receipt).

Everything else — validation, causation cycle and depth TTL, per-source rate
limiting, the ``signal_log`` outcome audit — applies unchanged.
"""

from dataclasses import dataclass

from kestrel_sdk.signals import SourceRegistration


@dataclass
class InFlightControlActionRegistration(SourceRegistration):
    """A trusted ACTION on already-running work; see the module docstring."""


__all__ = ["InFlightControlActionRegistration"]
