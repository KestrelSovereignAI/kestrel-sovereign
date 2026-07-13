"""Iron Rule gate for forged features.

The Iron Rule (see ``docs/principles/KESTREL_CONSTITUTION.md``): *"Each layer may
narrow the permissions granted by layers above it, but may never widen them."*
A forged feature is the lowest of the low — an agent extending itself — so it may
only compose capabilities the platform already grants the agent, narrowed. A spec
that requests a capability the agent does not hold (a *widen*) is rejected at
validation time, mirroring Book III Section 3's "narrow only, never widen."

This module is intentionally pure (no agent, no I/O) so the gate is trivially
unit-testable: pass the requested capabilities and the set the agent holds, get a
verdict back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set

# Capabilities every sovereign agent holds by default — reading/writing its own
# memory and graph, generating with its own LLM, reading its own audit trail,
# scheduling its own work, and searching the web. A forged feature composing
# only these is always a narrowing (it can request a subset, never more).
BASELINE_CAPABILITIES: frozenset = frozenset({
    "memory_read",
    "memory_write",
    "graph_read",
    "graph_write",
    "storage_read",
    "storage_write",
    "llm_generate",
    "audit_read",
    "schedule_tasks",
    "web_search",
})

# Capabilities that touch the world beyond the agent's own data. The platform
# only grants these when the corresponding feature/grant is actually present
# (Amendment IX governs the host-touching ones). A forged feature requesting one
# the agent does not hold is a widen attempt and is rejected.
PRIVILEGED_CAPABILITIES: frozenset = frozenset({
    "network_outbound",
    "filesystem_read",
    "filesystem_write",
    "shell_execution",
    "spawn_agent",
    "wallet_spend",
})

# The full catalogue the forge understands. A requested capability outside this
# set is not "widened" so much as *unknown* — the forge cannot compose something
# the platform has no concept of — and is rejected distinctly.
KNOWN_CAPABILITIES: frozenset = BASELINE_CAPABILITIES | PRIVILEGED_CAPABILITIES


@dataclass(frozen=True)
class IronRuleVerdict:
    """Result of running the Iron Rule gate over a forge spec's permissions."""

    valid: bool
    requested: List[str]
    granted: List[str]
    # Requested capabilities the agent does NOT hold — the widen attempts that
    # the Iron Rule rejects.
    widened: List[str] = field(default_factory=list)
    # Requested names the platform does not define at all.
    unknown: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "requested": self.requested,
            "granted": self.granted,
            "widened": self.widened,
            "unknown": self.unknown,
            "reason": self.reason,
        }


def _normalize(caps: Optional[Iterable[str]]) -> List[str]:
    """Lower-case, de-dup, and drop blanks while preserving first-seen order."""
    seen: Set[str] = set()
    out: List[str] = []
    for cap in caps or []:
        if not isinstance(cap, str):
            continue
        name = cap.strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def validate_narrowing(
    requested: Optional[Iterable[str]],
    granted: Optional[Iterable[str]],
    known: Optional[Iterable[str]] = None,
) -> IronRuleVerdict:
    """Return the Iron Rule verdict for ``requested`` against ``granted``.

    A spec is valid only when every requested capability is (a) a known platform
    capability and (b) already held by the agent (``granted``). Any requested
    capability outside ``granted`` is a widen attempt and fails the gate; any
    outside ``known`` is an unknown capability and also fails. Requesting a
    strict subset of the granted set (a narrowing) — including requesting nothing
    at all — passes.
    """
    known_set = set(_normalize(known)) if known is not None else set(KNOWN_CAPABILITIES)
    req = _normalize(requested)
    grant = _normalize(granted)
    grant_set = set(grant)

    unknown = [c for c in req if c not in known_set]
    # A capability that is unknown is reported only as unknown, not also as
    # widened, so the two failure modes stay distinct in the verdict.
    widened = [c for c in req if c in known_set and c not in grant_set]

    if unknown:
        reason = (
            "Rejected: spec requests capability the platform does not define: "
            + ", ".join(sorted(unknown))
            + ". A forged feature can only compose known platform capabilities."
        )
        return IronRuleVerdict(False, req, grant, widened, unknown, reason)

    if widened:
        reason = (
            "Rejected by the Iron Rule (narrow only, never widen): spec requests "
            "capability the agent does not hold: "
            + ", ".join(sorted(widened))
            + ". A forged feature may only narrow the capabilities the platform "
            "already grants the agent."
        )
        return IronRuleVerdict(False, req, grant, widened, unknown, reason)

    reason = (
        "Accepted: every requested capability is a narrowing of the "
        f"{len(grant_set)} capability(ies) the agent already holds."
    )
    return IronRuleVerdict(True, req, grant, widened, unknown, reason)
