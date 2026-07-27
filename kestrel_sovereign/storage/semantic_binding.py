"""Non-authorizing semantic metadata emitted by agent-bound storage facades."""

from __future__ import annotations

from dataclasses import dataclass

from kestrel_sovereign.knowledge import Visibility


@dataclass(frozen=True, slots=True)
class SemanticAssertionBinding:
    """The storage-owned fields an assertion adapter must not take from a tool.

    This is deliberately not an authorization capability.  The canonical
    assertion store still checks its private tenant capability on every write;
    this value only keeps adapters from accepting tenant, owner, or privacy
    fields as untrusted tool arguments.
    """

    tenant_id: str
    owning_agent_id: str
    privacy_classification: str
    release_policy_reference: str
    visibility: Visibility
