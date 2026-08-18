"""Per-agent constitution reanchor receipts: current pointer plus history.

A reanchor receipt answers a question about one governance *event*: under what
authority, from what source, verified how, this agent came to be governed by a
particular constitution hash. Events accumulate. That is why this module exists
alongside :func:`~kestrel_sovereign.constitution.genesis_audit.supersede_genesis_audit`
and mirrors it exactly — a genesis audit is *state* (pending or not), where
superseding is the right verb, but the receipt describing how that state changed
is not something the next change may destroy.

Before #2893, a superseded receipt's per-agent facts survived incidentally on
its own ``constitution_amendment_artifact`` node, because that node's id is the
hash of the artifact bytes — a v2→v3 reanchor writes a different artifact, so
the v2 node kept its copy. Making that node fleet-shareable moved those
per-agent fields off it (they are per-agent by nature: an operator filesystem
path, when *this* agent anchored, how *this* agent's trust root verified it), and
removed the only place a superseded receipt was retained. This module gives them
a home that is per-agent by construction, so the fix cannot reopen #2893.

The history lives on the agent node rather than in a fresh node of its own,
deliberately. A per-(agent, artifact) receipt *node* would be a fresh node
carrying free-text — precisely the channel ``privacy_wrapper`` default-denies
for ``constitution_amendment_artifact``. ``genesis_audit_history`` already
established that a governance ``*_history`` blob on the capability-gated agent
node is the reviewed home for this shape.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, MutableMapping

from kestrel_sovereign.constitution.genesis_audit import utc_timestamp

# The agent-node property holding the most recent receipt, and the append-only
# list of the ones it replaced. Named here so writers and the privacy
# classification cannot drift apart on a string literal.
CONSTITUTION_REANCHOR_KEY = "constitution_reanchor"
CONSTITUTION_REANCHOR_HISTORY_KEY = "constitution_reanchor_history"


def supersede_constitution_reanchor(
    properties: MutableMapping[str, Any],
    *,
    receipt: MutableMapping[str, Any],
    provenance: str,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Record ``receipt`` as current, preserving the receipt it replaces.

    ``properties`` is the agent node's property mapping, mutated in place the
    same way :func:`supersede_genesis_audit` mutates it, so both writers keep
    updating one node inside the reanchor's single transaction.

    The superseded receipt is stored verbatim under ``receipt`` rather than
    merged, because the two writers do not agree on field names for the same
    fact — ``setup`` records ``source_path`` where the runtime chat command
    records ``path``. Wrapping preserves what each writer actually claimed
    instead of inventing a reconciliation this function cannot justify.

    Returns the new current receipt.
    """
    changed_at = recorded_at or utc_timestamp()
    existing = properties.get(CONSTITUTION_REANCHOR_KEY)
    history = properties.get(CONSTITUTION_REANCHOR_HISTORY_KEY)
    history = list(history) if isinstance(history, list) else []
    if existing is not None:
        history.append(
            {
                "receipt": deepcopy(existing),
                "superseded_at": changed_at,
                "superseded_by_constitution_hash": receipt.get("new_hash"),
                "superseded_by_artifact_hash": receipt.get("signed_artifact_hash"),
                "provenance": provenance,
            }
        )
    if history:
        properties[CONSTITUTION_REANCHOR_HISTORY_KEY] = history

    current = dict(receipt)
    properties[CONSTITUTION_REANCHOR_KEY] = current
    return current
