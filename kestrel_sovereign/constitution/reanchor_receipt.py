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
from typing import Any, Mapping, MutableMapping
import re

from kestrel_sovereign.constitution.genesis_audit import utc_timestamp

# The agent-node property holding the most recent receipt, and the append-only
# list of the ones it replaced. Named here so writers and the privacy
# classification cannot drift apart on a string literal.
CONSTITUTION_REANCHOR_KEY = "constitution_reanchor"
CONSTITUTION_REANCHOR_HISTORY_KEY = "constitution_reanchor_history"
MAX_REANCHOR_RECEIPT_HISTORY = 128


def is_constitution_hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def reanchor_prior_pointer_fields(pointer: Any, historical_hash: str | None) -> dict:
    """Keep an observed corrupt pointer separate from a typed prior digest.

    Recovery evidence is not signing authority. Both public writers must first
    inspect readable historical bytes and independently verify the new artifact.
    Existing malformed receipts are never normalized by this new-event builder.
    """
    if pointer is None or pointer == "" or is_constitution_hash(pointer):
        return {"old_hash": pointer or None}
    if not isinstance(pointer, str) or len(pointer) > 256 or not is_constitution_hash(historical_hash):
        raise ValueError("Malformed anchor pointer has unreadable historical governance; restore its exact prior pointer before signed repair")
    return {"old_hash": historical_hash, "repaired_constitution_pointer": pointer}


def _receipt_hash(value: Any, field: str) -> None:
    if not is_constitution_hash(value):
        raise ValueError(f"Malformed or unreadable historical governance reanchor receipt {field}; existing evidence is preserved")


def validate_constitution_reanchor_receipt(receipt: Any) -> None:
    """Validate evidence shape, not signature authority or legacy prose.

    Legacy receipts can predate signed artifacts. Preserve their facts rather
    than fabricating signatures; an optional hash must nevertheless be valid.
    """
    if not isinstance(receipt, Mapping):
        raise ValueError("Malformed or unreadable historical governance reanchor receipt; existing evidence is preserved")
    _receipt_hash(receipt.get("new_hash"), "new_hash")
    # Offline first-anchor receipts use None; runtime uses the literal none.
    # Both describe absence, never a substitute destination hash.
    if "old_hash" in receipt and receipt["old_hash"] not in (None, "none"):
        _receipt_hash(receipt["old_hash"], "old_hash")
    if "signed_artifact_hash" in receipt:
        _receipt_hash(receipt["signed_artifact_hash"], "signed_artifact_hash")


def validate_constitution_reanchor_evidence(properties: Mapping[str, Any], *, superseding: bool = False) -> list:
    """One admission rule for complete current/history reads and both writers."""
    history = properties.get(CONSTITUTION_REANCHOR_HISTORY_KEY, [])
    if not isinstance(history, list) or len(history) > MAX_REANCHOR_RECEIPT_HISTORY:
        raise ValueError("Malformed, unbounded or unreadable historical governance reanchor history; existing evidence is preserved")
    for entry in history:
        if not isinstance(entry, Mapping):
            raise ValueError("Malformed or unreadable historical governance reanchor history entry; existing evidence is preserved")
        validate_constitution_reanchor_receipt(entry.get("receipt"))
        _receipt_hash(entry.get("superseded_by_constitution_hash"), "superseded_by_constitution_hash")
        if "superseded_by_artifact_hash" in entry:
            _receipt_hash(entry["superseded_by_artifact_hash"], "superseded_by_artifact_hash")
    if CONSTITUTION_REANCHOR_KEY in properties:
        validate_constitution_reanchor_receipt(properties[CONSTITUTION_REANCHOR_KEY])
        if superseding and len(history) >= MAX_REANCHOR_RECEIPT_HISTORY:
            raise ValueError("Reanchor history cannot archive another receipt; existing evidence is preserved")
    return history


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
    # Complete admission precedes every in-place mutation, even a same-hash
    # repair whose genesis receipt does not need another history entry.
    history = deepcopy(validate_constitution_reanchor_evidence(properties, superseding=True))
    validate_constitution_reanchor_receipt(receipt)
    changed_at = recorded_at or utc_timestamp()
    existing = properties.get(CONSTITUTION_REANCHOR_KEY)
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
