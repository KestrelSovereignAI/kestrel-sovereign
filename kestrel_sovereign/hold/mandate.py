"""Mandate-scoped Hold: an ancestor latches its signed descendants (#3168).

Hold authority is spawn's mandate relation, not a new model (#3135). A parent
may Hold a DID only when that DID is in its authoritative descendant set, the
one :meth:`AgentManager.authoritative_descendant_lease` proves from verified
parent-signed spawn receipts and active restart witnesses. Nothing else
confers it: not causation, not display metadata such as
``kestrel.orchestrator``, and not the unsigned ``_parent_children`` cache.

Three consequences follow from the set alone, with no special case:

* a child never Holds its parent, because a parent is not its descendant;
* a peer never Holds a peer, because there is no mandate between them;
* an expired, tampered, ambiguous, cyclic, or wrong-parent lineage fails
  closed, because the query either omits the edge or refuses the graph.

Authorization and the latch write happen under one topology execution lease,
so a concurrent terminate, unspawn, or mandate withdrawal cannot land between
"is my descendant" and "latched". Each holder owns exactly one latch per
target, keyed by ``(target, holder)``: two ancestors never share one, and no
holder can release a latch the sovereign or another holder set.
"""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator, Optional

from .state import (
    HoldAuthority,
    HoldMutation,
    HoldScope,
    HoldState,
)


class MandateHoldRefusal(RuntimeError):
    """A descendant Hold the holder's signed lineage does not authorize."""

    NOT_A_DESCENDANT = "not_a_signed_descendant"
    LINEAGE_UNVERIFIABLE = "lineage_unverifiable"
    NO_AUTHORITY_SOURCE = "no_lineage_authority"

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _require_did(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a concrete DID")
    return value


@asynccontextmanager
async def _authorized_descendant(
    manager: Any, holder_did: str, target_did: str
) -> AsyncIterator[None]:
    """Hold the lease with ``target_did`` proven a signed descendant."""

    from kestrel_sovereign.multi_agent.agent_manager import SpawnAuthorityGraphError

    open_lease = getattr(manager, "authoritative_descendant_lease", None)
    if not callable(open_lease):
        raise MandateHoldRefusal(
            MandateHoldRefusal.NO_AUTHORITY_SOURCE,
            "No durable spawn lineage is available to authorize a descendant Hold",
        )
    async with AsyncExitStack() as lease:
        try:
            descendants = await lease.enter_async_context(open_lease(holder_did))
        except SpawnAuthorityGraphError as error:
            raise MandateHoldRefusal(
                MandateHoldRefusal.LINEAGE_UNVERIFIABLE,
                "Signed spawn lineage could not be verified; refusing to Hold",
            ) from error
        if target_did not in {descendant.agent_id for descendant in descendants}:
            raise MandateHoldRefusal(
                MandateHoldRefusal.NOT_A_DESCENDANT,
                "The target is not a signed descendant of this agent",
            )
        yield


async def hold_descendant(
    *,
    manager: Any,
    store: Any,
    holder_did: str,
    target_did: str,
    reason: str,
    operation_id: str,
) -> HoldMutation:
    """Latch ``target_did`` under ``holder_did``'s verified mandate."""

    holder = _require_did(holder_did, "holder_did")
    target = _require_did(target_did, "target_did")
    async with _authorized_descendant(manager, holder, target):
        return await store.set_hold(
            scope=HoldScope.MANDATE,
            target_id=target,
            holder_id=holder,
            actor_id=holder,
            reason=reason,
            operation_id=operation_id,
            authority=HoldAuthority.MANDATE,
        )


async def release_descendant_hold(
    *,
    manager: Any,
    store: Any,
    holder_did: str,
    target_did: str,
    reason: str,
    operation_id: str,
) -> Optional[HoldMutation]:
    """Release the latch ``holder_did`` itself set on ``target_did``.

    Still requires the verified mandate: a holder whose mandate lapsed has no
    authority left, and its latch stays until the sovereign releases it.
    Returns ``None`` when this holder has no latch on the target.
    """

    holder = _require_did(holder_did, "holder_did")
    target = _require_did(target_did, "target_did")
    async with _authorized_descendant(manager, holder, target):
        current: Optional[HoldState] = await store.get_hold(
            HoldScope.MANDATE, target, holder_id=holder
        )
        if current is None:
            return None
        return await store.release_hold(
            scope=HoldScope.MANDATE,
            target_id=target,
            holder_id=holder,
            actor_id=holder,
            reason=reason,
            operation_id=operation_id,
            expected_hold_receipt_id=current.hold_receipt_id,
            authority=HoldAuthority.MANDATE,
        )


__all__ = [
    "MandateHoldRefusal",
    "hold_descendant",
    "release_descendant_hold",
]
