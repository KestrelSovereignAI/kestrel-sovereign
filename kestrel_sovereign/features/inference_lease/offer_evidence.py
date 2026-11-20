"""Capture a normal turn's exact prepared offer for the isolated Kite verifier."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kestrel_sovereign.agent.invocation import (
    InvocationProvenance,
    current_invocation_id,
    current_invocation_provenance,
)
from kestrel_sovereign.tools.effect_journal import observe_tool_effect_result
from kestrel_sovereign.turn_scope import turn_scoped

if TYPE_CHECKING:
    from kestrel_sdk.tools.result import ToolResult

    from kestrel_sovereign.llm.inference_leases import (
        InferenceLeaseCoordinator,
        PreparedInferenceLeaseOffer,
    )
    from kestrel_sovereign.tools.effect_journal import _ToolEffectResultObservation


@dataclass(frozen=True)
class _Candidate:
    coordinator: InferenceLeaseCoordinator
    offer: PreparedInferenceLeaseOffer
    receipt: _ToolEffectResultObservation


@dataclass
class _OfferCapture:
    owner: object
    invocation_id: str
    provenance: InvocationProvenance
    active: bool = True
    candidate: _Candidate | None = None
    ambiguous: bool = False

    async def observation(self) -> dict[str, Any] | None:
        candidate = self.candidate
        if (
            self.active
            or self.ambiguous
            or candidate is None
            or not candidate.receipt.confirmed
        ):
            return None
        if not await candidate.coordinator.is_current_prepared_offer(candidate.offer):
            return None
        return {
            "actor": self.provenance.actor,
            "invocation_id": self.invocation_id,
            "owner_id": candidate.offer.request.owner_id,
            "tool": "inference_lease_offer",
            "outcome": "RETURNED_OK",
            "offer": candidate.offer.to_public_dict(),
        }


_capture: ContextVar[_OfferCapture | None] = ContextVar(
    "kestrel_current_prepared_offer_capture",
    default=None,
)


@contextmanager
def _bind(capture: _OfferCapture | None) -> Iterator[None]:
    token = _capture.set(capture)
    try:
        yield
    finally:
        _capture.reset(token)


def _capture_for_owner(owner: object) -> _OfferCapture | None:
    capture = _capture.get()
    return (
        capture
        if capture is not None and capture.active and capture.owner is owner
        else None
    )


turn_scoped(
    "private_inference_offer_capture",
    variables=(_capture,),
    capture=_capture_for_owner,
    bind=_bind,
)


@contextmanager
def capture_prepared_offer(
    owner: object,
    invocation_id: str,
    provenance: InvocationProvenance,
) -> Iterator[_OfferCapture]:
    """Ask the original authenticated invoke to observe, never execute, a tool."""
    capture = _OfferCapture(owner, invocation_id, provenance)
    with _bind(capture):
        try:
            yield capture
        finally:
            capture.active = False


def observe_prepared_offer(
    owner: object,
    coordinator: InferenceLeaseCoordinator,
    offer: PreparedInferenceLeaseOffer,
    result: ToolResult,
) -> None:
    capture = _capture_for_owner(owner)
    if (
        capture is None
        or current_invocation_id() != capture.invocation_id
        or current_invocation_provenance() is not capture.provenance
    ):
        return
    receipt = observe_tool_effect_result(owner, result)
    if receipt.turn.invocation_id != capture.invocation_id:
        return
    if capture.candidate is not None:
        capture.ambiguous = True
        return
    capture.candidate = _Candidate(coordinator, offer, receipt)
