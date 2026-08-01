"""Opaque, one-shot authority for the fixed Kite core-erasure drill.

The release-evidence HTTP route receives a durable nonce receipt and exchanges
it for this capability.  Storage accepts no caller-selected operation string:
it consumes this exact capability once and derives its own correlation from
the binding retained here.
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
import hashlib
import re
import weakref

from kestrel_sovereign.knowledge.kite_evidence_signing import (
    claim_kite_evidence_nonce_receipt,
)


ERASURE_CORE_SNAPSHOT_OPERATION = "erasure_core_snapshot"
_NONCE_RE = re.compile(r"^[0-9a-f]{64}$")


class KiteErasureDrillAuthorityError(RuntimeError):
    """A Kite core-erasure capability was forged, malformed, or reused."""


class KiteErasureDrillCapability:
    """An opaque nominal capability for exactly one core-erasure operation."""

    __slots__ = ("__weakref__",)

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            "Kite erasure drill capabilities are issued only by the typed evidence endpoint"
        )


@dataclass(frozen=True, slots=True)
class _ErasureDrillBinding:
    operation: str
    operation_id: str
    correlation: str


_issued_capabilities: weakref.WeakKeyDictionary[
    KiteErasureDrillCapability, _ErasureDrillBinding
] = weakref.WeakKeyDictionary()
_endpoint_issuance_scope: contextvars.ContextVar[object | None] = (
    contextvars.ContextVar("kite_erasure_endpoint_issuance_scope", default=None)
)


@contextmanager
def _typed_kite_erasure_endpoint_issuance_scope():
    """Mark the authenticated typed endpoint's task during capability transit.

    A same-process feature can durably consume its own nonce, but it cannot
    exchange that receipt for erasure authority outside the task-local route
    scope.  Keeping the scope over storage consumption also rejects a leaked
    capability after the route returns.
    """
    token = _endpoint_issuance_scope.set(object())
    try:
        yield
    finally:
        _endpoint_issuance_scope.reset(token)


def _issue_kite_erasure_drill_capability(
    nonce_receipt: object, *, operation: str,
) -> KiteErasureDrillCapability:
    """Issue the one-shot core-drill authority after exact nonce commitment.

    The only accepted operation is the closed-set endpoint operation.  Claiming
    the receipt simultaneously prevents it from minting another capability.
    """
    if _endpoint_issuance_scope.get() is None:
        raise KiteErasureDrillAuthorityError(
            "Kite erasure drill authority may be issued only by the typed endpoint"
        )
    if operation != ERASURE_CORE_SNAPSHOT_OPERATION:
        raise KiteErasureDrillAuthorityError("Kite erasure drill operation is invalid")
    nonce = claim_kite_evidence_nonce_receipt(nonce_receipt)
    if not _NONCE_RE.fullmatch(nonce):
        raise KiteErasureDrillAuthorityError("Kite erasure drill nonce is malformed")
    operation_id = f"kite-erasure-{nonce}"
    capability = object.__new__(KiteErasureDrillCapability)
    _issued_capabilities[capability] = _ErasureDrillBinding(
        operation=operation,
        operation_id=operation_id,
        correlation=hashlib.sha256(operation_id.encode("ascii")).hexdigest(),
    )
    return capability


def _consume_kite_erasure_drill_capability(
    capability: object, *, expected_operation: str,
) -> _ErasureDrillBinding:
    """Atomically validate and consume one endpoint-issued drill capability."""
    if _endpoint_issuance_scope.get() is None:
        raise KiteErasureDrillAuthorityError(
            "Kite erasure drill authority may be consumed only by the typed endpoint"
        )
    if type(capability) is not KiteErasureDrillCapability:
        raise KiteErasureDrillAuthorityError("Kite erasure drill capability is invalid")
    try:
        binding = _issued_capabilities.pop(capability)
    except KeyError as error:
        raise KiteErasureDrillAuthorityError(
            "Kite erasure drill capability was already consumed"
        ) from error
    if binding.operation != expected_operation:
        raise KiteErasureDrillAuthorityError(
            "Kite erasure drill capability is bound to another operation"
        )
    nonce = binding.operation_id.removeprefix("kite-erasure-")
    expected_correlation = hashlib.sha256(binding.operation_id.encode("ascii")).hexdigest()
    if (
        not _NONCE_RE.fullmatch(nonce)
        or binding.correlation != expected_correlation
    ):
        raise KiteErasureDrillAuthorityError("Kite erasure drill capability binding is malformed")
    return binding
