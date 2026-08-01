"""Opaque, one-shot authority for the fixed Kite core-erasure drill.

The release-evidence HTTP route receives a durable nonce receipt and exchanges
it for this capability.  Storage accepts no caller-selected operation string:
it consumes this exact capability once and derives its own correlation from
the binding retained here.
"""

from __future__ import annotations

import asyncio
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
    owner_task: object


@dataclass(frozen=True, slots=True)
class _EndpointIssuanceScope:
    """The exact asynchronous endpoint task allowed to carry authority."""

    owner_task: object


_issued_capabilities: weakref.WeakKeyDictionary[
    KiteErasureDrillCapability, _ErasureDrillBinding
] = weakref.WeakKeyDictionary()
_endpoint_issuance_scope: contextvars.ContextVar[_EndpointIssuanceScope | None] = (
    contextvars.ContextVar("kite_erasure_endpoint_issuance_scope", default=None)
)


def _require_endpoint_owner_task() -> object:
    """Return the exact route task, rejecting copied ContextVar state."""
    try:
        current_task = asyncio.current_task()
    except RuntimeError as error:
        raise KiteErasureDrillAuthorityError(
            "Kite erasure drill authority requires an active endpoint task"
        ) from error
    scope = _endpoint_issuance_scope.get()
    if current_task is None or scope is None or scope.owner_task is not current_task:
        raise KiteErasureDrillAuthorityError(
            "Kite erasure drill authority may be used only by its endpoint task"
        )
    return current_task


@contextmanager
def _typed_kite_erasure_endpoint_issuance_scope():
    """Mark the authenticated typed endpoint's task during capability transit.

    A same-process feature can durably consume its own nonce, but it cannot
    exchange that receipt for erasure authority outside the exact endpoint
    task. Context variables are inherited by child tasks, so the scope stores
    the owner's ``asyncio.current_task()`` identity and every issuer/consumer
    verifies it. Keeping the scope over storage consumption also rejects a
    leaked capability after the route returns.
    """
    try:
        owner_task = asyncio.current_task()
    except RuntimeError as error:
        raise KiteErasureDrillAuthorityError(
            "Kite erasure drill authority requires an active endpoint task"
        ) from error
    if owner_task is None:
        raise KiteErasureDrillAuthorityError(
            "Kite erasure drill authority requires an active endpoint task"
        )
    token = _endpoint_issuance_scope.set(_EndpointIssuanceScope(owner_task))
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
    owner_task = _require_endpoint_owner_task()
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
        owner_task=owner_task,
    )
    return capability


def _consume_kite_erasure_drill_capability(
    capability: object, *, expected_operation: str,
) -> _ErasureDrillBinding:
    """Atomically validate and consume one endpoint-issued drill capability."""
    owner_task = _require_endpoint_owner_task()
    if type(capability) is not KiteErasureDrillCapability:
        raise KiteErasureDrillAuthorityError("Kite erasure drill capability is invalid")
    try:
        binding = _issued_capabilities[capability]
    except KeyError as error:
        raise KiteErasureDrillAuthorityError(
            "Kite erasure drill capability was already consumed"
        ) from error
    if binding.owner_task is not owner_task:
        raise KiteErasureDrillAuthorityError(
            "Kite erasure drill capability belongs to another endpoint task"
        )
    _issued_capabilities.pop(capability)
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
