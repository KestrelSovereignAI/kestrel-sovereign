"""Connection-free execution authority with irreversible task custody.

The host supplies schema-specific transactional validation. Core carries the
same mutable admission into owned children and foreign turn executors, and
enforces it at native storage/tool boundaries. A retired admission is never an
absent admission: copied children remain denied even after the owning ContextVar
has been reset. Causation and telemetry are not consulted for authority.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

from kestrel_sovereign.turn_scope import turn_scoped


class ExecutionAuthorityError(RuntimeError):
    """Execution no longer possesses its original admission authority."""


class ExecutionFence(Protocol):
    """Trusted host's immutable binding; validation uses the mutation's session.

    ``lock_and_validate`` must lock authority rows before native graph/file
    locks, validate the original generation, and retain locks through commit.
    The supplied executor is the actual backend connection, not a second
    checkout. Validators must not publish effects or start/commit transactions.
    """

    backend_type: str

    def require_work(self) -> None: ...

    async def lock_and_validate(self, connection: Any) -> None: ...


@dataclass(eq=False)
class ExecutionCustody:
    """One admission shared by reference across all captured task contexts."""

    fence: ExecutionFence
    _denial: str | None = field(default=None, init=False)

    def revoke(self, reason: str) -> None:
        if self._denial is None:
            self._denial = reason or "execution authority revoked"

    def require_work(self) -> None:
        if self._denial is not None:
            raise ExecutionAuthorityError(self._denial)
        self.fence.require_work()


_CURRENT_CUSTODY: ContextVar[tuple[ExecutionCustody, ...]] = ContextVar(
    "kestrel_execution_custody", default=(),
)


def current_execution_custody() -> tuple[ExecutionCustody, ...]:
    """Capture immutable scope membership, retaining shared revocation state."""

    return _CURRENT_CUSTODY.get()


def require_execution_work() -> None:
    for scope in current_execution_custody():
        scope.require_work()


def require_execution_backend(backend_type: str) -> None:
    for scope in current_execution_custody():
        scope.require_work()
        if scope.fence.backend_type != backend_type:
            raise ExecutionAuthorityError(
                f"execution authority requires {scope.fence.backend_type} backend, "
                f"not {backend_type}"
            )


async def lock_execution_authority(connection: Any, backend_type: str) -> None:
    require_execution_backend(backend_type)
    for scope in current_execution_custody():
        await scope.fence.lock_and_validate(connection)
        # Cancellation-resistant validators must not republish a lost scope.
        require_execution_backend(backend_type)


@contextmanager
def bind_execution_custody(fence: ExecutionFence) -> Iterator[ExecutionCustody]:
    require_execution_work()
    if fence.backend_type != "postgres":
        raise ExecutionAuthorityError(
            "transaction-bound hosted execution authority requires postgres backend"
        )
    scope = ExecutionCustody(fence)
    scope.require_work()
    token = _CURRENT_CUSTODY.set((*current_execution_custody(), scope))
    try:
        yield scope
    finally:
        scope.revoke("execution admission retired")
        _CURRENT_CUSTODY.reset(token)


@contextmanager
def _bind_captured_custody(
    captured: tuple[ExecutionCustody, ...],
) -> Iterator[None]:
    # Binding a foreign turn may add its own scopes; never discard a denying
    # ancestor that was already present on that task.
    existing = current_execution_custody()
    scopes = existing + tuple(scope for scope in captured if scope not in existing)
    token = _CURRENT_CUSTODY.set(scopes)
    try:
        require_execution_work()
        yield
    finally:
        _CURRENT_CUSTODY.reset(token)


turn_scoped(
    "execution_custody",
    variables=(_CURRENT_CUSTODY,),
    capture=lambda _agent: current_execution_custody(),
    bind=_bind_captured_custody,
)
