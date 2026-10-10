"""Connection-free execution authority with irreversible task custody.

The host supplies schema-specific transactional validation. Core carries the
same mutable admission into owned children and foreign turn executors, and
enforces it at native storage/tool boundaries. A retired admission is never an
absent admission: copied children remain denied even after the owning ContextVar
has been reset. Causation and telemetry are not consulted for authority.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol, TYPE_CHECKING

from kestrel_sovereign.turn_scope import turn_scoped
from kestrel_sdk.storage.database import TransactionError

if TYPE_CHECKING:
    from kestrel_sovereign.storage.db.postgres import AdvisoryLease


class ExecutionAuthorityError(RuntimeError):
    """Execution no longer possesses its original admission authority."""


class ExecutionCommitOutcomeError(ExecutionAuthorityError, TransactionError):
    """Denial at commit is not evidence that the transaction rolled back."""

    def __init__(self, outcome: str):
        if outcome not in {"committed", "unknown"}:
            raise ValueError("invalid execution commit outcome")
        self.commit_outcome = outcome
        super().__init__(
            f"execution transaction outcome is {outcome}; the operation may have "
            "committed; reconcile durable state before any retry"
        )


def execution_commit_outcome(error: BaseException) -> str | None:
    """Preserve the commit distinction through legacy storage error wrapping."""
    seen: set[int] = set()
    while id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, ExecutionCommitOutcomeError):
            return error.commit_outcome
        if error.__cause__ is None:
            return None
        error = error.__cause__
    return None


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


@dataclass(frozen=True)
class AdvisoryExecutionFence:
    """Carry a native advisory capability into storage and tool boundaries.

    This does not implement another lease or claim validator. Its caller still
    supplies its existing claim/owner checks; copied effect contexts retain
    immediate denial when the exact exclusion session is lost or retired.
    """

    lease: AdvisoryLease
    owner_live: Callable[[], bool] | None = None
    backend_type: str = field(default="postgres", init=False)

    def require_work(self) -> None:
        self.lease.require_live()
        if self.owner_live is not None and not self.owner_live():
            raise ExecutionAuthorityError("execution owner lease lost")

    async def lock_and_validate(self, connection: Any) -> None:
        self.require_work()


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
        try:
            self.fence.require_work()
        except ExecutionAuthorityError as error:
            self.revoke(str(error))
            raise


_CURRENT_CUSTODY: ContextVar[tuple[ExecutionCustody, ...]] = ContextVar(
    "kestrel_execution_custody", default=(),
)
_CLEANUP_ONLY: ContextVar[bool] = ContextVar("kestrel_execution_cleanup_only", default=False)


def current_execution_custody(owner: Any = None) -> tuple[ExecutionCustody, ...]:
    """Capture immutable scope membership, retaining shared revocation state."""

    ambient = _CURRENT_CUSTODY.get()
    retained = getattr(owner, "_execution_custody", None)
    if isinstance(retained, ExecutionCustody) and retained not in ambient:
        return (retained, *ambient)
    return ambient


def require_execution_work(owner: Any = None) -> None:
    if _CLEANUP_ONLY.get():
        raise ExecutionAuthorityError("execution is cleanup-only; ordinary work is denied")
    for scope in current_execution_custody(owner):
        scope.require_work()


def refuse_unfenced_executor(owner: Any = None) -> None:
    """SQLAlchemy cannot validate authority on its actual mutation session."""
    if _CLEANUP_ONLY.get() or current_execution_custody(owner):
        raise ExecutionAuthorityError(
            "custody-bound execution cannot use the unfenced SQLAlchemy executor"
        )


def require_execution_backend(
    backend_type: str, scopes: tuple[ExecutionCustody, ...] | None = None,
) -> None:
    if _CLEANUP_ONLY.get():
        raise ExecutionAuthorityError("execution is cleanup-only; ordinary work is denied")
    for scope in current_execution_custody() if scopes is None else scopes:
        scope.require_work()
        if scope.fence.backend_type != backend_type:
            raise ExecutionAuthorityError(
                f"execution authority requires {scope.fence.backend_type} backend, "
                f"not {backend_type}"
            )


async def lock_execution_authority(
    connection: Any, backend_type: str,
    scopes: tuple[ExecutionCustody, ...] | None = None,
) -> None:
    captured = current_execution_custody() if scopes is None else scopes
    require_execution_backend(backend_type, captured)
    for scope in captured:
        try:
            await scope.fence.lock_and_validate(connection)
        except ExecutionAuthorityError as error:
            scope.revoke(str(error))
            raise
        # Cancellation-resistant validators must not republish a lost scope.
        require_execution_backend(backend_type, captured)


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


@contextmanager
def bind_execution_runtime(owner: Any) -> Iterator[None]:
    """Present retained runtime custody without minting a fresh admission."""
    with _bind_captured_custody(current_execution_custody(owner)):
        yield


@contextmanager
def bind_execution_custody_snapshot(captured: tuple[ExecutionCustody, ...]) -> Iterator[None]:
    """Carry original live admissions to an already-owned control worker."""
    with _bind_captured_custody(captured):
        yield


@contextmanager
def bind_execution_cleanup(
    owner: Any, captured: tuple[ExecutionCustody, ...] = (),
) -> Iterator[None]:
    """Retain original denials while closing hosted work, never admit work.

    Standalone invocations without custody retain their existing semantics.
    Hosted cleanup can use only native fixed exact-identity terminal operations;
    storage, tools and provider calls still pass through ordinary work denial.
    Children inherit this restriction even after their closer leaves the scope.
    """
    existing = current_execution_custody(owner)
    scopes = existing + tuple(scope for scope in captured if scope not in existing)
    custody_token = _CURRENT_CUSTODY.set(scopes)
    cleanup_token = _CLEANUP_ONLY.set(_CLEANUP_ONLY.get() or bool(scopes))
    try:
        yield
    finally:
        _CLEANUP_ONLY.reset(cleanup_token)
        _CURRENT_CUSTODY.reset(custody_token)


turn_scoped(
    "execution_custody",
    variables=(_CURRENT_CUSTODY,),
    capture=current_execution_custody,
    bind=_bind_captured_custody,
)


@contextmanager
def _bind_cleanup_flag(captured: bool) -> Iterator[None]:
    token = _CLEANUP_ONLY.set(_CLEANUP_ONLY.get() or captured)
    try:
        yield
    finally:
        _CLEANUP_ONLY.reset(token)


turn_scoped(
    "execution_cleanup_only",
    variables=(_CLEANUP_ONLY,),
    capture=lambda owner: _CLEANUP_ONLY.get(),
    bind=_bind_cleanup_flag,
)
