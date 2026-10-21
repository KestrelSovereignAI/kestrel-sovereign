"""Connection-free execution authority with irreversible task custody.

The host supplies schema-specific transactional validation. Core carries the
same mutable admission into owned children and foreign turn executors, and
enforces it at native storage/tool boundaries. A retired admission is never an
absent admission: copied children remain denied even after the owning ContextVar
has been reset. Causation and telemetry are not consulted for authority.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import stat
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from httpx import AsyncByteStream as _NativeHTTPXAsyncByteStream
from httpx import AsyncClient as _NativeHTTPXAsyncClient
from httpx import Response as _NativeHTTPXResponse
from httpx2 import AsyncByteStream as _NativeHTTPX2AsyncByteStream
from httpx2 import AsyncClient as _NativeHTTPX2AsyncClient
from httpx2 import Response as _NativeHTTPX2Response
from kestrel_sdk.storage.database import TransactionError

from kestrel_sovereign.turn_scope import capture_turn_scope, turn_scoped

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


@dataclass
class ConfirmedLeaseDeadline:
    """Irreversible local deadline shared by original lease owners.

    This is not distributed authority or a token validator. A host combines
    this additional fail-closed lifetime with its existing exact-token CAS.
    Both synchronous admission and an independent monitor consult the SAME
    monotonic deadline, so a stalled renewal cannot extend an old interval.
    Only a completed, still-timely renewal may confirm a later interval.
    """

    lost: asyncio.Event = field(default_factory=asyncio.Event)
    error: BaseException | None = None
    confirmed_deadline: float | None = None
    deadline_changed: asyncio.Event = field(default_factory=asyncio.Event)

    def require_work(self) -> None:
        if not self.is_live():
            raise ExecutionAuthorityError("original confirmed lease lost")

    def is_live(self) -> bool:
        if self.confirmed_deadline is not None and time.monotonic() >= self.confirmed_deadline:
            self.mark_lost(ExecutionAuthorityError("original confirmed lease expired"))
        return not self.lost.is_set()

    def confirm(self, deadline: float) -> bool:
        if not self.is_live() or not math.isfinite(deadline) or deadline <= time.monotonic():
            self.mark_lost(ExecutionAuthorityError("renewal exceeded original confirmed lease"))
            return False
        self.confirmed_deadline = deadline
        self.deadline_changed.set()
        return True

    def mark_lost(self, error: BaseException | None = None) -> None:
        if self.error is None and error is not None:
            self.error = error
        self.lost.set()
        self.deadline_changed.set()

    async def monitor(self) -> None:
        """Expire independently of a blocked renewal checkout or row lock."""
        while self.is_live():
            self.deadline_changed.clear()
            deadline = self.confirmed_deadline
            if deadline is None:
                await self.deadline_changed.wait()
                continue
            try:
                async with asyncio.timeout(max(0.0, deadline - time.monotonic())):
                    await self.deadline_changed.wait()
            except TimeoutError:
                # A timely completion may have advanced the interval in this
                # event-loop turn. Never revoke that newer confirmed interval.
                self.is_live()


@dataclass(eq=False)
class NativeExecutionRetirement:
    """One original checkout's reset owner, shared by its native receipts."""

    owner: asyncio.Future | None = None


@dataclass(eq=False)
class AcknowledgedExecutionCommit:
    """Original native ACK and retirement owners, never authority to work.

    A caller can stop observing while COMMIT succeeds. Keep that durable fact
    on its original exception without inventing uncertainty on healthy scopes.
    Localization is allowed only after the SAME checkout's reset also succeeds.
    Native ACK/reset errors still use irreversible commit/debt evidence.
    """

    commit_owner: asyncio.Future
    retirement: NativeExecutionRetirement = field(default_factory=NativeExecutionRetirement)

    @property
    def retirement_owner(self) -> asyncio.Future | None:
        return self.retirement.owner

    @retirement_owner.setter
    def retirement_owner(self, owner: asyncio.Future) -> None:
        self.retirement.owner = owner

    @property
    def acknowledged(self) -> bool:
        from kestrel_sovereign._async_ownership import owned_task_error
        return self.commit_owner.done() and owned_task_error(self.commit_owner) is None

    @property
    def retired(self) -> bool:
        from kestrel_sovereign._async_ownership import owned_task_error
        return (self.acknowledged and self.retirement_owner is not None
                and self.retirement_owner.done()
                and owned_task_error(self.retirement_owner) is None)


def retain_acknowledged_execution_commit(
    error: BaseException, receipt: AcknowledgedExecutionCommit,
) -> BaseException:
    """Attach a native fact to the exact original carrier; grant no admission."""
    if not isinstance(receipt, AcknowledgedExecutionCommit) or not receipt.acknowledged:
        raise ValueError("native COMMIT acknowledgement has not settled successfully")
    existing = vars(error).get("_kestrel_acknowledged_commits", ())
    if receipt not in existing:
        error._kestrel_acknowledged_commits = (*existing, receipt)
        if not existing:
            error.add_note("execution transaction outcome is committed; reconcile durable state before retry")
    return error


def _acknowledged_commits(error: BaseException) -> tuple[AcknowledgedExecutionCommit, ...]:
    return tuple(receipt for receipt in vars(error).get("_kestrel_acknowledged_commits", ())
                 if isinstance(receipt, AcknowledgedExecutionCommit) and receipt.acknowledged)


def _explicit_control_chain(error: BaseException | None) -> Iterator[BaseException]:
    """Walk explicit carriers, never implicit cleanup/handling context."""
    seen: set[int] = set()
    pending = [error] if error is not None else []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if isinstance(current, BaseExceptionGroup):
            # Lifecycle owners explicitly group child cancellation and close
            # failures. Group members are evidence, unlike incidental context.
            pending.extend(current.exceptions)


def _enclosing_original_errors(errors: list[BaseException]) -> list[BaseException]:
    """Keep representatives covering every explicit component, including cycles.

    An error enclosed by a retained representative is redundant. A later
    enclosing error can replace that representative because its whole graph
    remains reachable. Unlike removing every node with an incoming edge,
    this also retains a representative of a cyclic component beside unrelated
    originals. No original graph is changed.
    """
    enclosing: list[BaseException] = []
    chains: dict[int, set[int]] = {}
    for error in errors:
        if any(id(error) in chains[id(prior)] for prior in enclosing):
            continue
        chain = {id(member) for member in _explicit_control_chain(error)}
        chains[id(error)] = chain
        enclosing = [prior for prior in enclosing if id(prior) not in chain]
        enclosing.append(error)
    return enclosing


def execution_commit_outcome(error: BaseException | None) -> str | None:
    """Preserve the strongest commit distinction through explicit carriers."""
    outcome = None
    for current in _explicit_control_chain(error):
        unresolved = vars(current).get("_kestrel_unresolved_commit_outcome")
        if unresolved == "unknown":
            return "unknown"
        if unresolved == "committed":
            outcome = "committed"
        if isinstance(current, ExecutionCommitOutcomeError):
            if current.commit_outcome == "unknown":
                return "unknown"
            outcome = current.commit_outcome
        if _acknowledged_commits(current):
            outcome = outcome or "committed"
    return outcome


def is_execution_control_error(error: BaseException | None) -> bool:
    """Original authority, commit, or physical-retirement evidence is control.

    An ordinary native close exception still controls retirement when it
    carries the actual retained resource/closer. Availability/tool-error
    normalization must never erase that original ownership evidence.
    """
    return any(isinstance(current, ExecutionAuthorityError)
               or bool(_acknowledged_commits(current))
               or bool(vars(current).get("_kestrel_native_retirement_debts"))
               for current in _explicit_control_chain(error))


def retained_execution_error(error: BaseException) -> BaseException:
    """Propagate the SAME receipt without retaining every subsequent reader.

    Retention freezes each explicit carrier's original traceback once. Reusing
    an exception otherwise prepends another consumer frame on every heartbeat
    or result read. Identity, explicit causes/groups and commit evidence stay
    unchanged; implicit context is neither authority nor a retention owner.
    """
    for carrier in _explicit_control_chain(error):
        state = vars(carrier)
        original = state.setdefault("_kestrel_retained_traceback", carrier.__traceback__)
        carrier.__traceback__ = original
    return error


def execution_cleanup_debts(error: BaseException | None) -> tuple:
    """Actual failed native resources/closers, not ordinary provider errors.

    An owner is None only when creation was refused before a closer existed;
    the resource record still retains the original physical connection. This
    absence is debt, never a fabricated successful retirement future.
    """
    debts = []
    if error is not None:
        for carrier in _explicit_control_chain(error):
            for debt in vars(carrier).get("_kestrel_native_retirement_debts", ()):
                if not any(debt[0] is prior[0] and debt[1] is prior[1] for prior in debts):
                    debts.append(debt)
    return tuple(debts)


def execution_denial_scopes(error: BaseException | None) -> tuple | None:
    """Exact denial provenance, or None for unscoped/unsettled control.

    Consumers may distinguish a retired occurrence from their runtime only
    when EVERY explicit authority carrier has original typed scope evidence.
    An empty tuple means an acknowledged, retired observer-only commit receipt:
    no authority scope was denied. UNKNOWN or failed native retirement is
    never localized, even alongside another successfully acknowledged commit.
    """
    scopes = []
    settled_commit = False
    if execution_cleanup_debts(error):
        return None
    for carrier in _explicit_control_chain(error):
        if vars(carrier).get("_kestrel_unresolved_commit_outcome"):
            return None
        if vars(carrier).get("_kestrel_unscoped_execution_denial"):
            return None
        receipts = _acknowledged_commits(carrier)
        if receipts:
            if not all(receipt.retired for receipt in receipts):
                return None
            settled_commit = True
        if isinstance(carrier, ExecutionCommitOutcomeError):
            return None
        if isinstance(carrier, ExecutionAuthorityError) or vars(carrier).get("_kestrel_denied_execution_scopes"):
            if vars(carrier).get("_kestrel_unscoped_execution_denial"):
                return None
            denied = vars(carrier).get("_kestrel_denied_execution_scopes")
            if not denied:
                return None
            scopes.extend(scope for scope in denied if scope not in scopes)
    return tuple(scopes) if scopes or settled_commit else None


def is_scope_local_execution_control(error: BaseException | None, captured_scopes: tuple) -> bool:
    """Classify a settled source refusal, never authorize replacement work.

    Only exact original typed admission objects may localize their own denial.
    A trusted caller supplies its immutable source snapshot, not identifiers,
    current reader scopes, or scopes inferred from the failure itself.
    """
    denied = execution_denial_scopes(error)
    # The canonical provenance reader already excludes cleanup debt, UNKNOWN
    # and unscoped control. A settled acknowledged COMMIT still forbids locality.
    return bool(denied and all(any(scope is admitted for admitted in captured_scopes)
                              for scope in denied)
                and execution_commit_outcome(error) is None)


_FINITE_NATIVE_SOURCES = object()


def _canonical_native_source_originals(error: BaseException | None) -> tuple[BaseException, ...] | None:
    """Only authenticated canonical finite provenance can split a source."""
    if error is None:
        return None
    provenance = vars(error).get("_kestrel_finite_native_sources")
    if (type(provenance) is tuple and len(provenance) == 2
            and provenance[0] is _FINITE_NATIVE_SOURCES):
        return provenance[1]
    return None


def _native_source_originals(error: BaseException | None) -> tuple[BaseException, ...]:
    """Flatten only canonical finite carriers, never genuine source groups."""
    if error is None:
        return ()
    sources = _canonical_native_source_originals(error)
    if sources is not None:
        return sources
    return (error,) if is_execution_control_error(error) else ()


def execution_observer_cancellations(error: BaseException | None) -> tuple[asyncio.CancelledError, ...]:
    """Read actual finite C only from authenticated Core composition.

    A genuine unmarked source/group cannot be split just because it contains
    cancellation. Canonical provenance identifies the native source graph;
    cancellation outside that graph belongs to this finite observation, not
    a shared native latch. Return exact objects without rewriting carriers.
    """
    sources = _canonical_native_source_originals(error)
    if sources is None:
        return ()
    source_ids = {id(member) for source in sources for member in _explicit_control_chain(source)}
    return tuple(member for member in _explicit_control_chain(error)
                 if isinstance(member, asyncio.CancelledError) and id(member) not in source_ids)


def execution_error_with_cause(wrapper: BaseException, error: BaseException) -> BaseException:
    """Core-owned wrappers preserve a canonical finite source projection.

    Raising a new diagnostic/classification wrapper must not turn its finite
    observer C into a genuine source. Unmarked originals, including source
    cancellations/groups, remain intact. A commit classifier additionally
    carries its NEW native outcome on a private source-only classifier.
    """
    wrapper.__cause__ = error
    sources = _canonical_native_source_originals(error)
    if sources is not None:
        if isinstance(wrapper, ExecutionCommitOutcomeError):
            classified_source = ExecutionCommitOutcomeError(wrapper.commit_outcome)
            classified_source.__cause__ = execution_native_source_error(error)
            sources = (classified_source,)
        wrapper._kestrel_finite_native_sources = (_FINITE_NATIVE_SOURCES, sources)
    return wrapper


def execution_native_source_error(*errors: BaseException | None) -> BaseException | None:
    """Project canonical native SOURCES, never a finite observer's withdrawal.

    Only our freshly allocated composition carries this provenance. An
    unmarked original source (including native cancellation/group) stays exact;
    arbitrary exception attributes or cancellation-shaped foreign leaves do
    not authorize rewriting its explicit graph. Flatten canonical compositions
    before merging a shared latch, so repeated projections of the SAME sources
    cannot accumulate wrappers or previous observers.
    """
    sources = []
    for error in errors:
        for source in _native_source_originals(error):
            if all(source is not prior for prior in sources):
                sources.append(source)
    return execution_terminal_error(*sources, retain_originals=True,
                                    isolate_control_originals=True)


def execution_observation_error(*errors: BaseException | None,
                                message: str = "original completion observation evidence") -> BaseException | None:
    """Compose a Core-owned join without making observers genuine sources.

    A fresh group preserves even ordinary physical errors without decorating
    their graphs. Only the originals' authenticated native-source projections
    cross a later shared boundary. Genuine unmarked source groups remain whole;
    this does not authorize flattening a foreign group containing cancellation.
    """
    present = []
    sources = []
    for error in errors:
        if error is None or any(error is prior for prior in present):
            continue
        present.append(error)
        for source in _native_source_originals(error):
            if all(source is not prior for prior in sources):
                sources.append(source)
    # Re-observing an already enclosing carrier is not another independent
    # failure. Keep that exact original envelope rather than growing wrapper
    # history or converting a sole typed outcome into a duplicate group.
    present = _enclosing_original_errors(present)
    if len(present) < 2:
        return present[0] if present else None
    terminal = BaseExceptionGroup(message, present)
    terminal._kestrel_finite_native_sources = (_FINITE_NATIVE_SOURCES, tuple(sources))
    return execution_terminal_error(terminal, retain_originals=True, isolate_control_originals=True)


def execution_finite_error(*errors: BaseException | None, retain_originals: bool = False,
                           prefer_cancellation: bool = False,
                           isolate_control_originals: bool = False) -> BaseException | None:
    """Finite lifecycle composition isolates native sources, not generic defaults.

    A native receipt cannot erase this caller's actual withdrawal or acquire
    independent failures in its original graph. Ordinary no-control callers
    retain their existing selection and caused-cancellation policy.
    """
    native = any(is_execution_control_error(error) for error in errors)
    return execution_terminal_error(
        *errors, retain_originals=retain_originals or native,
        prefer_cancellation=prefer_cancellation,
        isolate_control_originals=isolate_control_originals or native,
    )


def execution_terminal_error(*errors: BaseException | None, retain_originals: bool = False,
                             prefer_cancellation: bool = False,
                             isolate_control_originals: bool = False) -> BaseException | None:
    """Keep irreversible control evidence ahead of cancellation/cleanup noise.

    Native lifecycle boundaries can retain all original body/closer carriers
    explicitly. Ordinary repeated readers retain bounded irreversible facts,
    not a growing exception history. Unresolved legacy commit classification
    is one such fact: a successful ACK elsewhere never settles it.
    A lifecycle drain may prefer actual caller withdrawal over an ordinary
    opening failure after cleanup settles. Original native control still wins;
    retain_originals keeps the opening failure as explicit evidence.
    When a native receipt is also a shared strongest-writer latch, finite
    lifecycle callers can isolate_control_originals: retain unrelated originals
    in a fresh carrier instead of mutating that source receipt. Already-enclosing
    carriers are reused, so repeated observation does not grow wrapper history.
    """
    import asyncio
    present = tuple(error for error in errors if error is not None)
    if isolate_control_originals:
        retain_originals = True
        if any(is_execution_control_error(error) for error in present):
            unique = []
            for error in present:
                if all(error is not prior for prior in unique):
                    unique.append(error)
            enclosing = _enclosing_original_errors(unique)
            if len(enclosing) > 1:
                terminal = BaseExceptionGroup("original execution terminal evidence", enclosing)
                sources = []
                for error in enclosing:
                    for source in _native_source_originals(error):
                        if all(source is not prior for prior in sources):
                            sources.append(source)
                # Ordinary physical errors can be native only by association
                # with an actual retained owner. Preserve that exact original,
                # not an unrelated observer C in the same finite aggregate.
                from kestrel_sovereign._async_ownership import owned_task_error
                for error in enclosing:
                    for _resource, owner in execution_cleanup_debts(error):
                        if isinstance(owner, asyncio.Future) and owner.done():
                            physical = owned_task_error(owner)
                            if (physical is not None
                                and any(physical is member for member in _explicit_control_chain(terminal))
                            ):
                                # A physical owner can itself have returned a
                                # canonical finite observation. Its observer C
                                # is still not a shared native source.
                                for source in _native_source_originals(physical) or (physical,):
                                    if all(source is not prior for prior in sources):
                                        sources.append(source)
                terminal._kestrel_finite_native_sources = (_FINITE_NATIVE_SOURCES, tuple(sources))
                present = (terminal,)
            else:
                present = tuple(enclosing)
    def retain(selected):
        if retain_originals:
            # A selected cause cannot point back to its enclosing original
            # cancellation without a cycle. Keep that ORIGINAL enclosing
            # carrier instead; irreversible classification is retained below.
            promoted = {id(selected)}
            while True:
                parent = next((error for error in present
                    if id(error) not in promoted
                    and selected in tuple(_explicit_control_chain(error))), None)
                if parent is None:
                    break
                selected = parent
                promoted.add(id(parent))
        debts = []
        commits = []
        denied_scopes = []
        unscoped_denial = False
        unresolved_commit = None
        for error in present:
            for carrier in _explicit_control_chain(error):
                commits.extend(receipt for receipt in _acknowledged_commits(carrier) if receipt not in commits)
                unresolved = (carrier.commit_outcome if isinstance(carrier, ExecutionCommitOutcomeError)
                    else vars(carrier).get("_kestrel_unresolved_commit_outcome"))
                if unresolved in {"committed", "unknown"}:
                    if unresolved_commit is None or unresolved == "unknown":
                        unresolved_commit = unresolved
            for debt in execution_cleanup_debts(error):
                if not any(debt[0] is prior[0] and debt[1] is prior[1] for prior in debts):
                    debts.append(debt)
            for carrier in _explicit_control_chain(error):
                # Aggregation may select an original cancellation/ordinary
                # exception as the retained carrier. Its explicit provenance
                # remains authoritative at EVERY later aggregation boundary.
                state = vars(carrier)
                denied = state.get("_kestrel_denied_execution_scopes", ())
                unscoped_denial |= bool(state.get("_kestrel_unscoped_execution_denial"))
                if isinstance(carrier, ExecutionAuthorityError) and not denied:
                    unscoped_denial = True
                denied_scopes.extend(scope for scope in denied if scope not in denied_scopes)
        if debts:
            selected._kestrel_native_retirement_debts = tuple(debts)
        if commits:
            selected._kestrel_acknowledged_commits = tuple(commits)
        if denied_scopes:
            selected._kestrel_denied_execution_scopes = tuple(denied_scopes)
        if unscoped_denial:
            selected._kestrel_unscoped_execution_denial = True
        if unresolved_commit is not None:
            selected._kestrel_unresolved_commit_outcome = unresolved_commit
        if retain_originals:
            retained = tuple(_explicit_control_chain(selected))
            others = []
            for error in present:
                if (error not in retained and error not in others
                    and selected not in tuple(_explicit_control_chain(error))):
                    others.append(error)
            if others:
                if selected.__cause__ is not None:
                    others.insert(0, selected.__cause__)
                selected.__cause__ = (others[0] if len(others) == 1 else
                    BaseExceptionGroup("original execution terminal evidence", others))
        return retained_execution_error(selected)
    for outcome in ("unknown", "committed"):
        for error in present:
            if execution_commit_outcome(error) == outcome:
                return retain(retained_execution_error(error))
    for error in present:
        if is_execution_control_error(error):
            return retain(retained_execution_error(error))
    for error in present:
        # OwnedAsyncIterator carries source cleanup debt as the explicit cause
        # of cancellation. A failed checkpoint is not a successful Stop.
        if isinstance(error, asyncio.CancelledError) and error.__cause__ is not None:
            return retain(error.__cause__)
    if prefer_cancellation:
        for error in present:
            if isinstance(error, asyncio.CancelledError):
                return retain(error)
    if present and isinstance(present[0], GeneratorExit):
        # GeneratorExit is the source's ordinary close request, not caller
        # cancellation. A later cancellation while joining that close must
        # still reach the closer after cleanup settles. Control evidence was
        # already selected above, so it cannot be demoted by this rule.
        return retain(next((error for error in present[1:] if not isinstance(error, GeneratorExit)), present[0]))
    if present and isinstance(present[0], (asyncio.CancelledError, GeneratorExit)):
        for error in present[1:]:
            if not isinstance(error, (asyncio.CancelledError, GeneratorExit)):
                return retain(error)
    return retain(present[0]) if present else None


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
class ProcessRuntimeExecutionFence:
    """Original process-owned runtime lifetime for a standalone PG host.

    This is NOT a tenant row, distributed generation, Hold grant or scheduler
    occurrence. A trusted standalone host creates one per runtime construction;
    shutdown irrevocably retires it. Tenant hosts must supply their own native
    generation fence instead. Ordinary calls still retain ALL ambient leases.
    """

    agent_id: str
    backend_type: str = field(default="postgres", init=False)
    _retired: bool = field(default=False, init=False)

    def retire(self) -> None:
        self._retired = True

    def require_work(self) -> None:
        if self._retired:
            raise ExecutionAuthorityError("original process runtime retired")

    async def lock_and_validate(self, connection: Any) -> None:
        # No authority rows exist for a standalone process lifetime. Validate
        # this SAME local object on the mutation's session; never check out a
        # replacement lease or pretend it is distributed row authority.
        self.require_work()


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
    _uncertain_commit: str | None = field(default=None, init=False)
    _acknowledged_commits: tuple[AcknowledgedExecutionCommit, ...] = field(default=(), init=False)

    def preserve_commit_uncertainty(self, outcome: str) -> None:
        if outcome not in {"committed", "unknown"}:
            raise ValueError("invalid execution commit outcome")
        if self._uncertain_commit is None or outcome == "unknown":
            self._uncertain_commit = outcome

    def revoke(self, reason: str) -> None:
        if self._denial is None:
            self._denial = reason or "execution authority revoked"

    def retain_acknowledged_commit(self, receipt: AcknowledgedExecutionCommit) -> None:
        """Keep the denied scope's known native fact, not shared uncertainty."""
        if receipt not in self._acknowledged_commits:
            self._acknowledged_commits = (*self._acknowledged_commits, receipt)

    def _retain_denial(self, error: ExecutionAuthorityError) -> None:
        """Keep original scope provenance at both pure and native gates."""
        self.revoke(str(error))
        scopes = vars(error).get("_kestrel_denied_execution_scopes", ())
        if self not in scopes:
            error._kestrel_denied_execution_scopes = (*scopes, self)

    def require_work(self) -> None:
        try:
            if self._uncertain_commit is not None:
                raise ExecutionCommitOutcomeError(self._uncertain_commit)
            if self._denial is not None:
                raise ExecutionAuthorityError(self._denial)
            self.fence.require_work()
        except ExecutionAuthorityError as error:
            self._retain_denial(error)
            for receipt in self._acknowledged_commits:
                retain_acknowledged_execution_commit(error, receipt)
            raise


def retire_process_runtime_custody(custody: ExecutionCustody | None) -> None:
    """Retire only a locally owned standalone lifetime, never tenant authority."""
    if isinstance(custody, ExecutionCustody) and isinstance(
        custody.fence, ProcessRuntimeExecutionFence
    ):
        custody.fence.retire()


_CURRENT_CUSTODY: ContextVar[tuple[ExecutionCustody, ...]] = ContextVar(
    "kestrel_execution_custody", default=(),
)
_CLEANUP_ONLY: ContextVar[bool] = ContextVar("kestrel_execution_cleanup_only", default=False)
_NATIVE_RESPONSE_SINK: ContextVar["NativeExecutionStreamOwner | None"] = ContextVar(
    "kestrel_native_response_sink", default=None,
)


@dataclass(eq=False)
class _HostIdentityAnchorAccess:
    """Lexical host file custody, NOT replacement execution/tenant authority."""

    path: Path
    scopes: tuple[ExecutionCustody, ...]
    read_only: bool
    active: bool = True

    def require_live(self) -> None:
        if not self.active:
            raise ExecutionAuthorityError("host identity-anchor access retired")
        for scope in self.scopes:
            scope.require_work()


_HOST_IDENTITY_ANCHOR_ACCESS: ContextVar[_HostIdentityAnchorAccess | None] = ContextVar(
    "kestrel_host_identity_anchor_access", default=None,
)


@contextmanager
def _host_identity_anchor_access(path: Path, *, read_only: bool = False) -> Iterator[None]:
    """Authorize ONLY this host-owned inception/replay file until joined close.

    Core's two host identity consumers use this capability to open their local
    birth anchor while original PostgreSQL execution custody remains bound.
    Every original scope still governs liveness; destination PG writes retain
    their native transactional validators. No ordinary tenant SQLite database
    is authorized, and copied children/retained handles share irreversible
    retirement. Archives are exact read-only files, never directory grants.
    """
    require_execution_work()
    path = Path(path)
    try:
        retained = path.lstat()
    except FileNotFoundError:
        retained = None
    # Preserve the public native file-safety refusal before granting access.
    # A linked/non-regular file is not a lost execution generation, and must
    # never acquire a capability even when its basename looks canonical.
    if retained is not None and (
        not stat.S_ISREG(retained.st_mode) or retained.st_nlink != 1
    ):
        raise ValueError("Host identity-anchor access requires an exclusively retained regular database")
    if not (
        path.name == "kestrel_prime.db"
        or (read_only and path.name.startswith("kestrel_prime.db.backup-"))
    ):
        raise ExecutionAuthorityError("host identity-anchor access requires an exact native anchor")
    access = _HostIdentityAnchorAccess(path.resolve(), current_execution_custody(), read_only)
    token = _HOST_IDENTITY_ANCHOR_ACCESS.set(access)
    try:
        yield
    finally:
        access.active = False
        _HOST_IDENTITY_ANCHOR_ACCESS.reset(token)


def _capture_host_identity_anchor_access(path: Path | None, *, cold_read: bool) -> _HostIdentityAnchorAccess | None:
    """Bind file custody once at native backend construction, never reacquire."""
    access = _HOST_IDENTITY_ANCHOR_ACCESS.get()
    if access is None or path != access.path:
        return None
    access.require_live()
    if access.read_only and not cold_read:
        raise ExecutionAuthorityError("archived identity custody permits only a cold read")
    return access


@dataclass(eq=False)
class _StreamCleanupState:
    # Shared by reference with the producer and its children. The closer sets
    # this BEFORE cancellation, which cannot mutate another task's ContextVars.
    scopes: tuple[ExecutionCustody, ...]
    requested: bool = False


_STREAM_CLEANUP: ContextVar[tuple[_StreamCleanupState, ...]] = ContextVar(
    "kestrel_execution_stream_cleanup", default=(),
)


def _execution_cleanup_requested() -> bool:
    return _CLEANUP_ONLY.get() or any(
        state.requested and bool(state.scopes) for state in _STREAM_CLEANUP.get()
    )


def current_execution_custody(owner: Any = None) -> tuple[ExecutionCustody, ...]:
    """Capture immutable scope membership, retaining shared revocation state."""

    ambient = _CURRENT_CUSTODY.get()
    retained = getattr(owner, "_execution_custody", None)
    if isinstance(retained, ExecutionCustody) and retained not in ambient:
        return (retained, *ambient)
    return ambient


def require_execution_work(owner: Any = None) -> None:
    terminal = None
    if _execution_cleanup_requested():
        terminal = ExecutionAuthorityError("execution is cleanup-only; ordinary work is denied")
    for scope in current_execution_custody(owner):
        try:
            scope.require_work()
        except BaseException as denial:
            terminal = execution_terminal_error(terminal, denial)
    if terminal is not None:
        raise terminal


def refuse_unfenced_executor(owner: Any = None) -> None:
    """SQLAlchemy cannot validate authority on its actual mutation session."""
    if _execution_cleanup_requested() or current_execution_custody(owner):
        raise ExecutionAuthorityError(
            "custody-bound execution cannot use the unfenced SQLAlchemy executor"
        )


def require_execution_backend(
    backend_type: str, scopes: tuple[ExecutionCustody, ...] | None = None,
    *, _identity_anchor_access: _HostIdentityAnchorAccess | None = None,
) -> None:
    terminal = None
    if _execution_cleanup_requested():
        terminal = ExecutionAuthorityError("execution is cleanup-only; ordinary work is denied")
    if _identity_anchor_access is not None:
        try:
            _identity_anchor_access.require_live()
        except BaseException as denial:
            terminal = execution_terminal_error(terminal, denial)
    captured = current_execution_custody() if scopes is None else scopes
    for scope in captured:
        try:
            scope.require_work()
        except BaseException as denial:
            terminal = execution_terminal_error(terminal, denial)
    if terminal is not None:
        raise terminal
    for scope in captured:
        if scope.fence.backend_type != backend_type:
            if (
                backend_type == "sqlite"
                and _identity_anchor_access is not None
                and scope in _identity_anchor_access.scopes
            ):
                continue
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
            scope._retain_denial(error)
            raise
        # Cancellation-resistant validators must not republish a lost scope.
        require_execution_backend(backend_type, captured)


@contextmanager
def bind_execution_custody(
    fence: ExecutionFence, *,
    on_scope_created: Callable[[ExecutionCustody], None] | None = None,
) -> Iterator[ExecutionCustody]:
    """Retain an original scope before validation, without granting admission.

    A lifecycle owner may record the exact newly created scope even when its
    first work check denies entry. Registration is not a successful admission;
    both validation and the ordinary retirement rule still apply.
    """
    require_execution_work()
    if fence.backend_type != "postgres":
        raise ExecutionAuthorityError(
            "transaction-bound hosted execution authority requires postgres backend"
        )
    scope = ExecutionCustody(fence)
    try:
        if on_scope_created is not None:
            on_scope_created(scope)
        scope.require_work()
        token = _CURRENT_CUSTODY.set((*current_execution_custody(), scope))
        try:
            yield scope
        finally:
            _CURRENT_CUSTODY.reset(token)
    finally:
        scope.revoke("execution admission retired")


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


_Result = TypeVar("_Result")

_NATIVE_ADMISSION_VALIDATORS: ContextVar[tuple[Callable[[], Awaitable[None]], ...]] = ContextVar(
    "native_execution_admission_validators", default=())


@contextmanager
def bind_native_execution_admission(validator: Callable[[], Awaitable[None]]) -> Iterator[None]:
    """Carry a trusted original read-only gate to actual native dispatch.

    Provider preparation/discovery is not the effect boundary. HTTP requests
    retain this immutable callback inventory before observers and revalidate
    it immediately before the original SDK transport dispatch.
    """
    token = _NATIVE_ADMISSION_VALIDATORS.set((*_NATIVE_ADMISSION_VALIDATORS.get(), validator))
    try:
        yield
    finally:
        _NATIVE_ADMISSION_VALIDATORS.reset(token)


@dataclass(frozen=True)
class _NativeHTTPAdmission:
    client: object = field(repr=False)
    validator: Callable[[], Awaitable[None]] = field(repr=False)

    async def __call__(self) -> None:
        """Defer this client-specific gate until its original HTTP dispatch."""


@contextmanager
def bind_native_http_execution_admission(
    client: object, validator: Callable[[], Awaitable[None]],
) -> Iterator[None]:
    """Reauthorize one selected native HTTP client after request observers.

    An authorization lookup can use a different HTTP client. That lookup keeps
    all ordinary execution gates, without recursively invoking the peer gate
    whose decision it is computing. The immutable callback remains in copied
    contexts and applies to every actual dispatch through the selected client.
    """
    if not isinstance(client, (_NativeHTTPXAsyncClient, _NativeHTTPX2AsyncClient)):
        raise TypeError("Native HTTP admission requires an actual native client")
    if not callable(validator):
        raise TypeError("Native HTTP admission requires a trusted validator")
    with bind_native_execution_admission(_NativeHTTPAdmission(client, validator)):
        yield


def capture_native_execution_admission() -> tuple[Callable[[], Awaitable[None]], ...]:
    """Retain the original trusted gate inventory for a deferred resource."""
    return _NATIVE_ADMISSION_VALIDATORS.get()


@contextmanager
def bind_native_execution_admission_snapshot(captured: tuple[Callable[[], Awaitable[None]], ...]) -> Iterator[None]:
    """Join original native gates with caller gates; never replace either."""
    if not isinstance(captured, tuple) or not all(callable(validator) for validator in captured):
        raise TypeError("native admission snapshot must be an immutable callable tuple")
    existing = _NATIVE_ADMISSION_VALIDATORS.get()
    token = _NATIVE_ADMISSION_VALIDATORS.set(existing + tuple(
        validator for validator in captured if validator not in existing))
    try:
        yield
    finally:
        _NATIVE_ADMISSION_VALIDATORS.reset(token)


async def require_native_execution_admission(owner: Any = None) -> None:
    """Revalidate the current original dispatch gates at a final handoff.

    This checks retained custody and trusted validators; it never issues a
    replacement scope or turns a routing identifier into execution authority.
    """
    require_execution_work(owner)
    for validator in _NATIVE_ADMISSION_VALIDATORS.get():
        await validator()
    require_execution_work(owner)


async def await_execution_work(
    owner: Any, operation: Callable[[], Awaitable[_Result]], *,
    acquired_result_cleanup: Callable[[_Result], Awaitable[None]] | None = None,
) -> _Result:
    """Dispatch a lazy provider operation under its original runtime custody.

    The callable is evaluated only after admission, and its owned children
    inherit the same scopes. A failed provider cannot turn authority loss into
    a recoverable route failure; a successful provider cannot disclose a result
    after loss. This does not recall an already submitted remote operation.
    """
    validators = _NATIVE_ADMISSION_VALIDATORS.get()
    def retain_retirement(error):
        hook = getattr(type(owner), "_retain_native_execution_retirement", None)
        if execution_cleanup_debts(error) and callable(hook):
            hook(owner, error)
    with bind_execution_runtime(owner):
        try:
            # Owned provider children may start after their caller's entry
            # check. WebSocket dispatch has no HTTP transport hook, so the
            # same original validators must run BEFORE the lazy native effect.
            for validator in validators:
                await validator()
            require_execution_work(owner)
            result = await operation()
        except (ExecutionAuthorityError, asyncio.CancelledError) as error:
            retain_retirement(error)
            raise
        except Exception as error:
            retain_retirement(error)
            if is_execution_control_error(error):
                raise
            try:
                require_execution_work(owner)
            except BaseException as denial:
                # Failed physical retirement is not superseded by later
                # revocation. Retain its native owner/debt on the denial too.
                raise execution_terminal_error(error, denial)
            raise
        try:
            require_execution_work(owner)
            # Dispatch admission is not permission to publish credentials or
            # other results after the original constitutional gate changes.
            for validator in validators:
                await validator()
            require_execution_work(owner)
        except BaseException as denial:
            # A successful OPEN may already own an HTTP response. Admission
            # loss must not erase the sole native handle before its adapter
            # gets an opportunity to install the ordinary stream closer.
            if acquired_result_cleanup is not None:
                try:
                    # The SAME canonical physical-close boundary owns both
                    # ordinary streams and an acquired SDK context/result.
                    # Its lazy callback cannot run before Task admission, and
                    # refusal retains the original acquired value explicitly.
                    await close_native_execution_stream(
                        result, owner, close_operation=acquired_result_cleanup,
                        retain_originals=True, isolate_control_originals=True,
                    )
                except BaseException as cleanup_error:
                    terminal = execution_terminal_error(
                        denial, cleanup_error,
                        retain_originals=True, isolate_control_originals=True,
                    )
                    retain_retirement(terminal)
                    raise terminal
            raise
        return result


def retain_native_execution_stream_close(stream: Any, owner: Any = None) -> None:
    """Give SDK-implicit and adapter-explicit HTTP close the SAME native owner.

    HTTPX marks a response closed before awaiting its body retirement. A
    cancelled first close therefore cannot safely be repaired by calling the
    original close again. Retain its task before iteration can start it.
    """
    from kestrel_sovereign._async_ownership import OwnedTaskAdmission, await_owned_task, create_owned_task

    response_types = (_NativeHTTPXResponse, _NativeHTTPX2Response)
    response = stream if isinstance(stream, response_types) else getattr(stream, "response", None)
    if not isinstance(response, response_types):
        return
    if "kestrel.native_close_owner" in response.extensions:
        return
    native_close = response.aclose
    captured = current_execution_custody(owner)
    task = None
    admission = None
    opening = None

    async def joined_close():
        nonlocal task, admission, opening
        original = sys.exc_info()[1]
        if admission is None:
            admission = OwnedTaskAdmission()
            try:
                with bind_execution_cleanup(owner, captured):
                    task = create_owned_task(
                        native_close(), retirement_resource=response,
                        retain_originals=True, admission=admission,
                    )
            except BaseException as error:
                opening = error
                task = admission.task
        if task is None:
            raise execution_terminal_error(
                original, opening, retain_originals=True,
                isolate_control_originals=True,
            )
        outcome = await await_owned_task(task)
        retirement = None
        if outcome.error is not None:
            # The failed closer may itself report a shared native receipt.
            # Its private resource/Task debt belongs on this finite carrier,
            # never on the source exception owned by another operation.
            retirement = RuntimeError("original native HTTP response retirement failed")
            retirement._kestrel_native_retirement_debts = ((response, task),)
        if opening is not None or outcome.error is not None or outcome.cancellation is not None:
            if (outcome.error is None and outcome.cancellation is not None
                and (original is None or not is_execution_control_error(original))):
                raise outcome.cancellation
            raise execution_terminal_error(
                original, opening, outcome.error, retirement, outcome.cancellation,
                retain_originals=True, prefer_cancellation=True,
                isolate_control_originals=True,
            )

    response.extensions["kestrel.native_close_owner"] = joined_close
    response.aclose = joined_close


class _CustodyHTTPBody(_NativeHTTPXAsyncByteStream, _NativeHTTPX2AsyncByteStream):
    """Guard native body advances after arbitrary response observers await."""

    def __init__(self, response, scopes):
        self.response, self.stream, self.scopes = response, response.stream, scopes

    async def __aiter__(self):
        iterator = self.stream.__aiter__()
        try:
            while True:
                with bind_execution_custody_snapshot(self.scopes):
                    require_execution_work()
                    try:
                        chunk = await anext(iterator)
                    except StopAsyncIteration:
                        return
                    require_execution_work()
                yield chunk
        except BaseException as original:
            try:
                await close_native_execution_stream(self.response)
            except BaseException as cleanup:
                raise execution_terminal_error(
                    original, cleanup, retain_originals=True,
                    isolate_control_originals=True,
                )
            raise

    async def aclose(self):
        await self.stream.aclose()


def retain_native_execution_http_client_close(client: Any) -> None:
    """Acquire response custody for SDKs that hide it inside generators.

    HTTPX response hooks run before body iteration. Capture each request's
    own active custody there, never the first caller's scope on a shared
    client. The hook only installs the same per-response physical close owner
    used by exposed SDK streams; it grants no execution or retry authority.
    """
    if not isinstance(client, (_NativeHTTPXAsyncClient, _NativeHTTPX2AsyncClient)):
        return
    if getattr(client, "_kestrel_native_close_hook", None) is not None:
        return

    # Transport extensions are observer-mutable descriptive metadata. Capture
    # each original send's authority privately, including across redirects and
    # nested/concurrent calls on this same shared native client.
    send_admission = ContextVar("kestrel_original_native_http_send", default=None)

    def original_admission():
        admission = send_admission.get()
        if admission is None:
            raise ExecutionAuthorityError("native HTTP dispatch has no original send admission")
        return admission

    async def retain_response(response):
        captured, _validators = original_admission()
        if not isinstance(response.stream, _CustodyHTTPBody):
            response.stream = _CustodyHTTPBody(response, captured)
        retain_native_execution_stream_close(response)
        sink = _NATIVE_RESPONSE_SINK.get()
        if sink is not None:
            sink.retain(response)
        # Retain first so denial during SDK entry/error-body reading still
        # has an owner that physically closes the acquired native response.
        require_original_native_work()

    async def require_request_authority(request):
        # A native SDK must not spend an already-revoked admission, even if
        # a caller enabled internal retry or automatic function-call dispatch.
        # Check THAT request's active scope, never the client's first caller.
        captured, validators = original_admission()
        request.extensions["kestrel.execution_custody"] = captured
        request.extensions["kestrel.native_admission"] = validators
        require_original_native_work()

    native_dispatch = client._send_single_request
    native_send = client.send

    async def send_with_original_control(*args, **kwargs):
        # The SDK's retry/normalization boundary surrounds HTTPX.send, not
        # only the socket dispatch. Async response observers and eager body
        # reads can fail AFTER a real effect, before an SDK sees the result.
        token = send_admission.set((current_execution_custody(), _NATIVE_ADMISSION_VALIDATORS.get()))
        try:
            try:
                return await native_send(*args, **kwargs)
            except BaseException as original:
                sink = _NATIVE_RESPONSE_SINK.get()
                if sink is not None:
                    sink.retain_control(original)
                raise
        finally:
            send_admission.reset(token)

    async def dispatch_after_observers(request):
        # HTTPX runs every request observer before this native boundary. A
        # check at the beginning of that list cannot admit the final dispatch.
        captured, validators = original_admission()
        with bind_execution_custody_snapshot(captured):
            require_original_native_work()
            for validator in validators:
                if isinstance(validator, _NativeHTTPAdmission):
                    if validator.client is client:
                        await validator.validator()
                else:
                    await validator()
            require_original_native_work()
            return await native_dispatch(request)

    def require_original_native_work():
        sink = _NATIVE_RESPONSE_SINK.get()
        if sink is not None and sink.control is not None:
            # A retry belongs to this SAME operation, not a fresh admission.
            # Do not replace its original failure with a later scope denial.
            raise retained_execution_error(sink.control)
        try:
            require_execution_work()
        except BaseException as original:
            if sink is not None:
                sink.retain_control(original)
            raise

    # A caller's existing observer may itself read/close the response, or
    # cancellation may land while it awaits. Own the body before ANY observer.
    client.event_hooks["response"].insert(0, retain_response)
    client.event_hooks["request"].insert(0, require_request_authority)
    client._send_single_request = dispatch_after_observers
    client.send = send_with_original_control
    client._kestrel_native_close_hook = retain_response


async def close_native_execution_stream(
    stream: Any, owner: Any = None, *, retain_originals: bool = True,
    close_operation: Callable[[Any], Awaitable[None]] | None = None,
    isolate_control_originals: bool = True,
) -> None:
    """Physically join an acquired SDK/native close under cleanup-only custody.

    A physical epilogue is a finite observation: by default its opening,
    native closer and actual observer withdrawal all remain explicit, without
    appending private retirement history to a shared source receipt.
    An acquired SDK context/result can supply its exact physical exit instead
    of a method on that value; admission and failed-owner debt stay canonical.
    """
    from kestrel_sovereign._async_ownership import OwnedTaskAdmission, await_owned_task, create_owned_task

    close = None
    if close_operation is None:
        retain_native_execution_stream_close(stream, owner)
        close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
        if not callable(close):
            return  # Legacy in-memory iterators have no native response resource.
    original = sys.exc_info()[1]
    async def owned_close():
        with bind_execution_cleanup(owner):
            result = close() if close_operation is None else close_operation(stream)
            if inspect.isawaitable(result):
                await result
    admission = OwnedTaskAdmission()
    try:
        close_task = create_owned_task(
            owned_close(), retirement_resource=stream, retain_originals=retain_originals,
            admission=admission,
        )
    except BaseException as error:
        close_task = admission.task
        if close_task is None:
            raise execution_terminal_error(
                original, error, retain_originals=True,
                isolate_control_originals=isolate_control_originals,
            )
        # Canonical receipt publication failed after accepting a start-gated
        # Task. Join THAT failed original; its native resource is still debt,
        # never a successful or detached cleanup effect.
    outcome = await await_owned_task(close_task)
    retirement = None
    if outcome.error is not None:
        # Keep the exact failed attempt reachable even if the native SDK has
        # already set an optimistic "closed" flag. Never retry that closer.
        if isolate_control_originals:
            retirement = RuntimeError("original native stream retirement failed")
            retirement._kestrel_native_retirement_debts = ((stream, close_task),)
        else:
            outcome.error._kestrel_native_retirement_debts = (
                *getattr(outcome.error, "_kestrel_native_retirement_debts", ()),
                (stream, close_task),
            )
    if outcome.error is not None or outcome.cancellation is not None:
        if (outcome.error is None and outcome.cancellation is not None
            and (original is None or isinstance(original, GeneratorExit))
            and not is_execution_control_error(original)):
            # Plain GeneratorExit is the iterator's close protocol, not an
            # independent failure. Attaching it beneath actual withdrawal
            # lets later ordinary selectors incorrectly promote it over C.
            # A control-bearing GeneratorExit is still original evidence.
            raise outcome.cancellation
        raise execution_terminal_error(
            original, outcome.error, retirement, outcome.cancellation,
            retain_originals=retain_originals, prefer_cancellation=retain_originals,
            isolate_control_originals=isolate_control_originals,
        )


class NativeExecutionStreamOwner:
    """Retain every acquired SDK/native handle until physical retirement.

    A response can be acquired before an SDK context manager enters: its
    error-body read may fail first. The native response hook registers it
    here before any observer or read, so failed acquisition still has an owner.
    """

    def __init__(self):
        self.handles = []
        self.control = None
        self.retirement_error = None

    def retain(self, handle):
        if not any(handle is existing for existing in self.handles):
            self.handles.append(handle)
        return handle

    def retain_control(self, error):
        """Keep finite originals before an SDK can normalize or retry them."""
        if is_execution_control_error(error):
            self.control = execution_terminal_error(
                self.control, error, retain_originals=True,
                isolate_control_originals=True,
            )

    async def close(self, original=None):
        # SDKs can wrap a hook's original denial as APIConnectionError. The
        # physical response owner retains the exact pre-normalization receipt.
        terminal = execution_terminal_error(
            self.control, original, self.retirement_error,
            retain_originals=True, isolate_control_originals=True,
        )
        if self.retirement_error is not None:
            terminal._kestrel_native_retirement_owner = self
            raise terminal
        for handle in reversed(tuple(self.handles)):
            try:
                await close_native_execution_stream(
                    handle, retain_originals=True, isolate_control_originals=True,
                )
            except BaseException as error:
                if execution_cleanup_debts(error):
                    self.retirement_error = execution_terminal_error(
                        self.retirement_error, error, retain_originals=True,
                        isolate_control_originals=True,
                    )
                    terminal = execution_terminal_error(
                        terminal, error, retain_originals=True,
                        isolate_control_originals=True,
                    )
                else:
                    # Caller cancellation AFTER a successful physical close is
                    # a request receipt, not failed transport retirement. Keep
                    # canonical original control; an ordinary SDK status error
                    # must not mask this later observer cancellation.
                    self.handles[:] = [retained for retained in self.handles if retained is not handle]
                    if isinstance(error, asyncio.CancelledError) and not is_execution_control_error(terminal):
                        terminal = error
                    else:
                        terminal = execution_terminal_error(
                            terminal, error, retain_originals=True,
                            isolate_control_originals=True,
                        )
            else:
                self.handles[:] = [retained for retained in self.handles if retained is not handle]
        if self.retirement_error is not None:
            # The exception's original native close task owns failed handles;
            # this request owner also keeps its inventory for later shutdown.
            terminal._kestrel_native_retirement_owner = self
        if terminal is not None and (terminal is not original or original is None):
            raise terminal


@asynccontextmanager
async def native_execution_response_owner(client: Any):
    """Own hidden native responses through acquisition, reading and failure."""
    retain_native_execution_http_client_close(client)
    owner = NativeExecutionStreamOwner()
    token = _NATIVE_RESPONSE_SINK.set(owner)
    try:
        yield owner
    finally:
        original = sys.exc_info()[1]
        try:
            await owner.close(original)
        finally:
            _NATIVE_RESPONSE_SINK.reset(token)


def native_sdk_operation(function: Callable) -> Callable:
    """Keep one native-response owner around a complete non-stream SDK call.

    SDK retry/normalization happens inside the adapter operation. The same
    owner therefore spans those attempts, including asynchronously observed
    failures before an SDK has returned a response to its adapter. Streams
    keep their existing explicit acquisition/transfer boundary instead.
    """
    @wraps(function)
    async def guarded(self, *args, **kwargs):
        client = args[0] if args else kwargs.get("client")
        async with native_execution_response_owner(getattr(client, "_client", None)):
            return await function(self, *args, **kwargs)
    return guarded


@asynccontextmanager
async def owned_native_http_client(client: Any):
    """Own a temporary client through physical pool/transport retirement.

    HTTPX's context exit is separate from response closure. Retain the whole
    client close, so interrupted pool cleanup cannot erase a request receipt.
    """
    if not isinstance(client, (_NativeHTTPXAsyncClient, _NativeHTTPX2AsyncClient)):
        async with client as entered:
            yield entered
        return
    original = None
    try:
        await client.__aenter__()
        retain_native_execution_http_client_close(client)
        yield client
    except BaseException as error:
        original = error
    try:
        await close_native_execution_stream(client)
    except BaseException as cleanup:
        original = execution_terminal_error(original, cleanup)
    if original is not None:
        raise original


def execution_work_operation(function: Callable) -> Callable:
    """Keep finalization and its children inside the original runtime scope."""
    @wraps(function)
    async def guarded(self, *args, **kwargs):
        return await await_execution_work(self, lambda: function(self, *args, **kwargs))
    return guarded


async def await_execution_work_group(
    owner: Any,
    operations: Iterable[Callable[[], Awaitable[Any]]],
    *,
    return_exceptions: bool = False,
) -> list[Any]:
    """Own all fan-out children and retain irreversible cleanup evidence.

    Ordinary tolerant discovery may return ordinary failures as values, but
    control failure and cancellation always cancel/join every sibling. Lazy
    callables avoid creating undispatched coroutine objects after denial.
    """
    import asyncio

    from kestrel_sovereign._async_ownership import await_owned_task

    tasks = []

    async def run(operation):
        try:
            return await await_execution_work(owner, operation)
        except BaseException as error:
            if (
                isinstance(error, Exception)
                and return_exceptions
                and not is_execution_control_error(error)
                and not execution_cleanup_debts(error)
            ):
                return error
            # asyncio.gather reads a cancelled child's exception before its
            # later physical owner can join it. Retain the original receipt
            # inside the child, before asyncio consumes its cancellation.
            task = asyncio.current_task()
            if task is not None:
                task._kestrel_owned_terminal_error = retained_execution_error(error)
            raise

    try:
        require_execution_work(owner)
        for operation in operations:
            coroutine = run(operation)
            try:
                task = asyncio.create_task(coroutine)
            except BaseException:
                coroutine.close()
                raise
            tasks.append(task)
        results = list(await asyncio.gather(*tasks))
        require_execution_work(owner)
        return results
    except BaseException as error:
        for task in tasks:
            if not task.done():
                task.cancel()
        outcomes = [await await_owned_task(task) for task in tasks]
        terminal = execution_terminal_error(
            error,
            *(outcome.error for outcome in outcomes),
            *(outcome.cancellation for outcome in outcomes),
        )
        assert terminal is not None
        raise terminal


def execution_work_stream(function: Callable | None = None, *, lifecycle: Callable | None = None) -> Callable:
    """Monotonic custody across foreign consumers, errors, and fallback.

    Context is installed only during an advance/close, never across yield to
    the consumer. Every later consumer's admissions join the pinned union.
    """
    if function is None:
        return lambda source: execution_work_stream(source, lifecycle=lifecycle)
    @wraps(function)
    def guarded(self, *args, **kwargs) -> AsyncIterator[Any]:
        # Async-generator bodies run on first pull, too late to capture the
        # creator's authority. Construct only the lazy forwarder here.
        return execution_stream(self, function(self, *args, **kwargs),
                                lifecycle=None if lifecycle is None else lifecycle(self))
    return guarded


async def run_execution_blocking_operation(owner: Any, operation: Callable, *args):
    """Admit at the actual foreign-worker entry, then physically join it.

    Cleanup uses the separate unrestricted physical-join primitive. This
    work-only boundary is opt-in for native inference and model/file work.
    """
    from kestrel_sovereign._async_ownership import run_blocking_operation
    require_execution_work(owner)
    def run_original_work():
        require_execution_work(owner)
        return operation(*args)
    result = await run_blocking_operation(run_original_work)
    require_execution_work(owner)
    return result


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


@dataclass(frozen=True)
class ExecutionStreamLifecycle:
    """Opt-in physical producer custody supplied by its actual feature lifetime.

    The factory receives the lazy body and its original context. Registration
    finishes before any body step, including under an eager task factory.
    Nested forwarders inherit this same lifetime only inside source work.
    """

    owner_factory: Callable
    owner_started: Callable
    require_work: Callable


_STREAM_LIFECYCLE: ContextVar[tuple[asyncio.Task, ExecutionStreamLifecycle] | None] = ContextVar("execution_stream_lifecycle", default=None)


class _ExecutionForwarder:
    """One source task, monotonic consumer custody, and joined closure.

    The source can hold ContextVar tokens across yields: its advances and
    close always run in the SAME task/context. Consumers never hold those
    bindings, and later foreign consumers may add but never discard custody.
    """

    def __init__(self, owner: Any, stream: AsyncIterator[Any], captured_custody: tuple[ExecutionCustody, ...] = (), *, lifecycle: ExecutionStreamLifecycle | None = None):
        self._runtime_owner = owner
        self._source = aiter(stream)
        self._native_admission = capture_native_execution_admission()
        self._scopes = current_execution_custody(owner)
        if not isinstance(captured_custody, tuple) or not all(isinstance(scope, ExecutionCustody) for scope in captured_custody):
            raise TypeError("captured stream custody must be an immutable ExecutionCustody tuple")
        self._scopes += tuple(scope for scope in captured_custody if scope not in self._scopes)
        self._cleanup_state = _StreamCleanupState(self._scopes)
        self._source_exhausted = False
        # Task-relative capabilities (notably privacy-lock reentry) are not
        # supplied by copy_context alone. Capture them on the lock owner.
        self._turn_scope = capture_turn_scope(owner)
        import contextvars
        context = contextvars.copy_context()
        # The turn/session and privacy-reentry carriers above suffice for
        # source work. Do not delegate CONVERSATION: ordinary child wakes
        # must queue for their own turn, not inherit the parent's live hold.
        # Capture now, dispatch only on the first admitted consumer pull.
        # A deferred HTTP response may never be consumed at all.
        self._owner_context = context
        self._iterator = None
        inherited = _STREAM_LIFECYCLE.get()
        self._lifecycle = lifecycle if lifecycle is not None else (
            inherited[1] if inherited is not None and inherited[0] is asyncio.current_task() else None
        )

    def __aiter__(self):
        return self

    async def __anext__(self):
        additional = current_execution_custody(self._runtime_owner)
        self._scopes += tuple(scope for scope in additional if scope not in self._scopes)
        self._cleanup_state.scopes = self._scopes
        try:
            with _bind_captured_custody(self._scopes), bind_native_execution_admission_snapshot(self._native_admission):
                if self._lifecycle is not None:
                    self._lifecycle.require_work()
                # Trusted read-only checks retain original turn-relative lock
                # reentry only for validation, never across consumer delivery.
                with self._turn_scope.bind():
                    await require_native_execution_admission(self._runtime_owner)
                if self._iterator is None:
                    from kestrel_sovereign._async_ownership import create_owned_iterator
                    from kestrel_sovereign.llm.call_progress import (
                        inherit_call_progress,
                    )
                    inherit_call_progress(self._owner_context)
                    with self._turn_scope.bind():
                        self._iterator = await create_owned_iterator(
                            self._produce, operation="execution-stream-forward",
                            owner_context=self._owner_context,
                            owner_factory=None if self._lifecycle is None else self._lifecycle.owner_factory,
                            owner_started=None if self._lifecycle is None else self._lifecycle.owner_started,
                        )
                item = await anext(self._iterator)
                require_execution_work(self._runtime_owner)
                if self._lifecycle is not None:
                    self._lifecycle.require_work()
                with self._turn_scope.bind():
                    await require_native_execution_admission(self._runtime_owner)
        except BaseException as error:
            # Admission can fail while ENTERING the original binding, before
            # the next native pull. The suspended producer still has to retire.
            try:
                await self.aclose()
            except BaseException as cleanup_error:
                raise execution_terminal_error(error, cleanup_error, retain_originals=True, isolate_control_originals=True)
            if not is_execution_control_error(error):
                with _bind_captured_custody(self._scopes):
                    require_execution_work(self._runtime_owner)
            raise error
        return item

    async def _produce(self):
        error = None
        lifetime_token = _STREAM_LIFECYCLE.set(
            None if self._lifecycle is None else (asyncio.current_task(), self._lifecycle)
        )
        try:
            with self._turn_scope.bind(), _bind_stream_cleanup((self._cleanup_state,)):
                while True:
                    with _bind_captured_custody(self._scopes), bind_execution_runtime(self._runtime_owner), bind_native_execution_admission_snapshot(self._native_admission):
                        if self._lifecycle is not None:
                            self._lifecycle.require_work()
                        await require_native_execution_admission(self._runtime_owner)
                        try:
                            item = await anext(self._source)
                        except StopAsyncIteration:
                            require_execution_work(self._runtime_owner)
                            self._source_exhausted = True
                            return
                        require_execution_work(self._runtime_owner)
                        if self._lifecycle is not None:
                            self._lifecycle.require_work()
                        await require_native_execution_admission(self._runtime_owner)
                    yield item
        except BaseException as caught:
            error = caught
            self._cleanup_state.requested = True
            raise
        finally:
            _STREAM_LIFECYCLE.reset(lifetime_token)
            # Carrier admission itself can fail before source advancement.
            # Already-acquired handles still need physical cleanup in that case.
            close = getattr(self._source, "aclose", None)
            if callable(close):
                with bind_execution_cleanup(self._runtime_owner, self._scopes):
                    try:
                        await close()
                    except BaseException as cleanup_error:
                        self._source_exhausted = False
                        self._cleanup_state.requested = True
                        raise execution_terminal_error(error, cleanup_error, retain_originals=True, isolate_control_originals=True)

    async def aclose(self):
        # Normal EOF completes this source, not a queued child's independent
        # conversation turn. Aborted or failed closure still denies children
        # synchronously before the source owner is cancelled.
        if not self._source_exhausted:
            self._cleanup_state.requested = True
        await close_execution_stream(
            self._runtime_owner,
            self._source if self._iterator is None else self._iterator,
            self._scopes,
        )


async def close_execution_stream(
    owner: Any, stream: Any, captured: tuple[ExecutionCustody, ...] = (),
) -> None:
    """Join an owned source's close despite repeated caller cancellation."""
    from kestrel_sovereign._async_ownership import await_owned_operation

    close = getattr(stream, "aclose", None)
    if callable(close):
        with bind_execution_cleanup(owner, captured):
            outcome = await await_owned_operation(close(), retirement_resource=stream, retain_originals=True)
            error = execution_terminal_error(outcome.error, outcome.cancellation, retain_originals=True, isolate_control_originals=True)
            if error is not None:
                raise error


@asynccontextmanager
async def owned_execution_stream(owner: Any, stream: AsyncIterator[Any]):
    """Own every forwarding layer until the source's terminal classification."""
    iterator = _ExecutionForwarder(owner, stream)
    error = None
    try:
        yield iterator
    except BaseException as caught:
        error = caught
        raise
    finally:
        try:
            await iterator.aclose()
        except BaseException as cleanup_error:
            raise execution_terminal_error(error, cleanup_error, retain_originals=True, isolate_control_originals=True)


def execution_stream(owner: Any, stream: AsyncIterator[Any], *, captured_custody: tuple[ExecutionCustody, ...] = (), lifecycle: ExecutionStreamLifecycle | None = None) -> AsyncIterator[Any]:
    """Pin custody now for a deferred body and own its physical source.

    The caller must explicitly close this iterator (for example with Core's
    ClosingStreamingResponse). Construction does not dispatch source work.
    """
    return _ExecutionForwarder(owner, stream, captured_custody, lifecycle=lifecycle)


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


@contextmanager
def _bind_stream_cleanup(captured: tuple[_StreamCleanupState, ...]) -> Iterator[None]:
    existing = _STREAM_CLEANUP.get()
    token = _STREAM_CLEANUP.set(existing + tuple(state for state in captured if state not in existing))
    try:
        yield
    finally:
        _STREAM_CLEANUP.reset(token)


turn_scoped(
    "execution_stream_cleanup",
    variables=(_STREAM_CLEANUP,),
    capture=lambda owner: _STREAM_CLEANUP.get(),
    bind=_bind_stream_cleanup,
)
