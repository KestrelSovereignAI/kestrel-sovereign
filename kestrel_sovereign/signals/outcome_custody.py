"""Opaque permission for ONE accepted signal's fixed native terminal row.

Never installed on an executor, never carries a SQL callback, and never grants
ordinary storage, source, tool, provider, or UI work. Original denials remain
in the ambient context and native commit/cleanup evidence retains them.
"""

import asyncio
from copy import deepcopy
from dataclasses import dataclass, replace

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, current_execution_custody, require_execution_work,
)

_JOURNAL_ADMISSION = object()


class SignalOutcomeJournalPrivacyRefusal(ExecutionAuthorityError):
    """Terminal audit refused; the original completed effect did not roll back."""

    def __init__(self, reason, signal_result):
        super().__init__('Original signal journal privacy refusal: ' + reason)
        self.reason = reason
        self.signal_result = signal_result


class SignalOutcomeJournalWriteRefusal(ExecutionAuthorityError):
    """The genuine effect completed, but no exact terminal row can commit."""

    def __init__(self, reason, signal_result):
        super().__init__('Original signal journal write refusal: ' + reason)
        self.reason = reason
        self.signal_result = signal_result


@dataclass(frozen=True, eq=False)
class _AcceptedSignalOutcome:
    """Original dispatcher-issued envelope, not a causation-context proxy."""
    owner: asyncio.Task
    identity: tuple
    registration: object


@dataclass(frozen=True, eq=False, repr=False)
class _OriginalPostgresJournalPool:
    """Opaque native pool configuration accepted for this one fixed row.

    Public asyncpg reconfiguration replaces these objects; retaining the pool
    object alone would allow a late journal to acquire a different destination.
    Connection arguments may contain secrets and must never be serialized.
    """
    pool: object
    generation: int
    connect_args: object
    connect_kwargs: object
    wrapper: object

    @classmethod
    def capture(cls, backing):
        from kestrel_sovereign.storage.db.postgres import PostgresPoolAdapter
        adapted = isinstance(backing, PostgresPoolAdapter)
        native = backing.native_pool if adapted else backing
        return cls(native, native._generation, native._connect_args,
                   native._connect_kwargs, backing.wrap_checkout if adapted else None)

    def require_original(self, backing):
        from kestrel_sovereign.storage.db.postgres import PostgresPoolAdapter
        adapted = isinstance(backing, PostgresPoolAdapter)
        native = backing.native_pool if adapted else backing
        if (native is not self.pool
                or (backing.wrap_checkout if adapted else None) is not self.wrapper
                or native._generation != self.generation
                or native._connect_args is not self.connect_args
                or native._connect_kwargs is not self.connect_kwargs
                or native.is_closing()):
            raise ExecutionAuthorityError('Original signal journal pool configuration retired')


@dataclass(frozen=True)
class _OriginalPostgresJournalRelation:
    """Actual catalog identity resolved under the original live admission."""
    database_name: str
    namespace_oid: int
    relation_oid: int
    schema_name: str
    relation_name: str

    @property
    def qualified_name(self):
        return '.'.join('"' + value.replace('"', '""') + '"'
                        for value in (self.schema_name, self.relation_name))


class _OriginalSignalOutcome:
    def __init__(self, dispatcher, signal, registration, *, acceptance=None):
        from kestrel_sovereign.signals.store import SignalLogStore
        from kestrel_sovereign.storage.db.postgres import PostgresBackend
        from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

        admission = dispatcher._durable_admission.get()
        if (admission is None or admission[0] is not asyncio.current_task()
                or asyncio.current_task() not in dispatcher._durable_admission_owners):
            raise ExecutionAuthorityError('Original accepted dispatch required for terminal journal')
        if (type(acceptance) is not _AcceptedSignalOutcome
                or acceptance not in dispatcher._accepted_signal_outcomes
                or dispatcher._accepted_signal_outcomes[acceptance] is not None
                or acceptance.owner is not asyncio.current_task()
                or acceptance.identity != self._signal_identity(signal)
                or acceptance.registration is not registration):
            raise ExecutionAuthorityError('Original accepted dispatch record required, issued once')
        if type(dispatcher._store) is not SignalLogStore:
            raise ExecutionAuthorityError('Original native signal store required')
        backend = dispatcher._store._backend
        if type(backend) not in (SQLiteBackend, PostgresBackend):
            raise ExecutionAuthorityError('Original native signal backend required')
        require_execution_work(backend)
        # Admission identity for classifying this invocation's later surface
        # denial only. These references never authorize fresh UI/storage work.
        self._surface_scopes = current_execution_custody(dispatcher._agent)
        require_execution_work(dispatcher._agent)
        self._dispatcher = dispatcher
        self._store = dispatcher._store
        self._backend = backend
        self._agent = dispatcher._agent
        self._agent_id = dispatcher._agent.did
        self._agent_scope = getattr(self._agent, '_execution_custody', None)
        from kestrel_sovereign.storage.privacy_wrapper import (
            _resolve_durable_persistence_gate, _resolve_privacy_policy_epoch,
        )
        self._privacy_gate = _resolve_durable_persistence_gate(self._agent)
        self._privacy_epoch = _resolve_privacy_policy_epoch(self._agent)
        from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate
        self._privacy_exclusive_hold = (
            self._privacy_gate._capture_original_exclusive_hold()
            if type(self._privacy_gate) is DurablePersistenceGate else None)
        self._registration = registration
        # Payload containers and privacy settings belong to acceptance, not
        # to a handler's mutable arguments after revocation.
        self._signal_snapshot = deepcopy(signal)
        self._registration_snapshot = replace(registration,
            log_redaction=replace(registration.log_redaction))
        table = 'main.signal_log' if type(backend) is SQLiteBackend else 'signal_log'
        self._query = SignalLogStore._outcome_insert_query(table)
        self._match_query = SignalLogStore._outcome_match_query(table, postgres=type(backend) is PostgresBackend)
        self._acceptance = acceptance
        self._identity = self._signal_identity(signal)
        self._backend_scope = getattr(backend, '_execution_custody', None)
        self._backing = backend._connection if type(backend) is SQLiteBackend else backend._pool
        self._generation = backend.connection_generation if type(backend) is SQLiteBackend else None
        self._postgres_pool = (
            _OriginalPostgresJournalPool.capture(self._backing)
            if type(backend) is PostgresBackend else None)
        self._postgres_relation = None
        self._sqlite_relation = None
        self._entry = None
        self._completed_result = None
        self._acknowledgments = ()
        self._sqlite_committed_connection = None
        self._audit_loss_recorded = False
        self._outer_admission = self._writer_admission = None
        self._phase = 'accepted'
        self._observer_depth = 0
        self._require_backing()
        self._admission_key = _JOURNAL_ADMISSION
        dispatcher._accepted_signal_outcomes[acceptance] = self

    @staticmethod
    def _signal_identity(signal):
        return (signal.id, signal.source, signal.target_agent, signal.mode)

    def matches(self, signal, registration):
        return (self._identity == self._signal_identity(signal)
                and registration is self._registration)

    def privacy_refusal(self, reason, cause=None):
        return self._finite_refusal(reason, cause, privacy=True)

    def write_refusal(self, reason):
        return self._finite_refusal(reason, privacy=False)

    def _finite_refusal(self, reason, cause=None, *, privacy):
        from kestrel_sovereign.execution_custody import _FINITE_NATIVE_SOURCES, _native_source_originals
        if type(privacy) is not bool:
            raise TypeError('Original journal refusal kind must be a fixed boolean')
        kind = SignalOutcomeJournalPrivacyRefusal if privacy else SignalOutcomeJournalWriteRefusal
        original = kind(reason, self._completed_result)
        original.__cause__ = cause
        finite = BaseExceptionGroup('original signal terminal audit refusal', [original])
        finite._kestrel_finite_native_sources = (_FINITE_NATIVE_SOURCES, _native_source_originals(cause))
        return finite

    def _require_backing(self):
        # Identity/privacy getters are observers, never a reentrant native owner.
        # Keep this on the actual capability (not a copyable ContextVar).
        if self._observer_depth:
            raise ExecutionAuthorityError('Original journal observer reentry refused')
        self._observer_depth += 1
        try:
            backend = self._backend
            if (self._dispatcher._store is not self._store
                    or self._store._backend is not backend
                    or self._dispatcher._agent is not self._agent
                    or self._agent.did != self._agent_id
                    or getattr(self._agent, '_execution_custody', None) is not self._agent_scope
                    or getattr(backend, '_execution_custody', None) is not self._backend_scope):
                raise ExecutionAuthorityError('Original signal journal runtime identity changed')
            from kestrel_sovereign.storage.privacy_wrapper import (
                _resolve_durable_persistence_gate, _resolve_privacy_policy_epoch,
            )
            try:
                gate = _resolve_durable_persistence_gate(self._agent)
                epoch = _resolve_privacy_policy_epoch(self._agent)
            except BaseException as error:
                raise self.privacy_refusal('identity_unavailable', error)
            if gate is not self._privacy_gate or epoch != self._privacy_epoch:
                raise self.privacy_refusal('superseded')
            hold = self._privacy_exclusive_hold
            if hold is not None and (gate._exclusive_hold is not hold or not hold.live):
                raise self.privacy_refusal('exclusive_hold_retired')
            if backend.backend_type == 'sqlite':
                if (self._backing is None or backend._connection is not self._backing
                        or backend.connection_generation != self._generation or backend._closing):
                    raise ExecutionAuthorityError('Original signal journal connection retired')
                backend.assert_connected_file_still_valid()
            else:
                if self._backing is None or backend._pool is not self._backing:
                    raise ExecutionAuthorityError('Original signal journal pool retired')
                self._postgres_pool.require_original(self._backing)
        finally:
            self._observer_depth -= 1

    def require_preparation(self, result):
        """Check the sealed return before reservation or callback reentrancy."""
        if (self._observer_depth or asyncio.current_task() is not self._acceptance.owner
                or self._phase != 'accepted'
                or result is not self._completed_result
                or self._dispatcher._accepted_signal_outcomes.get(self._acceptance) is not self):
            raise ExecutionAuthorityError('Original pipeline result preparation unavailable or reentrant')

    def bind(self, signal, registration, result, prepared):
        from kestrel_sovereign.signals.store import _PreparedSignalOutcome
        self._require_backing()
        if (result is not self._completed_result
                or asyncio.current_task() is not self._acceptance.owner
                or self._phase != 'preparing' or not self.matches(signal, registration)
                or result.signal_id != self._identity[0]
                or type(prepared) is not _PreparedSignalOutcome
                or type(prepared.values) is not tuple or len(prepared.values) != 30
                or prepared.values[0] != self._identity[0]
                or prepared.values[1] != self._identity[1]
                or prepared.values[6] != self._identity[2]
                or prepared.values[3] != self._identity[3].value):
            raise ExecutionAuthorityError('Original signal journal outcome binding differs')
        self._entry = prepared
        self._phase = 'bound'

    def require_original(self, backend):
        from kestrel_sovereign._async_ownership import OwnedTaskAdmission
        if (type(self) is not _OriginalSignalOutcome
                or getattr(self, '_admission_key', None) is not _JOURNAL_ADMISSION
                or self._observer_depth
                or self._dispatcher._accepted_signal_outcomes.get(self._acceptance) is not self
                or backend is not self._backend
                or type(self._outer_admission) is not OwnedTaskAdmission
                or type(self._writer_admission) is not OwnedTaskAdmission
                or self._outer_admission.task is self._acceptance.owner
                or self._writer_admission.task is self._outer_admission.task
                or self._writer_admission.task is self._acceptance.owner
                or asyncio.current_task() is not self._writer_admission.task
                or self._phase not in ('bound', 'submitted')):
            raise ExecutionAuthorityError('Original native terminal journal permission required')
        self._require_backing()

    def begin(self, backend):
        self.require_original(backend)
        if self._phase != 'bound':
            raise ExecutionAuthorityError('Original signal journal already submitted')
        self._phase = 'submitted'
        return self._entry

    def acknowledged(self, backend):
        self.require_original(backend)
        if self._phase != 'submitted' or not self.has_acknowledged_commit():
            raise ExecutionAuthorityError('Original signal journal lacks native COMMIT evidence')
        self._phase = 'acknowledged'

    def has_acknowledged_commit(self):
        """A native COMMIT fact is independent of final identity admission."""
        return ((self._sqlite_committed_connection is not None
                    and self._sqlite_committed_connection is self._backing)
                or any(receipt.acknowledged for receipt in self._acknowledgments))

    def require_sqlite_cleanup(self, backend, connection):
        """Only this submitted writer's fixed connection; no fresh work grant."""
        from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
        if (self._observer_depth or type(backend) is not SQLiteBackend
                or backend is not self._backend or connection is not self._backing
                or self._phase != 'submitted'
                or self._writer_admission is None
                or asyncio.current_task() is not self._writer_admission.task):
            raise ExecutionAuthorityError('Original SQLite journal native owner differs')

    def retain_committed(self, error):
        """Retain the actual terminal INSERT fact, never new work authority."""
        if error is None or not self.has_acknowledged_commit():
            return error
        from kestrel_sovereign.execution_custody import (
            retain_acknowledged_execution_commit, execution_commit_outcome,
            ExecutionCommitOutcomeError, _FINITE_NATIVE_SOURCES, _native_source_originals,
        )
        if self._acknowledgments:
            # The ACK is a real native fact, but the waiter's withdrawal is
            # not a failed COMMIT/reset source. Keep both on a private finite
            # carrier without poisoning the shared native-source latch.
            finite = BaseExceptionGroup('original signal journal observer withdrawal', [error])
            finite._kestrel_finite_native_sources = (_FINITE_NATIVE_SOURCES, _native_source_originals(error))
            error = finite
        for receipt in self._acknowledgments:
            retain_acknowledged_execution_commit(error, receipt)
        if self._acknowledgments:
            return error
        if execution_commit_outcome(error) is not None:
            return error
        if not isinstance(error, Exception):
            # Preserve actual cancellation/Base control shape. These already
            # escape ordinary Exception catches, unlike an observation error.
            error._kestrel_unresolved_commit_outcome = 'committed'
            error.add_note('original signal terminal row COMMIT acknowledged; reconcile that row before retry')
            return error
        # SQLite's actual inline COMMIT is not an invented PG ACK/reset Task.
        # Use a typed finite classifier, not an ordinary exception annotated
        # with metadata that downstream best-effort catches could discard.
        finite = ExecutionCommitOutcomeError('committed')
        finite.__cause__ = error
        finite._kestrel_finite_native_sources = (_FINITE_NATIVE_SOURCES, _native_source_originals(error))
        return finite


def _require_original_signal_outcome(journal, backend):
    if type(journal) is not _OriginalSignalOutcome:
        raise ExecutionAuthorityError('Exact original native signal journal required')
    journal.require_original(backend)


async def _capture_original_outcome(dispatcher, signal, registration, *, acceptance):
    """Only a live original public dispatch can create this terminal unit."""
    from kestrel_sovereign.signals.store import SignalLogStore
    from kestrel_sovereign.storage.db import postgres as postgres_module
    from kestrel_sovereign.storage.db.postgres import PostgresBackend, PostgresPoolAdapter
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    admission = dispatcher._durable_admission.get()
    if (admission is None or admission[0] is not asyncio.current_task()
            or asyncio.current_task() not in dispatcher._durable_admission_owners):
        raise ExecutionAuthorityError('Original accepted dispatch required for terminal journal')
    store = dispatcher._store
    backend = store._backend
    native = (type(store) is SignalLogStore and store.TABLE == 'signal_log'
              and type(backend) in (SQLiteBackend, PostgresBackend))
    if native and type(backend) is PostgresBackend:
        native = postgres_module.ASYNCPG_AVAILABLE and isinstance(
            backend._pool, (postgres_module.asyncpg.Pool, PostgresPoolAdapter))
    if not native:
        if current_execution_custody(dispatcher._agent) or current_execution_custody(backend):
            raise ExecutionAuthorityError('Hosted terminal journal requires a supported native backend')
        return None  # legacy unscoped embeddings retain ordinary append only
    journal = _OriginalSignalOutcome(dispatcher, signal, registration, acceptance=acceptance)
    if type(backend) is PostgresBackend:
        # Resolve under the original live work admission, before any handler.
        # Later native checkout setup cannot redirect this fixed destination.
        journal._postgres_relation = await backend._capture_original_signal_outcome_relation(journal)
        journal._query = SignalLogStore._outcome_insert_query(journal._postgres_relation.qualified_name)
        journal._match_query = SignalLogStore._outcome_match_query(journal._postgres_relation.qualified_name, postgres=True)
    else:
        journal._sqlite_relation = await backend._read_original_signal_outcome_relation(journal, capture=True)
        if journal._sqlite_relation is None:
            raise ExecutionAuthorityError('Original SQLite journal requires its plain original audit table')
    return journal
