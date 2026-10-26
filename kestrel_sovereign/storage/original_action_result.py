"""One existing encrypted action row's physically owned negative result.

This owner is issued before dispatch under original current admission. It can
only settle the exact original no-dispatch fact; it grants no reads, ordinary
SQL, I/O, delivery, replacement identity or work after retirement.
"""
import asyncio
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from kestrel_sdk.security.aead import AEADCipher
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, current_execution_custody, require_execution_work,
    execution_observation_error,
)
from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission, create_owned_task, await_owned_task, owned_task_admission_error,
)
from kestrel_sovereign.signals.outcome_custody import _OriginalPostgresJournalPool

_ADMISSION = object()


@dataclass(frozen=True, eq=False, repr=False)
class _AcceptedActionResult:
    task: object
    origin: object
    identity: tuple
    payload: bytes
    stored_json: str


@dataclass(frozen=True, repr=False)
class _ActionResultEntry:
    values: tuple
    result_summary: object = None


@dataclass(frozen=True, repr=False)
class _ActionResultAdmission:
    task: object
    origin: object
    identity: tuple
    expected: bytes
    stored_json: str
    stored: object
    request: object
    privacy_holds: tuple
    writer: object = None
    entry: object = None


class _OriginalActionResult:
    def __init__(self, repository, origin, owner, session_key, payload, stored, *, admission):
        from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
        from kestrel_sovereign.storage.db.postgres import PostgresBackend
        from kestrel_sovereign.visual_computer.dispatch import _OriginalDispatch
        from kestrel_sovereign.visual_computer.repository import VisualCustodyRepository
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        if type(repository) is not VisualCustodyRepository or type(repository.db) is not AsyncDatabase:
            raise ExecutionAuthorityError('Original result requires its canonical native database')
        database = repository.db
        backend = database.backend
        if (type(repository) is not VisualCustodyRepository
                or type(admission) is not _AcceptedActionResult
                or admission not in repository._original_results or repository._original_results[admission] is not None
                or admission.task is not asyncio.current_task() or admission.origin is not origin
                or admission.identity != (owner,session_key,stored.request.action_id)
                or admission.payload != bytes(payload) or admission.stored_json != stored.model_dump_json()
                or type(backend) not in (SQLiteBackend, PostgresBackend)
                or type(origin) is not _OriginalDispatch or not origin.active
                or origin.request != stored.request or origin.trace is not None):
            raise ExecutionAuthorityError('Original prepared action result admission required')
        require_execution_work(backend)
        if backend._original_results_closing:
            raise ExecutionAuthorityError('Original result backend is closing')
        self._closed = asyncio.get_running_loop().create_future()
        self._acceptance = admission
        self._repository, self._key, self._origin = repository, repository._key, origin
        self._database = database
        self._backend, self._backend_scope = backend, getattr(backend, '_execution_custody', None)
        self._backend_work_scopes = current_execution_custody(backend)
        self._owner_task = asyncio.current_task()
        self._identity = (owner, session_key, stored.request.action_id)
        self._expected = bytes(payload)
        self._stored = stored.model_copy(deep=True)
        self._backing = backend._connection if type(backend) is SQLiteBackend else backend._pool
        self._generation = backend.connection_generation if type(backend) is SQLiteBackend else None
        self._postgres_pool = _OriginalPostgresJournalPool.capture(self._backing) if type(backend) is PostgresBackend else None
        self._postgres_relation = self._postgres_session = self._sqlite_relation = None
        self._entry = None
        self._writer_admission = None
        self._acknowledgments = ()
        self._sqlite_committed_connection = None
        self._observer_depth = 0
        self._phase = 'accepted'
        self._admission_key = _ADMISSION
        self._query = self._match_query = None
        from kestrel_sovereign.storage.privacy_wrapper import _DURABLE_GATE_SHARED_HOLDS
        privacy_holds = tuple((hold, hold.gate, hold.lease) for hold in _DURABLE_GATE_SHARED_HOLDS.get() if hold.live)
        self._registration = _ActionResultAdmission(self._owner_task, origin, self._identity,
            self._expected, self._stored.model_dump_json(), self._stored, self._stored.request, privacy_holds)
        repository._original_results[admission] = self
        backend._original_result_holds[self] = self._closed
        try:
            self._require_backing()
        except BaseException:
            self.close()
            raise

    def _require_backing(self):
        from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
        from kestrel_sovereign.storage.db.native_journal_programs import _NativeJournalSQLiteConnection
        backend = self._backend
        registration = self._registration
        from kestrel_sovereign.storage.privacy_wrapper import _SharedGateHold, _SharedGateLease
        if any(type(hold) is not _SharedGateHold or hold.live is not True or hold.gate is not gate
                or hold.lease is not lease or type(lease) is not _SharedGateLease
                or type(lease.holds) is not int or lease.holds <= 0
                for hold, gate, lease in registration.privacy_holds):
            raise ExecutionAuthorityError('Original action result privacy hold retired')
        if (self._observer_depth or self._repository._original_results.get(self._acceptance) is not self
                or self._owner_task is not registration.task or self._origin is not registration.origin
                or self._identity != registration.identity or self._expected != registration.expected
                or self._stored is not registration.stored or self._stored.request is not registration.request
                or self._writer_admission is not registration.writer
                or self._entry is not registration.entry
                or self._repository.db is not self._database or self._database.backend is not backend
                or self._repository._key is not self._key
                or getattr(backend, '_execution_custody', None) is not self._backend_scope):
            raise ExecutionAuthorityError('Original action result identity retired')
        if self._query is not None or self._match_query is not None:
            table = ('main.visual_computer_actions' if type(backend) is SQLiteBackend
                     else self._postgres_relation.qualified_name)
            cast = '?' if type(backend) is SQLiteBackend else 'CAST(? AS BYTEA)'
            if (self._query != f"UPDATE {table} SET payload = ?, pending = 0 WHERE owner = ? AND session_key = ? AND action_id = ? AND payload = ? AND pending = 1"
                    or self._match_query != f"SELECT 1 FROM {table} WHERE payload = ? AND owner = ? AND session_key = ? AND action_id = ? AND {cast} IS NOT NULL AND pending = 0"):
                raise ExecutionAuthorityError('Original action result fixed native program differs')
        if type(backend) is SQLiteBackend:
            if (backend._connection is not self._backing or backend.connection_generation != self._generation
                    or backend._closing or type(self._backing._conn) is not _NativeJournalSQLiteConnection
                    or not self._backing._conn._original_journal_programs_supported()):
                raise ExecutionAuthorityError('Original action result native connection retired')
            backend.assert_connected_file_still_valid()
        else:
            if backend._pool is not self._backing:
                raise ExecutionAuthorityError('Original action result pool retired')
            self._postgres_pool.require_original(self._backing)

    def require_admission(self, backend):
        if (type(self) is not _OriginalActionResult or getattr(self, '_admission_key', None) is not _ADMISSION
                or backend is not self._backend or asyncio.current_task() is not self._owner_task
                or self._phase != 'accepted'):
            raise ExecutionAuthorityError('Original action result live admission required')
        self._require_backing()
        require_execution_work(backend)

    def require_original(self, backend):
        if (type(self) is not _OriginalActionResult or getattr(self, '_admission_key', None) is not _ADMISSION
                or backend is not self._backend or self._phase not in {'bound','submitted'}
                or type(self._writer_admission) is not OwnedTaskAdmission
                or self._writer_admission.task is self._owner_task
                or asyncio.current_task() is not self._writer_admission.task):
            raise ExecutionAuthorityError('Original action result native writer required')
        self._require_backing()

    def dispatch_payload(self, original, payload, stored):
        self.require_admission(self._backend)
        if self._stored.dispatch_started or original != self._expected or stored.request != self._stored.request or stored.receipt is not None or not stored.dispatch_started:
            raise ExecutionAuthorityError('Original action result dispatch attempt differs')
        expected, snapshot = bytes(payload), stored.model_copy(deep=True)
        registration = replace(self._registration, expected=expected,
            stored_json=snapshot.model_dump_json(), stored=snapshot, request=snapshot.request)
        self._expected, self._stored, self._registration = expected, snapshot, registration

    def begin(self, backend):
        self.require_original(backend)
        if self._phase != 'bound':
            raise ExecutionAuthorityError('Original action result already submitted')
        self._phase = 'submitted'
        return self._entry

    def require_sqlite_cleanup(self, backend, connection):
        # Physical rollback/queue retirement remains owed after key, registry,
        # or observation loss. This is not admission to write/COMMIT/work.
        if (type(self) is not _OriginalActionResult or getattr(self, '_admission_key', None) is not _ADMISSION
                or backend is not self._backend or connection is not self._backing
                or self._phase != 'submitted' or type(self._writer_admission) is not OwnedTaskAdmission
                or asyncio.current_task() is not self._writer_admission.task):
            raise ExecutionAuthorityError('Original action result native cleanup owner differs')

    def has_acknowledged_commit(self):
        return self._sqlite_committed_connection is self._backing or any(r.acknowledged for r in self._acknowledgments)

    def acknowledged(self, backend):
        self.require_original(backend)
        if not self.has_acknowledged_commit():
            raise ExecutionAuthorityError('Original action result has no native commit evidence')
        self._phase = 'acknowledged'

    def write_refusal(self, reason, cause=None):
        from kestrel_sovereign.execution_custody import _FINITE_NATIVE_SOURCES, _native_source_originals
        refusal = ExecutionAuthorityError('Original action result write refused: '+reason)
        refusal.__cause__ = cause
        finite = BaseExceptionGroup('original action result write refusal', [refusal])
        finite._kestrel_finite_native_sources = (_FINITE_NATIVE_SOURCES, _native_source_originals(cause))
        return finite

    def retain_committed(self, error):
        # Shared native ACK/reset classification, without granting signal or
        # action authority. This method uses only actual native receipt facts.
        from kestrel_sovereign.signals.outcome_custody import _OriginalSignalOutcome
        return _OriginalSignalOutcome.retain_committed(self, error)

    async def not_dispatched(self, original, fact):
        if (type(self) is not _OriginalActionResult or getattr(self, '_admission_key', None) is not _ADMISSION
                or asyncio.current_task() is not self._owner_task or self._phase != 'accepted'
                or original is not self._origin):
            raise ExecutionAuthorityError('Original action result owner differs or closed')
        self._require_backing()
        if self._backend.owns_open_transaction:
            raise ExecutionAuthorityError('Original result refuses an enclosing transaction')
        original.require(fact)
        if original.request != self._stored.request:
            raise ExecutionAuthorityError('Original action result request differs')
        # The snapshot was normalized under live admission. Terminal work
        # never invokes a caller's model/copy/serializer or repository hook.
        stored = json.loads(self._registration.stored_json)
        request = stored['request']
        stored['receipt'] = dict(schema_version=1, action_id=request['action_id'],
            session_id=request['session_id'], observation_id=request['observation_id'],
            control_epoch=request['control_epoch'], outcome='denied',
            recorded_at=datetime.now(UTC).isoformat(), reason_code='host_not_dispatched')
        owner, session_key, action_id = self._identity
        payload = AEADCipher(self._key).encrypt(json.dumps(stored).encode(),
            aad=f'{owner}:{session_key}:{action_id}'.encode())
        self._entry = _ActionResultEntry((payload, owner, session_key, action_id, self._expected))
        self._phase = 'bound'
        admission = self._writer_admission = OwnedTaskAdmission()
        self._registration = replace(self._registration, writer=admission, entry=self._entry)
        opening = None
        try:
            writer = create_owned_task(self._backend._settle_original_action_result(self),
                admission=admission, retain_originals=True, name='original-action-result')
        except BaseException as error:
            opening = owned_task_admission_error(self, admission, error)
            writer = admission.task
            if writer is None:
                raise opening
        outcome = await await_owned_task(writer)
        error = execution_observation_error(opening, outcome.error, outcome.cancellation)
        if error is not None:
            raise self.retain_committed(error)

    def close(self):
        if asyncio.current_task() is not self._owner_task:
            raise ExecutionAuthorityError('Original action result closes only in original owner')
        self._phase = 'closed'
        self._repository._original_results.pop(self._acceptance, None)
        self._backend._original_result_holds.pop(self, None)
        if not self._closed.done():
            self._closed.set_result(None)
