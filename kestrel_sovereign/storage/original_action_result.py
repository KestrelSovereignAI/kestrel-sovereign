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
    ExecutionAuthorityError, require_execution_work,
    execution_observation_error,
)
from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission, OwnedTaskOutcome, create_owned_task, await_owned_task, owned_task_admission_error,
)
from .original_result_backing import _OriginalResultCustody, _OriginalResultBacking

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
    binding: object = None
    private_source: object = None
    writer: object = None
    entry: object = None
    native_backing: object = None
    join: object = None


@dataclass(frozen=True, eq=False, repr=False)
class _ActionResultJoin:
    owner: object
    admission: object
    task: object
    outcome: object


def _closed_action_result(result):
    """Finite Core owners, not an extensible profile or subclass admission."""
    from kestrel_sovereign.visual_computer.dispatch import _OriginalDispatch
    if type(result) is _OriginalActionResult:
        return _AcceptedActionResult, _OriginalDispatch, 'agent'
    from kestrel_sovereign.storage.original_human_action_result import (
        _AcceptedHumanActionResult, _OriginalHumanActionResult,
    )
    from kestrel_sovereign.visual_computer.dispatch import _OriginalHumanDispatch
    if type(result) is _OriginalHumanActionResult:
        return _AcceptedHumanActionResult, _OriginalHumanDispatch, 'human'
    raise ExecutionAuthorityError('Exact Core original action result owner required')


def _origin_action(origin, channel):
    return origin.request if channel == 'agent' else origin.request.action


class _OriginalActionResultCustody(_OriginalResultCustody):
    """Private mechanical engine for the two closed original action owners."""
    def _capture(self, repository, origin, owner, session_key, payload, stored, *, admission):
        from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
        from kestrel_sovereign.storage.db.postgres import PostgresBackend
        from kestrel_sovereign.visual_computer.repository import VisualCustodyRepository
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        if type(repository) is not VisualCustodyRepository or type(repository.db) is not AsyncDatabase:
            raise ExecutionAuthorityError('Original result requires its canonical native database')
        acceptance_type, origin_type, channel = _closed_action_result(self)
        database = repository.db
        backend = database.backend
        if (type(repository) is not VisualCustodyRepository
                or type(admission) is not acceptance_type
                or admission not in repository._original_results or repository._original_results[admission] is not None
                or admission.task is not asyncio.current_task() or admission.origin is not origin
                or admission.identity != (owner,session_key,stored.request.action_id)
                or admission.payload != bytes(payload) or admission.stored_json != stored.model_dump_json()
                or type(backend) not in (SQLiteBackend, PostgresBackend)
                or type(origin) is not origin_type or not origin.active
                or _origin_action(origin, channel) != stored.request or origin.trace is not None or origin.driver_entered
                or stored.channel != channel
                or stored.control_id != (None if channel == 'agent' else origin.request.control_id)):
            raise ExecutionAuthorityError('Original prepared action result admission required')
        if channel == 'human' and (type(origin.binding) is not tuple or len(origin.binding) != 4
                or admission.binding is not origin.binding
                or any(value is not original for value, original in zip(
                    (origin.session, origin.lease, origin.request, origin.frame), origin.binding))
                or admission.private_identity != (owner, repository._address(
                    'private', origin.session.session_id, 'human', origin.request.control_id))
                or type(admission.private_payload) is not bytes or not admission.private_payload):
            raise ExecutionAuthorityError('Original human result binding differs')
        self._capture_native_backing(repository)
        self._acceptance = admission
        self._origin = origin
        self._identity = (owner, session_key, stored.request.action_id)
        self._expected = bytes(payload)
        self._stored = stored.model_copy(deep=True)
        self._postgres_relation = self._postgres_session = self._sqlite_relation = None
        self._entry = None
        self._writer_admission = None
        self._join = None
        self._acknowledgments = ()
        self._sqlite_committed_connection = None
        self._observer_depth = 0
        self._phase = 'accepted'
        self._admission_key = _ADMISSION
        self._query = self._match_query = None
        privacy_holds = self._native_backing.privacy_holds
        self._registration = _ActionResultAdmission(self._owner_task, origin, self._identity,
            self._expected, self._stored.model_dump_json(), self._stored, self._stored.request, privacy_holds,
            binding=admission.binding if channel == 'human' else None,
            private_source=(admission.private_identity, admission.private_payload) if channel == 'human' else None,
            native_backing=self._native_backing)
        repository._original_results[admission] = self
        backend._original_result_holds[self] = self._closed
        try:
            self._require_backing()
        except BaseException:
            self.close()
            raise

    def _require_backing(self):
        from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
        acceptance_type, origin_type, channel = _closed_action_result(self)
        backend = self._backend
        registration = self._registration
        if (type(registration) is not _ActionResultAdmission
                or type(self._acceptance) is not acceptance_type
                or type(self._origin) is not origin_type):
            raise ExecutionAuthorityError('Original action result descriptor type differs')
        if channel == 'human' and (type(registration.binding) is not tuple or len(registration.binding) != 4
                or type(registration.private_source) is not tuple or len(registration.private_source) != 2
                or self._origin.binding is not registration.binding
                or self._acceptance.binding is not registration.binding
                or self._acceptance.private_identity is not registration.private_source[0]
                or self._acceptance.private_payload is not registration.private_source[1]
                or any(value is not original for value, original in zip(
                    (self._origin.session, self._origin.lease, self._origin.request, self._origin.frame),
                    registration.binding))):
            raise ExecutionAuthorityError('Original human result binding retired')
        self._require_native_backing()
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

    def require_admission(self, backend):
        _closed_action_result(self)
        if (getattr(self, '_admission_key', None) is not _ADMISSION
                or backend is not self._backend or asyncio.current_task() is not self._owner_task
                or self._phase != 'accepted'):
            raise ExecutionAuthorityError('Original action result live admission required')
        self._require_backing()
        require_execution_work(backend)

    def require_original(self, backend):
        _closed_action_result(self)
        if (getattr(self, '_admission_key', None) is not _ADMISSION
                or backend is not self._backend or self._phase not in {'bound','submitted'}
                or type(self._writer_admission) is not OwnedTaskAdmission
                or self._writer_admission.task is self._owner_task
                or asyncio.current_task() is not self._writer_admission.task):
            raise ExecutionAuthorityError('Original action result native writer required')
        self._require_backing()

    def dispatch_payload(self, original, payload, stored):
        self.require_admission(self._backend)
        if self._stored.dispatch_started or original != self._expected or stored.model_dump(exclude={'dispatch_started'}) != self._stored.model_dump(exclude={'dispatch_started'}) or stored.receipt is not None or not stored.dispatch_started:
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
        _closed_action_result(self)
        # Physical rollback/queue retirement remains owed after key, registry,
        # or observation loss. This is not admission to write/COMMIT/work.
        if (getattr(self, '_admission_key', None) is not _ADMISSION
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
        _closed_action_result(self)
        if (getattr(self, '_admission_key', None) is not _ADMISSION
                or asyncio.current_task() is not self._owner_task or self._phase != 'accepted'
                or original is not self._origin):
            raise ExecutionAuthorityError('Original action result owner differs or closed')
        self._require_backing()
        if self._backend.owns_open_transaction:
            raise ExecutionAuthorityError('Original result refuses an enclosing transaction')
        original.require(fact)
        if _origin_action(original, _closed_action_result(self)[2]) != self._stored.request:
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
        self._join = _ActionResultJoin(self, admission, writer, outcome)
        self._registration = replace(self._registration, join=self._join)
        error = execution_observation_error(opening, outcome.error, outcome.cancellation)
        if error is not None:
            raise self.retain_committed(error)

    def close(self):
        _closed_action_result(self)
        backing = self._native_backing
        if (asyncio.current_task() is not self._owner_task
                or type(backing) is not _OriginalResultBacking
                or backing.owner is not self
                or self._registration.native_backing is not backing
                or backing.closed is not self._closed
                or backing.owner_task is not self._owner_task
                or backing.repository is not self._repository
                or backing.backend is not self._backend
                or (self._writer_admission is not None and self._writer_admission.task is not None
                    and (type(self._join) is not _ActionResultJoin
                        or self._registration.join is not self._join
                        or self._join.owner is not self
                        or self._join.admission is not self._writer_admission
                        or self._join.task is not self._writer_admission.task
                        or type(self._join.outcome) is not OwnedTaskOutcome))):
            raise ExecutionAuthorityError('Original action result closes only after its original owner joins')
        if self._phase == 'closed':
            return
        if (self._repository._original_results.get(self._acceptance) is not self
                or self._backend._original_result_holds.get(self) is not self._closed):
            raise ExecutionAuthorityError('Original action result close registration differs')
        self._phase = 'closed'
        self._repository._original_results.pop(self._acceptance, None)
        self._backend._original_result_holds.pop(self, None)
        if not self._closed.done():
            self._closed.set_result(None)


class _OriginalActionResult(_OriginalActionResultCustody):
    """Exact original agent action owner; human custody cannot enter here."""
    def __init__(self, repository, origin, owner, session_key, payload, stored, *, admission):
        self._capture(repository, origin, owner, session_key, payload, stored, admission=admission)
