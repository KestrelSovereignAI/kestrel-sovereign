"""Closed original request-row acquisition custody, never an SQL permit.

One root holds the original backing across I/O. Every checkpoint uses a fresh
native writer and exact old ciphertext/revision; only its full native ACK can
advance the root. A dispatch checkpoint never permits another native create.
"""
import asyncio
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from pydantic import SecretBytes
from kestrel_sdk.computer_use.acquisition import (
    EnvironmentAcquisitionCheckpoint, EnvironmentAcquisitionPreparation,
    EnvironmentAcquisitionStorage,
)
from kestrel_sdk.resources import ResourceConflict
from kestrel_sdk.security.aead import AEADCipher
from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission, OwnedTaskOutcome, await_owned_task, create_owned_task, owned_task_admission_error,
)
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, execution_observation_error, execution_terminal_error, require_execution_work,
)
from .original_result_backing import _OriginalResultBacking, _OriginalResultCustody

_ADMISSION = object()


@dataclass(frozen=True, eq=False, repr=False)
class _AcceptedAcquisitionResult:
    task: object
    identity: tuple
    payload: bytes
    plain: bytes
    revision: int
    guard: object
    runtime_custody: object = None


@dataclass(frozen=True, eq=False, repr=False)
class _AcquisitionEntry:
    values: tuple
    plain: bytes
    storage: bytes
    kind: str
    ordinary: bool
    result_summary: object = None


@dataclass(frozen=True, eq=False, repr=False)
class _AcquisitionAdmission:
    task: object
    identity: tuple
    expected: bytes
    plain: bytes
    revision: int
    guard: object
    privacy_holds: tuple
    native_backing: object
    expires_at: datetime
    spending_expires_at: datetime
    runtime_custody: object = None
    writer: object = None
    entry: object = None
    native: object = None
    last_ack: object = None
    native_receipt: object = None
    join: object = None


@dataclass(frozen=True, eq=False, repr=False)
class _AcquisitionJoin:
    owner: object
    admission: object
    task: object
    outcome: OwnedTaskOutcome


@dataclass(frozen=True, eq=False, repr=False)
class _AcquisitionStageAck:
    owner: object
    entry: object
    writer: object
    sqlite_connection: object
    acknowledgments: tuple


class _OriginalAcquisitionResult(_OriginalResultCustody):
    """A finite existing-row owner distinct from agent and human results."""

    def __init__(self, repository, acceptance):
        if (type(self) is not _OriginalAcquisitionResult
                or type(acceptance) is not _AcceptedAcquisitionResult
                or repository._original_results.get(acceptance, False) is not None
                or acceptance.task is not asyncio.current_task()):
            raise ExecutionAuthorityError('Original acquisition admission required')
        self._capture_native_backing(repository)
        self._acceptance = acceptance
        self._identity = acceptance.identity
        self._expected, self._plain, self._revision = acceptance.payload, acceptance.plain, acceptance.revision
        self._guard = acceptance.guard
        self._runtime_custody = acceptance.runtime_custody
        self._entry = self._writer_admission = None
        self._postgres_relation = self._postgres_session = self._sqlite_relation = None
        self._query = self._match_query = None
        self._acknowledgments = ()
        self._sqlite_committed_connection = None
        self._observer_depth = 0
        self._phase, self._admission_key = 'accepted', _ADMISSION
        self._native = None
        self._last_ack = None
        self._join = None
        # Normalize only the already authenticated original row, under live
        # admission. Native worker checks never parse a caller DTO after Stop.
        original = json.loads(self._plain)
        self._expires_at = datetime.fromisoformat(original['request']['expires_at'])
        self._spending_expires_at = datetime.fromisoformat(original['quote']['expires_at'])
        self._registration = _AcquisitionAdmission(self._owner_task, self._identity,
            self._expected, self._plain, self._revision, self._guard,
            self._native_backing.privacy_holds, self._native_backing, self._expires_at, self._spending_expires_at,
            self._runtime_custody)
        repository._original_results[acceptance] = self
        self._backend._original_result_holds[self] = self._closed
        try:
            self._require_backing()
        except BaseException:
            self.close()
            raise

    def _fixed_queries(self):
        from .db.native_row_custody import _request_row_queries
        return _request_row_queries(self)

    def _require_registered(self):
        if type(self) is not _OriginalAcquisitionResult:
            raise ExecutionAuthorityError('Exact original acquisition owner required')
        r = self._registration
        if (type(r) is not _AcquisitionAdmission
                or type(self._acceptance) is not _AcceptedAcquisitionResult
                or self._admission_key is not _ADMISSION
                or self._repository._original_results.get(self._acceptance) is not self
                or self._owner_task is not r.task or self._identity is not r.identity
                or self._expected is not r.expected or self._plain is not r.plain
                or type(self._revision) is not int or self._revision != r.revision
                or self._guard is not r.guard or self._writer_admission is not r.writer
                or self._entry is not r.entry or self._native is not r.native
                or self._last_ack is not r.last_ack or self._join is not r.join or self._observer_depth
                or self._runtime_custody is not r.runtime_custody
                or self._runtime_custody is not self._acceptance.runtime_custody):
            raise ExecutionAuthorityError('Original acquisition identity retired')
        if self._runtime_custody is not None:
            from kestrel_sovereign.visual_computer.storage_acquisition import _CoreAcquisitionCustody
            if type(self._runtime_custody) is not _CoreAcquisitionCustody:
                raise ExecutionAuthorityError('Original runtime acquisition consumer changed')
            self._runtime_custody._require_issued()
        if self._expires_at is not r.expires_at or self._spending_expires_at is not r.spending_expires_at:
            raise ExecutionAuthorityError('Original acquisition lifetime changed')
        if (self._query is not None or self._match_query is not None) and (self._query, self._match_query) != self._fixed_queries():
            raise ExecutionAuthorityError('Original acquisition fixed native program differs')

    def _require_backing(self):
        self._require_registered()
        self._require_native_backing()

    def _require_native_producer_lifetime(self, native):
        from kestrel_sovereign.visual_computer.acquisition_native import _NativeAcquisitionHandle
        from kestrel_sovereign.execution_sync_http import _THREAD
        if (type(native) is not _NativeAcquisitionHandle or native is not self._native
                or native._root is not self or native._state != 'running'
                or not native._in_original_worker() or _THREAD.get() is not native._thread
                or native._thread.acquisition is not native or self._phase != 'accepted'):
            raise ExecutionAuthorityError('Original acquisition running native worker required')
        self._require_registered()
        self._require_captured_native_lifetime()

    def check_current(self):
        if (asyncio.current_task() is not self._owner_task or self._phase != 'accepted'):
            raise ExecutionAuthorityError('Original acquisition operation retired')
        self._require_backing()
        self._require_current_authority()

    def _require_current_authority(self, *, spending=False):
        """Original synchronous restriction, usable by its exact native job."""
        from kestrel_sovereign.execution_sync_http import _check_sync_restriction
        _check_sync_restriction(self._guard)
        require_execution_work(self._backend)
        now = datetime.now(UTC)
        if self._expires_at <= now or (spending and self._spending_expires_at <= now):
            raise ResourceConflict('Original acquisition request or spending quote expired')

    def require_admission(self, backend):
        if backend is not self._backend:
            raise ExecutionAuthorityError('Original acquisition backend differs')
        self.check_current()

    def require_original(self, backend):
        if (backend is not self._backend or self._phase not in {'bound', 'submitted'}
                or type(self._writer_admission) is not OwnedTaskAdmission
                or self._writer_admission.task is self._owner_task
                or asyncio.current_task() is not self._writer_admission.task):
            raise ExecutionAuthorityError('Original acquisition native writer required')
        self._require_backing()
        if self._entry.ordinary:
            self._require_current_authority(spending=self._entry.kind == 'dispatch')
        else:
            self._require_native_result()

    def _value(self):
        from kestrel_sovereign.visual_computer.repository import CustodyRecord
        return self._repository._open(self._identity[0], self._identity[1], self._expected, CustodyRecord)

    def _storage(self):
        from kestrel_sovereign.visual_computer.repository import _AcquisitionEnvelope
        value = json.loads(self._plain).get('acquisition')
        return None if value is None else EnvironmentAcquisitionStorage.deserialize_private(_AcquisitionEnvelope.from_private(value).storage)

    @property
    def checkpoint(self):
        self.check_current()
        return self._storage().checkpoint

    @property
    def preparation(self):
        self.check_current()
        return self._storage().preparation

    async def allocate(self, request, quote):
        self.check_current()
        value = self._value()
        if value.request != request or value.quote != quote:
            raise ResourceConflict('Original acquisition request or quote differs')
        storage = self._storage()
        if storage is None:
            import secrets
            checkpoint = EnvironmentAcquisitionCheckpoint.from_original('acquisition-'+secrets.token_hex(16), request, quote)
            await self._transition(EnvironmentAcquisitionStorage(checkpoint=checkpoint), 'allocate')
        else:
            storage.checkpoint.require_original(request, quote)

    async def prepare_once(self, value):
        self.check_current()
        value = EnvironmentAcquisitionPreparation.model_validate(value)
        storage = self._storage()
        checkpoint = storage.checkpoint.prepared(value)
        if storage.preparation is not None:
            return storage.preparation
        await self._transition(EnvironmentAcquisitionStorage(checkpoint=checkpoint, preparation=value), 'prepare')
        return self.preparation

    def _attach_native(self, native):
        from kestrel_sovereign.visual_computer.acquisition_native import _NativeAcquisitionHandle
        self.check_current()
        if type(native) is not _NativeAcquisitionHandle or native._root is not self or self._native is not None:
            raise ExecutionAuthorityError('Original acquisition native preparation differs')
        if self._runtime_custody is not None:
            self._runtime_custody._refuse_native_worker(self, native._binding._attempt)
        self._native = native
        self._registration = replace(self._registration, native=native)

    async def _prepare_native(self, native, value):
        from kestrel_sovereign.visual_computer.acquisition_native import _NativeAcquisitionHandle
        self.check_current()
        if (type(native) is not _NativeAcquisitionHandle or native._root is not self
                or self._native is not native or native._state != 'inactive'):
            raise ExecutionAuthorityError('Original acquisition native preparation differs')
        value = EnvironmentAcquisitionPreparation.model_validate(value)
        storage = self._storage()
        checkpoint = storage.checkpoint.prepared(value)
        try:
            await self._transition(EnvironmentAcquisitionStorage(checkpoint=checkpoint,
                preparation=storage.preparation or value), 'bind' if storage.preparation is not None else 'prepare')
            native._state = 'prepared'
        except BaseException:
            native._state = 'failed'
            raise

    async def begin_dispatch(self):
        self.check_current()
        self._require_current_authority(spending=True)
        storage = self._storage()
        if storage.checkpoint.state in {'dispatching', 'accepted', 'unknown'}:
            return False
        self._require_native_preparation()
        await self._transition(EnvironmentAcquisitionStorage(checkpoint=storage.checkpoint.dispatched(), preparation=storage.preparation), 'dispatch')
        self._native._arm_original_dispatch(self)
        return True

    def _require_native_preparation(self):
        if self._native is None:
            raise ExecutionAuthorityError('Original native acquisition producer absent')
        from kestrel_sovereign.visual_computer.acquisition_native import _NativeAcquisitionHandle
        if type(self._native) is not _NativeAcquisitionHandle:
            raise ExecutionAuthorityError('Original native acquisition producer absent')
        self._native._require_prepared(self)

    def _seal_native_result(self, native, receipt):
        from kestrel_sovereign.visual_computer.acquisition_native import _NativeAcquisitionReceipt
        self._require_backing()
        if (asyncio.current_task() is not self._owner_task or self._phase != 'accepted'
                or native is not self._native or native._root is not self
                or type(receipt) is not _NativeAcquisitionReceipt or native._receipt is not receipt
                or receipt.handle is not native or native._state != 'joined'
                or self._registration.native_receipt is not None):
            raise ExecutionAuthorityError('Original acquisition joined result seal differs')
        self._registration = replace(self._registration, native_receipt=receipt)

    def _require_native_result(self):
        if self._native is None:
            raise ExecutionAuthorityError('Original native acquisition result absent')
        from kestrel_sovereign.visual_computer.acquisition_native import _NativeAcquisitionHandle
        if type(self._native) is not _NativeAcquisitionHandle:
            raise ExecutionAuthorityError('Original native acquisition result absent')
        self._native._require_result(self)

    async def record_unknown(self, lease=None):
        self._require_native_result()
        if lease is not None and not self._native._matches_lease(lease, state='unknown'):
            raise ExecutionAuthorityError('Original acquisition native lease differs')
        storage = self._storage()
        if self._native._minimal_acknowledged:
            return storage.checkpoint
        checkpoint = storage.checkpoint.unknown(self._native._original_minimal_lease())
        await self._transition(EnvironmentAcquisitionStorage(checkpoint=checkpoint, preparation=storage.preparation), 'minimal', ordinary=False)
        self._native._ack_minimal(self, checkpoint)
        return checkpoint

    async def record_accepted(self, lease):
        self.check_current()
        self._require_native_result()
        lease = self._native._accepted_proposal(self, lease)
        storage = self._storage()
        if storage.checkpoint.state == 'accepted':
            return storage.checkpoint
        checkpoint = storage.checkpoint.accepted(lease)
        await self._transition(EnvironmentAcquisitionStorage(checkpoint=checkpoint, preparation=storage.preparation), 'accepted')
        return checkpoint

    async def _transition(self, storage, kind, *, ordinary=True):
        if asyncio.current_task() is not self._owner_task or self._phase != 'accepted':
            raise ExecutionAuthorityError('Original acquisition stage closed or already submitted')
        self._require_backing()
        if ordinary:
            self.check_current()
        else:
            self._require_native_result()
        if self._backend.owns_open_transaction:
            raise ExecutionAuthorityError('Original acquisition refuses an enclosing transaction')
        from kestrel_sovereign.visual_computer.repository import _AcquisitionEnvelope
        plain = json.loads(self._plain)
        old_envelope = plain.get('acquisition')
        previous = self._storage()
        allowed = {'allocate': ('unprepared', 0), 'prepare': ('prepared', 1), 'bind': ('prepared', 0),
            'dispatch': ('dispatching', 1), 'minimal': ('unknown', 1), 'accepted': ('accepted', 1)}
        if (type(storage) is not EnvironmentAcquisitionStorage or kind not in allowed
                or type(ordinary) is not bool or ordinary != (kind != 'minimal')
                or storage.checkpoint.state != allowed[kind][0]
                or storage.checkpoint.sequence != (previous.checkpoint.sequence if previous else 0)+allowed[kind][1]
                or (kind == 'allocate') != (previous is None)
                or (previous is not None and storage.checkpoint.attempt_id != previous.checkpoint.attempt_id)):
            raise ExecutionAuthorityError('Original acquisition fixed transition differs')
        # Original identities are validated before binding the immutable stage.
        value = self._value()
        storage.checkpoint.require_original(value.request, value.quote)
        private = storage.serialize_private()
        previous_envelope = _AcquisitionEnvelope.from_private(old_envelope) if old_envelope else None
        binding = previous_envelope.binding if previous_envelope is not None else None
        native_result = previous_envelope.native_result if previous_envelope is not None else None
        if self._native is not None and kind in {'prepare', 'bind'}:
            original_binding = SecretBytes(self._native._binding._binding_bytes)
            if binding is not None and binding != original_binding:
                raise ExecutionAuthorityError('Original acquisition prepared native binding changed')
            binding = original_binding
        if plain.get('remote_environment') is not None and kind in {'prepare', 'bind'}:
            # Original vendor correlation joins this SAME prepare_once ACK.
            # No separate metadata transaction or duplicate create body exists.
            from kestrel_sovereign.visual_computer.runpod_quote_storage import (
                _RunpodEnvironmentEnvelope, _RunpodPreparationCorrelation,
            )
            import hashlib
            if self._native is None:
                raise ExecutionAuthorityError('Original native remote preparation required')
            original = self._native._binding
            envelope = _RunpodEnvironmentEnvelope.from_private(plain['remote_environment'])
            correlation = _RunpodPreparationCorrelation(
                plan_sha256=original._prepared.plan_sha256.get_secret_value(),
                binding=original._prepared.binding.get_secret_value(),
                create_sha256=hashlib.sha256(storage.preparation.create_payload.get_secret_value()).hexdigest(),
                worker_identity_sha256=hashlib.sha256(storage.preparation.worker_identity.get_secret_value()).hexdigest())
            if envelope.prepared is not None and envelope.prepared != correlation:
                raise ExecutionAuthorityError('Original remote preparation correlation changed')
            plain['remote_environment'] = envelope.model_copy(update={'prepared': correlation}).private_value()
        if kind == 'minimal' and self._native._receipt.native_result is not None:
            original_result = SecretBytes(self._native._receipt.native_result)
            if native_result is not None and native_result != original_result:
                raise ExecutionAuthorityError('Original acquisition native result changed')
            native_result = original_result
        plain['acquisition'] = _AcquisitionEnvelope(storage=private, binding=binding, native_result=native_result).private_value()
        plain['revision'] = self._revision + 1
        if kind == 'dispatch':
            plain['phase'] = 'acquiring'
        if kind in {'minimal', 'accepted'}:
            plain['phase'] = 'leased' if kind == 'accepted' else 'acquiring'
            if storage.checkpoint.lease is not None:
                plain['lease'] = storage.checkpoint.lease.model_dump(mode='json')
        raw = json.dumps(plain).encode()
        owner, request_id, session_key = self._identity
        payload = AEADCipher(self._key).encrypt(raw, aad=f'{owner}:{request_id}'.encode())
        self._entry = _AcquisitionEntry((payload, self._revision+1, owner, request_id, session_key,
            self._expected, self._revision), raw, private.get_secret_value(), kind, ordinary)
        self._phase = 'bound'
        self._acknowledgments, self._sqlite_committed_connection = (), None
        admission = self._writer_admission = OwnedTaskAdmission()
        self._join = None
        self._registration = replace(self._registration, writer=admission, entry=self._entry, join=None)
        opening = None
        try:
            writer = create_owned_task(self._backend._settle_original_acquisition_result(self),
                admission=admission, retain_originals=True, name='original-acquisition-result')
        except BaseException as error:
            opening = owned_task_admission_error(self, admission, error)
            writer = admission.task
            if writer is None:
                self._phase = 'failed'
                raise opening
        outcome = await await_owned_task(writer)
        self._join = _AcquisitionJoin(self, admission, writer, outcome)
        self._registration = replace(self._registration, join=self._join)
        error = execution_observation_error(opening, outcome.error, outcome.cancellation)
        if error is not None:
            self._phase = 'failed'
            raise self.retain_committed(error)
        if self._phase != 'acknowledged':
            self._phase = 'failed'
            raise ExecutionAuthorityError('Original acquisition native ACK absent')
        entry = self._entry
        self._last_ack = _AcquisitionStageAck(self, entry, admission,
            self._sqlite_committed_connection, self._acknowledgments)
        self._expected, self._plain, self._revision = entry.values[0], entry.plain, entry.values[1]
        self._entry = self._writer_admission = None
        self._registration = replace(self._registration, expected=self._expected, plain=self._plain,
            revision=self._revision, writer=None, entry=None, last_ack=self._last_ack)
        self._phase = 'accepted'

    def begin(self, backend):
        self.require_original(backend)
        if self._phase != 'bound':
            raise ExecutionAuthorityError('Original acquisition stage already submitted')
        self._phase = 'submitted'
        return self._entry

    def require_sqlite_cleanup(self, backend, connection):
        if (type(self) is not _OriginalAcquisitionResult or self._admission_key is not _ADMISSION
                or backend is not self._backend or connection is not self._backing
                or self._phase != 'submitted' or type(self._writer_admission) is not OwnedTaskAdmission
                or asyncio.current_task() is not self._writer_admission.task):
            raise ExecutionAuthorityError('Original acquisition native cleanup owner differs')

    def has_acknowledged_commit(self):
        return self._sqlite_committed_connection is self._backing or any(r.acknowledged for r in self._acknowledgments)

    def acknowledged(self, backend):
        self.require_original(backend)
        if not self.has_acknowledged_commit():
            raise ExecutionAuthorityError('Original acquisition native ACK absent')
        self._phase = 'acknowledged'

    def write_refusal(self, reason, cause=None):
        from kestrel_sovereign.execution_custody import _FINITE_NATIVE_SOURCES, _native_source_originals
        refusal = ExecutionAuthorityError('Original acquisition write refused: '+reason)
        refusal.__cause__ = cause
        finite = BaseExceptionGroup('original acquisition write refusal', [refusal])
        finite._kestrel_finite_native_sources = (_FINITE_NATIVE_SOURCES, _native_source_originals(cause))
        return finite

    def retain_committed(self, error):
        from kestrel_sovereign.signals.outcome_custody import _OriginalSignalOutcome
        return _OriginalSignalOutcome.retain_committed(self, error)

    def _require_close_owner(self):
        """Only retire the original physical hold; acquire no current work."""
        backing = self._native_backing
        writer = self._writer_admission
        if writer is None and self._last_ack is not None:
            writer = self._last_ack.writer
        if (type(self) is not _OriginalAcquisitionResult
                or asyncio.current_task() is not self._owner_task
                or type(backing) is not _OriginalResultBacking
                or backing.owner is not self
                or self._registration.native_backing is not backing
                or backing.closed is not self._closed
                or backing.owner_task is not self._owner_task
                or backing.repository is not self._repository
                or backing.backend is not self._backend
                or (writer is not None and writer.task is not None
                    and (type(self._join) is not _AcquisitionJoin
                        or self._registration.join is not self._join
                        or self._join.owner is not self
                        or self._join.admission is not writer
                        or self._join.task is not writer.task
                        or type(self._join.outcome) is not OwnedTaskOutcome))):
            raise ExecutionAuthorityError('Original acquisition closes only after its original owner joins')
        if self._phase != 'closed' and (
                self._repository._original_results.get(self._acceptance) is not self
                or self._backend._original_result_holds.get(self) is not self._closed):
            raise ExecutionAuthorityError('Original acquisition close registration differs')

    def close(self):
        self._require_close_owner()
        if self._phase == 'closed':
            return
        if self._native is not None and self._native._state != 'closed':
            raise ExecutionAuthorityError('Original acquisition native owner must close first')
        self._phase = 'closed'
        # Physical retirement does not end the enclosing runtime invocation.
        # Its canonical repository context alone retires that restriction.
        if self._acceptance.runtime_custody is None:
            self._repository._original_results.pop(self._acceptance, None)
        self._backend._original_result_holds.pop(self, None)
        if not self._closed.done():
            self._closed.set_result(None)

    async def aclose(self):
        # A copied descriptor cannot close the genuine producer/client first.
        self._require_close_owner()
        failures = []
        if self._native is not None:
            try:
                await self._native._finalize_original()
            except BaseException as error:
                failures.append(error)
            try:
                await self._native._close_original()
            except BaseException as error:
                failures.append(error)
        try:
            self.close()
        except BaseException as error:
            failures.append(error)
        if failures:
            raise execution_terminal_error(*failures, retain_originals=True)
