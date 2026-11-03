"""Private SDK acquisition facade over the SAME original encrypted row.

No provider or authority resolver is registered here. The trusted host retains
its original admitted operation; request facts and run leases grant nothing.
"""
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import asyncio

from kestrel_sdk.computer_use import EnvironmentLease

from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown, ResourceUnsupported
from kestrel_sovereign.execution_custody import ExecutionAuthorityError


# The SAME immutable acceptance follows its logical provider invocation into
# joined child Tasks. This is only a restriction: its absence grants nothing,
# and repository/native attachment still require their original real owners.
_RUNTIME_ACQUISITION_RESTRICTION = ContextVar('runtime_acquisition_restriction', default=None)


class _CoreAcquisitionAttempt:
    def __init__(self, result):
        from kestrel_sovereign.storage.original_acquisition_result import _OriginalAcquisitionResult
        if type(result) is not _OriginalAcquisitionResult:
            raise ExecutionAuthorityError('Original acquisition owner required')
        self._result = result
        self._original = self
        self._runtime_custody = result._runtime_custody

    def _require_original(self):
        if (type(self) is not _CoreAcquisitionAttempt or self._original is not self
                or self._runtime_custody is not self._result._runtime_custody):
            raise ExecutionAuthorityError('Original acquisition attempt required')
        self._result._require_registered()
        if self._runtime_custody is not None:
            self._runtime_custody._require_attempt(self)

    @property
    def checkpoint(self):
        self._require_original()
        return self._result.checkpoint
    @property
    def preparation(self):
        self._require_original()
        return self._result.preparation
    def check_current(self):
        self._require_original()
        self._result.check_current()
    async def prepare_once(self, value):
        self._require_original()
        return await self._result.prepare_once(value)
    async def begin_dispatch(self):
        self._require_original()
        return await self._result.begin_dispatch()
    async def record_unknown(self, lease=None):
        self._require_original()
        return await self._result.record_unknown(lease)
    async def record_accepted(self, lease):
        self._require_original()
        return await self._result.record_accepted(lease)


@dataclass(frozen=True, eq=False, repr=False)
class _AcceptedAcquisitionCheckpoint:
    custody: object
    root: object
    ack: object
    lease_json: bytes


class _CoreAcquisitionCustody:
    api_version = 1

    def __init__(self, repository, scope, record, *, guard):
        self._repository, self._scope, self._record, self._guard = repository, scope, record, guard
        self._entered = False
        self._active = False
        self._sdk_open = False
        self._task = asyncio.current_task()
        self._completed = None
        self._attempt = None
        # A restriction issued by this exact runtime consumer, never a grant.
        self._issued = (self, repository, scope, record, guard, self._task)

    def _require_issued(self):
        if (type(self) is not _CoreAcquisitionCustody or type(self._issued) is not tuple
                or len(self._issued) != 6 or any(original is not actual for original, actual in zip(
                    self._issued, (self, self._repository, self._scope, self._record, self._guard, self._task), strict=True))):
            raise ExecutionAuthorityError('Original runtime acquisition consumer required')

    def _require_issuing(self, repository, scope, record):
        self._require_issued()
        if (asyncio.current_task() is not self._task or not self._active
                or self._attempt is not None or repository is not self._repository
                or scope != self._scope or record is not self._record):
            raise ExecutionAuthorityError('Original runtime acquisition issuance differs')

    def _bind_original_attempt(self, attempt):
        self._require_issuing(self._repository, self._scope, self._record)
        if (type(attempt) is not _CoreAcquisitionAttempt
                or attempt._result._runtime_custody is not self):
            raise ExecutionAuthorityError('Original runtime acquisition attempt differs')
        attempt._result.check_current()
        self._attempt = attempt

    def _require_invocation(self, root):
        self._require_issued()
        if (asyncio.current_task() is not self._task or not self._active
                or root._owner_task is not self._task or root._runtime_custody is not self
                or root._repository is not self._repository or root._guard is not self._guard
                or root._identity != (self._repository._owner(self._scope),
                    self._record.request.request_id, self._repository._address('session', self._record.session_id))
                or (self._attempt is not None and self._attempt._result is not root)):
            raise ExecutionAuthorityError('Original runtime provider invocation differs')

    def _require_attempt(self, attempt):
        self._require_invocation(attempt._result)
        if attempt is not self._attempt or not self._sdk_open:
            raise ExecutionAuthorityError('Original runtime acquisition attempt retired or differs')

    @asynccontextmanager
    async def _activate_original(self):
        self._require_issued()
        if asyncio.current_task() is not self._task or self._active or self._attempt is not None:
            raise ExecutionAuthorityError('Original runtime provider invocation already entered')
        self._active = True
        token = None
        try:
            # The canonical allocation COMMIT/reset ACK precedes provider
            # exposure. Its SAME registration restricts the entire invocation,
            # including before and after the provider's one-shot SDK scope.
            async with self._repository.original_acquisition_result(self._scope, self._record,
                    self._record.request, self._record.quote, guard=self._guard,
                    runtime_custody=self) as attempt:
                token = _RUNTIME_ACQUISITION_RESTRICTION.set(attempt._result._acceptance)
                yield
        finally:
            # Keep the restriction through the original physical close/join.
            # A child retains its inherited restriction after the parent exits.
            if token is not None:
                _RUNTIME_ACQUISITION_RESTRICTION.reset(token)
            self._active = False

    def _refuse_native_worker(self, root, attempt):
        self._require_issued()
        if (asyncio.current_task() is not self._task or not self._sdk_open
                or attempt is not self._attempt or type(attempt) is not _CoreAcquisitionAttempt
                or attempt._result is not root or root._runtime_custody is not self
                or root._repository is not self._repository or root._guard is not self._guard):
            raise ExecutionAuthorityError('Original runtime native acquisition consumer differs')
        # A genuine canonical native profile has reached attachment. Neither a
        # quote nor an SDK DTO supplies the missing worker/bootstrap contract.
        raise ResourceUnsupported(
            'Native worker bootstrap and transport are unverified; runtime admission cannot create a resource')

    @asynccontextmanager
    async def operation(self, request, quote):
        self._require_issued()
        if (asyncio.current_task() is not self._task or not self._active or self._entered
                or request != self._record.request or quote != self._record.quote):
            raise ResourceConflict('Original acquisition operation differs or already entered')
        self._entered = True
        attempt = self._attempt
        if type(attempt) is not _CoreAcquisitionAttempt:
            raise ExecutionAuthorityError('Original runtime acquisition attempt absent')
        self._sdk_open = True
        try:
            attempt.check_current()
            yield attempt
            root = attempt._result
            root.check_current()
            # SDK operation scopes may prepare or inspect without dispatch.
            # Only an accepted native result can later satisfy runtime open.
            if root._native is None or root._last_ack is None or root._last_ack.entry.kind != 'accepted':
                return
            root._require_native_result()
            from kestrel_sovereign.storage.original_acquisition_result import _AcquisitionStageAck
            from kestrel_sovereign._async_ownership import OwnedTaskAdmission
            ack, checkpoint = root._last_ack, root._storage().checkpoint
            if (type(ack) is not _AcquisitionStageAck or ack.owner is not root
                    or ack is not root._registration.last_ack or ack.entry.kind != 'accepted'
                    or ack.entry.values[0] is not root._expected or ack.entry.values[1] != root._revision
                    or type(ack.writer) is not OwnedTaskAdmission or ack.writer.task is None
                    or not ack.writer.task.done() or ack.writer.task.cancelled()
                    or checkpoint.state != 'accepted' or type(checkpoint.lease) is not EnvironmentLease
                    or not (ack.sqlite_connection is root._backing or any(value.acknowledged for value in ack.acknowledgments))):
                raise ResourceOutcomeUnknown('original custodied acquisition has no accepted native ACK')
            self._completed = _AcceptedAcquisitionCheckpoint(self, root, ack, checkpoint.lease.model_dump_json().encode())
        finally:
            self._sdk_open = False

    def _accepted(self, repository, scope, lease):
        """Factual ACK identity only; the repository still needs ordinary work."""
        from kestrel_sovereign.storage.original_acquisition_result import _AcquisitionStageAck
        proof = self._completed
        if (type(self) is not _CoreAcquisitionCustody or asyncio.current_task() is not self._task
                or repository is not self._repository or scope != self._scope
                or type(proof) is not _AcceptedAcquisitionCheckpoint or proof.custody is not self):
            raise ResourceOutcomeUnknown('original custodied acquisition has no accepted native ACK')
        root, ack = proof.root, proof.ack
        if (type(ack) is not _AcquisitionStageAck or ack is not root._last_ack
                or ack is not root._registration.last_ack or ack.owner is not root
                or root._repository is not repository or root._owner_task is not self._task
                or root._phase != 'closed' or not root._closed.done()
                or root._native is None or root._native._state != 'closed'
                or not root._native._client_closed or root._registration.native_receipt is not root._native._receipt
                or type(lease) is not EnvironmentLease or lease.model_dump_json().encode() != proof.lease_json):
            raise ResourceConflict('provider result differs from original accepted native checkpoint')
        return proof
