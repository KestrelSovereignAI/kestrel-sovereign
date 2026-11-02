"""Private SDK acquisition facade over the SAME original encrypted row.

No provider or authority resolver is registered here. The trusted host retains
its original admitted operation; request facts and run leases grant nothing.
"""
from contextlib import asynccontextmanager
from dataclasses import dataclass
import asyncio

from kestrel_sdk.computer_use import EnvironmentLease

from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from kestrel_sovereign.execution_custody import ExecutionAuthorityError


class _CoreAcquisitionAttempt:
    def __init__(self, result):
        from kestrel_sovereign.storage.original_acquisition_result import _OriginalAcquisitionResult
        if type(result) is not _OriginalAcquisitionResult:
            raise ExecutionAuthorityError('Original acquisition owner required')
        self._result = result

    @property
    def checkpoint(self): return self._result.checkpoint
    @property
    def preparation(self): return self._result.preparation
    def check_current(self): self._result.check_current()
    async def prepare_once(self, value): return await self._result.prepare_once(value)
    async def begin_dispatch(self): return await self._result.begin_dispatch()
    async def record_unknown(self, lease=None): return await self._result.record_unknown(lease)
    async def record_accepted(self, lease): return await self._result.record_accepted(lease)


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
        self._task = asyncio.current_task()
        self._completed = None

    @asynccontextmanager
    async def operation(self, request, quote):
        if (asyncio.current_task() is not self._task or self._entered
                or request != self._record.request or quote != self._record.quote):
            raise ResourceConflict('Original acquisition operation differs or already entered')
        self._entered = True
        async with self._repository.original_acquisition_result(self._scope, self._record,
                request, quote, guard=self._guard) as attempt:
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
