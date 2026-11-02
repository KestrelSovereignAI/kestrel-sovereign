"""Core-owned one-shot native producer receipts for a closed acquisition.

This is not a parser/transport registry. Only the explicit canonical Runpod
host binding can issue a handle. Native request/body/close events come from
the existing HTTPX boundary; a DTO or exception never supplies those facts.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime
import inspect
import threading

import httpx
from kestrel_sdk.computer_use import EnvironmentLease

from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission, await_owned_task, create_owned_task, owned_task_admission_error,
)
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, execution_terminal_error, require_execution_work,
)

from .native_response import _BoundedOriginalResponse, MAX_ORIGINAL_RESPONSE_BYTES

MAX_ACQUISITION_RESPONSE_BYTES = MAX_ORIGINAL_RESPONSE_BYTES


@dataclass(frozen=True, eq=False, repr=False)
class _NativeAcquisitionReceipt:
    handle: object
    admission: object
    thread: object
    request: object
    response: object
    phase: object
    lease: object
    lease_json: bytes | None
    native_result: bytes | None
    raw_bytes: int
    exhausted: bool
    response_closed: bool
    client_closed: bool
    terminal: object


class _NativeAcquisitionHandle(_BoundedOriginalResponse):
    def __init__(self, binding):
        from .runpod_acquisition import _RunpodBinding
        if type(self) is not _NativeAcquisitionHandle or type(binding) is not _RunpodBinding:
            raise ExecutionAuthorityError('Exact canonical acquisition binding required')
        binding._require_current()
        self._binding, self._root = binding, binding._root
        self._owner_task, self._loop = asyncio.current_task(), asyncio.get_running_loop()
        self._loop_thread = threading.current_thread()
        self._client = binding._client
        self._state = 'inactive'
        self._dispatch_ack = None
        self._admission = self._thread = self._worker = None
        self._worker_entered = self._worker_exited = False
        self._job_entered = False
        self._request = self._response = self._phase = None
        self._exhausted = self._response_closed = self._client_closed = False
        self._raw_bytes = 0
        self._sockets = set()
        self._receipt = None
        self._minimal_lease = self._accepted_lease = None
        self._captured_lease_json = self._captured_native_result = None
        self._minimal_attempted = self._minimal_acknowledged = False
        self._minimal_checkpoint = None

    def _require_identity(self, root):
        if (type(self) is not _NativeAcquisitionHandle or root is not self._root
                or root._native is not self or root._registration.native is not self
                or root._owner_task is not self._owner_task or self._state == 'closed'):
            raise ExecutionAuthorityError('Original acquisition producer retired')

    def _require_root(self, root):
        self._require_identity(root)
        if self._in_original_worker():
            root._require_native_producer_lifetime(self)
        else:
            root._require_backing()
        self._binding._require_integrity()

    def _require_prepared(self, root):
        self._require_root(root)
        root.check_current()
        if self._state != 'prepared':
            raise ExecutionAuthorityError('Original native producer is not prepared')
        self._binding._require_saved_preparation()

    def _arm_original_dispatch(self, root):
        from kestrel_sovereign.storage.original_acquisition_result import _AcquisitionStageAck
        self._require_prepared(root)
        ack = root._last_ack
        storage = root._storage()
        if (type(ack) is not _AcquisitionStageAck or ack.owner is not root
                or ack is not root._registration.last_ack or ack.entry.kind != 'dispatch'
                or ack.entry.values[0] is not root._expected or ack.entry.values[1] != root._revision
                or type(ack.writer) is not OwnedTaskAdmission or ack.writer.task is None
                or not ack.writer.task.done() or ack.writer.task.cancelled()
                or not (ack.sqlite_connection is root._backing or any(value.acknowledged for value in ack.acknowledgments))
                or storage.checkpoint.state != 'dispatching'
                or storage.serialize_private().get_secret_value() != ack.entry.storage):
            raise ExecutionAuthorityError('Original dispatch native ACK required')
        self._dispatch_ack, self._state = ack, 'armed'

    def _in_original_worker(self):
        return self._worker_entered and not self._worker_exited and threading.current_thread() is self._worker

    def check_parse_current(self):
        self._require_root(self._root)
        if self._in_original_worker():
            from kestrel_sovereign.execution_sync_http import _THREAD, _require_sync_work
            if self._state != 'running' or _THREAD.get() is not self._thread or not self._thread.active:
                raise ExecutionAuthorityError('Original native acquisition worker retired')
            _require_sync_work()
            self._root._require_current_authority()
            _require_sync_work()
        else:
            if (threading.current_thread() is not self._loop_thread
                    or asyncio.get_running_loop() is not self._loop
                    or asyncio.current_task() is not self._owner_task
                    or self._state != 'minimal-acknowledged'):
                raise ExecutionAuthorityError('Original acquisition parse owner differs')
            self._root.check_current()

    def _require_worker(self, original):
        from kestrel_sovereign.execution_sync_http import _THREAD
        self._require_root(self._root)
        if (self._state != 'running' or not self._in_original_worker()
                or original.thread is not self._thread or _THREAD.get() is not self._thread
                or self._thread.acquisition is not self or original.client is not self._client):
            raise ExecutionAuthorityError('Original acquisition native request owner differs')

    def _require_result(self, root):
        # Factual custody is narrower than current parse/input authority. It
        # consumes only the originally sealed joined receipt, fixed row and
        # still-live native DB/key/privacy holds; no vendor parser is entered.
        self._require_identity(root)
        root._require_backing()
        self._binding._canonical.decoder.require_storage_original()
        receipt = self._receipt
        if (asyncio.current_task() not in (self._owner_task, root._writer_admission.task if root._writer_admission else None)
                or type(receipt) is not _NativeAcquisitionReceipt or receipt.handle is not self
                or receipt is not root._registration.native_receipt
                or receipt.phase is not self._phase or receipt.lease is not self._minimal_lease
                or receipt.lease_json is not self._captured_lease_json
                or receipt.native_result is not self._captured_native_result
                or receipt.admission is not self._admission or self._admission.task is None
                or not self._admission.task.done() or not self._worker_exited
                or self._state not in {'joined', 'minimal-acknowledged'}
                or self._thread.sockets or not receipt.client_closed
                or (receipt.phase is not None and self._thread.requests)
                or (receipt.phase is not None and not (receipt.exhausted and receipt.response_closed))):
            raise ExecutionAuthorityError('Original acquisition native result is not physically joined')

    def _original_minimal_lease(self):
        self._require_result(self._root)
        value = self._receipt.lease_json
        # These exact private bytes were normalized before the original
        # canonical capture ended. This never decodes the HTTP body again.
        return None if value is None else EnvironmentLease.model_validate_json(value)

    async def persist_minimal(self):
        if asyncio.current_task() is not self._owner_task:
            raise ExecutionAuthorityError('Original acquisition fact owner differs')
        self._require_result(self._root)
        if self._minimal_acknowledged:
            return self._minimal_checkpoint.model_copy(deep=True)
        if self._minimal_attempted:
            raise ExecutionAuthorityError('Original acquisition minimal ACK is unresolved')
        self._minimal_attempted = True
        checkpoint = await self._root.record_unknown()
        return checkpoint

    def _ack_minimal(self, root, checkpoint):
        from kestrel_sovereign.storage.original_acquisition_result import _AcquisitionStageAck
        self._require_result(root)
        ack = root._last_ack
        if (type(ack) is not _AcquisitionStageAck or ack.owner is not root or ack.entry.kind != 'minimal'
                or ack is not root._registration.last_ack or ack.entry.values[0] is not root._expected):
            raise ExecutionAuthorityError('Original minimal native ACK required')
        self._minimal_checkpoint = checkpoint.model_copy(deep=True)
        self._minimal_acknowledged, self._state = True, 'minimal-acknowledged'

    def _matches_lease(self, lease, *, state):
        """Compare bounded original facts without touching optional cost.

        Late facts never invoke a caller serializer or a custom timezone,
        scope, string or equality callback. None remains the preferred proposal.
        """
        from kestrel_sdk.resources import ResourceScope
        expected = self._original_minimal_lease()
        if type(lease) is not EnvironmentLease or expected is None:
            return False
        actual_values = object.__getattribute__(lease, '__dict__')
        original_values = object.__getattribute__(expected, '__dict__')
        if actual_values.keys() != original_values.keys():
            return False
        for name, original in original_values.items():
            if name == 'actual_cost_usd':
                continue  # Optional observations never enter this minimal lane.
            value = actual_values[name]
            if name == 'state':
                if type(value) is not str or value != state:
                    return False
            elif name == 'scope':
                if type(value) is not ResourceScope or type(original) is not ResourceScope:
                    return False
                actual_scope, original_scope = value.__dict__, original.__dict__
                if actual_scope.keys() != original_scope.keys() or any(
                        type(actual_scope[key]) is not type(item) or actual_scope[key] != item
                        for key, item in original_scope.items()):
                    return False
            elif name == 'expires_at':
                if type(value) is not datetime or type(original) is not datetime or type(value.tzinfo) is not type(original.tzinfo) or value != original:
                    return False
            elif type(value) is not type(original) or value != original:
                return False
        return True

    def _accepted_proposal(self, root, proposal):
        self._require_result(root)
        root.check_current()
        self._binding._require_public_result()
        if not self._minimal_acknowledged or not self._matches_lease(proposal, state='pending'):
            raise ExecutionAuthorityError('Original accepted native lease facts differ')
        self._accepted_lease = self._original_minimal_lease().model_copy(update={'state': 'pending'})
        return self._accepted_lease

    def _check_invocation_current(self):
        # The exact owner keeps full native backing checks. Only its physical
        # worker uses the captured lifetime restriction already proved above.
        self._require_root(self._root)
        self._root._require_current_authority()

    def _check_invocation_write(self):
        self._check_invocation_current()
        self._root._require_current_authority(spending=True)
        self._binding._require_request_write()

    def _require_dispatch_owner(self, client, operation):
        from kestrel_sovereign.storage.original_acquisition_result import _AcquisitionStageAck
        self._require_root(self._root)
        self._root.check_current()
        if (asyncio.current_task() is not self._owner_task or asyncio.get_running_loop() is not self._loop
                or self._state != 'armed' or client is not self._client
                or type(self._dispatch_ack) is not _AcquisitionStageAck
                or self._dispatch_ack is not self._root._registration.last_ack
                or self._dispatch_ack.owner is not self._root
                or self._dispatch_ack.entry.kind != 'dispatch'
                or self._dispatch_ack.entry.values[0] is not self._root._expected
                or self._dispatch_ack.entry.values[1] != self._root._revision
                or operation is not self._binding._operation):
            raise ExecutionAuthorityError('Original acquisition dispatch owner differs')
        self._root._require_current_authority(spending=True)

    async def run_original(self):
        """Invoke only this already-issued original client and operation.

        No caller checker or DTO can supply authority. The retained original
        host restriction remains lexical through the same API2 worker/join.
        """
        from kestrel_sovereign.execution_sync_http import (
            await_native_sync_http, bind_native_sync_http_execution_admission,
        )
        self._require_dispatch_owner(self._client, self._binding._operation)
        self._binding._require_saved_preparation()
        self._check_invocation_write()
        with bind_native_sync_http_execution_admission(self._client,
                self._check_invocation_current,
                check_before_request_write=self._check_invocation_write):
            return await await_native_sync_http(self._client, self._binding._operation,
                _original_acquisition=self)

    async def _await_original(self, client, operation):
        from kestrel_sovereign import execution_sync_http as native
        self._require_dispatch_owner(client, operation)
        self._binding._require_native_entry()
        original = tuple(value for value in native._RESTRICTIONS.get() if value.client is client)
        if not original:
            raise ExecutionAuthorityError('Original acquisition requires its original restriction')
        for restriction in original:
            restriction.check()
        native._require_sync_work()
        thread = self._thread = native._ThreadOperation(acquisition=self)
        token = native._THREAD.set(thread)
        admission = self._admission = OwnedTaskAdmission()
        self._state = 'running'
        result = terminal = None

        def run():
            nonlocal result, terminal
            self._worker, self._worker_entered = threading.current_thread(), True
            try:
                try:
                    result = operation(client)
                    if inspect.isawaitable(result):
                        if inspect.iscoroutine(result):
                            result.close()
                        raise TypeError('Original native acquisition returned an awaitable')
                    # The canonical bounded phase is captured before SAME-run
                    # cleanup can hide an otherwise returned original value.
                    (self._phase, self._minimal_lease, self._captured_lease_json,
                        self._captured_native_result) = self._binding._capture_phase(result)
                except BaseException as error:
                    terminal = error
                for request in tuple(thread.requests):
                    try:
                        request.response.close()
                    except BaseException as error:
                        terminal = execution_terminal_error(terminal, error, retain_originals=True)
                self._sockets.update(borrow.socket for borrow in tuple(thread.sockets))
                try:
                    client.close()  # The ONE dedicated client close, in this native run.
                except BaseException as error:
                    terminal = execution_terminal_error(terminal, error, retain_originals=True)
                # Actual socket identities are held independently of pool-list
                # mutation. A thread return or is_closed flag is not an ACK.
                for sock in self._sockets:
                    if sock.fileno() != -1:
                        try:
                            sock.close()
                        except BaseException as error:
                            terminal = execution_terminal_error(terminal, error, retain_originals=True)
                for borrow in tuple(thread.sockets):
                    if borrow.socket.fileno() == -1:
                        borrow.retire()
                        thread.sockets.discard(borrow)
                self._client_closed = (client.is_closed and not self._binding._pool._requests and not thread.sockets
                    and all(sock.fileno() == -1 for sock in self._sockets))
                if not self._client_closed:
                    terminal = execution_terminal_error(terminal,
                        ExecutionAuthorityError('Original acquisition physical cleanup unresolved'), retain_originals=True)
            finally:
                self._worker_exited = True
            if terminal is not None:
                raise terminal
            return result

        opening = None
        async def native_job():
            # A settled failed start gate before this exact payload entered
            # proves no executor job was submitted. Task.done alone does not.
            self._job_entered = True
            return await asyncio.to_thread(run)
        try:
            try:
                task = create_owned_task(native_job(), admission=admission,
                    retain_originals=True, name='original-native-acquisition')
            except BaseException as error:
                opening = owned_task_admission_error(self, admission, error)
                task = admission.task
                if task is None:
                    self._state = 'failed'
                    raise opening
            outcome = await await_owned_task(task, cancel_on_caller_cancellation=thread.cancel)
            terminal = execution_terminal_error(terminal, opening, outcome.error, outcome.cancellation,
                retain_originals=True, isolate_control_originals=True)
            if terminal is None:
                try:
                    for restriction in original:
                        restriction.check()
                    native._require_sync_work()
                except BaseException as denial:
                    terminal = native.NativeSyncHTTPOutcomeUnknown(denial) if thread.sent else denial
            if self._worker_entered and self._worker_exited:
                self._receipt = _NativeAcquisitionReceipt(self, admission, self._worker,
                    self._request, self._response, self._phase, self._minimal_lease,
                    self._captured_lease_json, self._captured_native_result, self._raw_bytes,
                    self._exhausted, self._response_closed, self._client_closed, terminal)
                self._state = 'joined'
                try:
                    self._root._seal_native_result(self, self._receipt)
                except BaseException as sealing:
                    terminal = execution_terminal_error(terminal, sealing, retain_originals=True)
            else:
                self._state = 'failed'
            if terminal is not None:
                raise terminal
            self._require_result(self._root)
            return result
        finally:
            thread.active = False
            native._THREAD.reset(token)

    async def _close_original(self):
        """Drain only the originally transferred dedicated native resource."""
        if (type(self) is not _NativeAcquisitionHandle or asyncio.current_task() is not self._owner_task
                or self._client is not self._binding._client):
            raise ExecutionAuthorityError('Original acquisition close owner differs')
        error = None
        if self._admission is not None and self._admission._acceptance_unknown:
            raise ExecutionAuthorityError('Original acquisition task acceptance remains unresolved')
        if self._admission is not None and self._admission.task is not None:
            if not self._admission.task.done():
                self._thread.cancel(self._admission.task)
                outcome = await await_owned_task(self._admission.task)
                error = execution_terminal_error(outcome.error, outcome.cancellation, retain_originals=True)
            if self._job_entered and (not self._worker_exited or not self._client_closed):
                raise ExecutionAuthorityError('Original acquisition native cleanup remains unresolved')
        if not self._client_closed:
            # Before any native work was accepted, the exact pristine pool
            # has no borrowed socket or response. No used client is adopted.
            if self._binding._pool.connections or self._binding._pool._requests:
                raise ExecutionAuthorityError('Unstarted acquisition client is not pristine')
            self._client.close()
            self._client_closed = True
        self._state = 'closed'
        if error is not None:
            raise error
