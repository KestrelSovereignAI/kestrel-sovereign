"""Opaque permission for ONE accepted signal's fixed native terminal row.

Never installed on an executor, never carries a SQL callback, and never grants
ordinary storage, source, tool, provider, or UI work. Original denials remain
in the ambient context and native commit/cleanup evidence retains them.
"""

import asyncio
from dataclasses import dataclass

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, current_execution_custody, require_execution_work,
)

_JOURNAL_ADMISSION = object()


@dataclass(frozen=True, eq=False)
class _AcceptedSignalOutcome:
    """Original dispatcher-issued envelope, not a causation-context proxy."""
    owner: asyncio.Task
    identity: tuple
    registration: object


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
        self._dispatcher = dispatcher
        self._store = dispatcher._store
        self._backend = backend
        self._agent = dispatcher._agent
        self._agent_id = dispatcher._agent.did
        self._registration = registration
        self._acceptance = acceptance
        self._identity = self._signal_identity(signal)
        self._backend_scope = getattr(backend, '_execution_custody', None)
        self._backing = backend._connection if type(backend) is SQLiteBackend else backend._pool
        self._generation = backend.connection_generation if type(backend) is SQLiteBackend else None
        self._entry = None
        self._outer_admission = self._writer_admission = None
        self._phase = 'accepted'
        self._require_backing()
        self._admission_key = _JOURNAL_ADMISSION
        dispatcher._accepted_signal_outcomes[acceptance] = self

    @staticmethod
    def _signal_identity(signal):
        return (signal.id, signal.source, signal.target_agent, signal.mode)

    def matches(self, signal, registration):
        return (self._identity == self._signal_identity(signal)
                and registration is self._registration)

    def _require_backing(self):
        backend = self._backend
        if (self._dispatcher._store is not self._store
                or self._store._backend is not backend
                or self._dispatcher._agent is not self._agent
                or self._agent.did != self._agent_id
                or getattr(backend, '_execution_custody', None) is not self._backend_scope):
            raise ExecutionAuthorityError('Original signal journal runtime identity changed')
        if backend.backend_type == 'sqlite':
            if (self._backing is None or backend._connection is not self._backing
                    or backend.connection_generation != self._generation or backend._closing):
                raise ExecutionAuthorityError('Original signal journal connection retired')
            backend.assert_connected_file_still_valid()
        elif self._backing is None or backend._pool is not self._backing:
            raise ExecutionAuthorityError('Original signal journal pool retired')

    def bind(self, signal, registration, result, prepared):
        from kestrel_sovereign.signals.store import _PreparedSignalOutcome
        self._require_backing()
        if (asyncio.current_task() is not self._acceptance.owner
                or self._phase != 'accepted' or not self.matches(signal, registration)
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
        if self._phase != 'submitted':
            raise ExecutionAuthorityError('Original signal journal was not submitted')
        self._phase = 'acknowledged'


def _require_original_signal_outcome(journal, backend):
    if type(journal) is not _OriginalSignalOutcome:
        raise ExecutionAuthorityError('Exact original native signal journal required')
    journal.require_original(backend)


def _capture_original_outcome(dispatcher, signal, registration, *, acceptance):
    """Only a live original public dispatch can create this terminal unit."""
    from kestrel_sovereign.signals.store import SignalLogStore
    from kestrel_sovereign.storage.db.postgres import PostgresBackend
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    admission = dispatcher._durable_admission.get()
    if (admission is None or admission[0] is not asyncio.current_task()
            or asyncio.current_task() not in dispatcher._durable_admission_owners):
        raise ExecutionAuthorityError('Original accepted dispatch required for terminal journal')
    store = dispatcher._store
    if type(store) is not SignalLogStore or type(store._backend) not in (SQLiteBackend, PostgresBackend):
        if current_execution_custody(dispatcher._agent):
            raise ExecutionAuthorityError('Hosted terminal journal requires a supported native backend')
        return None  # legacy unscoped embeddings retain ordinary append only
    return _OriginalSignalOutcome(dispatcher, signal, registration, acceptance=acceptance)
