"""One CURRENT claim/promotion on the canonical encrypted action row.

This owner has no late-result privilege. Its native writer checks the same live
host permission and ordinary execution through actual COMMIT acceptance.
"""

import asyncio
import threading
from dataclasses import dataclass

from kestrel_sdk.security.aead import AEADCipher

from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission,
    OwnedTaskOutcome,
    await_owned_task,
    create_owned_task,
    owned_task_admission_error,
)
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    execution_observation_error,
    require_execution_work,
)

from .original_result_backing import _OriginalResultCustody


@dataclass(frozen=True, eq=False, repr=False)
class _AcceptedCurrentStep:
    operation: object
    task: object
    identity: tuple
    payload: bytes
    plain: bytes
    revision: int
    mode: str


@dataclass(frozen=True, eq=False, repr=False)
class _CurrentStepRegistration:
    acceptance: object
    native_backing: object
    privacy_holds: tuple
    identity: tuple
    expected: bytes
    plain: bytes
    revision: int
    operation: object
    writer: object = None
    entry: object = None
    mode: str = "claim"
    ack: object = None
    relations: object = None
    join: object = None


@dataclass(frozen=True, eq=False, repr=False)
class _CurrentStepEntry:
    values: tuple
    plain: bytes
    result_summary: object = None


@dataclass(frozen=True, eq=False, repr=False)
class _CurrentStepAck:
    owner: object
    entry: object
    writer: object
    sqlite_connection: object
    acknowledgments: tuple


@dataclass(frozen=True, eq=False, repr=False)
class _CurrentStepJoin:
    owner: object
    admission: object
    task: object
    outcome: object


class _CurrentVisualStepMutation(_OriginalResultCustody):
    def __init__(self, repository, acceptance):
        from kestrel_sovereign.visual_computer.step import _OriginalVisualStep

        if (
            type(self) is not _CurrentVisualStepMutation
            or type(acceptance) is not _AcceptedCurrentStep
            or type(acceptance.operation) is not _OriginalVisualStep
            or repository._original_results.get(acceptance, False) is not None
            or acceptance.task is not asyncio.current_task()
        ):
            raise ExecutionAuthorityError("Exact current step admission required")
        acceptance.operation._check_original()
        self._capture_native_backing(repository)
        self._acceptance, self._operation = acceptance, acceptance.operation
        self._identity = acceptance.identity
        self._expected, self._plain, self._revision = (
            acceptance.payload,
            acceptance.plain,
            acceptance.revision,
        )
        self._entry = self._writer_admission = None
        self._postgres_relation = self._postgres_session = self._sqlite_relation = None
        self._query = self._match_query = None
        self._acknowledgments, self._sqlite_committed_connection = (), None
        self._observer_depth, self._phase = 0, "accepted"
        self._mode = acceptance.mode
        self._aux_relations = ()
        self._native_commit_thread = None
        self._ack = None
        self._join = None
        self._registration = _CurrentStepRegistration(
            acceptance,
            self._native_backing,
            self._native_backing.privacy_holds,
            self._identity,
            self._expected,
            self._plain,
            self._revision,
            self._operation,
            mode=acceptance.mode,
        )
        repository._original_results[acceptance] = self
        self._backend._original_result_holds[self] = self._closed

    def bind_relations(self):
        from dataclasses import replace

        self.check_current()
        if self._registration.relations is not None or self._query is not None:
            raise ExecutionAuthorityError("Current step relations already captured")
        self._registration = replace(
            self._registration,
            relations=(
                self._sqlite_relation,
                self._postgres_relation,
                self._aux_relations,
            ),
        )

    def _fixed_queries(self):
        from .db.native_row_custody import _step_row_queries

        return _step_row_queries(self)

    def _require_registered(self):
        r = self._registration
        if (
            type(self) is not _CurrentVisualStepMutation
            or type(r) is not _CurrentStepRegistration
            or type(self._acceptance) is not _AcceptedCurrentStep
            or r.acceptance is not self._acceptance
            or self._repository._original_results.get(self._acceptance) is not self
            or self._identity is not r.identity
            or self._expected is not r.expected
            or self._plain is not r.plain
            or type(self._revision) is not int
            or self._revision != r.revision
            or self._operation is not r.operation
            or self._writer_admission is not r.writer
            or self._entry is not r.entry
            or self._mode != r.mode
            or self._mode not in {"claim", "update"}
            or self._observer_depth
            or self._ack is not r.ack
            or self._join is not r.join
            or self._owner_task is not self._acceptance.task
        ):
            raise ExecutionAuthorityError("Original current step identity retired")
        if r.relations is not None:
            sqlite, postgres, auxiliary = r.relations
            if (
                self._postgres_relation is not postgres
                or self._aux_relations is not auxiliary
                or (
                    sqlite is not None
                    and (
                        self._sqlite_relation is None
                        or self._sqlite_relation[1:] != sqlite[1:]
                    )
                )
            ):
                raise ExecutionAuthorityError(
                    "Current step original relations replaced"
                )
        if (
            self._query is not None
            and (self._query, self._match_query) != self._fixed_queries()
        ):
            raise ExecutionAuthorityError("Current step fixed program changed")

    def _require_backing(self):
        self._require_registered()
        self._require_native_backing()

    def _require_current_authority(self):
        self._operation._check_original()
        require_execution_work(self._backend)

    def check_current(self):
        if asyncio.current_task() is not self._owner_task or self._phase != "accepted":
            raise ExecutionAuthorityError("Original current step operation retired")
        self._require_backing()
        self._require_current_authority()

    def require_admission(self, backend):
        if backend is not self._backend:
            raise ExecutionAuthorityError("Current step backend differs")
        self.check_current()

    def require_original(self, backend):
        if (
            backend is not self._backend
            or self._phase not in {"bound", "submitted"}
            or type(self._writer_admission) is not OwnedTaskAdmission
            or self._writer_admission.task is self._owner_task
            or asyncio.current_task() is not self._writer_admission.task
        ):
            raise ExecutionAuthorityError("Original current step writer required")
        self._require_backing()
        self._require_current_authority()

    def _bind_native_commit(self, native):
        # This method runs ONLY in the original SQLite queue, after BEGIN.
        self._require_registered()
        if (
            native is not self._backing._conn
            or self._phase != "submitted"
            or self._native_commit_thread is not None
        ):
            raise ExecutionAuthorityError(
                "Original current step native commit binding differs"
            )
        self._native_commit_thread = threading.current_thread()
        self._require_native_commit(native)

    def _require_native_commit(self, native):
        self._require_registered()
        if (
            native is not self._backing._conn
            or self._phase != "submitted"
            or threading.current_thread() is not self._native_commit_thread
            or type(self._writer_admission) is not OwnedTaskAdmission
            or self._writer_admission.task is None
            or native._current_row_owner is not self
        ):
            raise ExecutionAuthorityError(
                "Original current step native commit owner differs"
            )
        self._require_captured_native_lifetime()
        # The worker borrows the SAME original restrictions. It never creates
        # a Work scope, binds a replacement caller or enters ordinary SQL.
        for scope in self._backend_work_scopes:
            scope.require_work()
        if self._backend_scope is not None:
            self._backend_scope.require_work()
        self._operation._check_original()

    async def commit_step(self):
        from dataclasses import replace

        self.check_current()
        raw, pending = self._operation._mutation_plain(self)
        owner, session_key, action_id = self._identity
        payload = AEADCipher(self._key).encrypt(
            raw, aad=f"{owner}:{session_key}:{action_id}".encode()
        )
        if self._mode == "claim":
            values = (
                owner,
                session_key,
                action_id,
                payload,
                self._repository.max_actions,
                owner,
                session_key,
                self._revision,
                owner,
            )
        else:
            values = (
                payload,
                pending,
                owner,
                session_key,
                action_id,
                self._expected,
                self._revision,
            )
        self._entry = _CurrentStepEntry(values, raw)
        self._phase = "bound"
        admission = self._writer_admission = OwnedTaskAdmission()
        self._registration = replace(
            self._registration, writer=admission, entry=self._entry
        )
        opening = None
        try:
            writer = create_owned_task(
                self._backend._commit_current_visual_step(self),
                admission=admission,
                retain_originals=True,
                name="current-visual-step",
            )
        except BaseException as error:  # noqa: BLE001 - physically join the accepted native owner.
            opening = owned_task_admission_error(self, admission, error)
            writer = admission.task
            if writer is None:
                self._phase = "failed"
                raise opening
        outcome = await await_owned_task(writer)
        self._join = _CurrentStepJoin(self, admission, writer, outcome)
        self._registration = replace(self._registration, join=self._join)
        error = execution_observation_error(
            opening, outcome.error, outcome.cancellation
        )
        if error is not None:
            self._phase = "failed"
            raise self.retain_committed(error)
        if self._phase != "acknowledged":
            self._phase = "failed"
            raise ExecutionAuthorityError("Original current step native ACK absent")
        self._require_current_authority()
        self._ack = _CurrentStepAck(
            self,
            self._entry,
            admission,
            self._sqlite_committed_connection,
            self._acknowledgments,
        )
        self._registration = replace(self._registration, ack=self._ack)
        return self._ack

    def begin(self, backend):
        self.require_original(backend)
        if self._phase != "bound":
            raise ExecutionAuthorityError("Current step was already submitted")
        self._phase = "submitted"
        return self._entry

    def require_sqlite_cleanup(self, backend, connection):
        if (
            type(self) is not _CurrentVisualStepMutation
            or backend is not self._backend
            or connection is not self._backing
            or self._phase != "submitted"
            or type(self._writer_admission) is not OwnedTaskAdmission
            or asyncio.current_task() is not self._writer_admission.task
        ):
            raise ExecutionAuthorityError("Original current step cleanup differs")

    def acknowledged(self, backend):
        self.require_original(backend)
        if not (
            self._sqlite_committed_connection is self._backing
            or any(r.acknowledged for r in self._acknowledgments)
        ):
            raise ExecutionAuthorityError("Original current step native ACK absent")
        self._phase = "acknowledged"

    def has_acknowledged_commit(self):
        return self._sqlite_committed_connection is self._backing or any(
            r.acknowledged for r in self._acknowledgments
        )

    def write_refusal(self, reason, cause=None):
        from .original_action_result import _OriginalActionResult

        return _OriginalActionResult.write_refusal(self, reason, cause)

    def retain_committed(self, error):
        from kestrel_sovereign.signals.outcome_custody import _OriginalSignalOutcome

        return _OriginalSignalOutcome.retain_committed(self, error)

    def close(self):
        if (
            type(self) is not _CurrentVisualStepMutation
            or asyncio.current_task() is not self._owner_task
            or self._native_backing.owner is not self
            or self._registration.native_backing is not self._native_backing
            or self._native_backing.closed is not self._closed
            or (
                self._writer_admission is not None
                and self._writer_admission.task is not None
                and (
                    type(self._join) is not _CurrentStepJoin
                    or self._registration.join is not self._join
                    or self._join.owner is not self
                    or self._join.admission is not self._writer_admission
                    or self._join.task is not self._writer_admission.task
                    or type(self._join.outcome) is not OwnedTaskOutcome
                )
            )
        ):
            raise ExecutionAuthorityError(
                "Current step closes only in its original owner"
            )
        self._phase = "closed"
        self._repository._original_results.pop(self._acceptance, None)
        self._backend._original_result_holds.pop(self, None)
        if not self._closed.done():
            self._closed.set_result(None)
