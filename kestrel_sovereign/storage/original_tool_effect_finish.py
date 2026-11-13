"""Original attempt retirement through the closed native result mechanism."""

import asyncio
from dataclasses import dataclass, replace
import weakref

from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission, await_owned_task, create_owned_task, owned_task_admission_error,
)
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, execution_observation_error, require_execution_work,
)
from .original_result_backing import _OriginalResultCustody


_ADMISSION = object()


@dataclass(frozen=True, eq=False, repr=False)
class _FinishEntry:
    values: tuple
    result_summary: bool = True


@dataclass(frozen=True, eq=False, repr=False)
class _FinishRegistration:
    native_backing: object
    privacy_holds: tuple
    identity: tuple
    timeout: float
    relations: object = None
    writer: object = None
    entry: object = None
    unconfirmed: bool = False


class _OriginalToolEffectFinish(_OriginalResultCustody):
    """One admitted backing and attempt; closes its row without admitting work."""

    @classmethod
    async def prepare(cls, repository, identity):
        if cls is not _OriginalToolEffectFinish:
            raise ExecutionAuthorityError('Exact original attempt owner required')
        result = cls()
        result._capture_native_backing(repository)
        result._registration = _FinishRegistration(result._native_backing,
            result._native_backing.privacy_holds, identity, repository.terminal_timeout_seconds)
        result._identity = identity
        result._phase, result._admission_key = 'accepted', _ADMISSION
        result._attempt = None
        result._entry = result._writer_admission = None
        result._query = result._match_query = None
        result._postgres_relation = result._postgres_session = result._sqlite_relation = None
        result._acknowledgments = ()
        result._aux_relations = ()
        result._sqlite_committed_connection = None
        result._observer_depth = 0
        result._writer = result._opening = None
        # Pure catalog capture runs under live ordinary admission. No native
        # writer/resource hold exists until actual finite retirement starts.
        result._require_backing()
        if result._backend.backend_type == 'sqlite':
            result._sqlite_relation = await result._backend._read_original_signal_outcome_relation(result, capture=True)
            auxiliary = await result._backend._read_original_signal_outcome_relation(result, capture=True, role='claims')
            relation = result._sqlite_relation
        else:
            result._postgres_relation = await result._backend._capture_original_signal_outcome_relation(result)
            auxiliary = await result._backend._capture_original_signal_outcome_relation(result, role='claims')
            relation = result._postgres_relation
        if relation is None or auxiliary is None:
            raise ExecutionAuthorityError('Original tool effect journal relation is unavailable or unsafe')
        result._aux_relations = (auxiliary,)
        result._registration = replace(result._registration,
            relations=(result._sqlite_relation, result._postgres_relation, result._aux_relations))
        result._require_backing()
        return result

    def admitted(self, attempt):
        from kestrel_sovereign.tool_effect_store import ToolEffectAttempt

        if (self._phase != 'accepted' or self._attempt is not None
                or type(attempt) is not ToolEffectAttempt or attempt._terminal is not self
                or (attempt.scope_digest, attempt.operation_digest, attempt.attempt_id) != self._identity
                or asyncio.current_task() is not self._owner_task):
            raise ExecutionAuthorityError('Original acknowledged tool effect admission required')
        self._require_backing()
        require_execution_work(self._backend)
        self._attempt = weakref.ref(attempt)

    def require_attempt(self, repository, attempt):
        if (type(self) is not _OriginalToolEffectFinish or self._admission_key is not _ADMISSION
                or repository is not self._repository or self._attempt is None
                or self._attempt() is not attempt or attempt._terminal is not self
                or (attempt.scope_digest, attempt.operation_digest, attempt.attempt_id) != self._identity):
            raise ExecutionAuthorityError('Original tool effect backing and attempt required')
        self._require_captured_native_lifetime(require_inventory=False)

    def _fixed_queries(self):
        table = ('main.tool_effect_operations' if self._backend.backend_type == 'sqlite'
            else self._postgres_relation.qualified_name)
        where = 'scope_digest = ? AND operation_digest = ? AND attempt_id = ?'
        latch = ", latched = 1" if self._registration.unconfirmed else ""
        return (f"UPDATE {table} SET state = 'CLOSED', revision = revision + 1{latch} WHERE {where}",
            f"SELECT 1 FROM {table} WHERE {where} AND state = 'CLOSED'"
            + (" AND latched = 1" if self._registration.unconfirmed else ""))

    def _require_backing(self):
        if (self._observer_depth or type(self._registration) is not _FinishRegistration
                or self._identity is not self._registration.identity
                or self._writer_admission is not self._registration.writer
                or self._entry is not self._registration.entry):
            raise ExecutionAuthorityError('Original tool effect terminal registration differs')
        self._observer_depth += 1
        try:
            self._require_native_backing(require_inventory=self._phase in {'bound', 'submitted', 'acknowledged'})
            if self._registration.relations is not None:
                sqlite, postgres, auxiliary = self._registration.relations
                if (self._postgres_relation is not postgres or self._aux_relations is not auxiliary
                        or (sqlite is not None and (self._sqlite_relation is None
                            or self._sqlite_relation[:-1] != sqlite[:-1]
                            or self._sqlite_relation[-1][:-1] != sqlite[-1][:-1]))):
                    raise ExecutionAuthorityError('Original tool effect catalog identity differs')
            if self._phase != 'accepted' and (self._query, self._match_query) != self._fixed_queries():
                raise ExecutionAuthorityError('Original fixed tool effect terminal program differs')
        finally:
            self._observer_depth -= 1

    def require_admission(self, backend):
        if (type(self) is not _OriginalToolEffectFinish or self._admission_key is not _ADMISSION
                or self._phase != 'accepted' or backend is not self._backend
                or asyncio.current_task() is not self._owner_task):
            raise ExecutionAuthorityError('Original live tool effect admission required')
        self._require_backing()
        require_execution_work(backend)

    def require_original(self, backend):
        if (type(self) is not _OriginalToolEffectFinish or self._admission_key is not _ADMISSION
                or backend is not self._backend or self._phase not in {'bound', 'submitted'}
                or type(self._writer_admission) is not OwnedTaskAdmission
                or asyncio.current_task() is not self._writer_admission.task
                or self._writer_admission.task is self._owner_task):
            raise ExecutionAuthorityError('Original native tool effect writer required')
        self._require_backing()

    def begin(self, backend):
        self.require_original(backend)
        if self._phase != 'bound':
            raise ExecutionAuthorityError('Original tool effect writer already submitted')
        self._phase = 'submitted'
        return self._entry

    def require_sqlite_cleanup(self, backend, connection):
        if (type(self) is not _OriginalToolEffectFinish or self._admission_key is not _ADMISSION
                or backend is not self._backend or connection is not self._backing
                or self._phase != 'submitted' or self._writer_admission is None
                or asyncio.current_task() is not self._writer_admission.task):
            raise ExecutionAuthorityError('Original tool effect native cleanup owner differs')

    def acknowledged(self, backend):
        self.require_original(backend)
        if not self.has_acknowledged_commit():
            raise ExecutionAuthorityError('Original tool effect finish lacks native COMMIT evidence')
        self._phase = 'acknowledged'

    def write_refusal(self, reason, cause=None):
        if reason == 'row_count':
            from kestrel_sovereign.tool_effect_store import ToolEffectAttemptSuperseded

            return ToolEffectAttemptSuperseded('tool effect attempt was superseded')
        return super().write_refusal(reason, cause)

    async def finish(self, *, unconfirmed=False):
        if type(unconfirmed) is not bool:
            raise ValueError('Tool effect uncertainty must be a boolean')
        if unconfirmed and self._writer is not None and not self._registration.unconfirmed:
            raise ExecutionAuthorityError('Original clean terminal writer cannot absorb later uncertainty')
        if self._writer is None:
            if self._phase != 'accepted' or self._attempt is None or self._attempt() is None:
                raise ExecutionAuthorityError('Original attempt terminal owner is unavailable')
            self._require_backing()
            if self._backend._original_results_closing:
                raise ExecutionAuthorityError('Original result inventory is closing')
            # No await separates this admission fence from hold enrollment.
            # Existing writers may retire during close; late writers cannot
            # enter after its original inventory snapshot.
            self._registration = replace(self._registration, unconfirmed=unconfirmed)
            self._query, self._match_query = self._fixed_queries()
            self._entry = _FinishEntry(self._identity)
            self._phase = 'bound'
            admission = self._writer_admission = OwnedTaskAdmission()
            self._registration = replace(self._registration, writer=admission, entry=self._entry)
            self._backend._original_result_holds[self] = self._closed

            async def write_original():
                # Bounds checkout/body work. The canonical native writer still
                # joins any accepted COMMIT, rollback and checkout retirement.
                async with asyncio.timeout(self._registration.timeout):
                    return await self._backend._write_original_result(self)

            try:
                self._writer = create_owned_task(write_original(), admission=admission,
                    retirement_resource=self, retain_originals=True, name='original-tool-effect-finish')
            except BaseException as error:
                self._opening = owned_task_admission_error(self, admission, error)
                self._writer = admission.task
                if self._writer is None:
                    self.close()
                    raise self._opening
        try:
            outcome = await await_owned_task(self._writer)
            error = execution_observation_error(self._opening, outcome.error, outcome.cancellation)
            if error is not None:
                raise self.retain_committed(error)
        finally:
            self.close()

    def close(self):
        self._backend._original_result_holds.pop(self, None)
        if not self._closed.done():
            self._closed.set_result(None)
