"""Physical holds shared only by Core's closed visual result owners.

This descriptor grants no SQL or result transition. Each concrete owner still
authenticates its original admission, row, producer, fixed program and writer.
"""

import asyncio
from dataclasses import dataclass

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    current_execution_custody,
    require_execution_work,
)


def _closed_visual_result(owner):
    from .db.native_row_custody import _native_row_profile

    if _native_row_profile(owner)[0] not in {
        "visual_computer_actions",
        "visual_computer_requests",
    }:
        raise ExecutionAuthorityError("Exact Core visual result owner required")


@dataclass(frozen=True, eq=False, repr=False)
class _OriginalResultBacking:
    owner: object
    repository: object
    database: object
    backend: object
    key: object
    scope: object
    work_scopes: tuple
    owner_task: object
    backing: object
    generation: object
    postgres_pool: object
    privacy_holds: tuple
    closed: object


class _OriginalResultCustody:
    """Private physical mechanism, with finite exact owner admission only."""

    def _capture_native_backing(self, repository):
        from kestrel_sovereign.signals.outcome_custody import (
            _OriginalPostgresJournalPool,
        )
        from kestrel_sovereign.visual_computer.repository import VisualCustodyRepository

        from .async_database import AsyncDatabase
        from .db.postgres import PostgresBackend
        from .db.sqlite import SQLiteBackend
        from .privacy_wrapper import _DURABLE_GATE_SHARED_HOLDS

        _closed_visual_result(self)
        if (
            type(repository) is not VisualCustodyRepository
            or type(repository.db) is not AsyncDatabase
        ):
            raise ExecutionAuthorityError(
                "Original result requires its canonical native database"
            )
        database = repository.db
        backend = database.backend
        if type(backend) not in (SQLiteBackend, PostgresBackend):
            raise ExecutionAuthorityError(
                "Original result requires its exact native backend"
            )
        require_execution_work(backend)
        if backend._original_results_closing:
            raise ExecutionAuthorityError("Original result backend is closing")
        backing = (
            backend._connection if type(backend) is SQLiteBackend else backend._pool
        )
        descriptor = _OriginalResultBacking(
            self,
            repository,
            database,
            backend,
            repository._key,
            getattr(backend, "_execution_custody", None),
            current_execution_custody(backend),
            asyncio.current_task(),
            backing,
            backend.connection_generation if type(backend) is SQLiteBackend else None,
            _OriginalPostgresJournalPool.capture(backing)
            if type(backend) is PostgresBackend
            else None,
            tuple(
                (hold, hold.gate, hold.lease)
                for hold in _DURABLE_GATE_SHARED_HOLDS.get()
                if hold.live
            ),
            asyncio.get_running_loop().create_future(),
        )
        self._native_backing = descriptor
        self._repository, self._database = repository, database
        self._backend, self._key = backend, descriptor.key
        self._backend_scope, self._backend_work_scopes = (
            descriptor.scope,
            descriptor.work_scopes,
        )
        self._owner_task, self._closed = descriptor.owner_task, descriptor.closed
        self._backing, self._generation = descriptor.backing, descriptor.generation
        self._postgres_pool = descriptor.postgres_pool

    def _require_captured_native_lifetime(self):
        """Exact physical holds only; grants no native SQL/program admission.

        Only the original captured physical worker may borrow these facts. Full
        connection programs and owning-loop checks remain in the SQL path.
        """
        from .db.sqlite import SQLiteBackend
        from .privacy_wrapper import _SharedGateHold, _SharedGateLease

        _closed_visual_result(self)
        descriptor = self._native_backing
        registration = self._registration
        if (
            type(descriptor) is not _OriginalResultBacking
            or descriptor.owner is not self
            or registration.native_backing is not descriptor
            or registration.privacy_holds is not descriptor.privacy_holds
            or self._repository is not descriptor.repository
            or self._database is not descriptor.database
            or self._backend is not descriptor.backend
            or self._key is not descriptor.key
            or self._backend_scope is not descriptor.scope
            or self._backend_work_scopes is not descriptor.work_scopes
            or self._owner_task is not descriptor.owner_task
            or self._closed is not descriptor.closed
            or self._backing is not descriptor.backing
            or type(self._generation) is not type(descriptor.generation)
            or self._generation != descriptor.generation
            or self._repository.db is not self._database
            or self._database.backend is not self._backend
            or self._repository._key is not self._key
            or getattr(self._backend, "_execution_custody", None)
            is not self._backend_scope
            or self._backend._original_result_holds.get(self) is not self._closed
        ):
            raise ExecutionAuthorityError(
                "Original result native backing identity retired"
            )
        if any(
            type(hold) is not _SharedGateHold
            or hold.live is not True
            or hold.gate is not gate
            or hold.lease is not lease
            or type(lease) is not _SharedGateLease
            or type(lease.holds) is not int
            or lease.holds <= 0
            for hold, gate, lease in descriptor.privacy_holds
        ):
            raise ExecutionAuthorityError("Original result privacy hold retired")
        backend = self._backend
        if type(backend) is SQLiteBackend:
            if (
                backend._connection is not self._backing
                or backend.connection_generation != self._generation
                or backend._closing
            ):
                raise ExecutionAuthorityError(
                    "Original result native connection retired"
                )
            backend.assert_connected_file_still_valid()
        else:
            native = descriptor.postgres_pool.pool
            current = getattr(self._backing, "native_pool", self._backing)
            if (
                backend._pool is not self._backing
                or current is not native
                or native._generation != descriptor.postgres_pool.generation
                or native._closed
                or native._closing
            ):
                raise ExecutionAuthorityError("Original result pool retired")

    def _require_native_backing(self):
        from .db.native_journal_programs import _NativeJournalSQLiteConnection
        from .db.sqlite import SQLiteBackend

        self._require_captured_native_lifetime()
        if type(self._backend) is SQLiteBackend:
            if (
                type(self._backing._conn) is not _NativeJournalSQLiteConnection
                or not self._backing._conn._original_journal_programs_supported()
            ):
                raise ExecutionAuthorityError(
                    "Original result native connection retired"
                )
        else:
            # Canonical native relation capture refines the same frozen pool
            # recipe with its actual endpoint/principal/TLS facts. It does not
            # replace the originally admitted pool, connector or recipe.
            original, actual = self._native_backing.postgres_pool, self._postgres_pool
            if (
                type(actual) is not type(original)
                or actual.pool is not original.pool
                or type(actual.generation) is not type(original.generation)
                or actual.generation != original.generation
                or actual.connect_args is not original.connect_args
                or actual.connect_kwargs is not original.connect_kwargs
                or actual.wrapper is not original.wrapper
                or actual.configuration is not original.configuration
            ):
                raise ExecutionAuthorityError("Original result pool recipe retired")
            self._postgres_pool.require_original(self._backing)
