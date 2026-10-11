"""Standalone process custody for the deployment's existing host-key store.

The process lifetime has no PostgreSQL authority rows. It may guard this
separate operator-owned SQLite store using the SAME original lifetime. This
does not make SQLite a tenant executor: any occurrence, advisory or generation
fence still requires its native PostgreSQL host store and is refused here.
Ordinary SQLiteBackend instances keep their original strict backend guard.
"""

import asyncio
import re
import sqlite3
from contextlib import asynccontextmanager

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    ProcessRuntimeExecutionFence,
    current_execution_custody,
    require_execution_work,
)
from kestrel_sovereign.storage.async_database import (
    AsyncDatabase,
    CORE_SCHEMA,
    _close_failed_database_initialization,
)
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from kestrel_sovereign.storage.db.sqlite import (
    _close_aiosqlite_connection, _owned_script_statements,
)


_HOST_TABLES = frozenset({
    "service_providers", "host_service_keys",
    "user_master_service_keys", "sponsor_master_service_keys",
})


class _ProcessHostBackend(SQLiteBackend):
    def __init__(self, path, scopes):
        super().__init__(path)
        self._host_process_scopes = scopes
        # SQLAlchemy must refuse this retained owner even outside the opener's
        # context. Its generic executor cannot preserve this host boundary.
        self._execution_custody = scopes[0] if scopes else None

    @staticmethod
    def _authorize(action, table, column, database, trigger):
        if action in {sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_DROP_INDEX}:
            return sqlite3.SQLITE_OK if column in _HOST_TABLES else sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and column == "load_extension":
            return sqlite3.SQLITE_DENY
        if action in {
            sqlite3.SQLITE_READ, sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE,
            sqlite3.SQLITE_DELETE, sqlite3.SQLITE_CREATE_TABLE,
        }:
            return sqlite3.SQLITE_OK if table in _HOST_TABLES or table in {
                "sqlite_master", "sqlite_schema",
            } else sqlite3.SQLITE_DENY
        if action in {
            sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH, sqlite3.SQLITE_PRAGMA,
            sqlite3.SQLITE_CREATE_TRIGGER, sqlite3.SQLITE_CREATE_VIEW,
            sqlite3.SQLITE_DROP_TABLE, sqlite3.SQLITE_DROP_TRIGGER,
            sqlite3.SQLITE_DROP_VIEW, sqlite3.SQLITE_ALTER_TABLE,
        }:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    async def _install_authorizer(self, connection):
        # aiosqlite owns this native connection on its worker thread. Never
        # manipulate it on the event loop or publish it to an agent callback.
        await connection._execute(connection._conn.set_authorizer, self._authorize)

    async def connect(self):
        await super().connect()
        await self._install_authorizer(self._connection)

    async def _open_snapshot_read_connection(self):
        connection = await super()._open_snapshot_read_connection()
        try:
            await self._install_authorizer(connection)
        except BaseException:
            await _close_aiosqlite_connection(
                connection, retained_closes=self._retired_connection_closes,
            )
            raise
        return connection

    async def backup_to(self, dest_path):
        raise ExecutionAuthorityError("agent host-key accessor cannot export the host database")

    def _require_execution_backend(self):
        # Check cleanup-only and every ambient admission without clearing or
        # replacing ContextVars. Retain construction custody after scope reset.
        require_execution_work()
        scopes = self._host_process_scopes + current_execution_custody()
        if not scopes:
            raise ExecutionAuthorityError("host store requires original process custody")
        for scope in scopes:
            scope.require_work()
            if not isinstance(scope.fence, ProcessRuntimeExecutionFence):
                raise ExecutionAuthorityError(
                    "row/advisory-bound execution requires a native PostgreSQL host store"
                )

    def _after_commit(self):
        try:
            self._require_execution_backend()
        except ExecutionAuthorityError as error:
            # SQLite auto-commit already succeeded; never claim rollback or
            # allow the caller to replay a provisioning write after retirement.
            for scope in self._host_process_scopes:
                scope.preserve_commit_uncertainty("committed")
            raise ExecutionCommitOutcomeError("committed") from error

    async def execute(self, query, params=()):
        result = await super().execute(query, params)
        if not self.owns_open_transaction:
            self._after_commit()
        return result

    async def execute_many(self, query, params_list):
        result = await super().execute_many(query, params_list)
        if not self.owns_open_transaction:
            self._after_commit()
        return result

    async def execute_script(self, script):
        # sqlite3.executescript auto-commits between statements. Use the
        # canonical owning transaction so retirement before COMMIT rolls back
        # every host statement rather than publishing a partial script.
        async with self.transaction():
            for statement in _owned_script_statements(script):
                await self.execute(statement)

    @asynccontextmanager
    async def transaction(self, *, immediate=False, savepoint=False):
        nested = self.owns_open_transaction
        async with super().transaction(immediate=immediate, savepoint=savepoint):
            yield
            self._require_execution_backend()
        if not nested:
            self._after_commit()

    async def fetch_one(self, query, params=()):
        result = await super().fetch_one(query, params)
        self._require_execution_backend()
        return result

    async def fetch_all(self, query, params=()):
        result = await super().fetch_all(query, params)
        self._require_execution_backend()
        return result


async def open_process_host_database(path):
    """Open only the host.db path already resolved by the host-store caller."""
    backend = _ProcessHostBackend(path, current_execution_custody())
    db = AsyncDatabase(backend)
    connect = asyncio.create_task(backend.connect(), name="process-host-store-connect")
    cancelled = False
    try:
        while not connect.done():
            try:
                await asyncio.shield(connect)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                break
        connect.result()
        if cancelled:
            raise asyncio.CancelledError()
    except BaseException as error:
        await _close_failed_database_initialization(db, error)
        raise
    async def initialize_host_schema(database):
        # Reuse the canonical four table declarations, without creating graph,
        # conversations, tenant storage or unrelated migrations in host.db.
        for table in sorted(_HOST_TABLES):
            statement = re.search(
                rf"CREATE TABLE IF NOT EXISTS {table} \(.*?\n\);",
                CORE_SCHEMA, re.DOTALL,
            )
            if statement is None:
                raise RuntimeError("canonical host-key table declaration missing")
            await database.execute(statement.group())

    return await AsyncDatabase.from_connected_backend(
        backend, schema_initializer=initialize_host_schema,
    )
