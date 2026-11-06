"""Fixed native attempt retirement, never a grant of ordinary SQL authority.

The only mutation closes an existing exact attempt. No query, executor,
callback, new attempt, claim, or successor token is accepted by this API.
"""

import asyncio
import re
import sqlite3
from uuid import UUID

from kestrel_sovereign._async_ownership import await_owned_operation
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError,
    execution_terminal_error,
)


def _identities(attempt):
    from kestrel_sovereign.tool_effect_store import ToolEffectAttempt

    if type(attempt) is not ToolEffectAttempt:
        raise ValueError('Tool effect retirement requires the exact attempt type')
    values = (attempt.scope_digest, attempt.operation_digest, attempt.attempt_id)
    if (any(type(value) is not str for value in values)
            or any(re.fullmatch('[0-9a-f]{64}', value) is None for value in values[:2])
            or str(UUID(values[2])) != values[2]):
        raise ValueError('Tool effect retirement requires canonical original identities')
    return values


def _sqlite_finish(backend, connection, values):
    """One must-join worker unit through physical COMMIT/rollback."""
    from kestrel_sovereign.tool_effect_store import TOOL_EFFECT_OPERATIONS_SQL
    from .native_journal_programs import (
        _NativeJournalSQLiteConnection, _original_sqlite_query,
    )

    if (backend._connection is not connection or backend._closing
            or type(connection._conn) is not _NativeJournalSQLiteConnection
            or connection._conn.in_transaction):
        raise ExecutionAuthorityError('Tool effect retirement lost its original SQLite connection')
    native = connection._conn
    backend.assert_connected_file_still_valid()
    native._begin_original_journal()
    try:
        cursor = sqlite3.Connection.cursor(native, factory=sqlite3.Cursor)
        cursor.row_factory = None
        try:
            cursor.execute("SELECT sql FROM main.sqlite_schema WHERE type='table' AND name='tool_effect_operations'")
            if cursor.fetchone() != (TOOL_EFFECT_OPERATIONS_SQL,):
                raise ExecutionAuthorityError('Tool effect retirement relation changed')
            cursor.execute("SELECT 1 FROM main.sqlite_schema WHERE type='trigger' AND tbl_name='tool_effect_operations' UNION ALL SELECT 1 FROM temp.sqlite_schema WHERE type='trigger' AND tbl_name='tool_effect_operations' LIMIT 1")
            if cursor.fetchone() is not None:
                raise ExecutionAuthorityError('Tool effect retirement refuses trigger programs')
            cursor.execute("PRAGMA main.index_list('tool_effect_operations')")
            if any(row[3] != 'pk' for row in cursor.fetchall()):
                raise ExecutionAuthorityError('Tool effect retirement refuses application indexes')
            query = _original_sqlite_query(
                "UPDATE main.tool_effect_operations SET state='CLOSED', revision=revision+1 WHERE scope_digest=? AND operation_digest=? AND attempt_id=?",
                values, sqlite3.Connection.getlimit(native, sqlite3.SQLITE_LIMIT_SQL_LENGTH),
                database_encoding=native._original_journal_configuration()[-2])
            cursor.execute(query)
            changed = cursor.rowcount == 1
        finally:
            cursor.close()
        try:
            native.commit()
        except BaseException as error:
            from .native_journal_programs import NativeJournalDurabilityRefusal

            if type(error) is NativeJournalDurabilityRefusal:
                raise
            raise ExecutionCommitOutcomeError('unknown') from error
        return changed
    except BaseException as error:
        try:
            native.rollback()
        except BaseException as rollback_error:
            raise execution_terminal_error(error, rollback_error, retain_originals=True)
        raise


async def finish_tool_effect_attempt(backend, attempt):
    """Close only this existing attempt on its original connected backend."""
    from .sqlite import SQLiteBackend
    from .postgres import PostgresBackend

    values = _identities(attempt)
    if type(backend) is SQLiteBackend:
        connection = backend._connection
        if connection is None or backend._closing or backend.owns_open_transaction:
            raise ExecutionAuthorityError('Tool effect retirement requires its original outermost backend')
        await backend._wait_for_cancelled_write_drain()
        async with backend._write_lock:
            backend._raise_cancelled_write_drain_error()
            with backend._write_operation():
                outcome = await await_owned_operation(
                    connection._execute(_sqlite_finish, backend, connection, values),
                    name='tool-effect-attempt-retirement', retirement_resource=connection,
                    retain_originals=True)
                error = execution_terminal_error(outcome.error, outcome.cancellation, retain_originals=True)
                if error is not None:
                    if outcome.error is None:
                        # The SAME worker physically returned from COMMIT;
                        # observation withdrawal cannot erase that native fact.
                        error._kestrel_unresolved_commit_outcome = 'committed'
                    raise error
                return outcome.result
    if type(backend) is not PostgresBackend or backend._pool is None or backend.owns_open_transaction:
        raise ExecutionAuthorityError('Tool effect retirement requires its original native backend')
    # Same native checkout/COMMIT/reset ownership as fixed Stop settlement.
    # No ordinary custody is rebound and no SQL executor escapes this unit.
    async with backend._tracked_pool_checkout(backend._pool, admit_work=False) as (connection, state):
        async with backend._tracked_native_transaction(connection, state.scopes, state):
            result = await connection.execute(
                "UPDATE tool_effect_operations SET state='CLOSED', revision=revision+1 WHERE scope_digest=$1 AND operation_digest=$2 AND attempt_id=$3",
                *values)
            return result == 'UPDATE 1'
