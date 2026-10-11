"""Fixed privacy destruction on the original retired process's native pool.

No cleanup executor is returned or installed on live storage. Only canonical
purge primitives receive this private connection-bound adapter; ordinary work
retains its denying original custody throughout and after the sweep.
"""

from contextlib import asynccontextmanager
from copy import copy

from kestrel_sovereign.storage.async_database import AsyncDatabase
from .placeholder import sqlite_to_postgres


class _EphemeralCleanupDatabase(AsyncDatabase):
    def __init__(self, backend, connection):
        super().__init__(backend)
        self._connection = connection
        self._initialized = True

    def _query(self, sql, params):
        query, _ = sqlite_to_postgres(sql)
        return query, self._backend._strip_tz(params)

    async def execute(self, sql, params=()):
        query, params = self._query(sql, params)
        result = await self._connection.execute(query, *params)
        return int(result.rsplit(" ", 1)[-1]) if result.rsplit(" ", 1)[-1].isdigit() else 0

    async def execute_commit(self, sql, params=()):
        return await self.execute(sql, params)

    async def fetchall(self, sql, params=()):
        query, params = self._query(sql, params)
        return await self._connection.fetch(query, *params)

    async def fetchone(self, sql, params=()):
        query, params = self._query(sql, params)
        return await self._connection.fetchrow(query, *params)

    async def fetchval(self, sql, params=()):
        query, params = self._query(sql, params)
        return await self._connection.fetchval(query, *params)

    async def table_exists(self, table_name):
        return await self._connection.fetchval("SELECT to_regclass($1) IS NOT NULL", table_name)

    @asynccontextmanager
    async def transaction(self, *, immediate=False, savepoint=False):
        async with self._connection.transaction():
            yield


async def _purge_original_ephemeral_session(backend, connection, wrapper, *, reason, agent_id, since):
    # Shallow copies preserve the canonical purge algorithms and destructive
    # audit sink without mutating a backend or replacing any runtime authority.
    # Neither files, providers, migrations nor ordinary reads are initialized.
    db = _EphemeralCleanupDatabase(backend, connection)
    storage = copy(wrapper._storage)
    storage.db = db
    storage.agent_id = agent_id
    storage.conversation = copy(storage.conversation)
    storage.conversation.db = db
    storage.conversation.agent_id = agent_id
    storage.conversation._lexical_index = copy(storage.conversation._lexical_index)
    storage.conversation._lexical_index.db = db
    storage.graph = copy(storage.graph)
    storage.graph.db = db
    storage.graph.agent_id = agent_id
    cleanup = copy(wrapper)
    cleanup._storage = storage
    cleanup._entered_ephemeral_at = since
    # Optional observability callback keeps its ordinary, denying backend; it
    # is never handed this private executor or upgraded to cleanup authority.
    report = await cleanup.purge_ephemeral_session(reason=reason)
    if not report.required_sweep_failed:
        wrapper._entered_ephemeral_at = cleanup._entered_ephemeral_at
    wrapper._session_conversations = []
    wrapper._session_files = {}
    return report
