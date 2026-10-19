"""Actual SQL journal namespace isolation; no cloud transport or model calls."""

from contextlib import asynccontextmanager

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    bind_execution_custody,
    execution_commit_outcome,
    execution_denial_scopes,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import QueryError
from kestrel_sovereign.storage.sovereign_adapter import SovereignStorageAdapter
from tests.integration.test_execution_authority_postgres import (
    GenerationFence,
)
from tests.integration.test_execution_authority_postgres import (
    native_pg as _native_pg,
)
from tests.utils.postgres_schema import (
    disposable_postgres_schema,
    postgres_test_url,
    with_search_path,
)

native_pg = _native_pg


async def _no_schema_ddl(_backend):
    """Connect without booting/reconciling a caller's public schema."""


@asynccontextmanager
async def _owned_database(backend, tmp_path):
    if backend == "sqlite":
        db = await AsyncDatabase.sqlite(str(tmp_path / "owned-journal.db"))
        try:
            yield db
        finally:
            await db.close()
        return
    url = postgres_test_url()
    if url is None:
        pytest.skip("TEST_POSTGRES_URL is required for the real PostgreSQL case")
    admin = await AsyncDatabase.postgres(url, schema_initializer=_no_schema_ddl)
    try:
        async with disposable_postgres_schema(
            admin, "owned_sovereign_journal"
        ) as schema:
            db = await AsyncDatabase.postgres(
                with_search_path(url, schema), schema_initializer=_no_schema_ddl
            )
            try:
                yield db
            finally:
                await db.close()
    finally:
        await admin.close()


async def _append(adapter, *, failure=None):
    await adapter._log_import_attempt(
        agent_did="did:kestrel:owned-source",
        source_did="did:kestrel:owned-source",
        package_hash="a" * 64,
        continuity_score=1.0,
        status="imported",
        reject_reason=None,
        messages_restored=1,
        shards_restored=1,
        manifest_version="3.0",
        source_cid="owned-cid",
        failure=failure,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend", ["sqlite", pytest.param("postgres", marks=pytest.mark.native_postgres)]
)
async def test_configured_journal_preserves_host_schema_and_rows(backend, tmp_path):
    async with _owned_database(backend, tmp_path) as db:
        await db.execute(
            "CREATE TABLE agent_import_log(id TEXT PRIMARY KEY,companion_id TEXT NOT NULL,source_system TEXT NOT NULL)"
        )
        await db.execute(
            "INSERT INTO agent_import_log VALUES ('owned-host-record','owned-companion','owned-host-provenance')"
        )
        adapter = SovereignStorageAdapter(
            db,
            "owned-journal-secret",
            object(),
            agent_id="did:kestrel:owned-target",
            import_log_table="sovereign_import_log",
        )
        await _append(adapter)
        rows = await adapter.get_import_log(agent_did="did:kestrel:owned-source")
        assert len(rows) == 1
        assert rows[0]["host_agent_id"] == "did:kestrel:owned-target"
        assert rows[0]["status"] == "imported" and rows[0]["messages_restored"] == 1
        assert (
            rows[0]["source_cid"] == "owned-cid" and rows[0]["package_hash"] == "a" * 64
        )
        assert await db.fetchall("SELECT * FROM agent_import_log") == [
            ("owned-host-record", "owned-companion", "owned-host-provenance")
        ]
        assert not await db.column_exists("agent_import_log", "grant_id")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend", ["sqlite", pytest.param("postgres", marks=pytest.mark.native_postgres)]
)
@pytest.mark.parametrize("ignored", [False, True])
async def test_refused_journal_insert_propagates_actual_sql_error(
    backend, tmp_path, ignored
):
    async with _owned_database(backend, tmp_path) as db:
        adapter = SovereignStorageAdapter(
            db,
            "owned-journal-secret",
            object(),
            agent_id="did:kestrel:owned-target",
            import_log_table="sovereign_import_log",
        )
        await adapter._ensure_import_log()
        if backend == "sqlite":
            action = (
                "SELECT RAISE(IGNORE)"
                if ignored
                else "SELECT RAISE(ABORT,'owned audit refused')"
            )
            await db.execute(
                f"CREATE TRIGGER owned_refuse_import BEFORE INSERT ON sovereign_import_log BEGIN {action}; END"
            )
        else:
            action = (
                "RETURN NULL" if ignored else "RAISE EXCEPTION 'owned audit refused'"
            )
            await db.execute(
                f"CREATE FUNCTION owned_refuse_import() RETURNS TRIGGER LANGUAGE plpgsql AS $$ BEGIN {action}; END $$"
            )
            await db.execute(
                "CREATE TRIGGER owned_refuse_import BEFORE INSERT ON sovereign_import_log FOR EACH ROW EXECUTE FUNCTION owned_refuse_import()"
            )
        with pytest.raises(
            QueryError, match="not acknowledged" if ignored else "owned audit refused"
        ):
            await _append(adapter)
        assert await db.fetchone("SELECT count(*) FROM sovereign_import_log") == (0,)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "a.b",
        'a";DROP TABLE x;--',
        "x" * 41,
        "space name",
        "x\n",
        "a-b",
        "MixedCase",
    ],
)
def test_journal_namespace_refuses_unsafe_identifiers_before_any_io(name):
    with pytest.raises(ValueError, match="simple SQL identifier"):
        SovereignStorageAdapter(
            object(), "owned-secret", object(), import_log_table=name
        )


@pytest.mark.parametrize("name", [None, 123, True])
def test_journal_namespace_requires_text_before_any_io(name):
    with pytest.raises(TypeError, match="simple SQL identifier"):
        SovereignStorageAdapter(
            object(), "owned-secret", object(), import_log_table=name
        )


@pytest.mark.asyncio
@pytest.mark.native_postgres
@pytest.mark.parametrize("outcome", ["denied", "committed", "unknown"])
@pytest.mark.parametrize("operation", ["append", "failed-append", "read"])
async def test_journal_preserves_original_native_control_before_ddl(
    native_pg, outcome, operation
):
    backend, control, schema = native_pg
    adapter = SovereignStorageAdapter(
        AsyncDatabase(backend),
        "owned-native-journal-secret",
        object(),
        agent_id="did:kestrel:owned-native-target",
        import_log_table="sovereign_import_log",
    )
    async with backend.advisory_locks([(3721, 1)]) as lease:
        with bind_execution_custody(GenerationFence(lease=lease)) as original:
            if outcome == "denied":
                original.revoke("owned original journal admission retired")
            else:
                original.preserve_commit_uncertainty(outcome)
            body_error = None
            if operation == "failed-append":
                with pytest.raises(ExecutionAuthorityError) as body:
                    original.require_work()
                body_error = body.value
            with pytest.raises(
                ExceptionGroup if body_error else ExecutionAuthorityError
            ) as caught:
                if operation in {"append", "failed-append"}:
                    await _append(adapter, failure=body_error)
                else:
                    await adapter.get_import_log()
            if body_error:
                assert body_error in caught.value.exceptions
            assert execution_commit_outcome(caught.value) == (
                None if outcome == "denied" else outcome
            )
            if outcome == "denied":
                assert execution_denial_scopes(caught.value) == (original,)
            else:
                assert execution_denial_scopes(caught.value) is None
    assert (
        await control.fetchval(
            "SELECT to_regclass($1)", f"{schema}.sovereign_import_log"
        )
        is None
    )
