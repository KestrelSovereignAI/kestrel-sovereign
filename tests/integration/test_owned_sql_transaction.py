"""Every actual adapter SQL surface retains its owner's commit boundary."""

from uuid import uuid4

import pytest


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    "method",
    [
        "execute",
        "execute_many",
        "execute_script",
        "fetch_one",
        "fetch_all",
        "fetch_val",
    ],
)
@pytest.mark.parametrize("control", ["COMMIT", "ROLLBACK"])
async def test_owned_sql_entrypoint_cannot_steal_commit(db_backend, method, control):
    await _assert_transaction_control_refused(db_backend, method, control)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["fetch_one_diagnostic", "fetch_all_diagnostic"])
@pytest.mark.parametrize("control", ["COMMIT", "ROLLBACK"])
async def test_sqlite_diagnostic_sql_cannot_steal_commit(
    sqlite_backend, method, control
):
    # PostgreSQL has no separate diagnostic SQL entry points; its facade uses
    # the ordinary methods covered above. Exercise SQLite's real extra path.
    await _assert_transaction_control_refused(sqlite_backend, method, control)


async def _assert_transaction_control_refused(backend, method, control):
    await backend.connect()
    table = "transaction_boundary_" + uuid4().hex
    await backend.execute(f"CREATE TABLE {table} (value INTEGER)")
    try:
        with pytest.raises(Exception, match="transaction control"):
            async with backend.transaction():
                await backend.execute(f"INSERT INTO {table} VALUES (1)")
                args = ("/* custody */ " + control + ";",)
                if method == "execute_many":
                    args += ([()],)
                await getattr(backend, method)(*args)
        assert await backend.fetch_val(f"SELECT COUNT(*) FROM {table}") == 0
        assert backend.owns_open_transaction is False
    finally:
        await backend.execute(f"DROP TABLE {table}")


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_native_script_refuses_control_before_any_statement_runs(db_backend):
    await db_backend.connect()
    table = "transaction_script_" + uuid4().hex
    await db_backend.execute(f"CREATE TABLE {table} (value INTEGER)")
    try:
        with pytest.raises(Exception, match="transaction control"):
            async with db_backend.transaction():
                await db_backend.execute_script(
                    f"INSERT INTO {table} VALUES (1); /* comment */ COMMIT;"
                )
        assert await db_backend.fetch_val(f"SELECT COUNT(*) FROM {table}") == 0
    finally:
        await db_backend.execute(f"DROP TABLE {table}")


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_native_script_retains_quoted_body_semantics(db_backend):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL dollar-quoted body grammar")
    await db_backend.connect()
    async with db_backend.transaction():
        await db_backend.execute_script(
            "DO $proof$ BEGIN PERFORM 'COMMIT; END;'; END; $proof$;"
        )
        await db_backend.execute_script(
            "DO $😀$ BEGIN PERFORM 'COMMIT; END;'; END; $😀$;"
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_escape_continuation_cannot_hide_commit(db_backend):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL escape-string continuation grammar")
    table = "continued_string_" + uuid4().hex
    await db_backend.execute(f"CREATE TABLE {table} (value INTEGER)")
    try:
        with pytest.raises(Exception, match="transaction control"):
            async with db_backend.transaction():
                await db_backend.execute(f"INSERT INTO {table} VALUES (1)")
                await db_backend.execute_script(
                    "SELECT E'a'\n'b\\'x';\nSELECT 'c\\';\nCOMMIT;"
                )
        assert await db_backend.fetch_val(f"SELECT COUNT(*) FROM {table}") == 0
    finally:
        await db_backend.execute(f"DROP TABLE {table}")
