"""Every actual adapter SQL surface retains its owner's commit boundary."""

from uuid import uuid4

import pytest

from tests.utils.postgres_schema import disposable_postgres_schema


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("trap", ["identifiers", "case-alias"])
async def test_postgres_atomic_identifiers_cannot_hide_commit(db_backend, trap):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL SQL-standard identifier grammar")
    async with disposable_postgres_schema(db_backend, "owned_atomic_names") as schema:
        await db_backend.execute(f'CREATE TABLE "{schema}".proof (value integer)')
        definition = (
            "CREATE DOMAIN atomic AS integer; CREATE TABLE function (begin atomic);"
            if trap == "identifiers"
            else f'CREATE FUNCTION "{schema}".owned_case_label() RETURNS integer '
            "LANGUAGE SQL BEGIN ATOMIC SELECT 1 AS case; END;"
        )
        # Use a session-local path, so the actual unquoted BEGIN/ATOMIC pair
        # belongs to the test's disposable type, never any shared schema.
        refusal = None
        try:
            async with db_backend.transaction():
                await db_backend.execute(f'SET LOCAL search_path TO "{schema}"')
                await db_backend.execute(f'INSERT INTO "{schema}".proof VALUES (1)')
                await db_backend.execute_script(definition + " COMMIT;")
        except Exception as exc:
            refusal = exc
        assert await db_backend.fetch_val(f'SELECT COUNT(*) FROM "{schema}".proof') == 0
        assert refusal is not None and "transaction control" in str(refusal)


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("alias", ["case", "end"])
async def test_postgres_atomic_keyword_alias_retains_outer_rollback(db_backend, alias):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL SQL-standard alias grammar")
    name = "owned_alias_" + uuid4().hex
    definition = f"CREATE FUNCTION {name}() RETURNS integer LANGUAGE SQL BEGIN ATOMIC SELECT 1 AS {alias}; END;"

    class RollbackProof(Exception):
        pass

    with pytest.raises(Exception, match="rollback proof"):
        async with db_backend.transaction():
            await db_backend.execute_script(definition)
            assert await db_backend.fetch_val(f"SELECT {name}()") == 1
            raise RollbackProof("rollback proof")
    assert (
        await db_backend.fetch_val("SELECT to_regprocedure(?)", (name + "()",)) is None
    )


@pytest.mark.asyncio
async def test_sqlite_carriage_return_keeps_comment_text(sqlite_backend):
    async with sqlite_backend.transaction():
        await sqlite_backend.execute_script(
            "-- note\rCOMMIT;\nCREATE TABLE comment_proof (value integer);"
        )
        await sqlite_backend.execute("INSERT INTO comment_proof VALUES (1)")
        assert await sqlite_backend.fetch_val("SELECT COUNT(*) FROM comment_proof") == 1


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_native_postgres_sql_standard_body_preserves_outer_custody(db_backend):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL SQL-standard routine grammar")
    name = "owned_atomic_" + uuid4().hex
    definition = f"CREATE FUNCTION {name}() RETURNS integer LANGUAGE SQL BEGIN ATOMIC SELECT CASE WHEN true THEN 1 ELSE 2 END; END;"

    class RollbackProof(Exception):
        pass

    with pytest.raises(Exception, match="rollback proof") as rolled_back:
        async with db_backend.transaction():
            await db_backend.execute_script(definition)
            assert await db_backend.fetch_val(f"SELECT {name}()") == 1
            raise RollbackProof("rollback proof")
    assert isinstance(rolled_back.value.__cause__, RollbackProof)
    assert (
        await db_backend.fetch_val("SELECT to_regprocedure(?)", (name + "()",)) is None
    )
    # The final routine END must not conceal a following real COMMIT.
    with pytest.raises(Exception, match="transaction control"):
        async with db_backend.transaction():
            await db_backend.execute_script(definition + " COMMIT;")
    assert (
        await db_backend.fetch_val("SELECT to_regprocedure(?)", (name + "()",)) is None
    )


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
