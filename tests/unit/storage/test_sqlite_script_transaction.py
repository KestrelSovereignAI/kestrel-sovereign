"""Native scripts must not steal an enclosing transaction's commit boundary."""

import pytest

from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


@pytest.mark.asyncio
async def test_script_and_prior_write_roll_back_with_owner(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "script.db"))
    await backend.connect()
    try:
        await backend.execute("CREATE TABLE proof (value TEXT)")
        with pytest.raises(Exception, match="owner refuses commit"):
            async with backend.transaction():
                await backend.execute("INSERT INTO proof VALUES ('prior')")
                await backend.execute_script("INSERT INTO proof VALUES ('script');")
                raise ValueError("owner refuses commit")
        assert await backend.fetch_all("SELECT value FROM proof") == []
    finally:
        await backend.close()


@pytest.mark.parametrize(
    "control", ["COMMIT", "END", "ROLLBACK", "BEGIN", "SAVEPOINT p", "RELEASE p"]
)
@pytest.mark.asyncio
async def test_script_cannot_take_transaction_control(tmp_path, control):
    backend = SQLiteBackend(str(tmp_path / "script.db"))
    await backend.connect()
    try:
        await backend.execute("CREATE TABLE proof (value TEXT)")
        with pytest.raises(Exception, match="transaction control"):
            async with backend.transaction():
                await backend.execute_script(
                    f"INSERT INTO proof VALUES ('before'); /* lead; */ -- ignored\n {control};"
                )
        assert await backend.fetch_all("SELECT value FROM proof") == []
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_owned_script_preserves_trigger_and_quoted_semicolons(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "script.db"))
    await backend.connect()
    try:
        async with backend.transaction():
            await backend.execute_script("""
                CREATE TABLE proof (value TEXT);
                CREATE TABLE copies (value TEXT);
                CREATE TRIGGER copy_proof AFTER INSERT ON proof BEGIN
                    INSERT INTO copies VALUES (NEW.value);
                    INSERT INTO copies VALUES ('trigger;literal');
                END;
                INSERT INTO proof VALUES ('quoted;literal');
                -- A final statement does not require a terminator.
                INSERT INTO proof VALUES ('last')
            """)
        assert await backend.fetch_all("SELECT value FROM proof") == [
            ("quoted;literal",),
            ("last",),
        ]
        assert await backend.fetch_val("SELECT COUNT(*) FROM copies") == 4
    finally:
        await backend.close()
