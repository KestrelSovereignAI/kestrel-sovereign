"""Native SQLite rollback/commit evidence through the shared setup boundary."""

import pytest
import pytest_asyncio

from kestrel_sdk.resources import ResourceError, ResourceOutcomeUnknown
from kestrel_sdk.storage.database.interface import QueryError, TransactionError
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError, execution_commit_outcome,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.initialization_errors import repository_initialization_errors


class OwnedIntegrityError(ResourceError):
    pass


@pytest_asyncio.fixture
async def database(tmp_path):
    async def no_core_schema(db):
        pass

    db = await AsyncDatabase.sqlite(str(tmp_path / "setup.db"), schema_initializer=no_core_schema)
    try:
        await db.execute("CREATE TABLE owned_setup (value TEXT)")
        yield db
    finally:
        await db.close()


def setup_errors():
    return repository_initialization_errors(integrity_error=OwnedIntegrityError, unknown_message="owned setup requires reconciliation")


@pytest.mark.asyncio
async def test_original_integrity_error_is_unwrapped_after_actual_rollback(database):
    original = OwnedIntegrityError("restore original owned root")
    with pytest.raises(OwnedIntegrityError) as error:
        with setup_errors():
            async with database.transaction():
                await database.execute("INSERT INTO owned_setup VALUES ('rolled back')")
                raise original
    assert error.value is original
    assert await database.fetchone("SELECT COUNT(*) FROM owned_setup") == (0,)


@pytest.mark.asyncio
async def test_lost_ack_after_real_commit_retains_the_original_uncertain_receipt(database):
    original = ExecutionCommitOutcomeError("unknown")
    with pytest.raises(ResourceOutcomeUnknown) as error:
        with setup_errors():
            async with database.transaction():
                await database.execute("INSERT INTO owned_setup VALUES ('committed')")
            raise original
    assert error.value.__cause__ is original
    assert execution_commit_outcome(error.value) == "unknown"
    assert await database.fetchone("SELECT COUNT(*) FROM owned_setup") == (1,)


@pytest.mark.asyncio
async def test_native_control_is_not_replaced_by_an_integrity_error_in_its_cause(database):
    class OwnedStop(TransactionError, ExecutionAuthorityError):
        pass

    original = OwnedStop("owned execution stopped")
    original.__cause__ = OwnedIntegrityError("an earlier setup failure")
    with pytest.raises(OwnedStop) as error:
        with setup_errors():
            async with database.transaction():
                await database.execute("INSERT INTO owned_setup VALUES ('must roll back')")
                raise original
    assert error.value is original
    assert await database.fetchone("SELECT COUNT(*) FROM owned_setup") == (0,)


@pytest.mark.asyncio
async def test_real_sql_failure_remains_a_transaction_failure(database):
    with pytest.raises(TransactionError) as error:
        with setup_errors():
            async with database.transaction():
                await database.execute("INSERT INTO missing_owned_table VALUES ('refused')")
    assert isinstance(error.value.__cause__, QueryError)
