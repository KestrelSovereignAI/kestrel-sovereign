"""Native custody checks for the upstream Lighthouse/cold-read merge."""

import hashlib

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ProcessRuntimeExecutionFence, bind_execution_custody,
    execution_denial_scopes, is_execution_control_error,
)
from kestrel_sovereign.kestrel_agent import _build_lighthouse_provider
from kestrel_sovereign.services.payer_resolver import open_host_db
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


@pytest.mark.asyncio
async def test_cold_host_reader_preserves_bytes_and_original_process_authority(tmp_path):
    root = tmp_path / "agent_data"
    root.mkdir()
    path = root / "host.db"
    seed = SQLiteBackend(str(path))
    await seed.connect()
    try:
        await seed.execute("CREATE TABLE service_providers(id TEXT, name TEXT)")
        await seed.execute("INSERT INTO service_providers VALUES ('native','native')")
    finally:
        await seed.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    fence = ProcessRuntimeExecutionFence("cold-host-reader")
    db = None
    try:
        with bind_execution_custody(fence) as scope:
            db = await open_host_db(agent_data_dir=root, cold_read=True)
            assert db.backend.cold_read
            assert await db.fetchone("SELECT name FROM service_providers") == ("native",)
            tables = await db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
            assert tables == [("service_providers",)], "cold inspection migrated operator host.db"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        with pytest.raises(Exception) as caught:
            await db.fetchone("SELECT name FROM service_providers")
        assert is_execution_control_error(caught.value)
        assert execution_denial_scopes(caught.value) == (scope,)
    finally:
        if db is not None:
            await db.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.asyncio
async def test_lighthouse_construction_never_formats_native_authority_denial(tmp_path):
    root = tmp_path / "agent_data"
    root.mkdir()
    seed = SQLiteBackend(str(root / "host.db"))
    await seed.connect()
    await seed.close()
    fence = ProcessRuntimeExecutionFence("lighthouse-construction")

    async def resolve_storage():
        # Exercise the actual guarded host-store acquisition, not a fake error.
        return await open_host_db(agent_data_dir=root, cold_read=True)

    with bind_execution_custody(fence) as scope:
        fence.retire()
        with pytest.raises(ExecutionAuthorityError) as caught:
            await _build_lighthouse_provider(resolve_storage)
        assert execution_denial_scopes(caught.value) == (scope,)
