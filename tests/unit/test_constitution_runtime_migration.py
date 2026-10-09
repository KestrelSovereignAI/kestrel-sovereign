"""Concurrent sovereign replicas must share one successful schema upgrade."""

import asyncio

import pytest

from kestrel_sovereign.constitution.runtime_state import ConstitutionRuntimeStateStore
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_column", ["revision", "safe_mode_cause", "generation"])
async def test_independent_sqlite_replicas_upgrade_missing_column_once(
    tmp_path, missing_column
):
    path = str(tmp_path / "legacy-runtime.db")
    first, second = SQLiteBackend(path), SQLiteBackend(path)
    await first.connect()
    await second.connect()
    try:
        await ConstitutionRuntimeStateStore(first).initialize()
        await first.execute("DROP TRIGGER constitution_runtime_revision_fence_v2")
        await first.execute(
            f"ALTER TABLE constitution_runtime_state DROP COLUMN {missing_column}"
        )
        inspected = 0
        both_inspected = asyncio.Event()

        class ConcurrentUpgradeStore(ConstitutionRuntimeStateStore):
            async def _has_column(self, name):
                nonlocal inspected
                found = await super()._has_column(name)
                # Synchronize the unreserved catalog reads, not the protected
                # recheck: that must serialize independent SQLite connections.
                if (
                    name == missing_column
                    and not found
                    and not self._backend._in_transaction
                ):
                    inspected += 1
                    if inspected == 2:
                        both_inspected.set()
                    await both_inspected.wait()
                return found

        async with asyncio.timeout(10):
            await asyncio.gather(
                ConcurrentUpgradeStore(first).initialize(),
                ConcurrentUpgradeStore(second).initialize(),
            )
        assert inspected == 2
        assert await ConstitutionRuntimeStateStore(first)._has_column(missing_column)
        assert await ConstitutionRuntimeStateStore(second)._has_column(missing_column)
    finally:
        await first.close()
        await second.close()
