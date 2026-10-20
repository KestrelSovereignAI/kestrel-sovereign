"""The canonical frozen fence preserves its exact native advisory lifetime."""

from dataclasses import FrozenInstanceError

import pytest

from kestrel_sovereign.execution_custody import (
    AdvisoryExecutionFence,
    ExecutionAuthorityError,
    bind_execution_custody,
)
from tests.integration import test_execution_authority_postgres

native_pg = test_execution_authority_postgres.native_pg

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_frozen_fence_uses_original_native_lease_after_retirement(native_pg):
    backend, observer, schema = native_pg

    async with backend.advisory_locks([(3763, 1)]) as lease:
        fence = AdvisoryExecutionFence(lease)
        assert fence.lease is lease
        assert fence.backend_type == "postgres"
        with pytest.raises(FrozenInstanceError):
            fence.backend_type = "sqlite"
        with bind_execution_custody(fence):
            await backend.execute(
                "INSERT INTO effects VALUES (1, 'original native fence')"
            )

    assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 1
    with pytest.raises(ExecutionAuthorityError):
        fence.require_work()

    async with backend.advisory_locks([(3763, 1)]) as replacement:
        assert replacement is not lease
        replacement.require_live()
        with pytest.raises(ExecutionAuthorityError):
            fence.require_work()
