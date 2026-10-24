"""Pure request/authority contracts for terminal physical exclusion (#3739)."""

import pytest

from kestrel_sovereign.storage.db.interface import ConnectionError
from kestrel_sovereign.storage.db.postgres import (
    PostgresBackend,
    TerminalAdvisoryBarrier,
)


@pytest.mark.parametrize("keys", [(), [], [(1, 2)]])
def test_terminal_keys_require_nonempty_immutable_tuple(keys):
    with pytest.raises(ValueError, match="nonempty immutable"):
        TerminalAdvisoryBarrier(keys)


@pytest.mark.parametrize("keys", [((1,),), ((1, 2, 3),), ([1, 2],)])
def test_terminal_keys_require_exact_immutable_pairs(keys):
    with pytest.raises(TypeError, match="immutable integer pairs"):
        TerminalAdvisoryBarrier(keys)


@pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
def test_terminal_keys_reject_noninteger_and_boolean_values(value):
    with pytest.raises(TypeError, match="not bool"):
        TerminalAdvisoryBarrier(((value, 2),))


@pytest.mark.parametrize("value", [-(2**31) - 1, 2**31])
def test_terminal_keys_reject_outside_signed_int32(value):
    with pytest.raises(ValueError, match="signed 32-bit"):
        TerminalAdvisoryBarrier(((1, value),))


def test_terminal_keys_keep_exact_signed_boundary_values():
    keys = ((-(2**31), 2**31 - 1), (0, 0))
    assert TerminalAdvisoryBarrier(keys).keys == keys


@pytest.mark.asyncio
async def test_terminal_rejects_opaque_operational_pool_before_advisory_allocation():
    backend = PostgresBackend.from_pool(
        object(), advisory_dsn="not-a-postgresql-connection-string"
    )
    with pytest.raises(ConnectionError, match="original connected backend native pool"):
        async with backend.advisory_locks(TerminalAdvisoryBarrier(((3739, 1),))):
            pytest.fail("opaque operational pool admitted a terminal physical barrier")
    assert backend._advisory_pool is None
