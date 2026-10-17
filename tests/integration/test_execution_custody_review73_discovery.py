"""A discovery timeout cannot turn a real committed receipt into an outage."""

from types import SimpleNamespace

import pytest

from kestrel_sovereign.execution_custody import (
    bind_execution_custody,
    execution_commit_outcome,
)
from kestrel_sovereign.llm.discovery_deadline import DiscoveryStepTimeout
from tests.integration.test_execution_authority_postgres import (
    GenerationFence,
)
from tests.integration.test_execution_authority_postgres import (
    native_pg as native_pg,  # noqa: PLC0414 - pytest fixture must be imported
)
from tests.integration.test_execution_custody_review30_pg import lose_ack
from tests.unit.test_execution_custody_review30 import service_owner

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("entry", ["chat", "decision"])
@pytest.mark.parametrize("carrier", ["timeout", "discovery_timeout"])
async def test_discovery_preserves_timeout_carried_native_commit_receipt(
    native_pg, monkeypatch, entry, carrier,
):
    backend, observer, schema = native_pg
    lose_ack(monkeypatch, cancelled=False)
    original = []

    async def listing(_client):
        try:
            await backend.execute("INSERT INTO effects VALUES (73, 'native discovery commit')")
        except Exception as error:
            assert execution_commit_outcome(error) == "unknown"
            timeout = (
                TimeoutError("original listing timeout")
                if carrier == "timeout"
                else DiscoveryStepTimeout(vendor="native", route="native:local",
                                          step="listing", seconds=5)
            )
            original.append(timeout)
            raise timeout from error
        pytest.fail("actual native COMMIT acknowledgement loss was not observed")

    service = service_owner()
    service.config = {}
    provider = {
        "name": "native:local", "vendor": "native", "client": None,
        "adapter": SimpleNamespace(list_models=listing, list_decision_models=listing),
    }
    with bind_execution_custody(GenerationFence()), pytest.raises(BaseException) as caught:
        if entry == "chat":
            await service._discover_vendor_catalog("native", provider)
        else:
            await service._discover_decision_route(provider, force=True)
    assert original and caught.value is original[0]
    assert execution_commit_outcome(caught.value) == "unknown"
    assert await observer.fetchval(
        f'SELECT value FROM "{schema}".effects WHERE id=73'
    ) == "native discovery commit"
