"""Finite caller withdrawal must not become a later reader's SafeMode source."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.kestrel_agent import require_feature_lifecycle_integrity
from tests.fixtures.sdk_contribution_fixture import SDKFixtureFeature
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_shared_lifecycle_latch_keeps_native_source_not_observer_withdrawal(
    native_pg, onboarding_owner, monkeypatch, source_cancel,
):
    _, agent, database = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    entered, release, withdrawals = asyncio.Event(), asyncio.Event(), []
    ordinary = OSError("foreign enable prose must not become shared native evidence")
    class Feature(SDKFixtureFeature):
        contribution_prefix = "shared-source-208"
        async def on_enable(self):
            raise ordinary
        async def on_disable(self):
            raise source.error
        async def shutdown(self):
            await database.execute("INSERT INTO host_rollback_effects VALUES (309)")
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError as error:
                withdrawals.append(error)
                raise
    feature = Feature(agent)
    owner = create_owned_task(agent._activate_feature_runtime(feature), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual current activation cleanup observer withdrew")
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        shared = agent._feature_lifecycle_control
        source.assert_retained(shared)
        chain = tuple(_explicit_control_chain(shared))
        assert ordinary not in chain
        assert all(error not in chain for error in withdrawals), "current caller withdrawal escaped into shared SafeMode source"
        with pytest.raises(BaseException) as later:
            require_feature_lifecycle_integrity(agent)
        source.assert_retained(later.value)
        assert all(error not in tuple(_explicit_control_chain(later.value)) for error in withdrawals)
        assert await database.fetchval("SELECT count(*) FROM host_rollback_effects WHERE id=309") == 1
    finally:
        release.set()
        await await_owned_task(owner)
