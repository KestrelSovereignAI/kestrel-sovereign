"""Actual proxy finalizers preserve independent local-native source graphs."""
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_authority_postgres import native_pg as _cleanup_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_isolated_feature_runtime import FakeIsolatedClient, _initialized_host_ingress_proxy
from tests.utils.original_failure import OriginalFailure

cleanup_pg = _cleanup_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("reconcile", [False, True])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_config_body_and_finalizer_keep_independent_originals(
    native_pg, cleanup_pg, tmp_path, monkeypatch, reconcile, source_cancel,
):
    opening = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    cleanup = OriginalFailure.capture(await _original_control(cleanup_pg[0], monkeypatch, source_cancel))
    feature, _ = await _initialized_host_ingress_proxy(monkeypatch, tmp_path, FakeIsolatedClient)
    database = await AsyncDatabase.sqlite(str(tmp_path / "config-finalizer-original.db"))
    await database.execute("CREATE TABLE finalized (id INTEGER)")
    actual_finalize = feature._finalize_external_ingress_transition
    async def finalize(quiesce):
        await actual_finalize(quiesce)
        await database.execute("INSERT INTO finalized VALUES (1)")
        raise cleanup.error
    async def body(*args, **kwargs):
        raise opening.error
    async def transition():
        if reconcile:
            await feature._reconcile_client_to_authoritative_config_with_ingress_fence({}, force=True)
        else:
            async with feature.config_transition_ingress_fence():
                raise opening.error
    try:
        with monkeypatch.context() as patch:
            patch.setattr(feature, "_finalize_external_ingress_transition", finalize)
            if reconcile:
                patch.setattr(feature, "_reconcile_client_to_authoritative_config", body)
            outcome = await await_owned_task(create_owned_task(transition(), retain_originals=True))
        assert await database.fetchval("SELECT count(*) FROM finalized") == 1
        opening.assert_retained(outcome.error)
        cleanup.assert_retained(outcome.error)
        assert not feature._config_ingress_live_turn_bypass_active
    finally:
        await feature.shutdown()
        await database.close()
