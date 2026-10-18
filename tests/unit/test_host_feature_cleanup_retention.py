"""Actual host contribution cleanup keeps failure and compatibility contracts."""

import pytest

from kestrel_sovereign.host_features import (
    SovereignHostContext,
    close_host_context,
    start_host_features,
    stop_host_features,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.fixtures.sdk_contribution_fixture import SDKFixtureHostFeature


@pytest.mark.asyncio
async def test_ordinary_optional_stop_failure_permits_a_clean_restart():
    calls = []

    class OptionalStop(SDKFixtureHostFeature):
        async def on_host_stop(self, ctx):
            calls.append(self)
            await super().on_host_stop(ctx)
            raise RuntimeError("ordinary optional stop failure")

    feature, ctx = OptionalStop(), SovereignHostContext()
    for _ in range(2):
        assert await start_host_features([feature], ctx) == [feature]
        await stop_host_features([feature], ctx)
        assert ctx.feature_contribution_runtime.active_owners() == ()
        assert ctx.started_host_features == ()
        assert not ctx._native_close_owners
    assert calls == [feature, feature]


@pytest.mark.asyncio
async def test_failed_declarative_teardown_survives_independent_sqlite_close(
    tmp_path, monkeypatch
):
    db = await AsyncDatabase.sqlite(str(tmp_path / "host-declarative-owner.db"))
    await db.execute("CREATE TABLE physical_close (id INTEGER)")
    connection = db.backend._connection
    worker = getattr(connection, "_thread", connection)
    ctx, feature = SovereignHostContext(db=db), SDKFixtureHostFeature()
    original = ValueError("original declarative teardown did not acknowledge")
    calls = []
    try:
        await start_host_features([feature], ctx)
        runtime = ctx.feature_contribution_runtime
        native_deactivate = runtime.deactivate

        def lose_deactivation_ack(owner):
            calls.append(owner)
            native_deactivate(owner)
            raise original

        monkeypatch.setattr(runtime, "deactivate", lose_deactivation_ack)
        with pytest.raises(ValueError) as first:
            await stop_host_features([feature], ctx)
        assert first.value is original
        assert runtime.active_owners() == () and feature.stopped
        with pytest.raises(ValueError) as repeated:
            await stop_host_features([feature], ctx)
        assert repeated.value is original and calls == [feature]
        with pytest.raises(ValueError) as closed:
            await close_host_context(ctx)
        assert closed.value is original
        assert db.backend._connection is None and not worker.is_alive()
        with pytest.raises(ValueError) as restarted:
            await start_host_features([feature], ctx)
        assert restarted.value is original and calls == [feature]
    finally:
        await db.close()
