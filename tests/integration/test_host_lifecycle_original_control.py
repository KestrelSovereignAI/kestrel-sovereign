"""Public host teardown retains native control while retiring all peers."""

import httpx
import pytest
from kestrel_sdk.features.host_base import HostFeature

from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    execution_commit_outcome,
)
from kestrel_sovereign.host_features import (
    SovereignHostContext,
    start_host_features,
    stop_host_features,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack


@pytest.mark.asyncio
async def test_public_host_stop_preserves_original_native_unknown_and_closes_peers(
    native_pg, monkeypatch
):
    backend, observer, schema = native_pg
    ctx = SovereignHostContext(db=backend)
    backend._execution_custody = ExecutionCustody(GenerationFence())
    entered = []

    class NativeHost(HostFeature):
        def __init__(self, name, key):
            self.name, self.key = name, key
            self.client = httpx.AsyncClient(trust_env=False)

        async def on_host_start(self, ctx):
            await backend.execute(
                "INSERT INTO effects VALUES (?, ?)", (self.key, self.name)
            )

        async def on_host_stop(self, ctx):
            entered.append(self.name)
            try:
                await backend.execute(
                    "INSERT INTO effects VALUES (?, ?)", (self.key + 10, self.name)
                )
            finally:
                await self.client.aclose()

    class IndependentNativeHost(NativeHost):
        """A distinct contribution owner, not a duplicate instance of one owner."""

    first, second = (
        NativeHost("first-native", 1),
        IndependentNativeHost("second-native", 2),
    )
    try:
        assert await start_host_features([first, second], ctx) == [first, second]
        receipts = lose_native_ack(
            monkeypatch,
            boundary="commit",
            original=OSError("original host stop COMMIT ACK lost"),
        )
        with pytest.raises(BaseException) as caught:
            await stop_host_features([first, second], ctx)
        assert receipts == [True]
        assert execution_commit_outcome(caught.value) == "unknown"
        assert entered == ["second-native", "first-native"], (
            "first control skipped an independent original peer"
        )
        assert first.client.is_closed and second.client.is_closed
        assert (
            ctx.started_host_features == ()
            and ctx.feature_contribution_runtime.active_owners() == ()
        )
        assert (
            await observer.fetchval(
                f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=12'
            )
            == 1
        )
        assert (
            await observer.fetchval(
                f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=11'
            )
            == 0
        ), "UNKNOWN quarantine must not admit another ordinary SQL write"
    finally:
        backend._execution_custody = None
        await first.client.aclose()
        await second.client.aclose()
        await stop_host_features([first, second], ctx)
