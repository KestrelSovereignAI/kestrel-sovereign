"""Public host teardown retains native control while retiring all peers."""

import httpx
import pytest
from fastapi import FastAPI
from kestrel_sdk.features.host_base import HostFeature

from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.host_features import (
    SovereignHostContext,
    close_host_context,
    start_host_features,
    stop_host_features,
)
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack


@pytest.mark.asyncio
@pytest.mark.parametrize("through_server", [False, True])
@pytest.mark.parametrize(
    "followup",
    [
        "repeat",
        "restart",
        "context-close",
        "independent-peer",
        "server-phases",
        "same-resource",
    ],
)
async def test_repeated_public_host_stop_keeps_original_native_receipt(
    native_pg, monkeypatch, through_server, followup
):
    backend, observer, schema = native_pg
    ctx = SovereignHostContext(db=backend)
    backend._execution_custody = ExecutionCustody(GenerationFence())
    entered, starts, closes = [], [], []

    class NativeHost(HostFeature):
        def __init__(self, name, key):
            self.name, self.key = name, key
            self.client = httpx.AsyncClient(trust_env=False)

        async def on_host_start(self, ctx):
            starts.append(self.name)
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

        async def close(self):
            closes.append(self)
            await backend.close()

    class LaterIndependentNativeHost(NativeHost):
        """A peer left active until a followup Stop, never a replacement."""

    third = LaterIndependentNativeHost("later-native", 3)
    first, second = (
        NativeHost("first-native", 1),
        IndependentNativeHost("second-native", 2),
    )
    try:
        initial = (
            [first, second, third]
            if followup == "independent-peer"
            else [first, second]
        )
        assert await start_host_features(initial, ctx) == initial
        if followup == "same-resource":
            # The protocol permits the feature itself to own the context's
            # backend lifetime. Its stop hook is not its independent close.
            ctx._db = second
        from tests.utils.original_failure import OriginalFailure
        from kestrel_sovereign.execution_custody import _explicit_control_chain
        native_failure = OSError("original host stop COMMIT ACK lost")
        native_snapshot = OriginalFailure.capture(native_failure)
        receipts = lose_native_ack(
            monkeypatch,
            boundary="commit",
            original=native_failure,
        )
        with pytest.raises(BaseException) as caught:
            if through_server:
                from kestrel_sovereign.server import _shutdown_host_features

                app = FastAPI()
                app.state.host_features = [first, second]
                app.state.host_context = ctx
                await _shutdown_host_features(app)
            else:
                await stop_host_features([first, second], ctx)
        assert receipts == [True]
        assert execution_commit_outcome(caught.value) == "unknown"
        sources = OriginalFailure.capture_sources(caught.value)
        native_snapshot.assert_retained(caught.value)
        assert entered == ["second-native", "first-native"], (
            "first control skipped an independent original peer"
        )
        assert first.client.is_closed and second.client.is_closed
        if followup == "independent-peer":
            assert ctx.started_host_features == (third,)
            assert ctx.feature_contribution_runtime.is_active(third)
        else:
            assert ctx.started_host_features == ()
            assert ctx.feature_contribution_runtime.active_owners() == ()
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
        if followup == "server-phases":
            from kestrel_sovereign.server import _shutdown_server_resources

            if not through_server:
                app = FastAPI()
                app.state.host_features = [first, second]
                app.state.host_context = ctx
            for _ in range(2):
                cancelled, failure = await _shutdown_server_resources(app)
                assert not cancelled
                for source in sources:
                    source.assert_retained(failure)
                    source.assert_retained(app.state._lifespan_teardown_error)
                assert execution_commit_outcome(failure) == "unknown"
            assert app.state.host_context is ctx
            assert app.state.host_features == [first, second]
        else:
            with pytest.raises(BaseException) as repeated:
                if followup == "independent-peer":
                    await stop_host_features([third], ctx)
                elif followup == "restart":
                    await start_host_features([first, second], ctx)
                elif followup in ("context-close", "same-resource"):
                    await close_host_context(ctx)
                elif through_server:
                    await _shutdown_host_features(app)
                else:
                    await stop_host_features([first, second], ctx)
            for source in sources:
                source.assert_retained(repeated.value)
            native_snapshot.assert_retained(repeated.value)
            assert execution_commit_outcome(repeated.value) == "unknown"
        expected_starts = ["first-native", "second-native"]
        expected_stops = ["second-native", "first-native"]
        if followup == "independent-peer":
            expected_starts.append("later-native")
            expected_stops.append("later-native")
            assert third.client.is_closed
            assert ctx.started_host_features == ()
        assert starts == expected_starts, (
            "failed retirement admitted a new host-start hook"
        )
        assert entered == expected_stops, (
            "followup replayed an original native stop hook or skipped independent cleanup"
        )
        assert ctx.feature_contribution_runtime.active_owners() == ()
        if followup == "same-resource":
            assert closes == [second] and backend._pool is None, (
                "failed feature hook suppressed the SAME object's independent backend close"
            )
        retained = [
            record
            for record in ctx._native_close_owners.values()
            if record.resource is second
        ]
        assert retained and any(
            native_failure in tuple(_explicit_control_chain(record.error)) and not record.closed
            for record in retained
        ), "deactivation discarded the original failed feature/resource owner"
        if through_server:
            assert app.state.host_context is ctx
            assert app.state.host_features == [first, second]
    finally:
        backend._execution_custody = None
        await first.client.aclose()
        await second.client.aclose()
        await third.client.aclose()
        try:
            await stop_host_features([first, second, third], ctx)
        except BaseException as error:
            if not is_execution_control_error(error):
                raise
