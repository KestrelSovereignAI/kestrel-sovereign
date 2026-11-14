"""Original local ACT caller survives revocation of its egress borrower."""
import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from kestrel_sdk.computer_use import ActionRequest
from kestrel_sdk.resources import ResourceOutcomeUnknown

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.visual_computer.local_worker import DRIVER, LocalTwoDomainDriver
from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.shared.installed_worker_requirement import worker_module


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["borrower", "caller"])
async def test_local_act_revocation_is_unknown_for_uncancelled_original_caller(
        harness, tmp_path, monkeypatch, interruption):
    worker_module("kestrel_cloud_runpod.remote_worker_tls",
                  required_callables=("PinnedWorkerTarget", "PinnedWorkerTLS"))
    h = harness
    context = h.context()
    entered, departed = asyncio.Event(), asyncio.Event()
    observed = {}
    with h.bind(context):
        session = (await h.runtime.admit(context, h.request(context))).model_copy(
            update={"driver_id": DRIVER})
        driver = LocalTwoDomainDriver(docker_socket=str(tmp_path / "never-opened.sock"),
                                       egress_guard=h.runtime.egress.operation)
        request = ActionRequest(action_id="original-local-revoked-action",
            session_id=session.session_id, observation_id="bounded-no-pixels",
            control_epoch=session.control_epoch,
            not_after=datetime.now(UTC)+timedelta(seconds=10),
            action={"kind": "click", "x": 1, "y": 1})

        async def bounded_rpc(*args, current, **kwargs):
            observed["borrower"] = asyncio.current_task()
            current()
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                departed.set()

        monkeypatch.setattr(driver, "_rpc", bounded_rpc)

        async def original_call():
            async with (
                h.runtime.operation(context, VisualAction.ACT) as check,
                h.runtime.egress.activate(session, request.action_id, check,
                                         deadline=request.not_after),
            ):
                try:
                    return await driver.act(session, request)
                except BaseException as error:
                    observed["original_error"] = error
                    observed["caller_cancelling"] = asyncio.current_task().cancelling()
                    raise

        caller = asyncio.create_task(original_call())
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if interruption == "borrower":
                await h.runtime.egress.revoke_session(session.scope, session.session_id)
            else:
                caller.cancel("original-local-caller-withdrawal")
            outcome = await await_owned_task(caller)
            if interruption == "borrower":
                assert observed["caller_cancelling"] == 0
                assert type(outcome.error) is ResourceOutcomeUnknown, (
                    "revoked local borrower must not cancel its original ACT caller")
                assert not caller.cancelled()
            else:
                assert observed["caller_cancelling"] > 0
                assert type(outcome.error) is asyncio.CancelledError
                assert outcome.error.args == ("original-local-caller-withdrawal",)
                assert caller.cancelled()
            assert outcome.error is observed["original_error"]
            assert departed.is_set() and observed["borrower"].done()
            assert not h.runtime.egress._calls
        finally:
            if not caller.done():
                caller.cancel()
            await await_owned_task(caller)
