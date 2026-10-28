"""Composition of original host custody with the actual shared worker engine."""

import asyncio
import json

import pytest
from kestrel_sdk.computer_use import ComputerSession, ControlRequest
from kestrel_sdk.computer_use.human_control import (
    HumanActionReceipt,
    HumanControlLease,
    HumanScreenObservation,
)
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound
from kestrel_sovereign.visual_computer.human import HumanComputerRuntime
from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.unit.visual_computer.test_human import setup, input_request


@pytest.mark.asyncio
@pytest.mark.parametrize("at", ["accept", "dispatch"])
async def test_manual_preparation_stop_retires_without_an_effect_or_agent_receipt(
    harness, monkeypatch, at
):
    h = harness
    ctx = h.context()
    operator = h.context()
    entered, resume = asyncio.Event(), asyncio.Event()
    with h.bind(ctx), h.bind(operator):
        session, lease, extension = await setup(h, ctx)
        request = await input_request(h, ctx, session, lease)
        original = h.repository.action

        async def interleave(*args, **kw):
            value = await original(*args, **kw)
            if (at == "accept" and kw.get("accept") is not None) or (
                at == "dispatch" and kw.get("dispatch_revision") is not None
            ):
                entered.set()
                await resume.wait()
            return value

        monkeypatch.setattr(h.repository, "action", interleave)
        active = asyncio.create_task(h.runtime.human.act(ctx, session.session_id, "human", request))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await h.runtime.control(
                operator,
                session.session_id,
                ControlRequest(
                    request_id="manual-stop",
                    session_id=session.session_id,
                    expected_control_epoch=session.control_epoch,
                    operation="pause",
                ),
                action=VisualAction.STOP,
            )
            resume.set()
            with pytest.raises(ResourceConflict):
                await active
        finally:
            resume.set()
            await asyncio.gather(active, return_exceptions=True)
        monkeypatch.setattr(h.repository, "action", original)
        assert extension.effects == 0 and h.runtime._inflight == 0
        assert not await h.repository.has_pending_actions(h.runtime.scope(ctx), session.session_id)
        receipt = await h.runtime.human.action_status(
            ctx, session.session_id, "human", request.action.action_id
        )
        assert (
            receipt.receipt.outcome == "denied"
            and receipt.receipt.reason_code == "host_not_dispatched"
        )
        assert not await h.repository.has_pending_actions(h.runtime.scope(ctx), session.session_id)
        with pytest.raises(ResourceNotFound):
            await h.runtime.action_status(ctx, session.session_id, request.action.action_id)


@pytest.mark.asyncio
async def test_actual_worker_manual_focus_denial_reconciles_then_fresh_input_completes(
    harness, tmp_path
):
    from tests.unit.test_visual_browser_controller import BrowserController, Page, Refused
    from tests.unit.test_visual_browser_human import module
    from tests.unit.visual_computer.test_runtime import opened

    h = harness
    ctx = h.context()
    page = Page()
    focus = ["BODY"]

    async def evaluate(_):
        return ["about:blank", True, focus[0], "", None]

    page.evaluate = evaluate

    def parse(model, value):
        return model.model_validate_json(json.dumps(value))

    with h.bind(ctx):
        session, _ = await opened(h, ctx)
        controller = BrowserController(
            page,
            {
                "schema_version": 1,
                "session_id": session.session_id,
                "scope": session.scope.model_dump(mode="json"),
                "lease_id": session.lease_id,
                "surface": "browser",
                "required_capabilities": ["screenshot", "click", "type"],
                "expires_at": session.expires_at.isoformat(),
            },
            driver_id=h.adapter.driver_id,
        )
        (tmp_path / "native").mkdir()
        native = module.HumanTransport(controller, tmp_path / "native", "b" * 64)
        controller.human = native

        class Extension:
            async def open(self, s, request):
                return parse(
                    HumanControlLease,
                    await native.open(s.model_dump(mode="json"), request.model_dump(mode="json")),
                )

            async def status(self, s, request):
                return parse(
                    HumanControlLease,
                    native.status(s.model_dump(mode="json"), request.model_dump(mode="json")),
                )

            async def observe(self, s, lease):
                value = await native.observe(
                    s.model_dump(mode="json"), lease.model_dump(mode="json")
                )
                from kestrel_sdk.computer_use import ScreenFrame, ScreenObservation
                import base64

                return HumanScreenObservation(
                    control_id=value["control_id"],
                    consumer_id=value["consumer_id"],
                    observation=ScreenObservation(
                        frame=parse(ScreenFrame, value["frame"]),
                        image=base64.b64decode(value["image_base64"]),
                    ),
                )

            async def act(self, s, lease, request):
                try:
                    return parse(
                        HumanActionReceipt,
                        await native.act(
                            s.model_dump(mode="json"),
                            lease.model_dump(mode="json"),
                            request.model_dump(mode="json"),
                        ),
                    )
                except Refused as error:
                    raise ResourceConflict(error.code) from None

            async def action_status(self, s, lease, aid):
                return parse(
                    HumanActionReceipt,
                    native.action_status(
                        s.model_dump(mode="json"), lease.model_dump(mode="json"), aid
                    ),
                )

            async def revoke(self, s, request):
                return parse(
                    HumanControlLease,
                    await native.revoke(s.model_dump(mode="json"), request.model_dump(mode="json")),
                )

        async def control(s, request):
            return parse(
                ComputerSession,
                await controller.control(
                    s.model_dump(mode="json"), request.model_dump(mode="json")
                ),
            )

        h.adapter.control = control
        h.adapter.human_control = lambda: Extension()
        h.runtime.human = HumanComputerRuntime(h.runtime)
        session = await h.runtime.control(
            ctx,
            session.session_id,
            ControlRequest(
                request_id="take",
                session_id=session.session_id,
                expected_control_epoch=0,
                operation="takeover",
            ),
            action=VisualAction.TAKEOVER,
        )
        lease = await h.runtime.human.open(ctx, session.session_id, "human")
        request = await input_request(h, ctx, session, lease)
        focus[0] = "INPUT"
        with pytest.raises(ResourceConflict, match="stale_observation"):
            await h.runtime.human.act(ctx, session.session_id, "human", request)
        denied = await h.runtime.human.action_status(
            ctx, session.session_id, "human", request.action.action_id
        )
        assert denied.receipt.outcome == "denied" and page.effects == []
        assert await h.runtime.human.act(ctx, session.session_id, "human", request) == denied
        with pytest.raises(ResourceNotFound):
            await h.runtime.action_status(ctx, session.session_id, request.action.action_id)
        fresh = await input_request(h, ctx, session, lease)
        fresh = fresh.model_copy(update={"action":fresh.action.model_copy(update={"action_id":"fresh-after-denial"})})
        result = await h.runtime.human.act(ctx, session.session_id, "human", fresh)
        assert result.receipt.outcome == "completed" and len(page.effects) == 1
        assert await h.runtime.human.act(ctx, session.session_id, "human", fresh) == result
        journal = native.journal.read_text()
        assert fresh.action.action.text == "PRIVATE-HUMAN-TEXT"
        assert fresh.action.action.text not in journal and "image_base64" not in journal


@pytest.mark.asyncio
async def test_released_original_human_channel_cannot_reopen(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, lease, extension = await setup(h, ctx)
        before = await h.runtime.human._record(h.runtime.scope(ctx), session.session_id, "human", check=lambda: None)
        assert (await h.runtime.release(ctx, session.session_id))["released"]
        with pytest.raises(ResourceConflict):
            await h.runtime.human.open(ctx, session.session_id, "human")
        after = await h.runtime.human._record(h.runtime.scope(ctx), session.session_id, "human", check=lambda: None)
        assert after == before and after.lease == lease
