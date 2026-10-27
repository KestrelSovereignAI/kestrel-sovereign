"""Original lost control replies reconcile through native encrypted custody."""
import json

import pytest
from kestrel_sdk.computer_use import ComputerSession, ControlRequest
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from kestrel_sovereign.visual_computer.repository import VisualCustodyRepository
from kestrel_sovereign.visual_computer.runtime import VisualAction, VisualComputerRuntime
from tests.integration import test_visual_custody_backends as native_backends
from tests.unit.test_visual_browser_controller import BrowserController, Page, Refused
from tests.unit.visual_computer import conftest as visual_fixtures
from tests.unit.visual_computer.test_runtime import opened

custody = native_backends.custody
harness = visual_fixtures.harness


@pytest.mark.parametrize("operation,action", [
    ("pause", VisualAction.STOP), ("takeover", VisualAction.TAKEOVER),
    ("resume", VisualAction.HANDBACK),
])
async def test_original_lost_worker_reply_survives_host_restart_without_second_control(
    harness, custody, operation, action,
):
    # Establish the harness before the native backend creates its encrypted
    # key row; both must use that SAME final test-owned encryption key.
    store, reopen = custody
    h, context = harness, harness.context()
    h.runtime.repository = h.repository = store
    with h.bind(context):
        session, old_input = await opened(h, context)
        request = {"schema_version": 1, "session_id": session.session_id,
                   "lease_id": session.lease_id, "scope": json.loads(session.scope.model_dump_json()),
                   "surface": "browser", "required_capabilities": ["screenshot"],
                   "expires_at": session.expires_at.isoformat()}
        worker = BrowserController(Page(), request, driver_id=session.driver_id)
        worker._session["capabilities"] = sorted(session.capabilities)
        calls, lose_reply = [], False

        async def control(original_session, original_request):
            calls.append(original_request)
            try:
                receipt = await worker.control(json.loads(original_session.model_dump_json()),
                                               json.loads(original_request.model_dump_json()))
            except Refused as refusal:
                raise ResourceConflict(refusal.code) from None
            if lose_reply and original_request.request_id == "original-control":
                raise ResourceOutcomeUnknown("original worker reply was lost")
            return ComputerSession.model_validate_json(json.dumps(receipt))

        h.adapter.control = control
        if operation == "resume":
            takeover = ControlRequest(request_id="first-takeover", session_id=session.session_id,
                expected_control_epoch=session.control_epoch, operation="takeover")
            session = await h.runtime.control(context, session.session_id, takeover, action=VisualAction.TAKEOVER)
            calls.clear()
        original = ControlRequest(request_id="original-control", session_id=session.session_id,
            expected_control_epoch=session.control_epoch, operation=operation)
        lose_reply = True
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.control(context, session.session_id, original, action=action)
        pending = await store.get(h.runtime.scope(context), session_id=session.session_id)
        assert pending.pending_control == original and pending.session == session
        assert worker._session["control_epoch"] == session.control_epoch + 1
        with pytest.raises(ResourceConflict):
            await h.runtime.act(context, old_input)

        # A fresh native connection and runtime read the SAME original pending
        # control; no reconstruction from a successful caller-side DTO.
        database = await reopen()
        try:
            repository = VisualCustodyRepository(database)
            restarted = VisualComputerRuntime(repository, h.runtime.authority,
                providers=h.runtime.providers, drivers=h.runtime.drivers)
            lose_reply = False
            receipt = await restarted.control(context, session.session_id, original, action=action)
            current = await repository.get(restarted.scope(context), session_id=session.session_id)
            assert current.pending_control is None and current.last_control == original
            assert current.session == current.control_receipt == receipt
            assert receipt.control_epoch == session.control_epoch + 1
            assert await restarted.control(context, session.session_id, original, action=action) == receipt
            assert calls == [original, original]
            assert worker._session["control_epoch"] == receipt.control_epoch
            assert worker.page.effects == []
            with pytest.raises(ResourceConflict):
                await restarted.control(context, session.session_id,
                    original.model_copy(update={"request_id": "different-stale"}), action=action)
            h.revoked.add(id(context))
            with pytest.raises(OperatorAuthorizationError):
                await restarted.control(context, session.session_id, original, action=action)
            assert calls == [original, original]
        finally:
            await database.close()
