"""Returned SDK bindings are untrusted facts; original host checks remain errors."""
import json

import pytest
from kestrel_sdk.computer_use import ControlRequest
from kestrel_sdk.computer_use.human_control import (
    HumanControlLease,
    HumanControlRequest,
)
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sdk.resources import ResourceOutcomeUnknown

from kestrel_sovereign.visual_computer.human import HumanComputerRuntime
from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.unit.visual_computer.test_human import Extension, input_request
from tests.unit.visual_computer.test_runtime import opened


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["open", "status", "observe", "act", "action_status", "revoke", "tombstone"])
async def test_foreign_sdk_result_is_unknown_for_non_docker_provider(harness, operation):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, _ = await opened(h, ctx)
        extension = Extension(h.runtime)
        h.adapter.human_control = lambda: extension
        h.runtime.human = HumanComputerRuntime(h.runtime)
        session = await h.runtime.control(ctx, session.session_id, ControlRequest(request_id="takeover",
            session_id=session.session_id, expected_control_epoch=0, operation="takeover"), action=VisualAction.TAKEOVER)
        lease = request = None
        name = "revoke" if operation == "tombstone" else operation
        original = getattr(extension, name)
        observed = []
        failure = release = None
        try:
            if operation != "open":
                lease = await h.runtime.human.open(ctx, session.session_id, "human")
            if operation in {"act", "action_status"}:
                request = await input_request(h, ctx, session, lease)
            if operation == "action_status":
                original_act = extension.act
                async def lost(*args):
                    await original_act(*args)
                    raise ResourceOutcomeUnknown("original provider reply lost")
                extension.act = lost
                with pytest.raises(ResourceOutcomeUnknown):
                    await h.runtime.human.act(ctx, session.session_id, "human", request)
                extension.act = original_act

            async def foreign(*args):
                value = await original(*args)
                observed.append(value)
                if operation in {"open", "status", "revoke"}:
                    old = value.request
                    changed = HumanControlRequest.model_validate_json(json.dumps({
                        **old.model_dump(mode="json"), "consumer_id": "foreign-consumer"}))
                    return HumanControlLease(request=changed, request_sha256=changed.digest(), state=value.state,
                        accepted_at=value.accepted_at, revoked_at=value.revoked_at)
                if operation == "tombstone":
                    return value.model_copy(update={"request": value.request.model_copy(update={"consumer_id": "foreign-consumer"})})
                if operation == "observe":
                    return value.model_copy(update={"consumer_id": "foreign-consumer"})
                return value.model_copy(update={"receipt": value.receipt.model_copy(update={"action_id": "foreign-action"})})

            setattr(extension, name, foreign)
            try:
                if operation in {"open", "status", "observe"}:
                    await getattr(h.runtime.human, operation)(ctx, session.session_id, "human")
                elif operation == "act":
                    await h.runtime.human.act(ctx, session.session_id, "human", request)
                elif operation == "action_status":
                    await h.runtime.human.action_status(ctx, session.session_id, "human", request.action.action_id)
                else:
                    await h.runtime.human.revoke(ctx, session.session_id,
                        "late" if operation == "tombstone" else "human", "original-revoke", session.control_epoch)
            except (ValueError, ResourceOutcomeUnknown) as error:
                failure = error
        finally:
            setattr(extension, name, original)
            release = await h.runtime.release(ctx, session.session_id)
        assert release["released"] and not h.runtime.egress._calls and h.runtime._inflight == 0
        assert not h.repository._original_results and not h.db.backend._original_result_holds
        assert len(observed) == 1
        assert type(failure) is ResourceOutcomeUnknown
        assert str(failure) == "invalid_private_worker_response"
        if operation in {"act", "action_status"}:
            row = await h.repository.action(h.runtime.scope(ctx), session.session_id, request.action.action_id)
            assert row.dispatch_started and row.receipt is None
            assert extension.effects == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["provider-value-error", "withdrawn-authority"])
async def test_original_provider_error_and_current_authority_are_not_peer_decode_errors(harness, failure_kind):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        from tests.unit.visual_computer.test_human import setup
        session, _, extension = await setup(h, ctx)
        original = extension.observe
        error = ValueError("original provider error")
        async def fail(*args):
            if failure_kind == "provider-value-error":
                raise error
            value = await original(*args)
            h.revoked.add(id(ctx))
            return value
        extension.observe = fail
        failure = None
        try:
            await h.runtime.human.observe(ctx, session.session_id, "human")
        except (ValueError, OperatorAuthorizationError) as caught:
            failure = caught
        finally:
            extension.observe = original
            h.revoked.discard(id(ctx))
            release = await h.runtime.release(ctx, session.session_id)
        assert release["released"] and not h.runtime.egress._calls
        if failure_kind == "provider-value-error":
            assert failure is error
        else:
            assert isinstance(failure, OperatorAuthorizationError)
        assert not isinstance(failure, ResourceOutcomeUnknown)
