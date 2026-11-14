"""Real acquiring ACK before first possible create, after read-only checks.

Daemon metadata and retirement I/O are SCRIPTED; no resource/native ACK proof.
The genuine Host checker, runtime and encrypted SQLite publication execute.
"""
import json
import pytest
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown
from kestrel_sovereign.visual_computer.local_worker import _LocalWorkerResources
from kestrel_sovereign.visual_computer.runtime import VisualAction
from kestrel_sovereign.visual_computer.worker_profile import CONTRACT, LABEL
from tests.unit.visual_computer.test_local_two_domain_admit import local_setup as local_setup


@pytest.mark.asyncio
async def test_original_acquiring_ack_precedes_create_and_requires_verified_cleanup(local_setup, monkeypatch):
    setup = local_setup
    observed, commands = [], []
    failure = ResourceOutcomeUnknown("scripted_possible_create_outcome_unknown")

    async def invoke(args, *, data=None, current=None):
        current()
        commands.append(list(args))
        if args == ["info", "--format", "{{json .}}"]:
            value = {"ServerVersion": "29.8.2", "OSType": "linux", "Architecture": "aarch64", "DefaultRuntime": "runc"}
        elif args[:2] == ["image", "inspect"]:
            value = [{"Id": args[2], "Os": "linux", "Architecture": "arm64",
                      "Config": {"Labels": {LABEL + ".contract": CONTRACT}}}]
        elif args[:2] == ["volume", "inspect"]:
            raise ResourceNotFound("scripted_readonly_exact_volume_absent")
        elif args[:2] == ["volume", "create"]:
            original = setup.host._current_setup()
            record = await setup.runtime.repository.get(original._record.request.scope,
                request_id=original._record.request.request_id, guard=original._check_original)
            observed.append(record.phase)
            # A durable acquiring row alone never proves non-dispatch or
            # retirement. Only the original physical cleanup can settle it.
            with pytest.raises(ResourceConflict):
                await setup.runtime.repository.retire_unacquired(record.request.scope, record,
                    at=setup.runtime._now(), guard=original._check_original)
            raise failure
        else:
            raise AssertionError("unexpected native effect: " + repr(args[:2]))
        current()
        return json.dumps(value).encode()

    async def retire(resource):
        resource._closed = True  # Controlled no-resource boundary, not physical ACK.

    monkeypatch.setattr(setup.provider.boundary, "invoke", invoke)
    monkeypatch.setattr(_LocalWorkerResources, "_retire", retire)
    async with setup.bound() as context:
        request = setup.request(context)
        with pytest.raises(ResourceOutcomeUnknown) as caught:
            await setup.runtime.admit(context, request)
        assert caught.value is failure
        assert setup.runtime.egress._calls == {}
        async with setup.runtime.operation(context, VisualAction.STATUS) as check:
            saved = await setup.runtime.repository.get(request.scope, request_id=request.request_id, guard=check)
            assert observed == ["acquiring"]
            assert sum(command[:2] == ["volume", "create"] for command in commands) == 1
            assert saved.phase == "retired" and saved.retired_from == "acquiring"
            assert saved.lease is None and saved.quote is not None
            with pytest.raises(ResourceConflict):
                await setup.runtime.repository.retire_unacquired(request.scope, saved,
                    at=setup.runtime._now(), guard=check)
