"""Setup cleanup and bounded request capacity: genuine Host/Runtime/SQLite.

The canonical controller constructs the healthy session. Controlled retirement
errors are physical UNKNOWN test inputs, never Docker/native ACK evidence.
"""
import asyncio
import json

import pytest
import pytest_asyncio
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown, ResourceUnsupported
from kestrel_sovereign.execution_custody import execution_terminal_error
from kestrel_sovereign.visual_computer.local_worker import _LocalWorkerResources
from tests.unit.visual_computer.test_local_two_domain_admit import local_setup as original_setup
from tests.unit.visual_computer.test_two_domain_lifecycle import lifecycle_transport as original_transport


def leaves(error):
    # Canonical execution_terminal_error retains ordinary originals through
    # explicit causes as well as groups. Implicit __context__ is not evidence.
    result, pending, seen = [], [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        else:
            result.append(current)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
    return result


@pytest_asyncio.fixture
async def controlled_setup(harness, monkeypatch, tmp_path):
    # Invoke the original fixture generators, retaining their actual shutdown,
    # release, close and database finalizers. Only a specifically retained
    # scripted UNKNOWN may be recorded as debt; every other teardown error fails.
    local = original_setup.__wrapped__(harness, monkeypatch, tmp_path)
    setup = await anext(local)
    transport = original_transport.__wrapped__(setup, monkeypatch)
    setup, state = await anext(transport)
    state["known_debt"] = []
    state["teardown_originals"] = []
    try:
        yield setup, state
    finally:
        errors = []
        for original in (transport, local):
            try:
                await anext(original)
            except StopAsyncIteration:
                pass
            except BaseException as error:
                state["teardown_originals"].append(error)
                if not all(any(leaf is debt for debt in state["known_debt"])
                           for leaf in leaves(error)):
                    errors.append(error)
        if errors:
            raise execution_terminal_error(None, *errors, retain_originals=True)


async def admit(setup, request_id):
    async with setup.bound() as context:
        return await setup.runtime.admit(context, setup.request(
            context, request_id=request_id,
            required_capabilities=frozenset({"screenshot"}),
        ))


@pytest.mark.asyncio
@pytest.mark.parametrize("sweep", ["fence", "maintenance"])
@pytest.mark.parametrize("admission_task", ["independent", "same-task"])
async def test_failed_setup_debt_does_not_abort_later_original_session(
    controlled_setup, monkeypatch, sweep, admission_task,
):
    setup, state = controlled_setup
    bootstrap = _LocalWorkerResources.bootstrap
    retire = _LocalWorkerResources._retire
    setup_error = ResourceOutcomeUnknown("scripted_original_setup_failure")
    cleanup_error = ResourceOutcomeUnknown("scripted_original_retirement_unknown")
    state["known_debt"].extend((setup_error, cleanup_error))

    async def fail_first_bootstrap(resource, request, credential):
        if resource.original._record.request.request_id == "0-original-failed-setup":
            raise setup_error
        return await bootstrap(resource, request, credential)

    async def fail_first_retirement(resource):
        if resource.original._record.request.request_id == "0-original-failed-setup":
            raise cleanup_error
        return await retire(resource)

    monkeypatch.setattr(_LocalWorkerResources, "bootstrap", fail_first_bootstrap)
    monkeypatch.setattr(_LocalWorkerResources, "_retire", fail_first_retirement)
    first = None
    try:
        await admit(setup, "0-original-failed-setup")
    except BaseException as error:
        first = error
    assert first is not None
    assert any(error is setup_error for error in leaves(first))
    assert any(error is cleanup_error for error in leaves(first))
    bad = state["resources"][0]
    assert bad._close_admission.task is not None
    try:
        await bad._close_admission.task  # SAME original Task, actual join.
    except ResourceOutcomeUnknown as error:
        assert error is cleanup_error
    assert bad.original._finished.exception() is cleanup_error
    assert setup.runtime.egress._calls.get(bad.original._key) is bad.original
    if admission_task == "independent":
        healthy_caller = asyncio.create_task(admit(setup, "1-original-healthy-session"))
        healthy = await healthy_caller  # Retain and actually join the original.
    else:
        # Same-Task fresh issuance must ignore retained inactive cleanup debt,
        # without removing that original from the existing egress map.
        healthy = await admit(setup, "1-original-healthy-session")
    scope = healthy.scope
    rows = await setup.runtime.repository.records(scope)
    assert [row.session_id for row in rows] == [bad.original._record.session_id, healthy.session_id]
    if sweep == "maintenance":
        # Genuine durable release intent, not a fake lease/ready result or clock.
        for row in rows:
            row.phase = "releasing"
            await setup.runtime.repository.save(scope, row)
    original_error, uncertain = None, None
    try:
        uncertain = await (setup.runtime.fence_owned_sessions(scope) if sweep == "fence"
                           else setup.runtime.maintain_owned_sessions(scope))
    except BaseException as error:
        original_error = error
    saved = await setup.runtime.repository.get(scope, session_id=healthy.session_id)
    expected = "fenced" if sweep == "fence" else "released"
    assert saved.phase == expected, (
        f"later original session was not processed: sweep={sweep}, phase={saved.phase}, "
        f"original_error={type(original_error).__name__}:{original_error}, uncertain={uncertain}"
    )
    assert original_error is None and uncertain == [bad.original._record.session_id]
    assert setup.runtime.egress._calls.get(bad.original._key) is bad.original
    assert bad._closed is False  # Unknown physical cleanup stays charged.


@pytest.mark.asyncio
@pytest.mark.parametrize("physical", ["success", "unknown"])
async def test_setup_completion_reports_retirement_not_closer_caller_cancellation(
    controlled_setup, monkeypatch, physical,
):
    setup, state = controlled_setup
    entered, finish = asyncio.Event(), asyncio.Event()
    setup_error = ResourceOutcomeUnknown("scripted_setup_before_controller")
    physical_error = ResourceOutcomeUnknown("scripted_retirement_still_unknown")
    state["known_debt"].append(setup_error)

    async def fail_bootstrap(resource, request, credential):
        raise setup_error

    async def gated_retirement(resource):
        entered.set()
        await finish.wait()
        if physical == "unknown":
            raise physical_error
        resource._closed = True  # Scripted successful physical boundary only.

    monkeypatch.setattr(_LocalWorkerResources, "bootstrap", fail_bootstrap)
    monkeypatch.setattr(_LocalWorkerResources, "_retire", gated_retirement)
    caller = asyncio.create_task(admit(setup, "cancelled-original-setup"))
    original = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        caller.cancel()
    finally:
        finish.set()
        try:
            await caller
        except BaseException as error:
            original = error
        # The caller's cancellation is not a substitute for native retirement.
        if state["resources"]:
            resource = state["resources"][0]
            try:
                await resource._close_admission.task
            except ResourceOutcomeUnknown as error:
                assert error is physical_error
                state["known_debt"].append(error)
            if resource.original._finished.done():
                error = resource.original._finished.exception()
                if error is not None:
                    state["known_debt"].extend(leaves(error))
    resource, = state["resources"]
    assert original is not None and any(error is setup_error for error in leaves(original))
    assert any(isinstance(error, asyncio.CancelledError) for error in leaves(original))
    assert resource.original._finished.done()
    expected = None if physical == "success" else physical_error
    assert resource.original._finished.exception() is expected, (
        "setup completion must report SAME physical retirement, not its caller: "
        f"{resource.original._finished.exception()!r}"
    )
    assert resource._closed is (physical == "success")
    if physical == "success":
        assert setup.runtime.egress._calls == {}
    else:
        assert setup.runtime.egress._calls.get(resource.original._key) is resource.original


@pytest.mark.asyncio
async def test_readonly_precreate_refusal_does_not_exhaust_original_request_capacity(
    harness, monkeypatch, tmp_path,
):
    local = original_setup.__wrapped__(harness, monkeypatch, tmp_path)
    setup = await anext(local)
    commands, errors, rows = [], [], []
    # Bound the actual repository's existing configurable request capacity.
    setup.runtime.repository.max_requests = 2

    async def wrong_platform(arguments, *, data=None, current=None):
        assert current is not None
        current()
        commands.append(list(arguments))
        assert arguments == ["info", "--format", "{{json .}}"]
        return json.dumps({"ServerVersion": "unmeasured-platform"}).encode()

    monkeypatch.setattr(setup.provider.boundary, "invoke", wrong_platform)
    try:
        for index in range(3):
            request_id = f"precreate-refusal-{index}"
            async with setup.bound() as context:
                scope = setup.runtime.scope(context)
                try:
                    await setup.runtime.admit(context, setup.request(context, request_id=request_id))
                except (ResourceUnsupported, ResourceConflict) as error:
                    errors.append(error)
                try:
                    row = await setup.runtime.repository.get(scope, request_id=request_id)
                except ResourceNotFound:
                    pass
                else:
                    rows.append(row)
        assert commands and all(command == ["info", "--format", "{{json .}}"] for command in commands)
        assert all(row.lease is row.connection is row.session is None for row in rows)
        assert setup.runtime.egress._calls == {}
        assert len(rows) == 3 and all(row.phase == "retired" for row in rows), (
            f"read-only refusal charged capacity: phases={[row.phase for row in rows]}, "
            f"errors={[(type(error).__name__, str(error)) for error in errors]}"
        )
        assert len(errors) == 3 and all(isinstance(error, ResourceUnsupported) for error in errors)
    finally:
        try:
            await anext(local)
        except StopAsyncIteration:
            pass
