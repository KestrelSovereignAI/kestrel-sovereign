"""Real native control must survive the original approval queue and policy I/O."""
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    is_execution_control_error,
)
from kestrel_sovereign.features.security.approval_queue import (
    ApprovalQueue,
    ApprovalRequest,
    ApprovalStatus,
)
from kestrel_sovereign.features.security.permissions import PermissionStore
from kestrel_sovereign.security.auto_approve import AutoApprovePolicy

pytestmark = pytest.mark.asyncio


def original_control(agent, carrier):
    try:
        agent._get_privacy_transition_lock().require_native_scope(None)
    except ExecutionAuthorityError as authority:
        if carrier == "group":
            error = ExceptionGroup("original native control", [authority])
        elif carrier == "timeout":
            try:
                raise TimeoutError("Original authority timeout carrier") from authority
            except TimeoutError as timeout:
                error = timeout
        else:
            error = authority
        assert is_execution_control_error(error)
        return error
    pytest.fail("Core admitted an absent native authority")


@pytest.mark.parametrize("carrier", ["direct", "group", "timeout"])
@pytest.mark.parametrize("boundary", [
    "permission_read", "dynamic_rules", "request_added", "request_wait",
    "decision_audit", "withdrawn", "sweep",
])
async def test_original_approval_boundaries_preserve_native_control(
    native_tool_effect_agent, tmp_path, monkeypatch, carrier, boundary,
):
    agent = await native_tool_effect_agent(did="did:test:approval-control")
    error = original_control(agent, carrier)
    store = PermissionStore(str(tmp_path / "owned-permissions.db"))
    await store.initialize()
    queue = ApprovalQueue()
    request_args = {"feature_name": "computer_use", "tool_name": "shell",
                    "tool_args": {"argv": ["gh", "issue", "view", "1", "-R", "KestrelSovereignAI/kestrel-sovereign"]}}
    async def reject(*args, **kwargs):
        raise error
    if boundary == "permission_read":
        original = store.get_permission
        async def original_read_then_fence(*args, **kwargs):
            await original(*args, **kwargs)
            raise error
        monkeypatch.setattr(store, "get_permission", original_read_then_fence)
        queue = ApprovalQueue(permission_store=store)
        operation = queue.request_approval(**request_args, allow_blocking=False)
    elif boundary == "dynamic_rules":
        original = store.list_auto_approve_rules
        async def original_rules_then_fence(*args, **kwargs):
            await original(*args, **kwargs)
            raise error
        monkeypatch.setattr(store, "list_auto_approve_rules", original_rules_then_fence)
        queue = ApprovalQueue(permission_store=store, auto_approve_policy=AutoApprovePolicy([], store))
        operation = queue.request_approval(**request_args, allow_blocking=False)
    elif boundary == "request_added":
        queue = ApprovalQueue(on_request_added=reject)
        operation = queue.request_approval(**request_args, timeout=0)
    elif boundary == "request_wait":
        async def arm_original_request(request):
            monkeypatch.setattr(request.resume_event, "wait", reject)
        queue = ApprovalQueue(on_request_added=arm_original_request)
        operation = queue.request_approval(**request_args, timeout=None)
    elif boundary == "decision_audit":
        original = store.log_decision
        async def original_audit_then_fence(*args, **kwargs):
            await original(*args, **kwargs)
            raise error
        monkeypatch.setattr(store, "log_decision", original_audit_then_fence)
        queue = ApprovalQueue(permission_store=store)
        operation = queue._persist_decision(**request_args, approved=False,
                                            scope="no_approver", audit_action="tool_execution")
    elif boundary == "withdrawn":
        queue = ApprovalQueue(on_request_withdrawn=reject)
        operation = queue.request_approval(**request_args, timeout=0)
    else:
        queue = ApprovalQueue(on_request_withdrawn=reject)
        request = ApprovalRequest(id=uuid4().hex, created_at=datetime.now(UTC), **request_args)
        queue._pending[request.id] = request
        operation = queue.sweep_stale(older_than_seconds=0)
    with pytest.raises(type(error)) as captured:
        await operation
    assert captured.value is error, "Approval replaced original native control"
    if boundary in {"request_added", "request_wait"}:
        assert len(queue.pending_requests) == 1
        assert queue.pending_requests[0].status is ApprovalStatus.AWAITER_GONE, (
            "Approval retained a live-looking request after native control ended its awaiter")
    elif boundary != "decision_audit":
        assert not queue.pending_requests


@pytest.mark.parametrize("carrier", ["direct", "group", "timeout"])
@pytest.mark.parametrize("secondary_kind", ["control", "cancellation"])
async def test_withdrawal_control_preserves_an_inflight_native_audit_failure(
    native_tool_effect_agent, tmp_path, monkeypatch, carrier, secondary_kind,
):
    agent = await native_tool_effect_agent(did="did:test:paired-approval-control")
    primary, secondary = original_control(agent, carrier), original_control(agent, carrier)
    if secondary_kind == "cancellation":
        import asyncio
        child = asyncio.create_task(asyncio.Event().wait())
        await asyncio.sleep(0)
        child.cancel()
        try:
            await child
        except asyncio.CancelledError as cancellation:
            secondary = cancellation
        assert child.done()
    store = PermissionStore(str(tmp_path / "paired-permissions.db"))
    await store.initialize()
    original = store.log_decision
    async def original_audit_then_fence(*args, **kwargs):
        await original(*args, **kwargs)
        raise primary
    async def withdrawal_fence(*args):
        raise secondary
    monkeypatch.setattr(store, "log_decision", original_audit_then_fence)
    queue = ApprovalQueue(permission_store=store, on_request_withdrawn=withdrawal_fence)
    with pytest.raises(BaseException) as captured:
        await queue.request_approval(feature_name="ReflectionFeature", tool_name="self_modify",
                                     tool_args={}, timeout=0)
    def contains_original(error, target):
        return (error is target
                or (isinstance(error, BaseExceptionGroup)
                    and any(contains_original(member, target) for member in error.exceptions))
                or (error.__cause__ is not None and contains_original(error.__cause__, target)))
    assert contains_original(captured.value, primary), "Withdrawal erased the original native audit control"
    assert contains_original(captured.value, secondary), "Withdrawal lost its own original native control"
    assert is_execution_control_error(captured.value)
    assert not queue.pending_requests


@pytest.mark.parametrize("carrier", ["direct", "group", "timeout"])
async def test_public_decision_failure_settles_the_original_waiter_without_approval(
    native_tool_effect_agent, tmp_path, monkeypatch, carrier,
):
    import asyncio
    agent = await native_tool_effect_agent(did="did:test:public-decision-control")
    error = original_control(agent, carrier)
    store = PermissionStore(str(tmp_path / "public-decision.db"))
    await store.initialize()
    original = store.log_decision
    async def original_audit_then_fence(*args, **kwargs):
        await original(*args, **kwargs)
        raise error
    monkeypatch.setattr(store, "log_decision", original_audit_then_fence)
    queued, completed = asyncio.Event(), asyncio.Event()
    async def notify(request):
        queued.set()
    queue = ApprovalQueue(permission_store=store, on_request_added=notify)
    task = asyncio.create_task(queue.request_approval(
        feature_name="ReflectionFeature", tool_name="self_modify", tool_args={}))
    task.add_done_callback(lambda _task: completed.set())
    try:
        async with asyncio.timeout(5):
            await queued.wait()
        request = queue.pending_requests[0]
        with pytest.raises(type(error)) as observer:
            await queue.submit_decision(request.id, approved=True, scope="once")
        assert observer.value is error
        try:
            async with asyncio.timeout(1):
                await completed.wait()
        except TimeoutError:
            pytest.fail("Native decision failure stranded the original approval waiter")
        with pytest.raises(type(error)) as receiver:
            await task
        assert receiver.value is error, "Original waiter lost the decision's native control"
        assert not queue.pending_requests
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("boundary", ["request_added", "request_wait"])
async def test_mixed_real_cancellation_and_control_marks_the_original_waiter_gone(
    native_tool_effect_agent, monkeypatch, boundary,
):
    import asyncio
    agent = await native_tool_effect_agent(did="did:test:mixed-approval-control")
    authority = original_control(agent, "direct")
    child = asyncio.create_task(asyncio.Event().wait())
    await asyncio.sleep(0)
    child.cancel()
    try:
        await child
    except asyncio.CancelledError as cancellation:
        mixed = BaseExceptionGroup("Original cancelled child and authority", [cancellation, authority])
    assert child.done() and is_execution_control_error(mixed)
    async def reject(*args):
        raise mixed
    async def arm_wait(request):
        monkeypatch.setattr(request.resume_event, "wait", reject)
    queue = ApprovalQueue(on_request_added=reject if boundary == "request_added" else arm_wait)
    with pytest.raises(BaseExceptionGroup) as captured:
        await queue.request_approval(feature_name="ReflectionFeature", tool_name="self_modify", tool_args={})
    assert captured.value is mixed
    assert queue.pending_requests[0].status is ApprovalStatus.AWAITER_GONE, (
        "Mixed native cancellation/control left a dead approval waiter PENDING")


async def test_cancelled_public_decision_settles_its_original_waiter(
    tmp_path, monkeypatch,
):
    import asyncio
    store = PermissionStore(str(tmp_path / "cancelled-decision.db"))
    await store.initialize()
    original = store.log_decision
    queued, entered, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def original_audit_then_wait(*args, **kwargs):
        await original(*args, **kwargs)
        entered.set()
        await asyncio.Event().wait()
    async def notify(request):
        queued.set()
    monkeypatch.setattr(store, "log_decision", original_audit_then_wait)
    queue = ApprovalQueue(permission_store=store, on_request_added=notify)
    waiter = asyncio.create_task(queue.request_approval(
        feature_name="ReflectionFeature", tool_name="self_modify", tool_args={}))
    waiter.add_done_callback(lambda _task: completed.set())
    deciding = None
    try:
        async with asyncio.timeout(5):
            await queued.wait()
            deciding = asyncio.create_task(queue.submit_decision(queue.pending_requests[0].id, True))
            await entered.wait()
        deciding.cancel()
        with pytest.raises(asyncio.CancelledError):
            await deciding
        async with asyncio.timeout(5):
            await completed.wait()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not queue.pending_requests
    finally:
        for task in [waiter, deciding]:
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*[t for t in [waiter, deciding] if t is not None], return_exceptions=True)


async def test_sweep_settles_an_original_live_waiter_with_timeout_provenance():
    import asyncio
    queued, completed = asyncio.Event(), asyncio.Event()
    async def notify(request):
        queued.set()
    queue = ApprovalQueue(on_request_added=notify)
    waiter = asyncio.create_task(queue.request_approval(
        feature_name="ReflectionFeature", tool_name="self_modify", tool_args={}))
    waiter.add_done_callback(lambda _task: completed.set())
    try:
        async with asyncio.timeout(5):
            await queued.wait()
        assert await queue.sweep_stale(older_than_seconds=0) == 1
        try:
            async with asyncio.timeout(1):
                await completed.wait()
        except TimeoutError:
            pytest.fail("Sweep removed the request but stranded its original live waiter")
        assert await waiter == (False, "timeout"), "Sweep mislabeled expiry as a user decision"
    finally:
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


async def test_handled_foreign_control_does_not_replace_a_successful_approval(native_tool_effect_agent):
    agent = await native_tool_effect_agent(did="did:test:ambient-approval-control")
    foreign = original_control(agent, "direct")
    async def approve(request):
        await queue.submit_decision(request.id, approved=True, scope="once")
    queue = ApprovalQueue(on_request_added=approve)
    try:
        raise foreign
    except ExecutionAuthorityError:
        result = await queue.request_approval(feature_name="ReflectionFeature", tool_name="self_modify", tool_args={})
    assert result == (True, "once"), "Ambient handled control replaced an unrelated successful approval"


@pytest.mark.parametrize("arrival", ["during_withdrawal", "after_withdrawal"])
async def test_native_decision_failure_during_withdrawal_reaches_the_original_waiter(
    native_tool_effect_agent, tmp_path, monkeypatch, arrival,
):
    import asyncio
    agent = await native_tool_effect_agent(did="did:test:late-decision-control")
    error = original_control(agent, "direct")
    store = PermissionStore(str(tmp_path / "late-decision.db"))
    await store.initialize()
    original = store.log_decision
    entered, release, withdrawing, withdrawn = (asyncio.Event() for _ in range(4))
    async def original_audit_then_wait(*args, **kwargs):
        await original(*args, **kwargs)
        if kwargs.get("decision") == "user_approved":
            entered.set()
            await release.wait()
            raise error
    monkeypatch.setattr(store, "log_decision", original_audit_then_wait)
    submissions = []
    async def notify(request):
        submissions.append(asyncio.create_task(queue.submit_decision(request.id, True)))
        await entered.wait()
    async def withdraw(*args):
        withdrawing.set()
        if arrival == "during_withdrawal":
            await withdrawn.wait()
    queue = ApprovalQueue(permission_store=store, on_request_added=notify, on_request_withdrawn=withdraw)
    waiter = asyncio.create_task(queue.request_approval(
        feature_name="ReflectionFeature", tool_name="self_modify", tool_args={}, timeout=0))
    try:
        async with asyncio.timeout(5):
            await withdrawing.wait()
        if arrival == "after_withdrawal":
            assert not waiter.done(), "Original waiter completed before its claimed decision settled"
        release.set()
        with pytest.raises(type(error)) as sender:
            await submissions[0]
        assert sender.value is error
        withdrawn.set()
        with pytest.raises(type(error)) as receiver:
            await waiter
        assert receiver.value is error, "Late native decision failure missed the original waiter"
        assert not queue.pending_requests
    finally:
        release.set()
        withdrawn.set()
        for task in [waiter, *submissions]:
            if not task.done():
                task.cancel()
        await asyncio.gather(waiter, *submissions, return_exceptions=True)


@pytest.mark.parametrize("carrier", ["direct", "group", "timeout"])
async def test_actual_security_permission_listing_preserves_native_control(
    native_tool_effect_agent, tmp_path, monkeypatch, carrier,
):
    from kestrel_sovereign.features.security.feature import SecurityFeature
    agent = await native_tool_effect_agent(did="did:test:security-list-control")
    store = PermissionStore(str(tmp_path / "security-list.db"))
    await store.initialize()
    security = SecurityFeature(agent)
    security.permission_store = store
    error = original_control(agent, carrier)
    read = store.get_permission_tree
    async def original_tree_then_fence():
        await read()
        raise error
    monkeypatch.setattr(store, "get_permission_tree", original_tree_then_fence)
    with pytest.raises(type(error)) as captured:
        await security.list_permissions()
    assert captured.value is error, "Security listing erased original native control"
